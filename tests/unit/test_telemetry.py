"""Request telemetry export: validated against the data contract before a byte is written.

Review gap: ``records_from_outputs`` and ``validate_records`` were only ever called from their own
tests, while the model card presented the contract as a control on request telemetry.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from paged_kv_serving.core.logging import configure_logging
from paged_kv_serving.engine.config import SloClass
from paged_kv_serving.engine.engine import RequestOutput, SampleOutput
from paged_kv_serving.engine.sequence import FinishReason
from paged_kv_serving.engine.telemetry import TelemetryExporter

pytestmark = pytest.mark.unit

WALL = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC).timestamp()


def output(index: int, *, cached: int = 0, arrival: float = 100.0) -> RequestOutput:
    return RequestOutput(
        request_id=f"{index:032x}",
        key_id="0123456789ab",
        slo_class=SloClass.BATCH if index % 2 else SloClass.INTERACTIVE,
        samples=(SampleOutput(0, (5, 6, 7), FinishReason.LENGTH, preemptions=0),),
        prompt_tokens=24,
        cached_prompt_tokens=cached,
        arrival_time=arrival,
        first_token_time=arrival + 0.2,
        finish_time=arrival + 0.5,
    )


def exporter(path: Path, *, batch_size: int = 64, max_bytes: int = 1 << 20) -> TelemetryExporter:
    # The engine clock reads 101.0 when the wall clock reads WALL, so arrival 100.0 is WALL - 1.
    return TelemetryExporter(
        path,
        engine_clock=lambda: 101.0,
        wall_clock=lambda: WALL,
        batch_size=batch_size,
        max_bytes=max_bytes,
    )


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_finished_requests_are_exported_as_contract_rows(tmp_path: Path) -> None:
    path = tmp_path / "telemetry" / "requests.jsonl"
    telemetry = exporter(path)
    for index in range(3):
        telemetry.record(output(index, cached=16), "a" * 64)
    assert telemetry.pending == 3
    assert telemetry.flush() == 3
    assert telemetry.pending == 0
    exported = rows(path)
    assert [row["request_id"] for row in exported] == [f"{i:032x}" for i in range(3)]
    first = exported[0]
    assert set(first) == {
        "request_id",
        "key_id",
        "prompt_sha256",
        "slo_class",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "cached_prompt_tokens",
        "preemptions",
        "ttft_seconds",
        "latency_seconds",
        "arrival_time",
    }
    # Engine timestamps are monotonic seconds; the row carries the wall-clock arrival in UTC.
    assert first["arrival_time"] == "2026-09-01T11:59:59+00:00"
    assert first["cached_prompt_tokens"] == 16
    assert first["completion_tokens"] == 3
    telemetry.record(output(7), "b" * 64)
    assert telemetry.flush() == 1
    assert len(rows(path)) == 4, "later flushes append"


@pytest.mark.usefixtures("restore_logging")
def test_a_batch_that_breaks_the_contract_is_dropped_whole_and_reported(tmp_path: Path) -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    path = tmp_path / "requests.jsonl"
    telemetry = exporter(path)
    telemetry.record(output(1), "a" * 64)
    telemetry.record(output(2, cached=99), "b" * 64)  # more cached tokens than the prompt has
    assert telemetry.flush() == 0
    assert not path.exists()
    assert telemetry.rejected_batches == 1
    logged = stream.getvalue()
    assert "telemetry_rejected" in logged
    assert "cached_tokens_fit_the_prompt" in logged


def test_the_file_is_rotated_once_it_reaches_its_size_cap(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    telemetry = exporter(path, max_bytes=600)
    for index in range(8):
        telemetry.record(output(index), "a" * 64)
        telemetry.flush()
    rotated = path.with_name("requests.jsonl.1")
    assert rotated.is_file()
    row_bytes = len(path.read_text(encoding="utf-8").splitlines()[0]) + 1
    # A file is rotated before the batch that would start past the cap, so it never holds more
    # than the cap plus one batch, and the export never more than two such files.
    assert path.stat().st_size < 600 + row_bytes
    assert rotated.stat().st_size < 600 + row_bytes
    kept = rows(rotated) + rows(path)
    assert len(kept) < 8, "the oldest rows were rotated away"
    identifiers = [str(row["request_id"]) for row in kept]
    assert identifiers == sorted(identifiers), "rotation keeps the rows in order"


def test_flushing_nothing_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    assert exporter(path).flush() == 0
    assert not path.exists()
    with pytest.raises(ValueError, match="batch_size"):
        exporter(path, batch_size=0)
    with pytest.raises(ValueError, match="max_bytes"):
        exporter(path, max_bytes=0)
