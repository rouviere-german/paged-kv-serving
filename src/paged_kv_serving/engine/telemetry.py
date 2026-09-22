"""Request telemetry: one JSONL row per finished request, validated before it is written.

This is the production path for the data contract in :mod:`paged_kv_serving.ml.contracts`. The
service hands every finished request to :class:`TelemetryExporter`; rows are built by
:func:`~paged_kv_serving.ml.contracts.records_from_outputs` and must pass the strict schema --
these columns and no others, cached tokens within the prompt, TTFT within latency -- before a byte
reaches disk. A batch that fails is dropped whole and reported as (column, check, count), never
with the offending values, so a bug that put text into a row cannot leak it into the file or the
log either.

What a row holds: request id, key id, a SHA-256 of the prompt, class, finish reason, token counts,
preemptions, TTFT, latency and the wall-clock arrival time in UTC. No prompt or completion text.

Rows are appended to ``<data_dir>/telemetry/requests.jsonl``. When the file passes ``max_bytes``
it is renamed to ``requests.jsonl.1`` (replacing the previous one), so the export never holds more
than about twice the cap -- the container keeps its data directory on a small tmpfs.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from paged_kv_serving.core.fileio import file_lock
from paged_kv_serving.core.logging import get_logger
from paged_kv_serving.engine.engine import RequestOutput
from paged_kv_serving.ml.contracts import DataContractError, records_from_outputs, validate_records

__all__ = ["TelemetryExporter"]

_log = get_logger("paged_kv_serving.telemetry")


@dataclass(frozen=True, slots=True)
class _Finished:
    output: RequestOutput
    prompt_sha256: str
    arrival: datetime


class TelemetryExporter:
    """Buffers finished requests and appends contract-validated rows in batches."""

    __slots__ = (
        "_pending",
        "batch_size",
        "engine_clock",
        "max_bytes",
        "path",
        "rejected_batches",
        "wall_clock",
        "written_rows",
    )

    def __init__(
        self,
        path: Path,
        *,
        engine_clock: Callable[[], float],
        wall_clock: Callable[[], float] = time.time,
        batch_size: int = 64,
        max_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        self.path = path
        self.engine_clock = engine_clock
        self.wall_clock = wall_clock
        self.batch_size = batch_size
        self.max_bytes = max_bytes
        self.rejected_batches = 0
        self.written_rows = 0
        self._pending: list[_Finished] = []

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def batch_ready(self) -> bool:
        return len(self._pending) >= self.batch_size

    def record(self, output: RequestOutput, prompt_sha256: str) -> None:
        """Queue one finished request. Its engine-clock arrival is converted to UTC now."""
        seconds_ago = self.engine_clock() - output.arrival_time
        arrival = datetime.fromtimestamp(self.wall_clock() - seconds_ago, tz=UTC)
        self._pending.append(_Finished(output, prompt_sha256, arrival))

    def take(self) -> list[_Finished]:
        """Hand over everything queued so far (so it can be written off the event loop)."""
        batch, self._pending = self._pending, []
        return batch

    def flush(self) -> int:
        """Validate and write everything queued. Returns the number of rows written."""
        return self.write(self.take())

    def write(self, batch: list[_Finished]) -> int:
        """Validate ``batch`` against the contract and append it. Returns rows written."""
        if not batch:
            return 0
        frame = records_from_outputs(
            [item.output for item in batch],
            [item.prompt_sha256 for item in batch],
            arrival_times=[pd.Timestamp(item.arrival) for item in batch],
        )
        try:
            validated = validate_records(frame)
        except DataContractError as exc:
            self.rejected_batches += 1
            _log.warning("telemetry_rejected", rows=len(batch), failures=exc.failures)
            return 0
        lines = []
        for record in validated.to_dict(orient="records"):
            record["arrival_time"] = record["arrival_time"].isoformat()
            lines.append(json.dumps(record) + "\n")
        payload = "".join(lines)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path):
            if self.path.is_file() and self.path.stat().st_size >= self.max_bytes:
                os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
        self.written_rows += len(batch)
        return len(batch)
