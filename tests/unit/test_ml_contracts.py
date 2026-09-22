"""The request-telemetry data contract: strict columns, cross-column facts, PII-safe failures."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from paged_kv_serving.engine.config import SloClass
from paged_kv_serving.engine.engine import RequestOutput, SampleOutput
from paged_kv_serving.engine.sequence import FinishReason
from paged_kv_serving.ml.contracts import (
    DataContractError,
    make_synthetic_records,
    records_from_outputs,
    validate_records,
)

pytestmark = pytest.mark.unit


def failures_for(frame: pd.DataFrame) -> DataContractError:
    with pytest.raises(DataContractError) as info:
        validate_records(frame)
    return info.value


def test_synthetic_records_satisfy_the_contract() -> None:
    assert len(validate_records(make_synthetic_records(200, seed=1))) == 200


@settings(max_examples=20)
@given(n=st.integers(min_value=1, max_value=40), seed=st.integers(min_value=0, max_value=10_000))
def test_generator_always_satisfies_the_contract(n: int, seed: int) -> None:
    validate_records(make_synthetic_records(n, seed=seed))


def test_violations_are_summarized_without_the_offending_values() -> None:
    frame = make_synthetic_records(20, seed=2)
    frame.loc[3, "ttft_seconds"] = -123456.789
    frame.loc[4, "key_id"] = "Robert'); DROP TABLE"
    error = failures_for(frame)
    assert {"ttft_seconds", "key_id"} <= {str(item["column"]) for item in error.failures}
    assert "123456" not in str(error)
    assert "123456" not in repr(error.failures)
    assert "DROP TABLE" not in repr(error.failures)


def test_a_prompt_text_column_cannot_be_added_to_the_telemetry_table() -> None:
    """Strict schema: a column holding prompt text fails validation the moment it appears."""
    frame = make_synthetic_records(5, seed=3)
    frame["prompt"] = "my password is hunter2"
    error = failures_for(frame)
    assert error.failures
    assert "hunter2" not in repr(error.failures)


def test_missing_columns_are_rejected() -> None:
    frame = make_synthetic_records(5, seed=3).drop(columns=["prompt_sha256"])
    assert failures_for(frame).failures


def test_timezone_naive_timestamps_are_rejected() -> None:
    frame = make_synthetic_records(5, seed=4)
    frame["arrival_time"] = frame["arrival_time"].dt.tz_localize(None)
    assert failures_for(frame).failures


def test_duplicate_request_ids_are_rejected() -> None:
    frame = make_synthetic_records(5, seed=5)
    frame.loc[1, "request_id"] = frame.loc[0, "request_id"]
    assert any(item["column"] == "request_id" for item in failures_for(frame).failures)


def test_cached_tokens_cannot_exceed_the_prompt() -> None:
    frame = make_synthetic_records(5, seed=6)
    frame.loc[0, "cached_prompt_tokens"] = int(frame["prompt_tokens"].iloc[0]) + 1
    assert failures_for(frame).failures


def test_time_to_first_token_cannot_exceed_latency() -> None:
    frame = make_synthetic_records(5, seed=7)
    frame.loc[0, "ttft_seconds"] = float(frame["latency_seconds"].iloc[0]) + 1.0
    assert failures_for(frame).failures


def test_unknown_finish_reasons_and_classes_are_rejected() -> None:
    frame = make_synthetic_records(5, seed=8)
    frame.loc[0, "finish_reason"] = "exploded"
    frame.loc[1, "slo_class"] = "platinum"
    columns = {str(item["column"]) for item in failures_for(frame).failures}
    assert {"finish_reason", "slo_class"} <= columns


def test_generator_rejects_empty_requests() -> None:
    with pytest.raises(ValueError, match="n must be"):
        make_synthetic_records(0, seed=1)


def _output(request_id: str, *, cached: int = 0) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        key_id="tenant-a",
        slo_class=SloClass.INTERACTIVE,
        samples=(SampleOutput(0, (5, 6, 7), FinishReason.LENGTH, preemptions=1),),
        prompt_tokens=24,
        cached_prompt_tokens=cached,
        arrival_time=10.0,
        first_token_time=10.2,
        finish_time=10.5,
        deadline=11.0,
    )


def test_engine_outputs_convert_into_valid_records() -> None:
    uuid_like = "0123456789abcdef0123456789abcdef"
    frame = records_from_outputs(
        [_output(uuid_like, cached=16), _output("req-0001")], ["a" * 64, "b" * 64]
    )
    validated = validate_records(frame)
    assert validated["request_id"].iloc[0] == uuid_like
    # Non-uuid ids (tests, benchmarks) are hashed into the contract's id shape.
    assert len(str(validated["request_id"].iloc[1])) == 32
    assert list(validated["completion_tokens"]) == [3, 3]
    assert list(validated["preemptions"]) == [1, 1]
    assert "prompt" not in validated.columns
