"""Data contract for request telemetry, with PII-safe failure reporting.

The table this schema describes is the one a serving team actually keeps: one row per completed
request, used for latency dashboards, capacity planning and the prompt-length drift monitor. The
schema is **strict**, so the contract is not only "these columns are well formed" but also "these
and no others" -- a column holding prompt or completion text cannot be added to this table by
accident, because validation fails the moment it appears.

Two cross-column checks encode facts the engine guarantees and a broken exporter would not:
cached prompt tokens can never exceed the prompt, and time to first token can never exceed total
latency. Failures are summarised as (column, check, count); the offending values are never copied
into the exception, since in a telemetry pipeline the offending value is exactly the thing that
should not be in a log line.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera.typing.pandas import Series

if TYPE_CHECKING:  # pragma: no cover - import only for type checking, never at runtime
    from paged_kv_serving.engine.engine import RequestOutput

__all__ = [
    "DataContractError",
    "RequestRecordSchema",
    "make_synthetic_records",
    "records_from_outputs",
    "validate_records",
]

FINISH_REASONS = ("length", "stop", "context_limit", "aborted")
SLO_CLASSES = ("interactive", "batch")


class DataContractError(ValueError):
    """A frame violated its contract. ``failures`` holds (column, check, count) summaries."""

    def __init__(self, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"data contract violated: {total} failure(s) across {len(failures)} check(s)"
        )
        self.failures = failures


class RequestRecordSchema(pa.DataFrameModel):
    """One row per completed request. No prompt or completion text is allowed in this table.

    Types are checked strictly and without coercion: ``arrival_time`` must already be
    timezone-aware UTC at nanosecond resolution, so a naive local timestamp is rejected rather
    than silently reinterpreted as UTC and quietly shifting every latency percentile.
    """

    request_id: Series[str] = pa.Field(unique=True, str_matches=r"^[0-9a-f]{32}$")
    key_id: Series[str] = pa.Field(str_matches=r"^[0-9a-z_-]{1,64}$")
    prompt_sha256: Series[str] = pa.Field(str_matches=r"^[0-9a-f]{64}$")
    slo_class: Series[str] = pa.Field(isin=list(SLO_CLASSES))
    finish_reason: Series[str] = pa.Field(isin=list(FINISH_REASONS))
    prompt_tokens: Series[int] = pa.Field(ge=1, le=1_000_000)
    completion_tokens: Series[int] = pa.Field(ge=0, le=1_000_000)
    cached_prompt_tokens: Series[int] = pa.Field(ge=0, le=1_000_000)
    preemptions: Series[int] = pa.Field(ge=0, le=100_000)
    ttft_seconds: Series[float] = pa.Field(ge=0.0, le=3_600.0)
    latency_seconds: Series[float] = pa.Field(ge=0.0, le=3_600.0)
    arrival_time: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "ns", "tz": "UTC"})

    class Config:
        strict = True
        name = "request_records"

    @pa.dataframe_check
    @classmethod
    def cached_tokens_fit_the_prompt(cls, frame: pd.DataFrame) -> Series[bool]:
        """A request cannot reuse more cached prompt tokens than its prompt has."""
        return Series[bool](frame["cached_prompt_tokens"] <= frame["prompt_tokens"])

    @pa.dataframe_check
    @classmethod
    def first_token_precedes_completion(cls, frame: pd.DataFrame) -> Series[bool]:
        """Time to first token is part of total latency, never larger than it."""
        return Series[bool](frame["ttft_seconds"] <= frame["latency_seconds"] + 1e-9)


def validate_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate lazily (all failures at once) and raise a summarized DataContractError."""
    try:
        return RequestRecordSchema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases[["column", "check"]].astype(str)
        counts = cases.value_counts().sort_index()
        failures: list[dict[str, str | int]] = [
            {"column": column, "check": check, "count": int(count)}
            for (column, check), count in counts.items()
        ]
        raise DataContractError(failures) from None


def records_from_outputs(
    outputs: Iterable[RequestOutput], prompt_hashes: Sequence[str]
) -> pd.DataFrame:
    """Turn engine ``RequestOutput`` objects into a contract-shaped telemetry frame.

    ``prompt_hashes`` is passed separately on purpose: an engine output does not carry the prompt,
    so nothing here can reach back for the text even if a future caller wanted it to.
    """
    rows: list[dict[str, object]] = []
    for index, output in enumerate(outputs):
        sample = output.samples[0]
        rows.append(
            {
                "request_id": _normalise_id(output.request_id),
                "key_id": output.key_id,
                "prompt_sha256": prompt_hashes[index],
                "slo_class": output.slo_class.label,
                "finish_reason": sample.finish_reason.value,
                "prompt_tokens": int(output.prompt_tokens),
                "completion_tokens": int(output.output_tokens),
                "cached_prompt_tokens": int(output.cached_prompt_tokens),
                "preemptions": int(output.preemptions),
                "ttft_seconds": float(output.ttft),
                "latency_seconds": float(output.latency),
                "arrival_time": pd.Timestamp("2026-01-01", tz="UTC"),
            }
        )
    frame = pd.DataFrame(rows, columns=list(_COLUMNS))
    return _with_dtypes(frame)


def make_synthetic_records(n: int, *, seed: int) -> pd.DataFrame:
    """Deterministic synthetic telemetry that satisfies :class:`RequestRecordSchema`."""
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    prompt_tokens = rng.integers(8, 512, size=n)
    cached = (prompt_tokens * rng.random(n) * 0.5).astype(np.int64)
    ttft = np.round(rng.gamma(2.0, 0.05, size=n), 6)
    completion = rng.integers(1, 128, size=n)
    latency = np.round(ttft + completion * rng.gamma(2.0, 0.005, size=n), 6)
    frame = pd.DataFrame(
        {
            "request_id": [f"{index:032x}" for index in range(n)],
            "key_id": [f"key-{value}" for value in rng.integers(0, 4, size=n)],
            "prompt_sha256": [f"{(index * 2654435761) % (16**64):064x}" for index in range(n)],
            "slo_class": rng.choice(SLO_CLASSES, size=n),
            "finish_reason": rng.choice(FINISH_REASONS, size=n),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "cached_prompt_tokens": cached,
            "preemptions": rng.integers(0, 3, size=n),
            "ttft_seconds": ttft,
            "latency_seconds": latency,
            "arrival_time": (
                pd.Timestamp("2026-01-01", tz="UTC")
                + pd.to_timedelta(rng.integers(0, 86_400, size=n), unit="s")
            ).astype("datetime64[ns, UTC]"),
        }
    )
    return _with_dtypes(frame)


def _with_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the contract's integer, float and timestamp dtypes (strings stay object)."""
    typed = frame.astype(_DTYPES)
    typed["arrival_time"] = pd.to_datetime(typed["arrival_time"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    return typed


def _normalise_id(request_id: str) -> str:
    """Engine ids are already 32 hex characters; anything else is hashed into that shape."""
    if len(request_id) == 32 and all(char in "0123456789abcdef" for char in request_id):
        return request_id
    import hashlib  # noqa: PLC0415 - only needed for non-uuid ids from tests and benchmarks

    return hashlib.blake2b(request_id.encode("utf-8"), digest_size=16).hexdigest()


_COLUMNS = (
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
)
_DTYPES: dict[str, str] = {
    "request_id": "object",
    "key_id": "object",
    "prompt_sha256": "object",
    "slo_class": "object",
    "finish_reason": "object",
    "prompt_tokens": "int64",
    "completion_tokens": "int64",
    "cached_prompt_tokens": "int64",
    "preemptions": "int64",
    "ttft_seconds": "float64",
    "latency_seconds": "float64",
}
