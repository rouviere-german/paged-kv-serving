"""Open-loop load generation over HTTP, for the served API rather than the in-process engine.

:mod:`loadgen` drives the engine directly, which is what the benchmarks need: one thread, no
sockets, numbers that describe scheduling and nothing else. This module drives the *server* --
authentication, rate limits, SSE streaming and all -- the way a real client population would, and
is what the compose file's ``loadgen`` service runs against the ``server`` service.

The same rule applies: arrival times are fixed before the run starts and every request is launched
as its own task at its scheduled moment, whether or not earlier ones have answered. A slow server
therefore faces the offered load instead of quietly receiving less of it (coordinated omission),
and time to first token is measured from the *scheduled* arrival, so time spent queueing inside
the client's own connection pool is charged to the server too.

Every outcome is kept: a 503 from admission control, a 429 from a per-key limit and a stream that
ended in a timeout event are all rows in the result, not silently missing samples.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from paged_kv_serving.engine.loadgen import burst_arrivals, percentile, poisson_arrivals

__all__ = [
    "HttpOutcome",
    "HttpRequestPlan",
    "HttpTrafficClass",
    "plan_http_workload",
    "run_http_open_loop",
    "summarize_http",
]


@dataclass(frozen=True, slots=True)
class HttpTrafficClass:
    """One population of callers: a key, an SLO class and the shape of what they ask for."""

    name: str
    api_key: str
    slo_class: str = "interactive"
    share: float = 1.0
    prompt_words: tuple[int, int] = (4, 24)
    max_tokens: tuple[int, int] = (4, 16)
    deadline_ms: int | None = None

    def __post_init__(self) -> None:
        if self.share <= 0:
            raise ValueError("share must be > 0")
        if self.slo_class not in {"interactive", "batch"}:
            raise ValueError("slo_class must be interactive or batch")
        if not 1 <= self.prompt_words[0] <= self.prompt_words[1]:
            raise ValueError("prompt_words must be a non-empty (low, high) range")
        if not 1 <= self.max_tokens[0] <= self.max_tokens[1]:
            raise ValueError("max_tokens must be a non-empty (low, high) range")


@dataclass(frozen=True, slots=True)
class HttpRequestPlan:
    """One scheduled request: when it is sent, as whom, and with which body."""

    offset: float
    traffic: str
    api_key: str
    body: dict[str, Any]


@dataclass(slots=True)
class HttpOutcome:
    """What happened to one planned request."""

    traffic: str
    status: int
    scheduled: float
    error: str | None = None
    first_token: float | None = None
    finished: float | None = None
    tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None and self.finished is not None

    @property
    def ttft(self) -> float | None:
        return None if self.first_token is None else self.first_token - self.scheduled

    @property
    def latency(self) -> float | None:
        return None if self.finished is None else self.finished - self.scheduled


_WORDS = (
    "batch", "block", "cache", "decode", "prefill", "token", "queue", "deadline", "memory",
    "table", "prefix", "stream", "kernel", "budget", "latency", "goodput", "sequence", "page",
)  # fmt: skip


def plan_http_workload(
    classes: Sequence[HttpTrafficClass],
    *,
    rate: float,
    count: int,
    seed: int = 0,
    burst: tuple[float, float, float] | None = None,
    system_prompt: str = "",
) -> list[HttpRequestPlan]:
    """Draw arrival offsets and request bodies from a seeded generator.

    ``burst`` is ``(burst_rate, start, duration)`` in requests per second and seconds. Prompts
    are made of a fixed vocabulary of words so their token counts are predictable, optionally
    behind a shared ``system_prompt`` so the prefix cache has something to reuse.
    """
    if not classes:
        raise ValueError("at least one traffic class is required")
    rng = np.random.default_rng(seed)
    if burst is None:
        offsets = poisson_arrivals(rate, count, rng)
    else:
        burst_rate, start, duration = burst
        offsets = burst_arrivals(
            base_rate=rate,
            burst_rate=burst_rate,
            duration=count / rate,
            burst_start=start,
            burst_duration=duration,
            rng=rng,
        )[:count]
    shares = np.asarray([item.share for item in classes], dtype=np.float64)
    shares = shares / shares.sum()
    plans: list[HttpRequestPlan] = []
    for offset in offsets:
        traffic = classes[int(rng.choice(len(classes), p=shares))]
        words = int(rng.integers(traffic.prompt_words[0], traffic.prompt_words[1] + 1))
        prompt = " ".join(str(rng.choice(_WORDS)) for _ in range(words))
        body: dict[str, Any] = {
            "prompt": system_prompt + prompt,
            "max_tokens": int(rng.integers(traffic.max_tokens[0], traffic.max_tokens[1] + 1)),
            "stream": True,
            "slo_class": traffic.slo_class,
            "seed": int(rng.integers(0, 2**31 - 1)),
        }
        if traffic.deadline_ms is not None:
            body["deadline_ms"] = traffic.deadline_ms
        plans.append(HttpRequestPlan(float(offset), traffic.name, traffic.api_key, body))
    return plans


async def _one(
    client: httpx.AsyncClient,
    plan: HttpRequestPlan,
    scheduled: float,
    clock: Callable[[], float],
) -> HttpOutcome:
    outcome = HttpOutcome(traffic=plan.traffic, status=0, scheduled=scheduled)
    try:
        async with client.stream(
            "POST", "/v1/completions", json=plan.body, headers={"X-API-Key": plan.api_key}
        ) as response:
            outcome.status = response.status_code
            if response.status_code != 200:
                raw = await response.aread()
                outcome.error = _error_code(raw) or f"http_{response.status_code}"
                return outcome
            event = ""
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    if event == "token":
                        outcome.tokens += 1
                        if outcome.first_token is None:
                            outcome.first_token = clock()
                    elif event == "error":
                        outcome.error = _error_field(line[5:].strip())
                    elif event == "summary":
                        outcome.finished = clock()
    except httpx.HTTPError as exc:
        outcome.error = type(exc).__name__
    return outcome


def _error_code(raw: bytes) -> str | None:
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return str(code) if code is not None else None


def _error_field(data: str) -> str:
    try:
        payload = json.loads(data)
    except ValueError:
        return "stream_error"
    return (
        str(payload.get("error", "stream_error")) if isinstance(payload, dict) else "stream_error"
    )


async def run_http_open_loop(
    client: httpx.AsyncClient,
    plans: Sequence[HttpRequestPlan],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> list[HttpOutcome]:
    """Launch every planned request at its offset from now, concurrently, and collect outcomes.

    Launching never waits for an earlier response: each request is its own task, created when
    its moment arrives.
    """
    ordered = sorted(plans, key=lambda plan: plan.offset)
    start = clock()
    tasks: list[asyncio.Task[HttpOutcome]] = []
    for plan in ordered:
        delay = start + plan.offset - clock()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(_one(client, plan, start + plan.offset, clock)))
    return list(await asyncio.gather(*tasks))


def summarize_http(outcomes: Sequence[HttpOutcome], *, wall_seconds: float) -> dict[str, Any]:
    """Per traffic class: completions, rejections by code, TTFT and latency percentiles."""
    groups: dict[str, list[HttpOutcome]] = {}
    for outcome in outcomes:
        groups.setdefault(outcome.traffic, []).append(outcome)
    summary: dict[str, Any] = {"wall_seconds": round(wall_seconds, 3), "classes": {}}
    for name, group in sorted(groups.items()):
        completed = [item for item in group if item.ok]
        errors: dict[str, int] = {}
        for item in group:
            if not item.ok:
                code = item.error or f"http_{item.status}"
                errors[code] = errors.get(code, 0) + 1
        ttfts = [item.ttft for item in completed if item.ttft is not None]
        latencies = [item.latency for item in completed if item.latency is not None]
        summary["classes"][name] = {
            "sent": len(group),
            "completed": len(completed),
            "errors": dict(sorted(errors.items())),
            "tokens": sum(item.tokens for item in completed),
            "ttft_p50": round(percentile(ttfts, 50), 4),
            "ttft_p99": round(percentile(ttfts, 99), 4),
            "latency_p50": round(percentile(latencies, 50), 4),
            "latency_p99": round(percentile(latencies, 99), 4),
        }
    return summary
