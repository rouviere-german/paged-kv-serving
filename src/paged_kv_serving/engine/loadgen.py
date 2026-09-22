"""Open-loop load generation and the statistics the benchmarks report.

The mistake this module exists to avoid is coordinated omission. A closed-loop client sends the
next request only after the previous one comes back, so when the server slows down the client
quietly *reduces its own offered load* and the recorded latencies describe a system that was never
actually overloaded. Here every arrival time is drawn before the run starts and a request is
submitted when its scheduled moment arrives, whatever the server is doing, and latency is measured
from that scheduled arrival -- including the time the request spent waiting to be accepted at all.
A request rejected by admission control is counted, not dropped from the statistics.

Arrivals come from a Poisson process (independent exponential gaps), optionally with a burst
window at a higher rate, both drawn from a seeded generator so a run is reproducible.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from paged_kv_serving.engine.admission import RejectionReason
from paged_kv_serving.engine.config import SloClass
from paged_kv_serving.engine.engine import Engine, RequestOutput
from paged_kv_serving.engine.sequence import Request

__all__ = [
    "LoadResult",
    "WorkloadSpec",
    "build_workload",
    "burst_arrivals",
    "percentile",
    "poisson_arrivals",
    "run_open_loop",
    "summarize",
]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. Returns 0.0 for an empty sample."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def poisson_arrivals(rate: float, count: int, rng: np.random.Generator) -> list[float]:
    """``count`` arrival times of a Poisson process of ``rate`` per second, starting near zero."""
    if rate <= 0:
        raise ValueError("rate must be > 0")
    if count < 0:
        raise ValueError("count must be >= 0")
    gaps = rng.exponential(1.0 / rate, size=count)
    return [float(value) for value in np.cumsum(gaps)]


def burst_arrivals(
    *,
    base_rate: float,
    burst_rate: float,
    duration: float,
    burst_start: float,
    burst_duration: float,
    rng: np.random.Generator,
) -> list[float]:
    """Poisson arrivals at ``base_rate`` with a window of ``burst_rate`` inside ``duration``.

    Drawn by thinning: sample at the higher rate throughout and keep an arrival outside the burst
    window with probability ``base_rate / burst_rate``. That keeps the two regimes independent of
    each other's random draws, so changing the burst does not reshuffle the baseline traffic.
    """
    if burst_rate < base_rate:
        raise ValueError("burst_rate must be at least base_rate")
    if duration <= 0 or burst_duration < 0:
        raise ValueError("duration must be > 0 and burst_duration >= 0")
    times: list[float] = []
    current = 0.0
    burst_end = burst_start + burst_duration
    keep = base_rate / burst_rate if burst_rate > 0 else 1.0
    while True:
        current += float(rng.exponential(1.0 / burst_rate))
        if current >= duration:
            return times
        if burst_start <= current < burst_end or rng.random() < keep:
            times.append(current)


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Shape of a synthetic workload, in tokens rather than text."""

    num_requests: int = 60
    rate: float = 20.0
    shared_prefix_tokens: int = 32
    prompt_tokens: tuple[int, int] = (16, 64)
    max_new_tokens: tuple[int, int] = (8, 32)
    interactive_fraction: float = 0.5
    interactive_deadline: float | None = 2.0
    batch_deadline: float | None = None
    shared_prefix_fraction: float = 0.5
    vocab_size: int = 256
    seed: int = 7
    keys: tuple[str, ...] = ("tenant-a",)
    burst: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.num_requests < 1:
            raise ValueError("num_requests must be >= 1")
        if not 0.0 <= self.interactive_fraction <= 1.0:
            raise ValueError("interactive_fraction must be in [0, 1]")
        if not 0.0 <= self.shared_prefix_fraction <= 1.0:
            raise ValueError("shared_prefix_fraction must be in [0, 1]")
        if self.prompt_tokens[0] < 1 or self.prompt_tokens[0] > self.prompt_tokens[1]:
            raise ValueError("prompt_tokens must be a non-empty (low, high) range")
        if self.max_new_tokens[0] < 1 or self.max_new_tokens[0] > self.max_new_tokens[1]:
            raise ValueError("max_new_tokens must be a non-empty (low, high) range")
        if not self.keys:
            raise ValueError("at least one key is required")


def build_workload(spec: WorkloadSpec) -> list[Request]:
    """Build requests with *relative* arrival times, ready for :func:`run_open_loop`.

    A configurable share of the prompts begin with the same block-aligned prefix, which is what
    gives the prefix cache something to hit; the rest are unique so the hit rate cannot be gamed.
    """
    from paged_kv_serving.engine.sequence import SamplingParams  # noqa: PLC0415 - avoids a cycle

    rng = np.random.default_rng(spec.seed)
    if spec.burst is None:
        arrivals = poisson_arrivals(spec.rate, spec.num_requests, rng)
    else:
        burst_rate, burst_start, burst_duration = spec.burst
        arrivals = burst_arrivals(
            base_rate=spec.rate,
            burst_rate=burst_rate,
            duration=spec.num_requests / spec.rate,
            burst_start=burst_start,
            burst_duration=burst_duration,
            rng=rng,
        )[: spec.num_requests]
    shared = [int(value) for value in rng.integers(1, spec.vocab_size, spec.shared_prefix_tokens)]

    requests: list[Request] = []
    for index, arrival in enumerate(arrivals):
        interactive = rng.random() < spec.interactive_fraction
        slo_class = SloClass.INTERACTIVE if interactive else SloClass.BATCH
        deadline = spec.interactive_deadline if interactive else spec.batch_deadline
        prompt_len = int(rng.integers(spec.prompt_tokens[0], spec.prompt_tokens[1] + 1))
        new_tokens = int(rng.integers(spec.max_new_tokens[0], spec.max_new_tokens[1] + 1))
        body = [int(value) for value in rng.integers(1, spec.vocab_size, prompt_len)]
        prompt = [*shared, *body] if rng.random() < spec.shared_prefix_fraction else body
        requests.append(
            Request(
                request_id=f"req-{index:04d}",
                prompt_token_ids=tuple(prompt),
                sampling=SamplingParams(max_new_tokens=new_tokens, seed=spec.seed + index),
                slo_class=slo_class,
                deadline_seconds=deadline,
                key_id=spec.keys[index % len(spec.keys)],
                arrival_time=float(arrival),
            )
        )
    return requests


@dataclass(slots=True)
class LoadResult:
    """Outcome of one open-loop run."""

    outputs: list[RequestOutput] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    submitted: int = 0
    wall_seconds: float = 0.0
    iterations: int = 0

    @property
    def rejected(self) -> int:
        return sum(self.rejections.values())

    @property
    def completed(self) -> int:
        return len(self.outputs)


def run_open_loop(
    engine: Engine,
    requests: Iterable[Request],
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    max_iterations: int = 1_000_000,
) -> LoadResult:
    """Drive ``engine`` with a pre-computed arrival schedule and collect every outcome.

    Single threaded on purpose: submission and iteration interleave in one loop, so a benchmark
    measures the engine rather than Python's thread scheduling. Arrival times are interpreted as
    offsets from the moment the run starts.
    """
    schedule = sorted(requests, key=lambda item: (item.arrival_time, item.request_id))
    result = LoadResult()
    start = clock()
    index = 0
    for _ in range(max_iterations):
        now = clock()
        while index < len(schedule) and start + schedule[index].arrival_time <= now:
            request = dataclasses.replace(
                schedule[index], arrival_time=start + schedule[index].arrival_time
            )
            decision = engine.submit(request)
            result.submitted += 1
            if not decision.admitted:
                reason = (decision.reason or RejectionReason.DEADLINE_INFEASIBLE).value
                result.rejections[reason] = result.rejections.get(reason, 0) + 1
            index += 1
        if engine.has_work:
            step = engine.step()
            result.outputs.extend(step.finished)
            result.iterations += 1
            continue
        if index >= len(schedule):
            break
        remaining = start + schedule[index].arrival_time - clock()
        if remaining > 0:
            sleep(remaining)
    else:  # pragma: no cover - only a runaway workload reaches the iteration cap
        raise RuntimeError("open-loop run exceeded max_iterations")
    result.wall_seconds = clock() - start
    return result


def summarize(result: LoadResult, *, label: str) -> dict[str, float | str]:
    """Latency, throughput and goodput of a run, as one flat row for the benchmark table."""
    ttfts = [output.ttft for output in result.outputs]
    tpots = [output.tpot for output in result.outputs if output.tpot > 0]
    latencies = [output.latency for output in result.outputs]
    output_tokens = sum(output.output_tokens for output in result.outputs)
    deadlined = [output for output in result.outputs if output.deadline is not None]
    on_time = sum(1 for output in deadlined if output.met_deadline)
    interactive = sum(1 for output in result.outputs if output.slo_class is SloClass.INTERACTIVE)
    interactive_ttfts = [
        output.ttft for output in result.outputs if output.slo_class is SloClass.INTERACTIVE
    ]
    wall = result.wall_seconds or 1e-9
    return {
        "label": label,
        "submitted": float(result.submitted),
        "completed": float(result.completed),
        "rejected": float(result.rejected),
        "rejection_rate": result.rejected / result.submitted if result.submitted else 0.0,
        "wall_seconds": round(wall, 4),
        "output_tokens": float(output_tokens),
        "tokens_per_second": round(output_tokens / wall, 2),
        "requests_per_second": round(result.completed / wall, 3),
        "ttft_p50": round(percentile(ttfts, 50), 4),
        "ttft_p99": round(percentile(ttfts, 99), 4),
        "tpot_p50": round(percentile(tpots, 50), 5),
        "tpot_p99": round(percentile(tpots, 99), 5),
        "latency_p50": round(percentile(latencies, 50), 4),
        "latency_p99": round(percentile(latencies, 99), 4),
        "interactive_completed": float(interactive),
        "interactive_ttft_p99": round(percentile(interactive_ttfts, 99), 4),
        "deadlined": float(len(deadlined)),
        "on_time": float(on_time),
        "slo_attainment": on_time / len(deadlined) if deadlined else 1.0,
        "goodput_per_second": round(on_time / wall, 3),
    }
