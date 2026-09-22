"""A simulated clock and executor for checking scheduling decisions without a stopwatch.

Whether admission control "works" is a question about the scheduler's decisions, not about how
fast the machine running the tests happens to be. :class:`SimulatedExecutor` replaces the model
runner with one that consumes exactly the time the latency model predicts for a batch and advances
a :class:`ManualClock` by it. Everything else -- the scheduler, the block allocator, the prefix
cache and admission control -- is the real code, so an engine built with :func:`simulated_engine`
behaves deterministically and "no admitted request misses its deadline" becomes a checkable claim.

The executor returns deterministic one-hot logits derived from the last token and the context
length, so greedy sampling still runs and sequences still grow; only the arithmetic is skipped.
The adversarial tests and the deterministic admission benchmark both use this module.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from paged_kv_serving.engine.admission import LatencyModel
from paged_kv_serving.engine.config import EngineConfig
from paged_kv_serving.engine.engine import Engine, RequestOutput
from paged_kv_serving.engine.keys import EntitlementGuard
from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.runner import ExecutionBatch
from paged_kv_serving.engine.sequence import Request

__all__ = ["ManualClock", "SimulatedExecutor", "drain_on_clock", "simulated_engine"]

FloatArray = npt.NDArray[np.float32]


class ManualClock:
    """A clock that only moves when told to."""

    __slots__ = ("now",)

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a clock cannot move backwards")
        self.now += seconds


class SimulatedExecutor:
    """Consumes exactly the modelled time of each batch and returns deterministic logits."""

    __slots__ = ("batches", "clock", "latency", "vocab")

    def __init__(self, clock: ManualClock, latency: LatencyModel, vocab: int) -> None:
        if vocab < 1:
            raise ValueError("vocab must be >= 1")
        self.clock = clock
        self.latency = latency
        self.vocab = vocab
        self.batches = 0

    def execute(self, batch: ExecutionBatch) -> dict[str, FloatArray]:
        self.batches += 1
        self.clock.advance(self.latency.predict(batch.num_tokens))
        logits: dict[str, FloatArray] = {}
        for entry in batch.entries:
            if entry.needs_logits:
                row = np.zeros(self.vocab, dtype=np.float32)
                row[(entry.token_ids[-1] + entry.context_len) % self.vocab] = 1.0
                logits[entry.seq_id] = row
        return logits


def simulated_engine(
    model: LanguageModel,
    config: EngineConfig,
    *,
    seconds_per_token: float = 1e-3,
    safety_factor: float = 1.0,
    entitlements: EntitlementGuard | None = None,
) -> tuple[Engine, ManualClock]:
    """A real engine with a simulated forward pass costing ``seconds_per_token`` per token.

    A purely proportional cost (no intercept) is the condition under which the admission test's
    iteration count is exact (docs/adr/0004-iteration-counting-admission.md). The latency model
    has its EWMA correction switched off, because the simulated cost *is* the model.
    """
    clock = ManualClock()
    latency = LatencyModel(0.0, seconds_per_token, ewma_alpha=0.0)
    if entitlements is not None:
        entitlements.clock = clock
    engine = Engine(
        model,
        dataclasses.replace(config, admission_safety_factor=safety_factor),
        executor=SimulatedExecutor(clock, latency, vocab=model.config.vocab_size),
        clock=clock,
        latency_model=latency,
        entitlements=entitlements,
    )
    return engine, clock


def drain_on_clock(
    engine: Engine,
    clock: ManualClock,
    requests: Iterable[Request],
    *,
    max_iterations: int = 100_000,
) -> list[RequestOutput]:
    """Submit each request when the manual clock reaches its arrival time, then run to idle.

    Open loop, like :func:`~paged_kv_serving.engine.loadgen.run_open_loop`: arrivals are fixed
    in advance and never wait for responses. When the engine is idle the clock jumps straight to
    the next arrival instead of sleeping.
    """
    outputs: list[RequestOutput] = []
    pending = sorted(requests, key=lambda item: (item.arrival_time, item.request_id))
    index = 0
    for _ in range(max_iterations):
        while index < len(pending) and pending[index].arrival_time <= clock.now:
            engine.submit(pending[index])
            index += 1
        if engine.has_work:
            outputs.extend(engine.step().finished)
            continue
        if index >= len(pending):
            return outputs
        clock.now = pending[index].arrival_time
    raise RuntimeError("the simulated engine did not drain")
