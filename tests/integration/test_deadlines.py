"""Deadline control, checked against a simulated clock rather than against a stopwatch.

Whether admission control "works" is a question about the scheduler's decisions, not about how
fast this laptop happens to be. So these tests replace the model runner with one that consumes
exactly the time the latency model predicts and advances a manual clock by it. The engine then
behaves deterministically, and the claim becomes checkable: with an accurate latency model, no
*admitted* request misses its deadline.

The simulated executor returns deterministic logits, so sampling still runs; only the arithmetic
is skipped. The scheduler, the block allocator, the prefix cache and admission control are all
the real ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from paged_kv_serving.engine.admission import RejectionReason
from paged_kv_serving.engine.config import EngineConfig, SloClass
from paged_kv_serving.engine.engine import Engine, RequestOutput
from paged_kv_serving.engine.loadgen import WorkloadSpec, build_workload, run_open_loop
from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.sequence import Request, SamplingParams
from paged_kv_serving.engine.simulation import ManualClock, drain_on_clock
from paged_kv_serving.engine.simulation import simulated_engine as make_simulated_engine

pytestmark = pytest.mark.integration

CONFIG = EngineConfig(
    block_size=8,
    num_blocks=256,
    max_batched_tokens=64,
    max_running_seqs=32,
    max_prompt_tokens=64,
    max_new_tokens=16,
    max_context_tokens=128,
    enable_prefix_cache=False,
)
#: Proportional cost per token: the machine then serves tokens at a rate that does not depend on
#: batch size, which is the condition under which the feasibility test is exact (docs/adr/0004).
SECONDS_PER_TOKEN = 1e-3
VOCAB = 64


def simulated_engine(
    model: LanguageModel, *, safety_factor: float = 1.0, config: EngineConfig = CONFIG
) -> tuple[Engine, ManualClock]:
    """A real engine -- real allocator, scheduler and admission -- with a simulated forward pass."""
    return make_simulated_engine(
        model, config, seconds_per_token=SECONDS_PER_TOKEN, safety_factor=safety_factor
    )


@pytest.fixture
def engine_and_clock(tiny_model: LanguageModel) -> tuple[Engine, ManualClock]:
    return simulated_engine(tiny_model)


def make(
    index: int,
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    deadline: float | None,
    arrival: float,
    slo_class: SloClass = SloClass.INTERACTIVE,
) -> Request:
    return Request(
        request_id=f"req-{index:03d}",
        prompt_token_ids=tuple((index + offset) % VOCAB for offset in range(prompt_tokens)),
        sampling=SamplingParams(max_new_tokens=max_new_tokens, seed=index),
        slo_class=slo_class,
        deadline_seconds=deadline,
        arrival_time=arrival,
    )


def drain(engine: Engine, clock: ManualClock, requests: list[Request]) -> list[RequestOutput]:
    """Submit each request at its arrival time on the manual clock, then run to completion."""
    return drain_on_clock(engine, clock, requests)


def test_no_admitted_request_misses_its_deadline_under_heavy_overload(
    engine_and_clock: tuple[Engine, ManualClock],
) -> None:
    engine, clock = engine_and_clock
    rng = np.random.default_rng(3)
    requests = [
        make(
            index,
            prompt_tokens=int(rng.integers(8, 48)),
            max_new_tokens=int(rng.integers(4, 16)),
            deadline=float(rng.choice([0.2, 0.5, 1.0, 2.0])),
            arrival=float(index) * 0.01,  # far faster than 1000 tokens/s can absorb
        )
        for index in range(120)
    ]
    outputs = drain(engine, clock, requests)
    assert outputs, "nothing completed"
    missed = [o for o in outputs if o.deadline is not None and not o.met_deadline]
    assert missed == [], f"{len(missed)} admitted requests missed their deadline"
    assert engine.slo_attainment == 1.0
    rejected = engine.admission.rejections.get(RejectionReason.DEADLINE_INFEASIBLE, 0)
    assert rejected > 0, "an overload with no rejections is not an overload"
    assert len(outputs) + rejected == len(requests)
    engine.check_invariants()


def test_batch_traffic_never_pushes_interactive_deadlines_over(
    engine_and_clock: tuple[Engine, ManualClock],
) -> None:
    """Class order is the guarantee: batch work is invisible to an interactive deadline."""
    engine, clock = engine_and_clock
    requests = [
        make(
            index,
            prompt_tokens=48,
            max_new_tokens=16,
            deadline=None,
            arrival=0.0,
            slo_class=SloClass.BATCH,
        )
        for index in range(40)
    ]
    requests += [
        make(
            100 + index,
            prompt_tokens=16,
            max_new_tokens=8,
            deadline=2.0,
            arrival=0.05 * index,
            slo_class=SloClass.INTERACTIVE,
        )
        for index in range(20)
    ]
    outputs = drain(engine, clock, requests)
    interactive = [o for o in outputs if o.slo_class is SloClass.INTERACTIVE]
    assert interactive
    assert all(o.met_deadline for o in interactive)


def test_a_request_that_fits_alone_is_admitted_and_met(
    engine_and_clock: tuple[Engine, ManualClock],
) -> None:
    engine, clock = engine_and_clock
    request = make(0, prompt_tokens=32, max_new_tokens=8, deadline=1.0, arrival=0.0)
    decision = engine.submit(request)
    assert decision.admitted
    # One iteration absorbs the 32-token prompt, eight more produce the answer; an iteration
    # is costed at a full 64-token batch, so 9 x 0.064s against a one-second deadline.
    assert decision.predicted_seconds == pytest.approx(9 * 0.064, rel=1e-6)
    outputs = engine.run_until_idle()
    assert outputs[0].met_deadline is True
    assert clock.now < 1.0


def test_a_tighter_safety_factor_rejects_more(tiny_model: LanguageModel) -> None:
    def counts(safety_factor: float) -> int:
        engine, clock = simulated_engine(tiny_model, safety_factor=safety_factor)
        requests = [
            make(index, prompt_tokens=32, max_new_tokens=8, deadline=1.2, arrival=index * 0.01)
            for index in range(30)
        ]
        drain(engine, clock, requests)
        return engine.admission.rejections.get(RejectionReason.DEADLINE_INFEASIBLE, 0)

    assert counts(4.0) > counts(1.0)


def test_the_open_loop_driver_submits_on_schedule_not_on_completion(
    tiny_model: LanguageModel,
) -> None:
    """Coordinated omission check: a slow engine must not quietly lower the offered load."""
    config = EngineConfig(
        block_size=8,
        num_blocks=128,
        max_batched_tokens=32,
        max_running_seqs=8,
        max_prompt_tokens=48,
        max_new_tokens=8,
        max_context_tokens=64,
    )
    spec = WorkloadSpec(
        num_requests=12,
        rate=1_000.0,  # everything arrives essentially at once
        prompt_tokens=(8, 24),
        max_new_tokens=(2, 6),
        interactive_deadline=None,
        vocab_size=64,
        seed=9,
    )
    requests = build_workload(spec)
    engine = Engine(tiny_model, config)
    result = run_open_loop(engine, requests)
    assert result.submitted == len(requests)
    assert result.completed + result.rejected == len(requests)
    # Latency is measured from the scheduled arrival, so queueing shows up in the numbers.
    assert all(output.latency >= output.ttft for output in result.outputs)
    assert result.wall_seconds > 0
    engine.check_invariants()
