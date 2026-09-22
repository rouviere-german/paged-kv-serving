from __future__ import annotations

import math

import pytest

from paged_kv_serving.engine.admission import (
    AdmissionController,
    LatencyModel,
    RejectionReason,
    WorkItem,
)

pytestmark = pytest.mark.unit

#: A controller with round numbers: one iteration costs 0.1s and absorbs 50 prompt tokens.
ITERATION_SECONDS = 0.1
PREFILL_PER_ITERATION = 50


def controller(**kwargs: object) -> AdmissionController:
    model = LatencyModel(0.0, ITERATION_SECONDS / 100)
    defaults: dict[str, object] = {"max_batched_tokens": 100, "max_running_seqs": 50}
    return AdmissionController(latency=model, **{**defaults, **kwargs})  # type: ignore[arg-type]


def item(rank: int, deadline: float | None, prefill: int, decode: int) -> WorkItem:
    return WorkItem(rank=rank, deadline=deadline, prefill_tokens=prefill, decode_tokens=decode)


# --- the latency model ---------------------------------------------------------------------


def test_latency_model_fits_an_affine_relationship() -> None:
    samples = [(size, 0.01 + 0.001 * size) for size in (1, 10, 50, 100)]
    model = LatencyModel.fit(samples)
    assert math.isclose(model.intercept, 0.01, abs_tol=1e-6)
    assert math.isclose(model.slope, 0.001, abs_tol=1e-8)
    assert math.isclose(model.predict(100), 0.11, abs_tol=1e-6)
    assert math.isclose(model.token_rate(100), 100 / 0.11, rel_tol=1e-6)


def test_a_single_batch_size_puts_all_the_cost_in_the_slope() -> None:
    """Without a second batch size the overhead is unidentifiable, so it is charged per token."""
    model = LatencyModel.fit([(10, 0.1), (10, 0.12)])
    assert model.intercept == 0.0
    assert math.isclose(model.slope, 0.011, rel_tol=1e-6)


def test_fit_rejects_empty_or_impossible_samples() -> None:
    with pytest.raises(ValueError, match="at least one latency sample"):
        LatencyModel.fit([])
    with pytest.raises(ValueError, match="positive sizes and durations"):
        LatencyModel.fit([(1, 0.0)])
    with pytest.raises(ValueError, match="must be >= 0 and slope"):
        LatencyModel(-1.0, 1.0)
    with pytest.raises(ValueError, match="ewma_alpha"):
        LatencyModel(0.0, 1.0, ewma_alpha=2.0)
    with pytest.raises(ValueError, match="batch_tokens must be"):
        LatencyModel(0.0, 1.0).predict(0)


def test_the_ewma_correction_follows_a_slower_machine() -> None:
    model = LatencyModel(0.0, 0.001, ewma_alpha=0.5)
    assert model.correction == 1.0
    for _ in range(10):
        model.observe(100, 0.2)  # twice the predicted 0.1
    assert math.isclose(model.correction, 2.0, rel_tol=1e-2)
    assert model.observations == 10
    assert model.predict(100) > 0.19
    # Nonsense observations are ignored rather than allowed to poison the model.
    model.observe(0, 1.0)
    model.observe(100, -1.0)
    assert model.observations == 10


def test_profiling_times_a_callable_at_several_batch_sizes() -> None:
    times = [0.0]

    def fake_clock() -> float:
        return times[0]

    def run(size: int) -> None:
        times[0] += 0.001 * size

    model = LatencyModel.profile(run, [1, 10, 100], repeats=2, warmup=1, clock=fake_clock)
    assert math.isclose(model.slope, 0.001, rel_tol=1e-6)
    with pytest.raises(ValueError, match="at least one batch size"):
        LatencyModel.profile(run, [], clock=fake_clock)


def test_update_replaces_the_fit_and_drops_the_old_correction() -> None:
    model = LatencyModel(0.01, 0.001)
    model.observe(100, 1.0)
    assert model.correction != 1.0
    model.update(0.02, 0.002)
    assert (model.intercept, model.slope, model.correction) == (0.02, 0.002, 1.0)
    with pytest.raises(ValueError, match="slope must be"):
        model.update(0.0, 0.0)


# --- the iteration model -------------------------------------------------------------------


def test_the_controller_costs_work_in_iterations() -> None:
    admission = controller()
    assert admission.prefill_tokens_per_iteration == PREFILL_PER_ITERATION
    assert math.isclose(admission.iteration_seconds, ITERATION_SECONDS)
    # 120 prompt tokens need three iterations to absorb, then eight more to decode eight tokens.
    assert admission.iterations_for(120, 8) == 11
    assert admission.iterations_for(0, 0) == 0
    assert math.isclose(admission.token_rate, PREFILL_PER_ITERATION / ITERATION_SECONDS)


def test_decode_is_serial_per_request_however_idle_the_machine_is() -> None:
    """A twenty-token answer needs twenty iterations, which a fluid model would miss entirely."""
    admission = controller()
    tight = admission.decide(
        now=0.0, candidate=item(0, deadline=1.0, prefill=10, decode=20), committed=[]
    )
    assert not tight.admitted, "21 iterations at 0.1s cannot fit in one second"
    roomy = admission.decide(
        now=0.0, candidate=item(0, deadline=3.0, prefill=10, decode=20), committed=[]
    )
    assert roomy.admitted
    assert math.isclose(roomy.predicted_seconds, 2.1)


def test_decode_of_different_requests_overlaps_instead_of_adding_up() -> None:
    """Ten requests of ten tokens each still take ten iterations, not a hundred."""
    admission = controller()
    committed = [item(0, deadline=5.0, prefill=0, decode=10) for _ in range(9)]
    decision = admission.decide(
        now=0.0, candidate=item(0, deadline=5.0, prefill=0, decode=10), committed=committed
    )
    assert decision.admitted
    assert math.isclose(decision.predicted_seconds, 1.0)


def test_prefill_of_different_requests_does_add_up() -> None:
    admission = controller()
    committed = [item(0, deadline=5.0, prefill=100, decode=1) for _ in range(5)]
    decision = admission.decide(
        now=0.0, candidate=item(0, deadline=5.0, prefill=100, decode=1), committed=committed
    )
    # 600 prompt tokens at 50 per iteration is twelve iterations, plus one to decode.
    assert math.isclose(decision.predicted_seconds, 1.3)


def test_a_request_that_cannot_fit_is_refused_with_a_truthful_retry_after() -> None:
    admission = controller()
    committed = [item(0, deadline=9.0, prefill=500, decode=4)]
    decision = admission.decide(
        now=0.0, candidate=item(0, deadline=0.5, prefill=200, decode=8), committed=committed
    )
    assert not decision.admitted
    assert decision.reason is RejectionReason.DEADLINE_INFEASIBLE
    # The backlog is ten prefill iterations plus four decode ones: 1.4 seconds.
    assert math.isclose(decision.retry_after_seconds, 1.4)
    assert decision.retry_after_header == "2"
    assert admission.rejections[RejectionReason.DEADLINE_INFEASIBLE] == 1


def test_admitting_a_request_never_breaks_a_deadline_already_accepted() -> None:
    """The demand test looks at every committed deadline, not only the candidate's."""
    admission = controller()
    committed = [item(0, deadline=1.0, prefill=50, decode=8)]
    # Behind that request there is room for a lot more work before thirty seconds.
    assert admission.decide(
        now=0.0, candidate=item(0, deadline=30.0, prefill=500, decode=20), committed=committed
    ).admitted
    # In front of it there is not: the earlier deadline jumps the queue and pushes it over.
    assert not admission.decide(
        now=0.0, candidate=item(0, deadline=0.5, prefill=500, decode=20), committed=committed
    ).admitted


def test_a_request_already_past_its_deadline_does_not_block_every_new_one() -> None:
    """A late commitment is hopeless, but it must not turn the server into a permanent 503."""
    admission = controller()
    late = [item(0, deadline=1.0, prefill=500, decode=5)]
    assert admission.decide(
        now=9.0, candidate=item(0, deadline=30.0, prefill=100, decode=5), committed=late
    ).admitted
    # Its work still counts against anything queued behind it.
    assert not admission.decide(
        now=9.0, candidate=item(0, deadline=9.5, prefill=100, decode=5), committed=late
    ).admitted


def test_a_request_arriving_after_its_own_deadline_is_refused() -> None:
    admission = controller()
    decision = admission.decide(
        now=10.0, candidate=item(0, deadline=9.0, prefill=1, decode=1), committed=[]
    )
    assert not decision.admitted
    assert decision.reason is RejectionReason.DEADLINE_INFEASIBLE


def test_higher_priority_work_counts_against_a_lower_class_deadline() -> None:
    admission = controller()
    interactive = [item(0, deadline=None, prefill=500, decode=20)]
    assert not admission.decide(
        now=0.0, candidate=item(1, deadline=1.0, prefill=10, decode=1), committed=interactive
    ).admitted, "batch work must wait behind every interactive request"
    # The other way round, batch work is invisible to an interactive deadline.
    batch = [item(1, deadline=None, prefill=500, decode=20)]
    assert admission.decide(
        now=0.0, candidate=item(0, deadline=1.0, prefill=10, decode=1), committed=batch
    ).admitted


def test_best_effort_requests_are_admitted_whenever_the_queue_has_room() -> None:
    admission = controller(max_queued_sequences=3)
    committed = [item(0, deadline=1.0, prefill=50, decode=5)]
    huge = admission.decide(
        now=0.0, candidate=item(0, deadline=None, prefill=10_000, decode=500), committed=committed
    )
    assert huge.admitted, "best-effort work runs last and cannot make a deadline infeasible"
    full = admission.decide(
        now=0.0,
        candidate=item(0, deadline=None, prefill=1, decode=1),
        committed=[],
        queued_sequences=3,
    )
    assert not full.admitted
    assert full.reason is RejectionReason.QUEUE_FULL


def test_the_safety_factor_tightens_the_iteration_cost() -> None:
    strict = controller(safety_factor=2.0)
    assert math.isclose(strict.iteration_seconds, 2 * ITERATION_SECONDS)
    relaxed = controller()
    candidate = item(0, deadline=1.0, prefill=50, decode=8)
    assert relaxed.decide(now=0.0, candidate=candidate, committed=[]).admitted
    assert not strict.decide(now=0.0, candidate=candidate, committed=[]).admitted


def test_controller_and_work_item_validation() -> None:
    with pytest.raises(ValueError, match="safety_factor"):
        controller(safety_factor=0.0)
    with pytest.raises(ValueError, match="max_batched_tokens"):
        controller(max_batched_tokens=0)
    with pytest.raises(ValueError, match="max_running_seqs"):
        controller(max_running_seqs=0)
    with pytest.raises(ValueError, match="token counts must be"):
        item(0, None, prefill=-1, decode=0)
    with pytest.raises(ValueError, match="token counts must be"):
        item(0, None, prefill=0, decode=-1)
    assert item(0, None, prefill=3, decode=4).tokens == 7


def test_a_running_set_wider_than_the_budget_still_leaves_room_for_prefill() -> None:
    """With more sequence slots than budget, prefill is squeezed to one token per iteration."""
    admission = controller(max_batched_tokens=8, max_running_seqs=64)
    assert admission.prefill_tokens_per_iteration == 1
    assert admission.iterations_for(5, 2) == 7


def test_a_lower_class_is_costed_at_its_prefill_allowance() -> None:
    """A batch deadline is checked against the share it is guaranteed, not the whole budget."""
    controller = AdmissionController(
        latency=LatencyModel(0.0, 1e-3, ewma_alpha=0.0),
        max_batched_tokens=64,
        max_running_seqs=8,
        lower_class_prefill_tokens=16,
    )
    assert controller.prefill_tokens_per_iteration_for(0) == 56
    assert controller.prefill_tokens_per_iteration_for(1) == 16
    assert controller.iterations_for(112, 4, rank=0) == 2 + 4
    assert controller.iterations_for(112, 4, rank=1) == 7 + 4
    uncapped = AdmissionController(
        latency=LatencyModel(0.0, 1e-3, ewma_alpha=0.0), max_batched_tokens=64, max_running_seqs=8
    )
    assert uncapped.prefill_tokens_per_iteration_for(1) == 56

    # Interactive: 2 + 4 iterations of 64 ms = 0.384 s fits 0.5 s; batch: 7 + 4 = 0.704 s does not.
    def item(rank: int) -> WorkItem:
        return WorkItem(rank=rank, deadline=0.5, prefill_tokens=112, decode_tokens=4)

    assert controller.decide(now=0.0, candidate=item(0), committed=[]).admitted
    assert not controller.decide(now=0.0, candidate=item(1), committed=[]).admitted
