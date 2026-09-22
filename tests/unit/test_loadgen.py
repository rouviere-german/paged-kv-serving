from __future__ import annotations

import numpy as np
import pytest

from paged_kv_serving.engine.config import SloClass
from paged_kv_serving.engine.loadgen import (
    LoadResult,
    WorkloadSpec,
    build_workload,
    burst_arrivals,
    percentile,
    poisson_arrivals,
    summarize,
)

pytestmark = pytest.mark.unit


def test_poisson_arrivals_are_increasing_and_match_the_rate() -> None:
    rng = np.random.default_rng(0)
    arrivals = poisson_arrivals(rate=50.0, count=4_000, rng=rng)
    assert arrivals == sorted(arrivals)
    gaps = np.diff([0.0, *arrivals])
    assert abs(float(np.mean(gaps)) - 1 / 50.0) < 0.002
    assert poisson_arrivals(rate=1.0, count=0, rng=rng) == []
    with pytest.raises(ValueError, match="rate must be"):
        poisson_arrivals(rate=0.0, count=1, rng=rng)
    with pytest.raises(ValueError, match="count must be"):
        poisson_arrivals(rate=1.0, count=-1, rng=rng)


def test_a_burst_window_concentrates_arrivals() -> None:
    rng = np.random.default_rng(1)
    times = burst_arrivals(
        base_rate=5.0,
        burst_rate=200.0,
        duration=4.0,
        burst_start=1.0,
        burst_duration=0.5,
        rng=rng,
    )
    inside = sum(1 for value in times if 1.0 <= value < 1.5)
    outside = len(times) - inside
    assert inside / 0.5 > 4 * (outside / 3.5), "the burst window should be far denser"
    with pytest.raises(ValueError, match="burst_rate"):
        burst_arrivals(
            base_rate=10.0,
            burst_rate=1.0,
            duration=1.0,
            burst_start=0.0,
            burst_duration=0.1,
            rng=rng,
        )
    with pytest.raises(ValueError, match="duration must be"):
        burst_arrivals(
            base_rate=1.0,
            burst_rate=1.0,
            duration=0.0,
            burst_start=0.0,
            burst_duration=0.1,
            rng=rng,
        )


def test_the_workload_is_reproducible_and_mixes_classes_and_prefixes() -> None:
    spec = WorkloadSpec(num_requests=60, rate=20.0, seed=3, interactive_fraction=0.5)
    first = build_workload(spec)
    assert [r.request_id for r in first] == [r.request_id for r in build_workload(spec)]
    assert [r.prompt_token_ids for r in first] == [r.prompt_token_ids for r in build_workload(spec)]
    classes = {r.slo_class for r in first}
    assert classes == {SloClass.INTERACTIVE, SloClass.BATCH}
    interactive = [r for r in first if r.slo_class is SloClass.INTERACTIVE]
    assert all(r.deadline_seconds == spec.interactive_deadline for r in interactive)
    shared = first[0].prompt_token_ids[: spec.shared_prefix_tokens]
    with_prefix = sum(1 for r in first if r.prompt_token_ids[: len(shared)] == shared)
    assert 0 < with_prefix < len(first), "some but not all prompts should share the prefix"


def test_arrival_times_are_relative_and_monotonic() -> None:
    requests = build_workload(WorkloadSpec(num_requests=20, rate=10.0, seed=4))
    times = [r.arrival_time for r in requests]
    assert times == sorted(times)
    assert times[0] >= 0.0


def test_a_burst_spec_produces_at_most_the_requested_count() -> None:
    spec = WorkloadSpec(num_requests=30, rate=10.0, seed=5, burst=(100.0, 0.5, 0.3))
    assert len(build_workload(spec)) <= 30


def test_workload_spec_validation() -> None:
    for kwargs, message in (
        ({"num_requests": 0}, "num_requests"),
        ({"interactive_fraction": 2.0}, "interactive_fraction"),
        ({"shared_prefix_fraction": -1.0}, "shared_prefix_fraction"),
        ({"prompt_tokens": (10, 2)}, "prompt_tokens"),
        ({"max_new_tokens": (0, 5)}, "max_new_tokens"),
        ({"keys": ()}, "at least one key"),
    ):
        with pytest.raises(ValueError, match=message):
            WorkloadSpec(**kwargs)


def test_percentiles_of_an_empty_sample_are_zero() -> None:
    assert percentile([], 99) == 0.0
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_summarize_handles_a_run_where_everything_was_rejected() -> None:
    result = LoadResult(
        rejections={"deadline_infeasible": 5}, submitted=5, wall_seconds=1.0, iterations=0
    )
    row = summarize(result, label="0.5x")
    assert row["label"] == "0.5x"
    assert row["completed"] == 0.0
    assert row["rejected"] == 5.0
    assert row["rejection_rate"] == 1.0
    assert row["slo_attainment"] == 1.0  # no deadlined request finished, so nothing was missed
    assert row["goodput_per_second"] == 0.0
