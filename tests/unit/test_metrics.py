from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from paged_kv_serving.engine.metrics import create_engine_metrics

pytestmark = pytest.mark.unit


def sample(registry: CollectorRegistry, name: str, **labels: str) -> float:
    value = registry.get_sample_value(name, labels or None)
    assert value is not None, f"{name} is not exposed"
    return value


def test_every_metric_the_readme_names_is_exposed() -> None:
    registry = CollectorRegistry()
    metrics = create_engine_metrics(registry)
    metrics.observe_iteration(seconds=0.01, batched_tokens=32, prefill_tokens=24, decode_tokens=8)
    metrics.observe_queues(queue_depth=3, running=2, blocks_used_ratio=0.25, prefix_hit_ratio=0.5)
    metrics.preemptions_total.inc(2)
    metrics.aborted_requests_total.inc()
    metrics.rejections_total.labels("deadline_infeasible").inc()
    metrics.ttft_seconds.observe(0.1)
    metrics.tpot_seconds.observe(0.01)
    metrics.request_latency_seconds.observe(0.4)
    metrics.slo_attainment.set(0.98)
    metrics.prompt_length_psi.set(0.03)

    assert sample(registry, "engine_ttft_seconds_count") == 1
    assert sample(registry, "engine_tpot_seconds_count") == 1
    assert sample(registry, "engine_request_latency_seconds_count") == 1
    assert sample(registry, "engine_kv_blocks_used_ratio") == 0.25
    assert sample(registry, "engine_prefix_cache_hit_ratio") == 0.5
    assert sample(registry, "engine_preemptions_total") == 2
    assert sample(registry, "engine_aborted_requests_total") == 1
    assert sample(registry, "engine_rejections_total", reason="deadline_infeasible") == 1
    assert sample(registry, "engine_queue_depth") == 3
    assert sample(registry, "engine_running_sequences") == 2
    assert sample(registry, "engine_slo_attainment") == 0.98
    assert sample(registry, "engine_prompt_length_psi") == 0.03
    assert sample(registry, "engine_generated_tokens_total") == 8
    assert sample(registry, "engine_prefill_tokens_total") == 24
    assert sample(registry, "engine_iteration_seconds_count") == 1
    assert sample(registry, "engine_batched_tokens_count") == 1


def test_metrics_default_to_a_private_registry() -> None:
    first = create_engine_metrics()
    second = create_engine_metrics()
    assert first.registry is not second.registry
