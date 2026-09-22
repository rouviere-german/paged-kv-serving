"""Prometheus metrics for the serving engine, on a caller-supplied registry.

The set is the one an on-call engineer actually needs for an LLM server: time to first token and
time per output token separate queueing from streaming, ``kv_blocks_used_ratio`` and
``preemptions_total`` say whether the memory pool is the bottleneck, ``prefix_cache_hit_ratio``
says whether reuse is working, and ``rejections_total`` is labelled by reason so a burst of
deadline rejections is never confused with a burst of authentication failures.

Histogram buckets are chosen for a CPU engine running a tiny model: milliseconds matter and
anything past a few seconds is already a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = ["EngineMetrics", "create_engine_metrics"]

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_TOKEN_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class EngineMetrics:
    registry: CollectorRegistry
    ttft_seconds: Histogram
    tpot_seconds: Histogram
    request_latency_seconds: Histogram
    iteration_seconds: Histogram
    batched_tokens: Histogram
    kv_blocks_used_ratio: Gauge
    prefix_cache_hit_ratio: Gauge
    queue_depth: Gauge
    running_sequences: Gauge
    slo_attainment: Gauge
    prompt_length_psi: Gauge
    preemptions_total: Counter
    aborted_requests_total: Counter
    rejections_total: Counter
    generated_tokens_total: Counter
    prefill_tokens_total: Counter

    def observe_iteration(
        self, *, seconds: float, batched_tokens: int, prefill_tokens: int, decode_tokens: int
    ) -> None:
        self.iteration_seconds.observe(seconds)
        self.batched_tokens.observe(batched_tokens)
        self.prefill_tokens_total.inc(prefill_tokens)
        self.generated_tokens_total.inc(decode_tokens)

    def observe_queues(
        self,
        *,
        queue_depth: int,
        running: int,
        blocks_used_ratio: float,
        prefix_hit_ratio: float,
    ) -> None:
        self.queue_depth.set(queue_depth)
        self.running_sequences.set(running)
        self.kv_blocks_used_ratio.set(blocks_used_ratio)
        self.prefix_cache_hit_ratio.set(prefix_hit_ratio)


def create_engine_metrics(
    registry: CollectorRegistry | None = None, namespace: str = "engine"
) -> EngineMetrics:
    """Create every engine metric family on ``registry`` (a fresh one when omitted)."""
    reg = registry if registry is not None else CollectorRegistry()
    return EngineMetrics(
        registry=reg,
        ttft_seconds=Histogram(
            "ttft_seconds",
            "Seconds from request arrival to its first generated token.",
            namespace=namespace,
            registry=reg,
            buckets=_LATENCY_BUCKETS,
        ),
        tpot_seconds=Histogram(
            "tpot_seconds",
            "Mean seconds per output token after the first, per finished request.",
            namespace=namespace,
            registry=reg,
            buckets=_TOKEN_BUCKETS,
        ),
        request_latency_seconds=Histogram(
            "request_latency_seconds",
            "Seconds from request arrival to the last generated token.",
            namespace=namespace,
            registry=reg,
            buckets=_LATENCY_BUCKETS,
        ),
        iteration_seconds=Histogram(
            "iteration_seconds",
            "Wall-clock seconds of one engine iteration (forward pass).",
            namespace=namespace,
            registry=reg,
            buckets=_TOKEN_BUCKETS,
        ),
        batched_tokens=Histogram(
            "batched_tokens",
            "Tokens executed in one engine iteration.",
            namespace=namespace,
            registry=reg,
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
        ),
        kv_blocks_used_ratio=Gauge(
            "kv_blocks_used_ratio",
            "Fraction of KV blocks referenced by a sequence or the prefix cache.",
            namespace=namespace,
            registry=reg,
        ),
        prefix_cache_hit_ratio=Gauge(
            "prefix_cache_hit_ratio",
            "Cumulative fraction of queried prompt blocks served from the prefix cache.",
            namespace=namespace,
            registry=reg,
        ),
        queue_depth=Gauge(
            "queue_depth",
            "Sequences waiting for admission.",
            namespace=namespace,
            registry=reg,
        ),
        running_sequences=Gauge(
            "running_sequences",
            "Sequences currently resident in the KV cache.",
            namespace=namespace,
            registry=reg,
        ),
        slo_attainment=Gauge(
            "slo_attainment",
            "Fraction of finished requests with a deadline that met it.",
            namespace=namespace,
            registry=reg,
        ),
        prompt_length_psi=Gauge(
            "prompt_length_psi",
            "Population Stability Index of recent prompt lengths against the reference window.",
            namespace=namespace,
            registry=reg,
        ),
        preemptions_total=Counter(
            "preemptions_total",
            "Sequences evicted from the running set and scheduled for recomputation.",
            namespace=namespace,
            registry=reg,
        ),
        aborted_requests_total=Counter(
            "aborted_requests_total",
            "Requests cancelled before finishing (client disconnect or timeout); blocks freed.",
            namespace=namespace,
            registry=reg,
        ),
        rejections_total=Counter(
            "rejections_total",
            "Requests refused before admission, by reason.",
            ["reason"],
            namespace=namespace,
            registry=reg,
        ),
        generated_tokens_total=Counter(
            "generated_tokens_total",
            "Decode tokens executed.",
            namespace=namespace,
            registry=reg,
        ),
        prefill_tokens_total=Counter(
            "prefill_tokens_total",
            "Prompt tokens executed, cached prefixes excluded.",
            namespace=namespace,
            registry=reg,
        ),
    )
