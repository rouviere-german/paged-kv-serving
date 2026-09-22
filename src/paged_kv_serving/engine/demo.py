"""Offline end-to-end demo: ``uv run python tasks.py demo``.

Eight steps, in the order a reviewer would want to check them:

1. build a deterministic random-init checkpoint and load it back through manifest verification;
2. prove the engine computes what a cache-free recompute computes -- first one prompt through
   arbitrary chunk splits and a scattered block table, then a whole mixed workload through the
   real scheduler on a pool small enough to force preemption;
3. show what block-level prefix caching saves on a shared system prompt;
4. show admission control refusing work it cannot finish instead of accepting it and timing out;
5. abort a request mid-generation and show its blocks come straight back;
6. run one open-loop burst through continuous batching and through static batching;
7. start the real server on a loopback port, stream over SSE, hang up mid-stream, and send an
   open-loop burst of interactive and batch traffic from two keys through it;
8. run an audited admin operation and verify the hash chain.

Nothing here reaches the network (the server binds 127.0.0.1), needs an API key or downloads
anything. Every number printed is measured during the run.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from pathlib import Path
from typing import Any

import httpx
from prometheus_client import CollectorRegistry

from paged_kv_serving.core.config import Settings
from paged_kv_serving.core.security import hash_api_key
from paged_kv_serving.engine.audit import AuditLog
from paged_kv_serving.engine.config import EngineConfig
from paged_kv_serving.engine.corpus import SHARED_SYSTEM_PROMPT, USER_TURNS
from paged_kv_serving.engine.engine import Engine
from paged_kv_serving.engine.equivalence import (
    RecordingExecutor,
    engine_logit_gap,
    max_logit_gap,
    sequence_tokens,
)
from paged_kv_serving.engine.http_loadgen import (
    HttpTrafficClass,
    plan_http_workload,
    run_http_open_loop,
    summarize_http,
)
from paged_kv_serving.engine.loadgen import (
    WorkloadSpec,
    build_workload,
    percentile,
    run_open_loop,
    summarize,
)
from paged_kv_serving.engine.localserver import serve_in_thread
from paged_kv_serving.engine.sequence import Request, SamplingParams
from paged_kv_serving.engine.static import StaticBatchingEngine
from paged_kv_serving.engine.tokenizer import ByteBPETokenizer
from paged_kv_serving.engine.weights import Checkpoint, build_random_checkpoint

__all__ = ["DEMO_ENGINE_CONFIG", "render_report", "run_serving_demo"]

_ADMIN_KEY = "demo-admin-key"
_USER_KEY = "demo-user-key"
_BATCH_KEY = "demo-batch-key"

DEMO_ENGINE_CONFIG = EngineConfig(
    block_size=16,
    num_blocks=192,
    max_batched_tokens=128,
    max_running_seqs=12,
    max_prompt_tokens=192,
    max_new_tokens=24,
    max_context_tokens=256,
)


def _with_prefix_cache(enabled: bool) -> EngineConfig:
    return dataclasses.replace(DEMO_ENGINE_CONFIG, enable_prefix_cache=enabled)


def _uneven(total: int) -> list[int]:
    """A deliberately ragged split whose boundaries do not line up with blocks."""
    sizes: list[int] = []
    remaining = total
    for size in (7, 3, 11, 5):
        if remaining <= size:
            break
        sizes.append(size)
        remaining -= size
    sizes.append(remaining)
    return sizes


def _equivalence(checkpoint: Checkpoint, prompts: list[list[int]]) -> dict[str, Any]:
    """Max logit gap against a cache-free recompute: single prompts, then a whole workload."""
    worst = 0.0
    block_size = DEMO_ENGINE_CONFIG.block_size
    for prompt in prompts:
        for chunks in ([len(prompt)], [1] * len(prompt), _uneven(len(prompt))):
            worst = max(
                worst,
                max_logit_gap(checkpoint.model, prompt, block_size=block_size, chunks=chunks),
            )
        blocks = -(-len(prompt) // block_size)
        worst = max(
            worst,
            max_logit_gap(
                checkpoint.model,
                prompt,
                block_size=block_size,
                block_order=list(reversed(range(blocks))),
            ),
        )

    # The same check through the real scheduler: a pool of 24 blocks for 12 requests forces
    # preemption and recomputation, a 40-token budget forces chunked prefill, and a shared prefix
    # means some sequences read their first blocks from the prefix cache.
    cramped = dataclasses.replace(
        DEMO_ENGINE_CONFIG, num_blocks=24, max_batched_tokens=40, watermark=0.0
    )
    engine = Engine(checkpoint.model, cramped)
    recorder = RecordingExecutor(engine.runner)
    engine.executor = recorder
    shared = [int(token) for token in prompts[0][:32]]
    requests = [
        Request(
            request_id=f"eq-{index}",
            prompt_token_ids=tuple(
                [*shared, *range(1 + index, 40 + 3 * index)] if index % 2 else [*prompts[1], index]
            ),
            sampling=SamplingParams(max_new_tokens=12 + index % 5, seed=index),
        )
        for index in range(12)
    ]
    outputs = engine.generate(requests)
    engine.check_invariants()
    tokens = sequence_tokens(
        requests,
        {output.request_id: [s.token_ids for s in output.samples] for output in outputs},
    )
    engine_gap, rows = engine_logit_gap(checkpoint.model, recorder.records, tokens)
    return {
        "single_prompt_max_abs_logit_gap": float(f"{worst:.3g}"),
        "engine_max_abs_logit_gap": float(f"{engine_gap:.3g}"),
        "engine_logit_rows_checked": rows,
        "engine_preemptions": engine.scheduler.stats.preemptions,
        "engine_prefix_cache_hit_blocks": engine.prefix_cache.stats.hit_blocks,
        "prompts": len(prompts),
    }


def _chat_requests(tokenizer: ByteBPETokenizer, engine: Engine) -> list[Request]:
    shared = tokenizer.encode(SHARED_SYSTEM_PROMPT)[: DEMO_ENGINE_CONFIG.max_prompt_tokens // 2]
    return [
        Request(
            request_id=f"chat-{index}",
            prompt_token_ids=(*shared, *tokenizer.encode(turn))[
                : DEMO_ENGINE_CONFIG.max_prompt_tokens
            ],
            sampling=SamplingParams(max_new_tokens=8, seed=index),
            key_id="tenant-a",
            arrival_time=engine.clock(),
        )
        for index, turn in enumerate(USER_TURNS)
    ]


def _prefix_cache_comparison(
    checkpoint: Checkpoint, tokenizer: ByteBPETokenizer
) -> dict[str, float]:
    """Run the same shared-system-prompt conversation with the prefix cache on and off."""
    results: dict[str, float] = {}
    for label, enabled in (("off", False), ("on", True)):
        engine = Engine(checkpoint.model, _with_prefix_cache(enabled))
        ttfts: list[float] = []
        for request in _chat_requests(tokenizer, engine):
            arrived = dataclasses.replace(request, arrival_time=engine.clock())
            ttfts.extend(output.ttft for output in engine.generate([arrived]))
        engine.check_invariants()
        results[f"ttft_p50_cache_{label}"] = round(percentile(ttfts, 50), 5)
        results[f"hit_ratio_cache_{label}"] = round(engine.prefix_cache.stats.hit_ratio, 4)
        results[f"prefill_tokens_cache_{label}"] = float(engine.scheduler.stats.prefill_tokens)
    saved = results["prefill_tokens_cache_off"] - results["prefill_tokens_cache_on"]
    results["prefill_tokens_saved_fraction"] = round(
        saved / max(results["prefill_tokens_cache_off"], 1.0), 4
    )
    results["ttft_reduction_fraction"] = round(
        1.0 - results["ttft_p50_cache_on"] / max(results["ttft_p50_cache_off"], 1e-9), 4
    )
    return results


def _admission_demo(checkpoint: Checkpoint) -> dict[str, Any]:
    """Offer far more deadlined work than the engine can finish and count what it refuses."""
    engine = Engine(checkpoint.model, DEMO_ENGINE_CONFIG)
    engine.profile_latency(repeats=3, warmup=1)
    iteration = engine.admission.iteration_seconds
    # A deadline of 40 worst-case iterations: one request fits easily, a burst of 60 cannot.
    deadline = round(40 * iteration, 3)
    spec = WorkloadSpec(
        num_requests=60,
        rate=400.0,
        prompt_tokens=(24, 48),
        max_new_tokens=(8, 16),
        interactive_fraction=1.0,
        interactive_deadline=deadline,
        shared_prefix_tokens=16,
        vocab_size=min(checkpoint.model.config.vocab_size, 256),
        seed=11,
    )
    result = run_open_loop(engine, build_workload(spec))
    engine.check_invariants()
    row = summarize(result, label="admission")
    return {
        "deadline_seconds": deadline,
        "worst_case_iteration_seconds": round(iteration, 5),
        "submitted": int(row["submitted"]),
        "completed": int(row["completed"]),
        "rejected": int(row["rejected"]),
        "rejection_reasons": dict(result.rejections),
        "admitted_on_time_fraction": round(float(row["slo_attainment"]), 4),
    }


def _sequence_blocks(engine: Engine) -> int:
    """Blocks held by live sequences. Prefix-cache entries are deliberately excluded."""
    return sum(len(sequence.block_table) for sequence in engine.live_sequences())


def _cancellation(checkpoint: Checkpoint) -> dict[str, Any]:
    """Abort a request mid-generation and show its blocks come straight back.

    Done on the synchronous engine so the measurement is deterministic; the HTTP path in step 7
    adds the disconnect plumbing on top, but the guarantee itself belongs to ``Engine.abort``.
    """
    engine = Engine(checkpoint.model, DEMO_ENGINE_CONFIG)
    request = Request(
        request_id="cancel-me",
        prompt_token_ids=tuple(range(1, 97)),
        sampling=SamplingParams(max_new_tokens=DEMO_ENGINE_CONFIG.max_new_tokens, seed=1),
    )
    engine.submit(request)
    engine.step()  # prefill, so the sequence really owns blocks
    held = _sequence_blocks(engine)
    removed = engine.abort(request.request_id)
    engine.check_invariants()
    # Full prompt blocks stay referenced by the prefix cache, which is reuse rather than a leak;
    # flushing the cache shows the pool really is whole again.
    flushed = engine.flush_prefix_cache()
    return {
        "blocks_held_before_abort": held,
        "sequences_aborted": removed,
        "blocks_held_after_abort": _sequence_blocks(engine),
        "blocks_retained_by_prefix_cache": flushed,
        "free_blocks_after_cache_flush": engine.blocks.num_free_blocks,
        "pool_size": engine.blocks.num_blocks,
    }


def _batching_comparison(checkpoint: Checkpoint) -> list[dict[str, float | str]]:
    """One arrival schedule with a burst, two schedulers, the same model and sampler."""
    spec = WorkloadSpec(
        num_requests=48,
        rate=60.0,
        prompt_tokens=(24, 96),
        max_new_tokens=(4, 24),
        interactive_fraction=0.5,
        interactive_deadline=1.0,
        shared_prefix_tokens=16,
        shared_prefix_fraction=0.5,
        vocab_size=min(checkpoint.model.config.vocab_size, 256),
        seed=5,
        burst=(240.0, 0.2, 0.2),
    )
    requests = build_workload(spec)

    engine = Engine(checkpoint.model, DEMO_ENGINE_CONFIG)
    engine.profile_latency(repeats=2, warmup=1)
    continuous = run_open_loop(engine, requests)
    engine.check_invariants()
    row = summarize(continuous, label="continuous")
    row["preemptions"] = float(engine.scheduler.stats.preemptions)
    row["prefix_cache_hit_ratio"] = round(engine.prefix_cache.stats.hit_ratio, 4)
    return [row, _static_row(checkpoint, requests)]


def _static_row(checkpoint: Checkpoint, requests: list[Request]) -> dict[str, float | str]:
    static = StaticBatchingEngine(
        checkpoint.model,
        batch_size=8,
        max_context_tokens=DEMO_ENGINE_CONFIG.max_context_tokens,
        block_size=DEMO_ENGINE_CONFIG.block_size,
    )
    start = static.clock()
    scheduled = [
        dataclasses.replace(request, arrival_time=start + request.arrival_time)
        for request in requests
    ]
    outputs = static.generate(scheduled)
    wall = static.clock() - start
    deadlines = {
        request.request_id: request.deadline_seconds
        for request in requests
        if request.deadline_seconds is not None
    }
    on_time = sum(
        1
        for output in outputs
        if output.request_id in deadlines and output.latency <= deadlines[output.request_id]
    )
    output_tokens = sum(output.output_tokens for output in outputs)
    ttfts = [output.ttft for output in outputs]
    latencies = [output.latency for output in outputs]
    return {
        "label": "static",
        "submitted": float(len(requests)),
        "completed": float(len(outputs)),
        "rejected": 0.0,
        "wall_seconds": round(wall, 4),
        "output_tokens": float(output_tokens),
        "tokens_per_second": round(output_tokens / max(wall, 1e-9), 2),
        "ttft_p50": round(percentile(ttfts, 50), 4),
        "ttft_p99": round(percentile(ttfts, 99), 4),
        "latency_p50": round(percentile(latencies, 50), 4),
        "latency_p99": round(percentile(latencies, 99), 4),
        "deadlined": float(len(deadlines)),
        "on_time": float(on_time),
        "slo_attainment": on_time / len(deadlines) if deadlines else 1.0,
        "goodput_per_second": round(on_time / max(wall, 1e-9), 3),
        "padding_tokens": float(static.total_padding_tokens),
        "wasted_decode_tokens": float(static.total_wasted_decode_tokens),
    }


def demo_settings(workdir: Path) -> Settings:
    """Settings for the served demo: an interactive user key, a batch-only key and an admin."""
    user_digest = hash_api_key(_USER_KEY)
    batch_digest = hash_api_key(_BATCH_KEY)
    admin_digest = hash_api_key(_ADMIN_KEY)
    return Settings(
        _env_file=None,
        api_key_sha256=[user_digest, batch_digest, admin_digest],
        key_policies=[
            f"{user_digest}:interactive|batch:60000:8:192:24:0",
            f"{batch_digest}:batch:20000:4:192:24:0",
            f"{admin_digest}:interactive|batch:60000:4:192:24:1",
        ],
        rate_limit_per_second=500.0,
        rate_limit_burst=500,
        engine_block_size=DEMO_ENGINE_CONFIG.block_size,
        engine_num_blocks=DEMO_ENGINE_CONFIG.num_blocks,
        engine_max_batched_tokens=DEMO_ENGINE_CONFIG.max_batched_tokens,
        engine_max_running_seqs=DEMO_ENGINE_CONFIG.max_running_seqs,
        engine_max_prompt_tokens=DEMO_ENGINE_CONFIG.max_prompt_tokens,
        engine_max_new_tokens=DEMO_ENGINE_CONFIG.max_new_tokens,
        engine_max_context_tokens=DEMO_ENGINE_CONFIG.max_context_tokens,
        engine_profile_on_start=True,
        engine_blas_threads=None,
        request_timeout_seconds=30.0,
        drain_timeout_seconds=5.0,
        data_dir=workdir,
    )


async def _served(base_url: str, engine: Engine, audit: AuditLog) -> dict[str, Any]:
    """Everything a client can observe, over a real socket."""
    summary: dict[str, Any] = {}
    user = {"X-API-Key": _USER_KEY}
    admin = {"X-API-Key": _ADMIN_KEY}
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        summary["healthz"] = (await client.get("/healthz")).status_code
        summary["readyz"] = (await client.get("/readyz")).json()["status"]
        summary["unauthenticated_status"] = (
            await client.post("/v1/completions", json={"prompt": "hi"})
        ).status_code
        batch_key_interactive = await client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 4, "slo_class": "interactive"},
            headers={"X-API-Key": _BATCH_KEY},
        )
        summary["batch_key_asking_for_interactive_status"] = batch_key_interactive.status_code

        events = 0
        async with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": USER_TURNS[0], "max_tokens": 8, "stream": True},
            headers=user,
        ) as response:
            summary["stream_status"] = response.status_code
            async for line in response.aiter_lines():
                events += line.startswith("event: token")
        summary["streamed_token_events"] = events

        # Hang up after the first token of a 24-token answer.
        aborted_before = engine.aborted_requests
        async with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": USER_TURNS[1], "max_tokens": 24, "stream": True},
            headers=user,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("event: token"):
                    break
        for _ in range(500):
            if engine.active_requests == 0:
                break
            await asyncio.sleep(0.01)
        summary["hangup_aborted_requests"] = engine.aborted_requests - aborted_before
        summary["active_requests_after_hangup"] = engine.active_requests
        summary["sequence_blocks_after_hangup"] = _sequence_blocks(engine)

        classes = [
            HttpTrafficClass(
                "interactive", _USER_KEY, "interactive", share=1.0, max_tokens=(4, 12),
                deadline_ms=3_000,
            ),
            HttpTrafficClass(
                "batch", _BATCH_KEY, "batch", share=1.0, prompt_words=(8, 20),
                max_tokens=(16, 24),
            ),
        ]  # fmt: skip
        plans = plan_http_workload(
            classes, rate=40.0, count=60, seed=3, burst=(200.0, 0.3, 0.3),
            system_prompt=SHARED_SYSTEM_PROMPT,
        )  # fmt: skip
        started = time.perf_counter()
        outcomes = await run_http_open_loop(client, plans)
        summary["open_loop_burst"] = summarize_http(
            outcomes, wall_seconds=time.perf_counter() - started
        )

        summary["admin_without_scope_status"] = (
            await client.post("/v1/admin/prefix-cache/flush", headers=user)
        ).status_code
        flushed = await client.post("/v1/admin/prefix-cache/flush", headers=admin)
        summary["admin_flush_status"] = flushed.status_code
        head = flushed.json()["audit_head"]
        summary["audit_verified"] = audit.verify(expect_head=head) == head
        metrics = (await client.get("/metrics", headers=admin)).text
        summary["metrics_exported"] = sorted(
            name
            for name in (
                "engine_ttft_seconds",
                "engine_tpot_seconds",
                "engine_kv_blocks_used_ratio",
                "engine_prefix_cache_hit_ratio",
                "engine_preemptions_total",
                "engine_rejections_total",
                "engine_queue_depth",
                "engine_slo_attainment",
                "engine_prompt_length_psi",
            )
            if name in metrics
        )
    engine.check_invariants()
    return summary


def _api_demo(checkpoint: Checkpoint, workdir: Path) -> dict[str, Any]:
    from paged_kv_serving.engine.api import (  # noqa: PLC0415 - avoids an import cycle
        build_service,
        create_api,
    )

    settings = demo_settings(workdir)
    audit_path = workdir / "admin-audit.jsonl"
    audit_path.unlink(missing_ok=True)
    audit = AuditLog(audit_path)
    registry = CollectorRegistry()
    service = build_service(checkpoint, settings, registry=registry)
    app = create_api(checkpoint, settings, service=service, audit_log=audit, registry=registry)
    with serve_in_thread(app) as base_url:
        return asyncio.run(_served(base_url, service.engine, audit))


def run_serving_demo(workdir: Path) -> dict[str, Any]:
    """Run every step and return a JSON-serialisable summary."""
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint = build_random_checkpoint(workdir / "checkpoint", seed=3, max_seq_len=512)
    tokenizer = checkpoint.tokenizer
    prompts = [
        tokenizer.encode(turn)[: DEMO_ENGINE_CONFIG.max_prompt_tokens] for turn in USER_TURNS[:3]
    ]
    summary: dict[str, Any] = {
        "parameters": checkpoint.model.config.parameter_count,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "manifest_sha256": checkpoint.manifest.sha256(),
        "equivalence": _equivalence(checkpoint, prompts),
        "prefix_cache": _prefix_cache_comparison(checkpoint, tokenizer),
        "admission": _admission_demo(checkpoint),
        "cancellation": _cancellation(checkpoint),
        "batching": _batching_comparison(checkpoint),
    }
    summary["api"] = _api_demo(checkpoint, workdir)
    return summary


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_report(summary: dict[str, Any]) -> str:
    """A short human-readable report of the demo summary, ending in the comparison table."""
    eq = summary["equivalence"]
    pc = summary["prefix_cache"]
    ad = summary["admission"]
    ca = summary["cancellation"]
    api = summary["api"]
    lines = [
        f"model: {summary['parameters']:,} parameters, vocab {summary['tokenizer_vocab_size']}, "
        f"manifest sha256 {summary['manifest_sha256'][:16]}...",
        f"equivalence: single-prompt max |dlogit| {eq['single_prompt_max_abs_logit_gap']:.3g}; "
        f"engine max |dlogit| {eq['engine_max_abs_logit_gap']:.3g} over "
        f"{eq['engine_logit_rows_checked']} rows with {eq['engine_preemptions']} preemptions",
        f"prefix cache: hit ratio {pc['hit_ratio_cache_on']:.3f}, prefill tokens saved "
        f"{pc['prefill_tokens_saved_fraction']:.1%}, TTFT p50 reduction "
        f"{pc['ttft_reduction_fraction']:.1%}",
        f"admission: {ad['submitted']} offered, {ad['completed']} admitted, {ad['rejected']} "
        f"refused early, {ad['admitted_on_time_fraction']:.0%} of admitted met the "
        f"{ad['deadline_seconds']}s deadline",
        f"cancellation: {ca['blocks_held_before_abort']} blocks held -> "
        f"{ca['blocks_held_after_abort']} after abort; pool whole after cache flush: "
        f"{ca['free_blocks_after_cache_flush'] == ca['pool_size']}",
        f"server: healthz {api['healthz']}, readyz {api['readyz']}, no key "
        f"{api['unauthenticated_status']}, batch key asking for interactive "
        f"{api['batch_key_asking_for_interactive_status']}, hang-up aborted "
        f"{api['hangup_aborted_requests']} request(s), sequence blocks after hang-up "
        f"{api['sequence_blocks_after_hangup']}, audit verified {api['audit_verified']}",
    ]
    for name, row in api["open_loop_burst"]["classes"].items():
        lines.append(
            f"http burst [{name}]: sent {row['sent']}, completed {row['completed']}, errors "
            f"{row['errors'] or '{}'}, TTFT p50/p99 {row['ttft_p50']}/{row['ttft_p99']}s"
        )
    columns = (
        "label", "completed", "tokens_per_second", "ttft_p50", "ttft_p99", "latency_p99",
        "on_time", "goodput_per_second",
    )  # fmt: skip
    rows = [[_fmt(row.get(column, "")) for column in columns] for row in summary["batching"]]
    widths = [max(len(column), *(len(row[i]) for row in rows)) for i, column in enumerate(columns)]
    lines.append("")
    lines.append("continuous vs static batching (same arrivals, same model):")
    lines.append(
        "  ".join(column.ljust(width) for column, width in zip(columns, widths, strict=True))
    )
    for row in rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))
    return "\n".join(lines)
