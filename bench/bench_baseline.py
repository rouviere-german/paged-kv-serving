"""Micro-benchmarks for the engine's hot paths, with CI regression thresholds.

Usage:
  uv run python bench/bench_baseline.py           # run, write bench/results/latest.json
  uv run python bench/bench_baseline.py --check   # also exit 1 if bench/thresholds.json is exceeded

These are the operations one iteration is made of -- allocate and free a block table, hash a
prompt for the prefix cache, run the paged attention kernel, sample a token, decide admission --
plus the correctness gap between the paged path and a cache-free recompute, which is tracked here
as a number so a regression in it fails CI like any other.

Thresholds are deliberately loose (several times the numbers measured on a laptop) because shared
CI runners are noisy; they catch order-of-magnitude regressions, not small drifts. The one tight
threshold is the equivalence gap, which is a correctness bound rather than a timing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paged_kv_serving.engine.admission import (
    AdmissionController,
    LatencyModel,
    WorkItem,
)
from paged_kv_serving.engine.attention import paged_attention
from paged_kv_serving.engine.blocks import BlockManager, RecordingBlockStore
from paged_kv_serving.engine.config import EngineConfig
from paged_kv_serving.engine.engine import Engine
from paged_kv_serving.engine.equivalence import (
    RecordingExecutor,
    engine_logit_gap,
    max_logit_gap,
    sequence_tokens,
)
from paged_kv_serving.engine.prefix_cache import block_hash_chain
from paged_kv_serving.engine.runner import BatchEntry, ExecutionBatch, ModelRunner
from paged_kv_serving.engine.sampling import sample_token
from paged_kv_serving.engine.sequence import Request, SamplingParams
from paged_kv_serving.engine.threads import limit_blas_threads
from paged_kv_serving.engine.weights import Checkpoint, build_random_checkpoint

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "latest.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

BENCH_ENGINE = EngineConfig(
    block_size=16,
    num_blocks=192,
    max_batched_tokens=128,
    max_running_seqs=12,
    max_prompt_tokens=192,
    max_new_tokens=24,
    max_context_tokens=256,
)


def _timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = np.asarray(samples, dtype=np.float64) / 1_000.0
    return {
        "runs": runs,
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
    }


def bench_block_allocation() -> dict[str, float]:
    """Allocate and free a 16-block table: the per-request cost of the paged allocator."""
    manager = BlockManager(num_blocks=4096, block_size=16, store=RecordingBlockStore())

    def cycle() -> None:
        blocks = manager.allocate_many(16)
        manager.free_many(blocks)

    return _timed(cycle, runs=5_000)


def bench_copy_on_write() -> dict[str, float]:
    """Share a block three ways and take a private copy: the cost of divergence."""
    manager = BlockManager(num_blocks=4096, block_size=16, store=RecordingBlockStore())

    def cycle() -> None:
        block = manager.allocate()
        manager.incref(block)
        manager.incref(block)
        private = manager.prepare_for_write(block)
        manager.free(private)
        manager.free(block)
        manager.free(block)

    return _timed(cycle, runs=5_000)


def bench_prefix_hash_chain() -> dict[str, float]:
    """Hash a 512-token prompt into 32 chained block keys."""
    tokens = list(range(512))
    return _timed(
        lambda: block_hash_chain(tokens, block_size=16, num_blocks=32, scope=b"tenant"),
        runs=2_000,
    )


def bench_paged_attention_decode() -> dict[str, float]:
    """One decode query over a 256-token context spread across 16 blocks."""
    rng = np.random.default_rng(0)
    heads, dim, block_size, blocks = 4, 32, 16, 16
    keys = rng.standard_normal((blocks, block_size, heads, dim)).astype(np.float32)
    values = rng.standard_normal((blocks, block_size, heads, dim)).astype(np.float32)
    query = rng.standard_normal((1, heads, dim)).astype(np.float32)
    positions = np.asarray([blocks * block_size - 1], dtype=np.int64)
    table = np.arange(blocks, dtype=np.int64)
    return _timed(
        lambda: paged_attention(
            query, positions, keys, values, table, blocks * block_size, dim**-0.5
        ),
        runs=2_000,
    )


def bench_sampling_top_p() -> dict[str, float]:
    """Nucleus sampling over a 512-token vocabulary: one uniform draw per token."""
    rng = np.random.default_rng(1)
    logits = rng.standard_normal(512).astype(np.float32)
    params = SamplingParams(max_new_tokens=1, temperature=0.8, top_p=0.9, seed=3)
    generator = np.random.default_rng(3)
    return _timed(lambda: sample_token(logits, params, generator), runs=5_000)


def bench_admission_decision() -> dict[str, float]:
    """The EDF feasibility test against 64 committed requests."""
    controller = AdmissionController(
        latency=LatencyModel(1e-3, 1e-4), max_batched_tokens=128, max_queued_sequences=4096
    )
    committed = [
        WorkItem(rank=index % 2, deadline=1.0 + index * 0.05, prefill_tokens=48, decode_tokens=16)
        for index in range(64)
    ]
    candidate = WorkItem(rank=0, deadline=5.0, prefill_tokens=96, decode_tokens=32)
    return _timed(
        lambda: controller.decide(now=0.0, candidate=candidate, committed=committed), runs=2_000
    )


def bench_iteration(checkpoint: Checkpoint) -> dict[str, float]:
    """A full 128-token forward pass: 16 decode sequences plus one prefill chunk."""
    runner = ModelRunner(checkpoint.model, num_blocks=256, block_size=16)
    entries = [
        BatchEntry(
            seq_id=f"decode-{index}",
            token_ids=(1,),
            positions=(63,),
            block_table=tuple(range(index * 4, index * 4 + 4)),
            logits_index=0,
        )
        for index in range(16)
    ]
    entries.append(
        BatchEntry(
            seq_id="prefill",
            token_ids=tuple(range(1, 113)),
            positions=tuple(range(112)),
            block_table=tuple(range(128, 135)),
            logits_index=111,
        )
    )
    batch = ExecutionBatch(tuple(entries))
    return _timed(lambda: runner.execute(batch), runs=20, warmup=3)


def bench_engine_throughput(checkpoint: Checkpoint) -> dict[str, float]:
    """End-to-end: 24 queued requests through the continuous engine, tokens per second."""
    rng = np.random.default_rng(4)
    vocab = min(checkpoint.model.config.vocab_size, 256)
    requests = [
        Request(
            request_id=f"bench-{index}",
            prompt_token_ids=tuple(int(v) for v in rng.integers(1, vocab, 48)),
            sampling=SamplingParams(max_new_tokens=12, seed=index),
        )
        for index in range(24)
    ]
    engine = Engine(checkpoint.model, BENCH_ENGINE)
    start = time.perf_counter()
    outputs = engine.generate(requests)
    elapsed = max(time.perf_counter() - start, 1e-9)
    engine.check_invariants()
    tokens = sum(output.output_tokens for output in outputs)
    return {
        "runs": 1,
        "requests": float(len(outputs)),
        "output_tokens": float(tokens),
        "seconds": round(elapsed, 4),
        "tokens_per_second": round(tokens / elapsed, 2),
        "seconds_per_request": round(elapsed / max(len(outputs), 1), 5),
    }


def bench_equivalence(checkpoint: Checkpoint) -> dict[str, float]:
    """Correctness, tracked as a number: paged and chunked versus a cache-free recompute."""
    rng = np.random.default_rng(5)
    vocab = min(checkpoint.model.config.vocab_size, 256)
    worst = 0.0
    for _ in range(4):
        length = int(rng.integers(20, 120))
        tokens = [int(value) for value in rng.integers(1, vocab, length)]
        for chunks in ([length], [1] * length, [7, 5, 13, length - 25]):
            if any(size < 1 for size in chunks):
                continue
            worst = max(
                worst, max_logit_gap(checkpoint.model, tokens, block_size=16, chunks=chunks)
            )
        blocks = -(-length // 16)
        worst = max(
            worst,
            max_logit_gap(
                checkpoint.model, tokens, block_size=16, block_order=list(reversed(range(blocks)))
            ),
        )
    return {"runs": 4, "max_abs_logit_gap": float(f"{worst:.4g}")}


def bench_engine_equivalence(checkpoint: Checkpoint) -> dict[str, float]:
    """The same gap through the real scheduler, on a pool small enough to force preemption.

    Every logits row the engine samples from -- mixed prefill/decode batches, chunk boundaries,
    prefix-cache hits, recomputation after preemption, forked samples -- is compared against a
    dense recompute of that sequence.
    """
    import dataclasses  # noqa: PLC0415 - only this benchmark needs it

    config = dataclasses.replace(BENCH_ENGINE, num_blocks=20, max_batched_tokens=48, watermark=0.0)
    engine = Engine(checkpoint.model, config)
    recorder = RecordingExecutor(engine.runner)
    engine.executor = recorder
    rng = np.random.default_rng(6)
    vocab = min(checkpoint.model.config.vocab_size, 256)
    shared = [int(value) for value in rng.integers(1, vocab, 32)]
    requests = []
    for index in range(16):
        tail = [int(value) for value in rng.integers(1, vocab, int(rng.integers(8, 90)))]
        requests.append(
            Request(
                request_id=f"eq-{index}",
                prompt_token_ids=tuple([*shared, *tail] if index % 2 else tail),
                sampling=SamplingParams(
                    max_new_tokens=int(rng.integers(4, 24)),
                    temperature=0.0 if index % 3 else 0.9,
                    seed=index,
                ),
                n=2 if index % 5 == 0 else 1,
            )
        )
    outputs = engine.generate(requests)
    engine.check_invariants()
    tokens = sequence_tokens(
        requests, {output.request_id: [s.token_ids for s in output.samples] for output in outputs}
    )
    gap, rows = engine_logit_gap(checkpoint.model, recorder.records, tokens)
    return {
        "runs": 1,
        "max_abs_logit_gap": float(f"{gap:.4g}"),
        "rows_checked": float(rows),
        "preemptions": float(engine.scheduler.stats.preemptions),
        "prefix_hit_blocks": float(engine.prefix_cache.stats.hit_blocks),
    }


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
        "blas_threads": 1,
    }


def run() -> dict[str, dict[str, float]]:
    limit_blas_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = build_random_checkpoint(Path(tmp) / "checkpoint", seed=3, max_seq_len=512)
        return {
            "block_allocate_free_16": bench_block_allocation(),
            "copy_on_write_block": bench_copy_on_write(),
            "prefix_hash_chain_512_tokens": bench_prefix_hash_chain(),
            "paged_attention_decode_256ctx": bench_paged_attention_decode(),
            "sample_top_p_512_vocab": bench_sampling_top_p(),
            "admission_decide_64_committed": bench_admission_decision(),
            "engine_iteration_128_tokens": bench_iteration(checkpoint),
            "engine_throughput_24_requests": bench_engine_throughput(checkpoint),
            "paged_vs_reference_equivalence": bench_equivalence(checkpoint),
            "engine_vs_reference_equivalence": bench_engine_equivalence(checkpoint),
        }


def check_thresholds(results: dict[str, dict[str, float]]) -> list[str]:
    """Compare against bench/thresholds.json: ``min_`` keys are floors, the rest ceilings."""
    thresholds: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems = []
    for name, limits in thresholds.items():
        if name not in results:
            problems.append(f"{name}: benchmark missing")
            continue
        for metric, limit in limits.items():
            floor = metric.startswith("min_")
            key = metric[4:] if floor else metric
            value = results[name][key]
            if floor and value < limit:
                problems.append(f"{name}.{key} = {value} is below the floor {limit}")
            elif not floor and value > limit:
                problems.append(f"{name}.{key} = {value} exceeds threshold {limit}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="engine micro-benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine(), "results": results}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("| benchmark | runs | p50 (us) | p95 (us) | other |")
    print("|---|---:|---:|---:|---|")
    for name, stats in results.items():
        extra = ", ".join(
            f"{key}={value}"
            for key, value in stats.items()
            if key not in {"runs", "p50_us", "p95_us"}
        )
        p50 = stats.get("p50_us", float("nan"))
        p95 = stats.get("p95_us", float("nan"))
        print(f"| {name} | {stats['runs']:.0f} | {p50} | {p95} | {extra} |")
    if args.check:
        problems = check_thresholds(results)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
