"""The serving comparison: continuous batching against static batching on one arrival schedule.

Usage:
  uv run python bench/bench_serving.py           # full sweep, writes bench/results/serving.json
  uv run python bench/bench_serving.py --quick   # fewer load points, for a local smoke check
  uv run python bench/bench_serving.py --check   # full sweep, enforce serving_thresholds.json

Method, and the parts of it that matter:

* **One schedule, two engines.** The arrival times are drawn once per load point from a seeded
  Poisson process and replayed against both engines, so the two runs see the same work in the same
  order rather than the same *distribution* of work.
* **Open loop.** Requests are submitted at their scheduled time whatever the server is doing, and
  every latency is measured from that scheduled time. A slow run therefore looks slow instead of
  quietly lowering its own offered load (coordinated omission).
* **Load points relative to capacity.** Capacity is measured first, as the request rate the
  continuous engine sustains when saturated, and the sweep then offers 0.5x, 1x, 1.5x and 3x of
  it. "Three times capacity" means three times *this machine's* capacity, so the table means the
  same thing on a faster one.
* **Goodput, not just throughput.** A request counts only if it finished inside its deadline.

Absolute numbers are small: this is float32 NumPy in Python, and the per-iteration overhead of
that would dominate any comparison against a real GPU engine. Only the ratios between the two
schedulers on the same machine are claimed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paged_kv_serving.engine.admission import RejectionReason
from paged_kv_serving.engine.config import EngineConfig, SloClass
from paged_kv_serving.engine.engine import Engine
from paged_kv_serving.engine.loadgen import (
    LoadResult,
    WorkloadSpec,
    build_workload,
    percentile,
    run_open_loop,
    summarize,
)
from paged_kv_serving.engine.sequence import Request, SamplingParams
from paged_kv_serving.engine.simulation import drain_on_clock, simulated_engine
from paged_kv_serving.engine.static import StaticBatchingEngine
from paged_kv_serving.engine.threads import limit_blas_threads
from paged_kv_serving.engine.weights import (
    MANIFEST_FILE,
    TOKENIZER_FILE,
    WEIGHTS_FILE,
    Checkpoint,
    build_random_checkpoint,
)
from paged_kv_serving.ml.manifest import create_manifest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "serving.json"
THRESHOLDS = ROOT / "bench" / "serving_thresholds.json"

ENGINE = EngineConfig(
    block_size=16,
    num_blocks=256,
    max_batched_tokens=128,
    max_running_seqs=48,
    max_prompt_tokens=192,
    max_new_tokens=32,
    max_context_tokens=256,
)
#: Both schedulers get the same KV memory. Static batching must reserve a whole context per slot,
#: so the same pool buys it 16 slots; the paged allocator spends the same tokens on whatever the
#: sequences actually use. Deriving the batch size rather than fixing it is what keeps the
#: comparison about scheduling instead of about who was given more memory.
STATIC_BATCH_SIZE = (ENGINE.num_blocks * ENGINE.block_size) // ENGINE.max_context_tokens
BASE_SPEC = WorkloadSpec(
    num_requests=48,
    rate=20.0,
    prompt_tokens=(24, 128),
    max_new_tokens=(4, 32),
    interactive_fraction=0.5,
    # Tight enough that head-of-line blocking actually costs goodput: roughly three times the
    # median latency the continuous engine reaches at its own capacity.
    interactive_deadline=1.0,
    shared_prefix_tokens=32,
    shared_prefix_fraction=0.5,
    vocab_size=256,
    seed=17,
)


def _engine(checkpoint: Checkpoint, *, prefix_cache: bool = True) -> Engine:
    engine = Engine(checkpoint.model, dataclasses.replace(ENGINE, enable_prefix_cache=prefix_cache))
    engine.profile_latency(repeats=2, warmup=1)
    return engine


def measure_capacity(checkpoint: Checkpoint, spec: WorkloadSpec) -> float:
    """Requests per second the continuous engine sustains when the queue is never empty.

    Everything arrives at once, so the engine is saturated from the first iteration and the result
    is a service rate rather than an arrival rate.
    """
    saturated = dataclasses.replace(spec, rate=10_000.0, interactive_deadline=None)
    engine = _engine(checkpoint)
    requests = [
        dataclasses.replace(request, arrival_time=0.0, deadline_seconds=None)
        for request in build_workload(saturated)
    ]
    start = time.perf_counter()
    outputs = engine.generate(requests)
    elapsed = max(time.perf_counter() - start, 1e-9)
    engine.check_invariants()
    return len(outputs) / elapsed


def _static_row(
    checkpoint: Checkpoint, requests: list[Request], label: str
) -> dict[str, float | str]:
    static = StaticBatchingEngine(
        checkpoint.model,
        batch_size=STATIC_BATCH_SIZE,
        max_context_tokens=ENGINE.max_context_tokens,
        block_size=ENGINE.block_size,
    )
    start = static.clock()
    outputs = static.generate(
        [
            dataclasses.replace(request, arrival_time=start + request.arrival_time)
            for request in requests
        ]
    )
    wall = max(static.clock() - start, 1e-9)
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
    tokens = sum(output.output_tokens for output in outputs)
    ttfts = [output.ttft for output in outputs]
    latencies = [output.latency for output in outputs]
    return {
        "label": label,
        "scheduler": "static",
        "submitted": float(len(requests)),
        "completed": float(len(outputs)),
        "rejected": 0.0,
        "rejection_rate": 0.0,
        "wall_seconds": round(wall, 4),
        "output_tokens": float(tokens),
        "tokens_per_second": round(tokens / wall, 2),
        "ttft_p50": round(percentile(ttfts, 50), 4),
        "ttft_p99": round(percentile(ttfts, 99), 4),
        "latency_p50": round(percentile(latencies, 50), 4),
        "latency_p99": round(percentile(latencies, 99), 4),
        "deadlined": float(len(deadlines)),
        "on_time": float(on_time),
        "slo_attainment": round(on_time / len(deadlines), 4) if deadlines else 1.0,
        "goodput_per_second": round(on_time / wall, 3),
        "padding_tokens": float(static.total_padding_tokens),
        "wasted_decode_tokens": float(static.total_wasted_decode_tokens),
        "kv_tokens_reserved": float(static.kv_tokens_reserved),
    }


def _continuous_row(
    checkpoint: Checkpoint, requests: list[Request], label: str
) -> tuple[dict[str, float | str], LoadResult, Engine]:
    engine = _engine(checkpoint)
    result = run_open_loop(engine, requests)
    engine.check_invariants()
    row: dict[str, float | str] = {**summarize(result, label=label), "scheduler": "continuous"}
    row["preemptions"] = float(engine.scheduler.stats.preemptions)
    row["prefix_cache_hit_ratio"] = round(engine.prefix_cache.stats.hit_ratio, 4)
    row["kv_tokens_peak_used"] = float(engine.blocks.num_used_blocks * ENGINE.block_size)
    row["mean_batch_tokens"] = round(engine.scheduler.stats.mean_batch_size, 2)
    return row, result, engine


def load_sweep(
    checkpoint: Checkpoint, capacity: float, multipliers: tuple[float, ...], spec: WorkloadSpec
) -> list[dict[str, float | str]]:
    """Run both schedulers at each multiple of the measured capacity."""
    rows: list[dict[str, float | str]] = []
    for multiplier in multipliers:
        point = dataclasses.replace(spec, rate=max(capacity * multiplier, 0.1))
        requests = build_workload(point)
        label = f"{multiplier:g}x"
        continuous, _, _ = _continuous_row(checkpoint, requests, label)
        continuous["offered_rate"] = round(point.rate, 2)
        static = _static_row(checkpoint, requests, label)
        static["offered_rate"] = round(point.rate, 2)
        rows.extend((continuous, static))
    return rows


def prefix_cache_effect(checkpoint: Checkpoint, spec: WorkloadSpec) -> dict[str, float]:
    """TTFT and prefill work on a shared-system-prompt workload, cache on versus off.

    Deliberately run well below capacity: at saturation, time to first token is dominated by
    queueing and the cache's effect on *prefill* would be hidden behind it. The prompts are a
    128-token shared system prompt plus a short user turn, which is the shape a chat deployment
    actually has.
    """
    shared = dataclasses.replace(
        spec,
        shared_prefix_fraction=1.0,
        shared_prefix_tokens=128,
        prompt_tokens=(16, 48),
        max_new_tokens=(4, 12),
        rate=3.0,
        num_requests=20,
        interactive_deadline=None,
    )
    requests = build_workload(shared)
    measured: dict[str, float] = {}
    for label, enabled in (("off", False), ("on", True)):
        engine = _engine(checkpoint, prefix_cache=enabled)
        result = run_open_loop(engine, requests)
        engine.check_invariants()
        ttfts = [output.ttft for output in result.outputs]
        measured[f"ttft_p50_{label}"] = round(percentile(ttfts, 50), 5)
        measured[f"ttft_p99_{label}"] = round(percentile(ttfts, 99), 5)
        measured[f"prefill_tokens_{label}"] = float(engine.scheduler.stats.prefill_tokens)
        measured[f"hit_ratio_{label}"] = round(engine.prefix_cache.stats.hit_ratio, 4)
    measured["ttft_p50_reduction"] = round(
        1.0 - measured["ttft_p50_on"] / max(measured["ttft_p50_off"], 1e-9), 4
    )
    measured["prefill_tokens_saved"] = round(
        1.0 - measured["prefill_tokens_on"] / max(measured["prefill_tokens_off"], 1.0), 4
    )
    return measured


def memory_utilization(checkpoint: Checkpoint, spec: WorkloadSpec) -> dict[str, float]:
    """How much of the KV memory each scheduler holds actually contains a live token.

    Paged: after every iteration, tokens resident in live sequences divided by the tokens their
    blocks can hold (the prefix cache is off, so every used block belongs to a sequence). The only
    waste is the unfilled tail of each sequence's last block.

    Static: each of the ``batch_size`` slots reserves a whole ``max_context_tokens`` region when
    the batch starts. Its utilisation is measured at the batch's *end*, when every row is as long
    as it will ever get -- the most favourable moment for static batching.
    """
    requests = build_workload(dataclasses.replace(spec, num_requests=24, rate=30.0))
    engine = _engine(checkpoint, prefix_cache=False)
    for request in requests:
        engine.submit(dataclasses.replace(request, arrival_time=engine.clock()))
    paged_ratios: list[float] = []
    peak_blocks = 0
    while engine.has_work:
        engine.step()
        used = engine.blocks.num_used_blocks
        peak_blocks = max(peak_blocks, used)
        live = sum(sequence.num_computed_tokens for sequence in engine.live_sequences())
        if used:
            paged_ratios.append(live / (used * ENGINE.block_size))
    engine.check_invariants()

    static = StaticBatchingEngine(
        checkpoint.model,
        batch_size=STATIC_BATCH_SIZE,
        max_context_tokens=ENGINE.max_context_tokens,
        block_size=ENGINE.block_size,
    )
    outputs = static.generate(
        [dataclasses.replace(request, arrival_time=0.0) for request in requests]
    )
    static_ratios = []
    for offset in range(0, len(outputs), STATIC_BATCH_SIZE):
        group = outputs[offset : offset + STATIC_BATCH_SIZE]
        final_tokens = sum(output.prompt_tokens + output.output_tokens for output in group)
        static_ratios.append(final_tokens / static.kv_tokens_reserved)
    return {
        "paged_mean_utilization": round(statistics.fmean(paged_ratios), 4),
        "paged_min_utilization": round(min(paged_ratios), 4),
        "static_end_of_batch_utilization": round(statistics.fmean(static_ratios), 4),
        "paged_peak_kv_tokens": float(peak_blocks * ENGINE.block_size),
        "static_reserved_kv_tokens": float(static.kv_tokens_reserved),
    }


def admission_accuracy(checkpoint: Checkpoint) -> dict[str, float]:
    """Deadline admission with a fake clock and an exact latency model, with and without it.

    The executor consumes exactly the time the latency model predicts, so what is measured is the
    admission test itself. At each load level the same 300 arrivals are run twice: with admission
    control, and with it effectively disabled (a vanishing safety factor admits everything). The
    claims are that no *admitted* request misses its deadline, and that refusing early buys more
    on-time completions than accepting everything.

    Simulated service rate: 2e-4 s per token and 128-token iterations, about 5,000 tokens/s, so
    with ~110 tokens per request capacity is roughly 45 requests/s.
    """
    rows: dict[str, float] = {}
    total_missed = 0
    vocab = min(checkpoint.model.config.vocab_size, 256)
    for spacing_ms, load in ((44, "0.5x"), (22, "1x"), (11, "2x"), (5.5, "4x")):
        rng = np.random.default_rng(int(spacing_ms * 10))
        requests = [
            Request(
                request_id=f"adm-{index:04d}",
                prompt_token_ids=tuple(
                    int(v) for v in rng.integers(1, vocab, int(rng.integers(16, 160)))
                ),
                sampling=SamplingParams(max_new_tokens=int(rng.integers(4, 32)), seed=index),
                slo_class=SloClass.INTERACTIVE if index % 3 else SloClass.BATCH,
                deadline_seconds=float(rng.choice([0.75, 1.5, 3.0])),
                arrival_time=index * spacing_ms / 1000.0,
            )
            for index in range(300)
        ]
        for mode, safety in (("admission", 1.0), ("accept_all", 1e-9)):
            engine, clock = simulated_engine(
                checkpoint.model, ENGINE, seconds_per_token=2e-4, safety_factor=safety
            )
            outputs = drain_on_clock(engine, clock, requests)
            engine.check_invariants()
            missed = sum(1 for output in outputs if output.met_deadline is False)
            on_time = sum(1 for output in outputs if output.met_deadline is True)
            label = f"{load}_{mode}"
            rows[f"{label}_admitted"] = float(len(outputs))
            rows[f"{label}_rejected"] = float(
                engine.admission.rejections.get(RejectionReason.DEADLINE_INFEASIBLE, 0)
            )
            rows[f"{label}_missed"] = float(missed)
            rows[f"{label}_on_time"] = float(on_time)
            if mode == "admission":
                total_missed += missed
    rows["admitted_missed_total"] = float(total_missed)
    return rows


def ratios(rows: list[dict[str, float | str]]) -> dict[str, float]:
    """Continuous over static, per load point, for the metrics the README quotes."""
    out: dict[str, float] = {}
    by_label: dict[str, dict[str, dict[str, float | str]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), {})[str(row["scheduler"])] = row
    for label, pair in by_label.items():
        if {"continuous", "static"} - set(pair):  # pragma: no cover - both are always present
            continue
        continuous, static = pair["continuous"], pair["static"]
        for metric in ("goodput_per_second", "tokens_per_second"):
            denominator = float(static[metric]) or 1e-9
            out[f"{label}_{metric}_ratio"] = round(float(continuous[metric]) / denominator, 3)
        out[f"{label}_ttft_p99_ratio"] = round(
            float(static["ttft_p99"]) / max(float(continuous["ttft_p99"]), 1e-9), 3
        )
    return out


CHECKPOINT_SEED = 3
MANIFEST_PACKAGES = ("numpy", "safetensors", "fastapi", "pydantic", "threadpoolctl")


def run_manifest(
    checkpoint: Checkpoint, *, params: dict[str, Any], metrics: dict[str, float]
) -> dict[str, Any]:
    """Reproducibility record for one run: seed, packages and the SHA-256 of every checkpoint file.

    The checkpoint is rebuilt from ``CHECKPOINT_SEED`` on every run, so equal file digests across
    two result files mean both runs measured the same weights and tokenizer.
    """
    manifest = create_manifest(
        seed=CHECKPOINT_SEED,
        datasets=[
            checkpoint.directory / name for name in (WEIGHTS_FILE, TOKENIZER_FILE, MANIFEST_FILE)
        ],
        params=params,
        metrics=metrics,
        packages=MANIFEST_PACKAGES,
    )
    return manifest.model_dump(mode="json")


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": __import__("os").cpu_count(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
    }


def check_thresholds(payload: dict[str, Any]) -> list[str]:
    if not THRESHOLDS.is_file():  # pragma: no cover - the file ships with the repo
        return ["bench/serving_thresholds.json is missing"]
    limits = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems: list[str] = []
    measured = {*limits.get("min", {}), *limits.get("max", {})}
    if not measured:  # pragma: no cover - the file always lists thresholds
        return ["bench/serving_thresholds.json lists no thresholds"]
    for key, floor in limits.get("min", {}).items():
        value = _lookup(payload, key)
        if value is None:
            problems.append(f"{key}: not measured")
        elif value < floor:
            problems.append(f"{key} = {value} is below the floor {floor}")
    for key, ceiling in limits.get("max", {}).items():
        value = _lookup(payload, key)
        if value is None:
            problems.append(f"{key}: not measured")
        elif value > ceiling:
            problems.append(f"{key} = {value} exceeds the ceiling {ceiling}")
    return problems


def _lookup(payload: dict[str, Any], path: str) -> float | None:
    """Follow a ``section/key`` path into the result payload. ``/`` separates, because the keys
    themselves contain dots (``1.5x_goodput_per_second_ratio``)."""
    node: Any = payload
    for part in path.split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) else None


def run(*, quick: bool) -> dict[str, Any]:
    multipliers = (1.0, 1.5) if quick else (0.5, 1.0, 1.5, 3.0)
    spec = dataclasses.replace(BASE_SPEC, num_requests=16 if quick else BASE_SPEC.num_requests)
    blas = limit_blas_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = build_random_checkpoint(
            Path(tmp) / "checkpoint", seed=CHECKPOINT_SEED, max_seq_len=512
        )
        started = time.perf_counter()
        capacity = statistics.median(
            measure_capacity(checkpoint, spec) for _ in range(1 if quick else 3)
        )
        rows = load_sweep(checkpoint, capacity, multipliers, spec)
        payload: dict[str, Any] = {
            "machine": machine(),
            "model": {
                "parameters": checkpoint.model.config.parameter_count,
                "layers": checkpoint.model.config.n_layers,
                "d_model": checkpoint.model.config.d_model,
                "vocab_size": checkpoint.model.config.vocab_size,
            },
            "engine": dataclasses.asdict(ENGINE),
            "static_batch_size": STATIC_BATCH_SIZE,
            "kv_tokens_per_scheduler": ENGINE.num_blocks * ENGINE.block_size,
            "capacity_requests_per_second": round(capacity, 2),
            "load_sweep": rows,
            "ratios": ratios(rows),
            "prefix_cache": prefix_cache_effect(checkpoint, spec),
            "memory": memory_utilization(checkpoint, spec),
            "admission": admission_accuracy(checkpoint),
            "bench_seconds": round(time.perf_counter() - started, 1),
        }
        payload["machine"]["blas_threads"] = blas
        payload["run_manifest"] = run_manifest(
            checkpoint,
            params={
                "engine": dataclasses.asdict(ENGINE),
                "static_batch_size": STATIC_BATCH_SIZE,
                "workload_seed": spec.seed,
                "quick": quick,
            },
            metrics={key: float(value) for key, value in payload["ratios"].items()},
        )
    return payload


def _print_table(payload: dict[str, Any]) -> None:
    print(f"capacity (continuous, saturated): {payload['capacity_requests_per_second']} req/s")
    print()
    print(
        "| load | scheduler | completed | rejected | tokens/s | TTFT p50 | TTFT p99 "
        "| latency p99 | SLO | goodput/s |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["load_sweep"]:
        print(
            f"| {row['label']} | {row['scheduler']} | {row['completed']:.0f} "
            f"| {row['rejected']:.0f} | {row['tokens_per_second']} | {row['ttft_p50']} "
            f"| {row['ttft_p99']} | {row['latency_p99']} | {row['slo_attainment']} "
            f"| {row['goodput_per_second']} |"
        )
    print()
    print("| ratio (continuous / static) | value |")
    print("|---|---:|")
    for key, value in payload["ratios"].items():
        print(f"| {key} | {value} |")
    print()
    cache = payload["prefix_cache"]
    print("| prefix cache (shared system prompt) | off | on |")
    print("|---|---:|---:|")
    print(f"| TTFT p50 (s) | {cache['ttft_p50_off']} | {cache['ttft_p50_on']} |")
    print(
        f"| prefill tokens | {cache['prefill_tokens_off']:.0f} | {cache['prefill_tokens_on']:.0f} |"
    )
    print(f"| block hit ratio | {cache['hit_ratio_off']} | {cache['hit_ratio_on']} |")
    print(
        f"| TTFT p50 reduction | | {cache['ttft_p50_reduction']:.1%} "
        f"(prefill tokens saved {cache['prefill_tokens_saved']:.1%}) |"
    )
    print()
    memory = payload["memory"]
    print("| KV memory | value |")
    print("|---|---:|")
    for key, value in memory.items():
        print(f"| {key} | {value} |")
    print()
    print("| admission (fake clock, exact latency model) | value |")
    print("|---|---:|")
    for key, value in payload["admission"].items():
        print(f"| {key} | {value:.0f} |")
    print()
    print(f"bench wall time: {payload['bench_seconds']} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="continuous versus static batching")
    parser.add_argument("--quick", action="store_true", help="fewer load points and requests")
    parser.add_argument("--check", action="store_true", help="enforce serving_thresholds.json")
    args = parser.parse_args(argv)

    payload = run(quick=args.quick)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    _print_table(payload)
    if args.check:
        problems = check_thresholds(payload)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all serving benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
