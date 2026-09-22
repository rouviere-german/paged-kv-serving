"""The serving benchmark embeds a reproducibility manifest in its results."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from paged_kv_serving.engine.weights import Checkpoint
from paged_kv_serving.ml.manifest import RunManifest, sha256_file

pytestmark = pytest.mark.unit


def load_bench(repo_root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bench_serving", repo_root / "bench" / "bench_serving.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_run_manifest_pins_seed_packages_and_checkpoint_digests(
    repo_root: Path, checkpoint: Checkpoint
) -> None:
    bench = load_bench(repo_root)
    payload = bench.run_manifest(
        checkpoint, params={"quick": True}, metrics={"1.5x_goodput_per_second_ratio": 2.0}
    )
    manifest = RunManifest.model_validate(payload)
    assert manifest.seed == bench.CHECKPOINT_SEED
    assert set(manifest.packages) >= {"numpy", "safetensors"}
    digests = {dataset.name: dataset.sha256 for dataset in manifest.datasets}
    assert digests["weights.safetensors"] == sha256_file(
        checkpoint.directory / "weights.safetensors"
    )
    assert set(digests) == {"weights.safetensors", "tokenizer.json", "manifest.json"}
    assert manifest.metrics == {"1.5x_goodput_per_second_ratio": 2.0}


def test_ratios_and_rows_are_averaged_over_seeds_with_their_range(repo_root: Path) -> None:
    bench = load_bench(repo_root)
    means, ranges = bench.aggregate_ratios([{"a": 1.0}, {"a": 3.0}, {"a": 2.0}])
    assert means == {"a": 2.0}
    assert ranges["a"] == {"mean": 2.0, "min": 1.0, "max": 3.0, "per_seed": [1.0, 3.0, 2.0]}
    rows = [
        {
            "label": "1x",
            "scheduler": "static",
            "seed": float(seed),
            "goodput_per_second": goodput,
            "slo_attainment": slo,
            "ttft_p99": ttft,
        }
        for seed, goodput, slo, ttft in ((17, 1.0, 0.5, 2.0), (18, 3.0, 1.0, 4.0))
    ]
    (row,) = bench.aggregate_rows(rows)
    assert (row["label"], row["scheduler"], row["seeds"]) == ("1x", "static", 2)
    assert row["goodput_per_second"] == 2.0
    assert "seed" not in row
    assert row["slo_attainment_range"] == {"mean": 0.75, "min": 0.5, "max": 1.0}
