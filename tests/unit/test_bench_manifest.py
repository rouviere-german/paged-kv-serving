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
