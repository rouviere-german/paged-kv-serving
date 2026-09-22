from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paged_kv_serving.engine.config import ModelConfig
from paged_kv_serving.engine.corpus import DEMO_CORPUS
from paged_kv_serving.engine.model import random_weights
from paged_kv_serving.engine.tokenizer import ByteBPETokenizer
from paged_kv_serving.engine.weights import (
    MANIFEST_FILE,
    TOKENIZER_FILE,
    WEIGHTS_FILE,
    CheckpointIntegrityError,
    WeightsManifest,
    build_random_checkpoint,
    load_checkpoint,
    write_checkpoint,
)

pytestmark = pytest.mark.unit

CONFIG = ModelConfig(vocab_size=300, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=64)


@pytest.fixture
def written(tmp_path: Path) -> Path:
    tokenizer = ByteBPETokenizer.train(["paged kv serving " * 20], vocab_size=290)
    write_checkpoint(tmp_path, CONFIG, random_weights(CONFIG, seed=1), tokenizer)
    return tmp_path


def test_a_written_checkpoint_loads_and_verifies(written: Path) -> None:
    checkpoint = load_checkpoint(written)
    assert checkpoint.model.config == CONFIG
    assert checkpoint.manifest.parameter_count == CONFIG.parameter_count
    assert set(written.iterdir()) >= {
        written / WEIGHTS_FILE,
        written / TOKENIZER_FILE,
        written / MANIFEST_FILE,
    }
    # The same bytes give the same logits as the in-memory model they came from.
    reference = random_weights(CONFIG, seed=1)
    assert np.array_equal(checkpoint.model.weights.token_embedding, reference.token_embedding)


def test_pinning_the_manifest_digest_detects_a_swapped_checkpoint(written: Path) -> None:
    manifest = WeightsManifest.load(written / MANIFEST_FILE)
    load_checkpoint(written, expect_manifest_sha256=manifest.sha256())
    with pytest.raises(CheckpointIntegrityError, match="pinned value"):
        load_checkpoint(written, expect_manifest_sha256="0" * 64)


def test_tampered_weights_are_refused_before_they_are_parsed(written: Path) -> None:
    path = written / WEIGHTS_FILE
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF  # flip one bit of one float
    path.write_bytes(bytes(raw))
    with pytest.raises(CheckpointIntegrityError, match="does not match its manifest digest"):
        load_checkpoint(written)


def test_a_resized_file_is_refused(written: Path) -> None:
    path = written / TOKENIZER_FILE
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="unexpected size"):
        load_checkpoint(written)


def test_a_missing_file_or_manifest_is_refused(written: Path, tmp_path: Path) -> None:
    (written / TOKENIZER_FILE).unlink()
    with pytest.raises(CheckpointIntegrityError, match="missing checkpoint file"):
        load_checkpoint(written)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CheckpointIntegrityError, match=r"no manifest\.json"):
        load_checkpoint(empty)


def test_a_manifest_that_lies_about_the_architecture_is_refused(written: Path) -> None:
    manifest_path = written / MANIFEST_FILE
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["architecture"]["d_model"] = 32
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected shape"):
        load_checkpoint(written)


def test_a_manifest_with_the_wrong_parameter_count_is_refused(written: Path) -> None:
    manifest_path = written / MANIFEST_FILE
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["parameter_count"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="parameter count"):
        load_checkpoint(written)


def test_a_manifest_with_the_wrong_tokenizer_size_is_refused(written: Path) -> None:
    manifest_path = written / MANIFEST_FILE
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["tokenizer_vocab_size"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="tokenizer vocabulary"):
        load_checkpoint(written)


def test_a_tokenizer_larger_than_the_embedding_table_is_refused(tmp_path: Path) -> None:
    tokenizer = ByteBPETokenizer.train(DEMO_CORPUS, vocab_size=400)
    assert tokenizer.vocab_size == 400
    small = ModelConfig(vocab_size=300, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=64)
    with pytest.raises(ValueError, match="does not fit the model"):
        write_checkpoint(tmp_path, small, random_weights(small, seed=1), tokenizer)


def test_manifest_digest_covers_every_field(written: Path) -> None:
    manifest = WeightsManifest.load(written / MANIFEST_FILE)
    assert manifest.digest_of(WEIGHTS_FILE).size_bytes > 0
    with pytest.raises(CheckpointIntegrityError, match="does not cover"):
        manifest.digest_of("nope.bin")
    altered = manifest.model_copy(update={"created_at": "2027-01-01T00:00:00+00:00"})
    assert altered.sha256() != manifest.sha256()


def test_build_random_checkpoint_is_reproducible(tmp_path: Path) -> None:
    first = build_random_checkpoint(tmp_path / "a", seed=2, n_layers=1, d_model=16, n_heads=2)
    second = build_random_checkpoint(tmp_path / "b", seed=2, n_layers=1, d_model=16, n_heads=2)
    assert first.manifest.sha256() == second.manifest.sha256()
    different = build_random_checkpoint(tmp_path / "c", seed=3, n_layers=1, d_model=16, n_heads=2)
    assert different.manifest.sha256() != first.manifest.sha256()


def test_the_checkpoint_format_contains_no_pickle(written: Path) -> None:
    """safetensors is a length-prefixed header plus raw bytes: no opcodes to interpret."""
    raw = (written / WEIGHTS_FILE).read_bytes()
    assert b"__reduce__" not in raw
    assert not raw.startswith(b"\x80")  # a pickle protocol marker
    header_length = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_length])
    assert all(entry["dtype"] == "F32" for entry in header.values() if isinstance(entry, dict))


def _rewrite_manifest(directory: Path, **changes: object) -> None:
    path = directory / MANIFEST_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("kept", [(), (TOKENIZER_FILE,), (WEIGHTS_FILE,)])
def test_a_manifest_must_cover_the_weights_and_the_tokenizer(
    written: Path, kept: tuple[str, ...]
) -> None:
    """Review regression: a manifest listing no artifacts vouched for any weights at all.

    With ``artifacts`` emptied and the weights edited, the checkpoint used to load.
    """
    data = json.loads((written / MANIFEST_FILE).read_text(encoding="utf-8"))
    _rewrite_manifest(
        written, artifacts=[item for item in data["artifacts"] if item["name"] in kept]
    )
    with pytest.raises(CheckpointIntegrityError, match="does not cover"):
        load_checkpoint(written)


@pytest.mark.parametrize(
    "name", ["../weights.safetensors", "sub/weights.safetensors", "a\\b", "..", ".hidden", ""]
)
def test_artifact_names_are_plain_file_names(written: Path, name: str) -> None:
    data = json.loads((written / MANIFEST_FILE).read_text(encoding="utf-8"))
    extra = {**data["artifacts"][0], "name": name}
    _rewrite_manifest(written, artifacts=[*data["artifacts"], extra])
    with pytest.raises(ValueError, match=r"plain file name|validation error"):
        load_checkpoint(written)


def test_an_artifact_may_not_be_listed_twice(written: Path) -> None:
    data = json.loads((written / MANIFEST_FILE).read_text(encoding="utf-8"))
    _rewrite_manifest(written, artifacts=[*data["artifacts"], data["artifacts"][0]])
    with pytest.raises(ValueError, match="more than once"):
        load_checkpoint(written)
