"""Checkpoint format: safetensors weights, a JSON tokenizer and a hash manifest.

Loading a model is loading untrusted bytes. The classic PyTorch checkpoint is a pickle, so a
tampered file is remote code execution before a single token is generated; ``numpy.load`` with
pickling switched on is the same hole. This repository never uses either. Weights are stored as
safetensors -- a length-prefixed header of shapes and dtypes followed by raw little-endian bytes,
with no opcodes to interpret -- and the tokenizer is plain JSON.

Format alone is not integrity, so a checkpoint also carries ``manifest.json``: the model shape,
the parameter count, and the SHA-256 and size of every other file. :func:`load_checkpoint`
recomputes those digests and refuses the checkpoint on any mismatch -- and refuses a manifest that
does not list the weights and the tokenizer at all, since a manifest that vouches for nothing would
otherwise pass every check it contains. Artifact names are plain file names, so a manifest cannot
point verification at a file outside the checkpoint directory. The manifest is the root of
that trust and is not self-protecting, so a deployment that cares pins its digest out of band and
passes it as ``expect_manifest_sha256``; the admin weight-swap endpoint requires exactly that.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from safetensors.numpy import load_file, save_file

from paged_kv_serving.core.canonical import canonical_json, sha256_hex
from paged_kv_serving.core.fileio import atomic_write_text
from paged_kv_serving.engine.config import ModelConfig
from paged_kv_serving.engine.corpus import DEMO_CORPUS
from paged_kv_serving.engine.model import LanguageModel, TransformerWeights, random_weights
from paged_kv_serving.engine.tokenizer import ByteBPETokenizer
from paged_kv_serving.ml.manifest import sha256_file

__all__ = [
    "MANIFEST_FILE",
    "REQUIRED_ARTIFACTS",
    "TOKENIZER_FILE",
    "WEIGHTS_FILE",
    "ArtifactDigest",
    "Checkpoint",
    "CheckpointIntegrityError",
    "WeightsManifest",
    "build_random_checkpoint",
    "load_checkpoint",
    "write_checkpoint",
]

WEIGHTS_FILE = "weights.safetensors"
TOKENIZER_FILE = "tokenizer.json"
MANIFEST_FILE = "manifest.json"
#: Files a loadable checkpoint must have digests for. Everything else a manifest lists is checked
#: too, but these are the ones load_checkpoint parses.
REQUIRED_ARTIFACTS = (WEIGHTS_FILE, TOKENIZER_FILE)
# No separators, no leading dot (so neither ".." nor hidden files), bounded length.
_PLAIN_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CheckpointIntegrityError(ValueError):
    """A checkpoint file does not match the digest recorded in its manifest."""


class ArtifactDigest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("name")
    @classmethod
    def _plain_file_name(cls, value: str) -> str:
        if not _PLAIN_FILE_NAME.fullmatch(value) or ".." in value:
            raise ValueError("an artifact name must be a plain file name inside the checkpoint")
        return value


class WeightsManifest(BaseModel):
    """What a checkpoint claims to be, and the digests that prove the files match."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    created_at: str
    architecture: dict[str, JsonValue]
    parameter_count: int = Field(ge=1)
    tokenizer_vocab_size: int = Field(ge=1)
    artifacts: tuple[ArtifactDigest, ...]

    @field_validator("artifacts")
    @classmethod
    def _unique_names(cls, value: tuple[ArtifactDigest, ...]) -> tuple[ArtifactDigest, ...]:
        names = [artifact.name for artifact in value]
        if len(names) != len(set(names)):
            raise ValueError("an artifact is listed more than once")
        return value

    def sha256(self) -> str:
        """Digest of the manifest itself: the value to pin out of band."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))

    def digest_of(self, name: str) -> ArtifactDigest:
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        raise CheckpointIntegrityError(f"manifest does not cover {name}")

    def to_model_config(self) -> ModelConfig:
        return ModelConfig.from_dict(dict(self.architecture))

    def write(self, path: Path) -> None:
        atomic_write_text(path, self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> WeightsManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A verified checkpoint: model, tokenizer and the manifest that vouched for them."""

    model: LanguageModel
    tokenizer: ByteBPETokenizer
    manifest: WeightsManifest
    directory: Path


def write_checkpoint(
    directory: Path,
    config: ModelConfig,
    weights: TransformerWeights,
    tokenizer: ByteBPETokenizer,
    *,
    created_at: str | None = None,
) -> WeightsManifest:
    """Write weights, tokenizer and manifest into ``directory`` and return the manifest."""
    weights.validate(config)
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError("the tokenizer vocabulary does not fit the model's embedding table")
    directory.mkdir(parents=True, exist_ok=True)
    tensors: Mapping[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]] = weights.to_tensors()
    save_file(dict(tensors), str(directory / WEIGHTS_FILE))
    tokenizer.save(directory / TOKENIZER_FILE)
    manifest = WeightsManifest(
        created_at=created_at or datetime.now(UTC).isoformat(),
        architecture=dict(config.to_dict()),
        parameter_count=config.parameter_count,
        tokenizer_vocab_size=tokenizer.vocab_size,
        artifacts=tuple(
            ArtifactDigest(
                name=name,
                sha256=sha256_file(directory / name),
                size_bytes=(directory / name).stat().st_size,
            )
            for name in (WEIGHTS_FILE, TOKENIZER_FILE)
        ),
    )
    manifest.write(directory / MANIFEST_FILE)
    return manifest


def load_checkpoint(directory: Path, *, expect_manifest_sha256: str | None = None) -> Checkpoint:
    """Verify every digest in the manifest, then load the model and tokenizer.

    Nothing is deserialised before the digests match, so a swapped weights file is rejected
    without being parsed.
    """
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise CheckpointIntegrityError(f"no {MANIFEST_FILE} in {directory}")
    manifest = WeightsManifest.load(manifest_path)
    if expect_manifest_sha256 is not None and manifest.sha256() != expect_manifest_sha256:
        raise CheckpointIntegrityError("manifest digest does not match the pinned value")
    for name in REQUIRED_ARTIFACTS:
        manifest.digest_of(name)  # raises when the manifest does not vouch for a file we parse
    for artifact in manifest.artifacts:
        path = directory / artifact.name
        if not path.is_file():
            raise CheckpointIntegrityError(f"missing checkpoint file {artifact.name}")
        if path.stat().st_size != artifact.size_bytes:
            raise CheckpointIntegrityError(f"{artifact.name} has an unexpected size")
        if sha256_file(path) != artifact.sha256:
            raise CheckpointIntegrityError(f"{artifact.name} does not match its manifest digest")

    config = manifest.to_model_config()
    tensors = load_file(str(directory / WEIGHTS_FILE))
    weights = TransformerWeights.from_tensors(
        {name: np.asarray(value, dtype=np.float32) for name, value in tensors.items()}, config
    )
    if config.parameter_count != manifest.parameter_count:
        raise CheckpointIntegrityError("parameter count does not match the manifest")
    tokenizer = ByteBPETokenizer.load(directory / TOKENIZER_FILE)
    if tokenizer.vocab_size != manifest.tokenizer_vocab_size:
        raise CheckpointIntegrityError("tokenizer vocabulary does not match the manifest")
    return Checkpoint(
        model=LanguageModel(config, weights),
        tokenizer=tokenizer,
        manifest=manifest,
        directory=directory,
    )


def build_random_checkpoint(
    directory: Path,
    *,
    seed: int = 0,
    n_layers: int = 4,
    d_model: int = 128,
    n_heads: int = 4,
    d_ff: int = 320,
    max_seq_len: int = 1024,
    tokenizer_vocab_size: int = 512,
    corpus: tuple[str, ...] = DEMO_CORPUS,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> Checkpoint:
    """Create a deterministic random-init checkpoint. No training, no download, no network.

    The claim this repository makes is numerical equivalence between a paged, continuously batched
    engine and a naive recompute, and that claim is independent of whether the weights were ever
    trained. Random weights keep the fixtures reproducible and the test suite fast.
    """
    tokenizer = ByteBPETokenizer.train(corpus, vocab_size=tokenizer_vocab_size)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
    )
    weights = random_weights(config, seed=seed)
    write_checkpoint(directory, config, weights, tokenizer, created_at=created_at)
    return load_checkpoint(directory)
