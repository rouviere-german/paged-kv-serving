"""A decoder-only transformer in float32 NumPy, plus the naive reference it is checked against.

The block is the now-conventional one: pre-norm RMSNorm, rotary position embeddings on queries
and keys, multi-head causal attention, and a SwiGLU feed-forward network, with the output
projection tied to the token embedding. References: Zhang and Sennrich, "Root Mean Square Layer
Normalization" (2019); Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
(2021); Shazeer, "GLU Variants Improve Transformer" (2020).

Quality is explicitly not the point here -- the weights are random. What matters is that this
module defines *the* answer: :func:`reference_forward` recomputes a sequence from scratch with
dense attention and no cache at all, and the paged, continuously batched engine has to produce
the same logits to within float32 noise no matter how it batches, chunks, shares or preempts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from paged_kv_serving.engine.config import ModelConfig

__all__ = [
    "FloatArray",
    "LanguageModel",
    "LayerWeights",
    "TransformerWeights",
    "apply_rope",
    "dense_causal_attention",
    "random_weights",
    "rms_norm",
    "rope_tables",
    "silu",
    "softmax_last",
]

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


def rms_norm(x: FloatArray, weight: FloatArray, eps: float) -> FloatArray:
    """x / sqrt(mean(x^2) + eps) * weight, accumulated in float32 over the last axis."""
    scale = np.sqrt(np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True) + eps)
    return ((x / scale) * weight).astype(np.float32)


def silu(x: FloatArray) -> FloatArray:
    """x * sigmoid(x), written so large negative inputs cannot overflow ``exp``."""
    return (x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))).astype(np.float32)


def softmax_last(x: FloatArray) -> FloatArray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return (exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)).astype(np.float32)


def rope_tables(max_seq_len: int, head_dim: int, theta: float) -> tuple[FloatArray, FloatArray]:
    """Cosine and sine tables of shape ``(max_seq_len, head_dim // 2)``."""
    if head_dim % 2:
        raise ValueError("head_dim must be even for rotary embeddings")
    half = head_dim // 2
    inverse_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float64) * 2.0 / head_dim))
    angles = np.outer(np.arange(max_seq_len, dtype=np.float64), inverse_freq)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope(x: FloatArray, positions: IntArray, cos: FloatArray, sin: FloatArray) -> FloatArray:
    """Rotate ``x`` of shape ``(tokens, heads, head_dim)`` by the angle of each token's position.

    Rotary embeddings encode *relative* position in the dot product, which is exactly why a paged
    cache is allowed to store a token's key once and reuse it from any batch: the key depends only
    on the token's own absolute position, never on the other sequences it is batched with.
    """
    half = x.shape[-1] // 2
    cos_p = cos[positions][:, None, :]
    sin_p = sin[positions][:, None, :]
    left, right = x[..., :half], x[..., half:]
    return np.concatenate([left * cos_p - right * sin_p, right * cos_p + left * sin_p], axis=-1)


@dataclass(frozen=True, slots=True)
class LayerWeights:
    attn_norm: FloatArray
    qkv_proj: FloatArray
    out_proj: FloatArray
    ffn_norm: FloatArray
    gate_proj: FloatArray
    up_proj: FloatArray
    down_proj: FloatArray


@dataclass(frozen=True, slots=True)
class TransformerWeights:
    """Every parameter of the model. Arrays are float32 and never mutated after loading."""

    token_embedding: FloatArray
    layers: tuple[LayerWeights, ...]
    final_norm: FloatArray

    def validate(self, config: ModelConfig) -> None:
        """Raise ValueError unless every array has the shape, dtype and finiteness expected."""
        d, ff, layers = config.d_model, config.d_ff, config.n_layers
        expected: list[tuple[str, FloatArray, tuple[int, ...]]] = [
            ("token_embedding", self.token_embedding, (config.vocab_size, d)),
            ("final_norm", self.final_norm, (d,)),
        ]
        if len(self.layers) != layers:
            raise ValueError(f"expected {layers} layers, found {len(self.layers)}")
        for index, layer in enumerate(self.layers):
            expected += [
                (f"layers.{index}.attn_norm", layer.attn_norm, (d,)),
                (f"layers.{index}.qkv_proj", layer.qkv_proj, (d, 3 * d)),
                (f"layers.{index}.out_proj", layer.out_proj, (d, d)),
                (f"layers.{index}.ffn_norm", layer.ffn_norm, (d,)),
                (f"layers.{index}.gate_proj", layer.gate_proj, (d, ff)),
                (f"layers.{index}.up_proj", layer.up_proj, (d, ff)),
                (f"layers.{index}.down_proj", layer.down_proj, (ff, d)),
            ]
        for name, array, shape in expected:
            if array.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, found {array.shape}")
            if array.dtype != np.float32:
                raise ValueError(f"{name}: expected float32, found {array.dtype}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name}: contains NaN or infinite values")

    def to_tensors(self) -> dict[str, FloatArray]:
        """Flat name -> array mapping, the form written to and read from safetensors."""
        tensors: dict[str, FloatArray] = {
            "token_embedding": self.token_embedding,
            "final_norm": self.final_norm,
        }
        for index, layer in enumerate(self.layers):
            for field_name in LayerWeights.__slots__:
                tensors[f"layers.{index}.{field_name}"] = getattr(layer, field_name)
        return tensors

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, FloatArray], config: ModelConfig
    ) -> TransformerWeights:
        """Rebuild from a flat mapping, rejecting missing or unexpected tensor names."""
        names = {"token_embedding", "final_norm"}
        for index in range(config.n_layers):
            names |= {f"layers.{index}.{name}" for name in LayerWeights.__slots__}
        missing = sorted(names - set(tensors))
        if missing:
            raise ValueError(f"missing tensors: {', '.join(missing)}")
        unexpected = sorted(set(tensors) - names)
        if unexpected:
            raise ValueError(f"unexpected tensors: {', '.join(unexpected)}")
        layers = tuple(
            LayerWeights(
                **{
                    name: np.ascontiguousarray(tensors[f"layers.{index}.{name}"])
                    for name in LayerWeights.__slots__
                }
            )
            for index in range(config.n_layers)
        )
        weights = cls(
            token_embedding=np.ascontiguousarray(tensors["token_embedding"]),
            layers=layers,
            final_norm=np.ascontiguousarray(tensors["final_norm"]),
        )
        weights.validate(config)
        return weights


def random_weights(config: ModelConfig, seed: int = 0) -> TransformerWeights:
    """Deterministic random initialisation, scaled so activations stay in a sane range.

    The model is never trained in this repository, so this is not an initialisation scheme meant
    to train well -- it only has to give a well-conditioned, reproducible function of the input.
    """
    rng = np.random.default_rng(seed)

    def normal(*shape: int, scale: float) -> FloatArray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    d, ff = config.d_model, config.d_ff
    layers = tuple(
        LayerWeights(
            attn_norm=np.ones(d, dtype=np.float32),
            qkv_proj=normal(d, 3 * d, scale=d**-0.5),
            out_proj=normal(d, d, scale=d**-0.5),
            ffn_norm=np.ones(d, dtype=np.float32),
            gate_proj=normal(d, ff, scale=d**-0.5),
            up_proj=normal(d, ff, scale=d**-0.5),
            down_proj=normal(ff, d, scale=ff**-0.5),
        )
        for _ in range(config.n_layers)
    )
    weights = TransformerWeights(
        token_embedding=normal(config.vocab_size, d, scale=0.05),
        layers=layers,
        final_norm=np.ones(d, dtype=np.float32),
    )
    weights.validate(config)
    return weights


def dense_causal_attention(
    queries: FloatArray, keys: FloatArray, values: FloatArray, scale: float
) -> FloatArray:
    """Textbook attention over a whole sequence: one softmax over one score matrix.

    Shapes are ``(tokens, heads, head_dim)``. This is the reduction order the paged kernel's
    tiled online softmax has to agree with, and the only place in the package where the whole
    score matrix exists at once.
    """
    length = queries.shape[0]
    scores = np.matmul(queries.transpose(1, 0, 2), keys.transpose(1, 2, 0)) * np.float32(scale)
    causal = np.triu(np.ones((length, length), dtype=bool), k=1)
    scores = np.where(causal[None, :, :], np.float32(-np.inf), scores)
    weights = softmax_last(scores)
    attended: FloatArray = np.matmul(weights, values.transpose(1, 0, 2)).transpose(1, 0, 2)
    return attended.astype(np.float32)


class LanguageModel:
    """Config, weights and the rotary tables, with the cache-free reference forward pass."""

    __slots__ = ("_cos", "_sin", "config", "weights")

    def __init__(self, config: ModelConfig, weights: TransformerWeights) -> None:
        weights.validate(config)
        self.config = config
        self.weights = weights
        self._cos, self._sin = rope_tables(config.max_seq_len, config.head_dim, config.rope_theta)

    @property
    def rope(self) -> tuple[FloatArray, FloatArray]:
        return self._cos, self._sin

    @property
    def attention_scale(self) -> float:
        return 1.0 / math.sqrt(self.config.head_dim)

    def project_qkv(
        self, hidden: FloatArray, layer: LayerWeights, positions: IntArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Normalise, project to q/k/v of shape ``(tokens, heads, head_dim)`` and apply RoPE."""
        config = self.config
        normed = rms_norm(hidden, layer.attn_norm, config.norm_eps)
        fused = normed @ layer.qkv_proj
        shaped = fused.reshape(-1, 3, config.n_heads, config.head_dim)
        queries = apply_rope(shaped[:, 0], positions, self._cos, self._sin)
        keys = apply_rope(shaped[:, 1], positions, self._cos, self._sin)
        values = np.ascontiguousarray(shaped[:, 2])
        return queries, keys, values

    def feed_forward(self, hidden: FloatArray, layer: LayerWeights) -> FloatArray:
        normed = rms_norm(hidden, layer.ffn_norm, self.config.norm_eps)
        gated = silu(normed @ layer.gate_proj) * (normed @ layer.up_proj)
        return (gated @ layer.down_proj).astype(np.float32)

    def logits_from_hidden(self, hidden: FloatArray) -> FloatArray:
        """Final norm and the tied output projection."""
        normed = rms_norm(hidden, self.weights.final_norm, self.config.norm_eps)
        return (normed @ self.weights.token_embedding.T).astype(np.float32)

    def embed(self, token_ids: IntArray) -> FloatArray:
        return np.ascontiguousarray(self.weights.token_embedding[token_ids])

    def reference_forward(self, token_ids: list[int] | IntArray) -> FloatArray:
        """Logits for every position, recomputed from scratch with no KV cache at all.

        This is the oracle for the equivalence tests and for the correctness benchmark.
        """
        tokens = np.asarray(token_ids, dtype=np.int64)
        if tokens.ndim != 1 or tokens.size == 0:
            raise ValueError("token_ids must be a non-empty 1-D sequence")
        if tokens.size > self.config.max_seq_len:
            raise ValueError("sequence longer than max_seq_len")
        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab_size):
            raise ValueError("token id outside the vocabulary")
        positions = np.arange(tokens.size, dtype=np.int64)
        hidden = self.embed(tokens)
        for layer in self.weights.layers:
            queries, keys, values = self.project_qkv(hidden, layer, positions)
            attended = dense_causal_attention(queries, keys, values, self.attention_scale)
            hidden = hidden + attended.reshape(tokens.size, self.config.d_model) @ layer.out_proj
            hidden = (hidden + self.feed_forward(hidden, layer)).astype(np.float32)
        return self.logits_from_hidden(hidden)
