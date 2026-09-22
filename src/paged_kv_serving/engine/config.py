"""Model and engine configuration.

Both dataclasses are frozen and fully validated at construction, because a bad shape only shows
up thousands of matrix multiplications later, and a bad block-size/context combination silently
corrupts the KV cache instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

__all__ = ["EngineConfig", "ModelConfig", "SloClass"]


class SloClass(Enum):
    """Service class of a request. Lower ``rank`` is scheduled first."""

    INTERACTIVE = ("interactive", 0)
    BATCH = ("batch", 1)

    def __init__(self, label: str, rank: int) -> None:
        self.label = label
        self.rank = rank

    @classmethod
    def parse(cls, value: str) -> SloClass:
        for member in cls:
            if member.label == value:
                return member
        raise ValueError(f"unknown SLO class {value!r}")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shape of the decoder-only transformer.

    Pre-norm RMSNorm blocks, rotary position embeddings, multi-head attention with one KV head
    per query head, and a SwiGLU feed-forward network. The output projection is tied to the token
    embedding, which is what keeps a 1M-parameter model from being mostly vocabulary.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.rope_theta <= 1.0:
            raise ValueError("rope_theta must be > 1")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be > 0")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def parameter_count(self) -> int:
        """Number of float32 parameters, counting the tied embedding once."""
        embedding = self.vocab_size * self.d_model
        attention = self.d_model * (3 * self.d_model) + self.d_model * self.d_model
        feed_forward = 3 * self.d_model * self.d_ff
        norms = 2 * self.d_model
        return embedding + self.n_layers * (attention + feed_forward + norms) + self.d_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        allowed = set(cls.__slots__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown model config fields: {', '.join(unknown)}")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Scheduling, memory and admission limits for one engine instance.

    ``max_batched_tokens`` is the per-iteration token budget shared by decode and chunked
    prefill. Decode tokens are taken first (one per running sequence), so a long prompt can never
    stall sequences that are already generating -- that is the head-of-line blocking that static
    batching suffers from.
    """

    block_size: int = 16
    num_blocks: int = 512
    max_batched_tokens: int = 256
    max_running_seqs: int = 32
    max_prompt_tokens: int = 512
    max_new_tokens: int = 128
    max_context_tokens: int = 1024
    enable_prefix_cache: bool = True
    # Fraction of blocks kept free before a prefill is admitted into the running set, so decode
    # steps of already-running sequences do not immediately run out of memory.
    watermark: float = 0.02
    # Exponentially weighted correction of the profiled latency model (0 disables correction).
    latency_ewma_alpha: float = 0.2
    # Multiplier applied to the predicted completion time before comparing against the deadline.
    admission_safety_factor: float = 1.0
    # While a higher SLO class has work in the engine, prefill of lower classes may take at most
    # this share of each iteration's token budget. Iteration time grows with batch tokens, so an
    # uncapped flood of batch prompts would stretch every interactive decode step to a full-budget
    # iteration (docs/adr/0005-lower-class-prefill-share.md).
    lower_class_prefill_share: float = 0.25

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.max_batched_tokens < 1:
            raise ValueError("max_batched_tokens must be >= 1")
        if self.max_running_seqs < 1:
            raise ValueError("max_running_seqs must be >= 1")
        if self.max_prompt_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("max_prompt_tokens and max_new_tokens must be >= 1")
        if self.max_context_tokens < self.max_prompt_tokens + self.max_new_tokens:
            raise ValueError("max_context_tokens must cover max_prompt_tokens + max_new_tokens")
        if not 0.0 <= self.watermark < 1.0:
            raise ValueError("watermark must be in [0, 1)")
        if not 0.0 <= self.latency_ewma_alpha <= 1.0:
            raise ValueError("latency_ewma_alpha must be in [0, 1]")
        if self.admission_safety_factor <= 0.0:
            raise ValueError("admission_safety_factor must be > 0")
        if not 0.0 < self.lower_class_prefill_share <= 1.0:
            raise ValueError("lower_class_prefill_share must be in (0, 1]")
        if self.num_blocks * self.block_size < self.max_context_tokens:
            raise ValueError("the KV cache cannot hold a single max-length sequence")

    @property
    def cache_capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def lower_class_prefill_tokens(self) -> int:
        """Per-iteration prefill allowance of classes below the highest one with work."""
        return max(1, int(self.max_batched_tokens * self.lower_class_prefill_share))
