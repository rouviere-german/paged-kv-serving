"""Requests, sampling parameters and the per-sample scheduling state.

One bookkeeping rule governs prefill, chunked prefill, decode and recomputation alike:

* ``token_ids`` is the prompt followed by everything sampled so far;
* ``num_computed_tokens`` counts the leading tokens whose keys and values are in the KV cache;
* an iteration runs ``token_ids[num_computed : num_computed + chunk]`` at those same positions;
* the iteration produces logits exactly when the chunk reaches the end of ``token_ids``, and the
  token sampled from them is appended.

Prefill (chunk of many), decode (chunk of one) and a preempted sequence re-running its prompt
*and* its own output are then the same operation with different chunk sizes, which is why the
scheduler can mix them freely inside one token budget.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from paged_kv_serving.engine.config import SloClass

__all__ = [
    "GREEDY_TEMPERATURE",
    "FinishReason",
    "Request",
    "SamplingParams",
    "Sequence",
    "SequenceStatus",
    "finish_reason_for",
    "new_sequence_id",
]

_SEQUENCE_COUNTER = itertools.count()
#: Temperatures below this sample greedily. Dividing float32 logits by a smaller temperature
#: overflows to inf (and 1e-300 rounds to 0.0 in float32), and the argmax is what such a
#: temperature asks for anyway.
GREEDY_TEMPERATURE = 1e-5


def new_sequence_id(request_id: str, index: int) -> str:
    return f"{request_id}#{index}"


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    ABORTED = "aborted"


class FinishReason(Enum):
    LENGTH = "length"
    STOP = "stop"
    CONTEXT_LIMIT = "context_limit"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Per-request sampling. A temperature below ``GREEDY_TEMPERATURE`` (including 0) means
    greedy, which ignores ``top_p``."""

    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

    @property
    def greedy(self) -> bool:
        return self.temperature < GREEDY_TEMPERATURE


@dataclass(frozen=True, slots=True)
class Request:
    """A submitted completion request. ``n`` samples share the prompt's KV blocks."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingParams = SamplingParams()
    n: int = 1
    slo_class: SloClass = SloClass.INTERACTIVE
    # Seconds after arrival by which the whole request must finish. None means best effort.
    deadline_seconds: float | None = None
    key_id: str = "anonymous"
    arrival_time: float = 0.0

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("the prompt must contain at least one token")
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be > 0")

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def deadline(self) -> float | None:
        """Absolute deadline, or None for best effort."""
        if self.deadline_seconds is None:
            return None
        return self.arrival_time + self.deadline_seconds

    def prompt_sha256(self) -> str:
        """Fingerprint used in logs and audit records instead of the prompt text."""
        payload = b"".join(token.to_bytes(4, "little") for token in self.prompt_token_ids)
        return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class Sequence:
    """One sample of a request, and the unit the scheduler moves between queues."""

    seq_id: str
    request: Request
    sample_index: int = 0
    token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: FinishReason | None = None
    # Number of leading blocks taken from the prefix cache on the most recent (re)admission.
    cached_prefix_blocks: int = 0
    # Running slots held for samples that will fork from this one when its prompt completes
    # (``n - 1`` on the first sample of an n-way request until it forks, else 0). Admission
    # counts them, so the forks never push the running set past its cap.
    fork_slots: int = 0
    # Chained hash of every full block of this sequence, in block-table order. Its length is the
    # number of blocks that have been offered to the prefix cache.
    block_hashes: list[bytes] = field(default_factory=list)
    preemption_count: int = 0
    first_token_time: float | None = None
    finish_time: float | None = None
    # Monotonic tiebreaker so equal (class, deadline) keys keep first-come order.
    admission_order: int = field(default_factory=lambda: next(_SEQUENCE_COUNTER))
    rng: np.random.Generator | None = None

    @classmethod
    def from_request(cls, request: Request, sample_index: int) -> Sequence:
        sequence = cls(
            seq_id=new_sequence_id(request.request_id, sample_index),
            request=request,
            sample_index=sample_index,
            token_ids=list(request.prompt_token_ids),
        )
        # A per-sample generator makes output depend only on (seed, sample index), never on how
        # the sequence happened to be batched with others.
        sequence.rng = np.random.default_rng(
            np.random.SeedSequence(entropy=request.sampling.seed, spawn_key=(sample_index,))
        )
        return sequence

    # --- lengths -------------------------------------------------------------------------

    @property
    def num_tokens(self) -> int:
        """Prompt plus sampled tokens."""
        return len(self.token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.request.prompt_len

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.request.prompt_len :]

    @property
    def is_decoding(self) -> bool:
        """True once the prompt is fully computed and each iteration runs a single token."""
        return self.num_computed_tokens > 0 and self.num_uncomputed_tokens == 1

    @property
    def is_finished(self) -> bool:
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.ABORTED)

    def blocks_needed(self, block_size: int, up_to_tokens: int | None = None) -> int:
        """Total blocks a block table must hold to cover ``up_to_tokens`` (default: all)."""
        total = self.num_tokens if up_to_tokens is None else up_to_tokens
        return -(-total // block_size)

    # --- state transitions ---------------------------------------------------------------

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def reset_for_recompute(self) -> None:
        """Forget cached KV so the sequence can be re-prefilled from position zero.

        Sampled tokens are kept: they become part of the text that is recomputed, so preemption
        is invisible in the output. Only the work is lost, never a token.
        """
        self.num_computed_tokens = 0
        self.block_table.clear()
        self.block_hashes.clear()
        self.cached_prefix_blocks = 0
        self.status = SequenceStatus.PREEMPTED
        self.preemption_count += 1

    def finish(self, reason: FinishReason, now: float) -> None:
        self.status = (
            SequenceStatus.ABORTED if reason is FinishReason.ABORTED else SequenceStatus.FINISHED
        )
        self.finish_reason = reason
        self.finish_time = now

    def priority_key(self) -> tuple[int, float, int]:
        """Queue order: SLO class first, then earliest deadline, then arrival order.

        Requests without a deadline sort after every deadlined request of the same class, so an
        interactive request with a deadline is never overtaken by a best-effort one.
        """
        deadline = self.request.deadline
        return (
            self.request.slo_class.rank,
            float("inf") if deadline is None else deadline,
            self.admission_order,
        )


def finish_reason_for(
    sequence: Sequence, token: int, *, eos_token_id: int | None, max_context_tokens: int
) -> FinishReason | None:
    """Why this sequence is done, or None if it should keep generating.

    The order matters: a stop token ends a request even on its last allowed step, and the context
    cap is the last line of defence. Admission already refuses a request whose prompt plus
    ``max_new_tokens`` would not fit, so in a correctly configured engine the cap is reached only
    by a sequence whose limits changed under it -- but a serving loop that silently overruns its
    KV reservation corrupts other requests, so the check stays.
    """
    sampling = sequence.request.sampling
    if eos_token_id is not None and token == eos_token_id and not sampling.ignore_eos:
        return FinishReason.STOP
    if sequence.num_output_tokens >= sampling.max_new_tokens:
        return FinishReason.LENGTH
    if sequence.num_tokens >= max_context_tokens:
        return FinishReason.CONTEXT_LIMIT
    return None
