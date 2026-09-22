"""Deadline-aware admission control: reject early instead of accepting work that cannot fit.

A server that accepts everything under a burst does not serve more requests, it just turns
timeouts into the common case: every client waits, every client gives up, and the work already
done is thrown away. Saying "503, retry in 2s" while the answer is still useful is strictly better
for the caller and for the queue behind it.

The test is a processor-demand check in the style of earliest-deadline-first schedulability
analysis (Liu and Layland, 1973; Horn, 1974; Baruah, Mok and Rosier, 1990), but counted in
*iterations* rather than in seconds of fluid work, because an LLM engine is not a fluid server:

* **Prefill is shared and divisible.** Prompt tokens from every queued request compete for the
  part of the per-iteration token budget that decode does not take, so they add up like work on a
  single machine.
* **Decode is serial per request and parallel across requests.** One iteration advances *every*
  running sequence by exactly one token. A request asking for 20 output tokens therefore needs 20
  iterations no matter how idle the machine is -- and twenty such requests still need 20
  iterations between them, not 400.

Treating decode as divisible work is the mistake that makes a fluid model promise deadlines it
cannot keep: it predicts that an idle machine finishes a 20-token answer in the time of two
iterations. The bound used here is

    iterations(D) = ceil(prefill tokens ahead of D / prefill tokens per iteration)
                    + the largest number of output tokens still owed by any request due by D

and a candidate is admitted only if, for every class and every committed deadline, that many
iterations still fit before the deadline. The scheduler runs classes strictly in rank order and
orders within a class by deadline, so "ahead of D" is all work of higher-ranked classes plus
same-class work due no later than D.

An iteration is costed at its worst case, ``L(max_batched_tokens)``, from a latency model profiled
at start-up and corrected by an EWMA of measured against predicted times. The deliberate
consequence is that admission is pessimistic at low load: it would rather refuse a request that
might have fitted than accept one that will not (docs/adr/0004). Deadlines that have already
passed are excluded from the test but not from the demand -- a late request still delays the ones
behind it, but letting its unsatisfiable constraint fail the test would refuse every new request
until it drained.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = [
    "AdmissionController",
    "AdmissionDecision",
    "LatencyModel",
    "RejectionReason",
    "WorkItem",
]


class RejectionReason(Enum):
    """Why a request was refused. Also the label of ``rejections_total``."""

    DEADLINE_INFEASIBLE = "deadline_infeasible"
    QUEUE_FULL = "queue_full"
    PROMPT_TOO_LONG = "prompt_too_long"
    CONTEXT_TOO_LONG = "context_too_long"
    KEY_CONCURRENCY = "key_concurrency"
    KEY_TOKEN_RATE = "key_token_rate"  # noqa: S105  # nosec B105 - a metric label
    CLASS_NOT_ENTITLED = "class_not_entitled"
    TOO_MANY_SAMPLES = "too_many_samples"


@dataclass(frozen=True, slots=True)
class WorkItem:
    """Outstanding work of one request, split the way the engine actually spends it.

    ``prefill_tokens`` are prompt tokens still to be computed, which share the per-iteration
    budget with every other request. ``decode_tokens`` are output tokens still to be produced,
    which cost one iteration each and overlap with every other request's decode.
    """

    rank: int
    deadline: float | None
    prefill_tokens: int
    decode_tokens: int

    def __post_init__(self) -> None:
        if self.prefill_tokens < 0 or self.decode_tokens < 0:
            raise ValueError("token counts must be >= 0")

    @property
    def tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    reason: RejectionReason | None = None
    retry_after_seconds: float = 0.0
    predicted_seconds: float = 0.0

    @property
    def retry_after_header(self) -> str:
        """``Retry-After`` value in whole seconds, never below one."""
        return str(max(1, int(self.retry_after_seconds + 0.999)))


class LatencyModel:
    """Affine iteration-latency model ``L(n) = intercept + slope * n``, corrected by an EWMA.

    ``intercept`` is the per-iteration overhead that does not depend on how many tokens are in the
    batch (Python dispatch, the per-layer norms and projections that run whatever the batch size);
    ``slope`` is the marginal cost of one more token. Both come from least squares over a short
    startup profile.
    """

    __slots__ = ("_alpha", "_correction", "_observations", "intercept", "slope")

    def __init__(self, intercept: float, slope: float, *, ewma_alpha: float = 0.2) -> None:
        if intercept < 0.0 or slope <= 0.0:
            raise ValueError("intercept must be >= 0 and slope must be > 0")
        if not 0.0 <= ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in [0, 1]")
        self.intercept = intercept
        self.slope = slope
        self._alpha = ewma_alpha
        self._correction = 1.0
        self._observations = 0

    @property
    def correction(self) -> float:
        """Current multiplicative correction (1.0 until the first observation)."""
        return self._correction

    @property
    def observations(self) -> int:
        return self._observations

    def update(self, intercept: float, slope: float) -> None:
        """Replace the fitted coefficients and drop the correction learned for the old ones."""
        if intercept < 0.0 or slope <= 0.0:
            raise ValueError("intercept must be >= 0 and slope must be > 0")
        self.intercept = intercept
        self.slope = slope
        self._correction = 1.0
        self._observations = 0

    def predict(self, batch_tokens: int) -> float:
        """Predicted seconds for one iteration of ``batch_tokens`` tokens."""
        if batch_tokens < 1:
            raise ValueError("batch_tokens must be >= 1")
        return (self.intercept + self.slope * batch_tokens) * self._correction

    def observe(self, batch_tokens: int, seconds: float) -> None:
        """Fold a measured iteration into the correction term."""
        if batch_tokens < 1 or seconds <= 0.0:
            return
        base = self.intercept + self.slope * batch_tokens
        if base <= 0.0:  # pragma: no cover - guarded by the constructor
            return
        ratio = seconds / base
        self._correction = (1.0 - self._alpha) * self._correction + self._alpha * ratio
        self._observations += 1

    def token_rate(self, batch_tokens: int) -> float:
        """Tokens per second sustained at a batch of ``batch_tokens``."""
        return batch_tokens / self.predict(batch_tokens)

    @classmethod
    def fit(cls, samples: Sequence[tuple[int, float]], *, ewma_alpha: float = 0.2) -> LatencyModel:
        """Least-squares fit of ``(batch_tokens, seconds)`` samples.

        A single sample, or samples that all share one batch size, cannot separate overhead from
        marginal cost, so the whole cost is attributed to the slope -- the conservative choice for
        admission, since it makes large batches look no cheaper per token than small ones.
        """
        if not samples:
            raise ValueError("at least one latency sample is required")
        sizes = np.asarray([size for size, _ in samples], dtype=np.float64)
        times = np.asarray([seconds for _, seconds in samples], dtype=np.float64)
        if np.any(sizes < 1) or np.any(times <= 0):
            raise ValueError("latency samples must have positive sizes and durations")
        if np.unique(sizes).size < 2:
            return cls(0.0, float(np.mean(times / sizes)), ewma_alpha=ewma_alpha)
        design = np.stack([np.ones_like(sizes), sizes], axis=1)
        (intercept, raw_slope), *_ = np.linalg.lstsq(design, times, rcond=None)
        slope = max(float(raw_slope), 1e-9)
        return cls(max(float(intercept), 0.0), slope, ewma_alpha=ewma_alpha)

    @classmethod
    def profile(
        cls,
        run_batch: Callable[[int], None],
        batch_sizes: Sequence[int],
        *,
        repeats: int = 3,
        warmup: int = 1,
        clock: Callable[[], float] = time.perf_counter,
        ewma_alpha: float = 0.2,
    ) -> LatencyModel:
        """Time ``run_batch`` at several batch sizes and fit the model to the medians."""
        if not batch_sizes:
            raise ValueError("at least one batch size is required")
        samples: list[tuple[int, float]] = []
        for size in batch_sizes:
            for _ in range(warmup):
                run_batch(size)
            durations = []
            for _ in range(repeats):
                start = clock()
                run_batch(size)
                durations.append(max(clock() - start, 1e-9))
            samples.append((size, float(np.median(durations))))
        return cls.fit(samples, ewma_alpha=ewma_alpha)


@dataclass(slots=True)
class AdmissionController:
    """Iteration-counting feasibility test against the engine's worst-case iteration cost."""

    latency: LatencyModel
    max_batched_tokens: int
    max_running_seqs: int = 32
    max_queued_sequences: int = 1024
    safety_factor: float = 1.0
    # Prefill allowance of every class below the top one (EngineConfig.lower_class_prefill_tokens).
    # None means the scheduler does not cap lower classes.
    lower_class_prefill_tokens: int | None = None
    _rejections: dict[RejectionReason, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_batched_tokens < 1:
            raise ValueError("max_batched_tokens must be >= 1")
        if self.max_running_seqs < 1:
            raise ValueError("max_running_seqs must be >= 1")
        if self.safety_factor <= 0.0:
            raise ValueError("safety_factor must be > 0")

    @property
    def iteration_seconds(self) -> float:
        """Worst-case cost of one iteration: a full token budget, scaled by the safety factor."""
        return self.latency.predict(self.max_batched_tokens) * self.safety_factor

    @property
    def prefill_tokens_per_iteration(self) -> int:
        """Budget left for prompt tokens once every running sequence has taken its decode token."""
        reserved = min(self.max_running_seqs, self.max_batched_tokens - 1)
        return max(1, self.max_batched_tokens - reserved)

    def prefill_tokens_per_iteration_for(self, rank: int) -> int:
        """Guaranteed prefill progress per iteration for work that includes class ``rank``.

        The top class (rank 0) always gets the budget left after decode. A lower class may be
        held to its allowance whenever higher-class work exists -- which can start after the
        request was admitted -- so its guarantee uses the allowance unconditionally. Combined
        prefill of the higher classes and this one advances by at least that much in every
        iteration: either higher-class prefill is running (and takes at least the full share)
        or the lower class has the allowance to itself.
        """
        full = self.prefill_tokens_per_iteration
        if rank <= 0 or self.lower_class_prefill_tokens is None:
            return full
        return max(1, min(full, self.lower_class_prefill_tokens))

    @property
    def token_rate(self) -> float:
        """Prefill tokens per second: what the benchmarks and /v1/engine/stats report."""
        return self.prefill_tokens_per_iteration / self.iteration_seconds

    @property
    def rejections(self) -> dict[RejectionReason, int]:
        return dict(self._rejections)

    def record_rejection(self, reason: RejectionReason) -> None:
        self._rejections[reason] = self._rejections.get(reason, 0) + 1

    def iterations_for(self, prefill_tokens: int, decode_tokens: int, rank: int = 0) -> int:
        """Iterations needed to absorb ``prefill_tokens`` and then produce ``decode_tokens``."""
        per_iteration = self.prefill_tokens_per_iteration_for(rank)
        return math.ceil(prefill_tokens / per_iteration) + decode_tokens

    def decide(
        self,
        *,
        now: float,
        candidate: WorkItem,
        committed: Iterable[WorkItem],
        queued_sequences: int = 0,
    ) -> AdmissionDecision:
        """Admit or reject ``candidate`` given the work already committed.

        A candidate without a deadline is best effort: it is admitted whenever the queue has room,
        and because the scheduler runs it behind every deadlined request of its class it cannot
        make an existing deadline infeasible.
        """
        items = list(committed)
        candidate_deadline = candidate.deadline
        if candidate_deadline is not None and candidate_deadline <= now:
            # Arriving already late. Accepting it would only spend capacity on an answer nobody
            # is still waiting for.
            self.record_rejection(RejectionReason.DEADLINE_INFEASIBLE)
            return AdmissionDecision(
                False, RejectionReason.DEADLINE_INFEASIBLE, self._backlog_seconds(items)
            )
        if queued_sequences >= self.max_queued_sequences:
            self.record_rejection(RejectionReason.QUEUE_FULL)
            return AdmissionDecision(
                False, RejectionReason.QUEUE_FULL, self._backlog_seconds(items)
            )
        if not self._feasible(now, [*items, candidate]):
            self.record_rejection(RejectionReason.DEADLINE_INFEASIBLE)
            return AdmissionDecision(
                False, RejectionReason.DEADLINE_INFEASIBLE, self._backlog_seconds(items)
            )
        return AdmissionDecision(True, predicted_seconds=self._response_seconds(items, candidate))

    def _backlog_seconds(self, items: Sequence[WorkItem]) -> float:
        """How long the work already accepted takes: a truthful ``Retry-After``."""
        prefill = sum(item.prefill_tokens for item in items)
        decode = max((item.decode_tokens for item in items), default=0)
        rank = max((item.rank for item in items), default=0)
        return self.iterations_for(prefill, decode, rank) * self.iteration_seconds

    def _response_seconds(self, items: Sequence[WorkItem], candidate: WorkItem) -> float:
        """Predicted seconds until the candidate finishes, under the scheduler's own order."""
        ahead = [item for item in items if _runs_before(item, candidate)]
        prefill = sum(item.prefill_tokens for item in ahead) + candidate.prefill_tokens
        decode = max([candidate.decode_tokens, *(item.decode_tokens for item in ahead)])
        return self.iterations_for(prefill, decode, candidate.rank) * self.iteration_seconds

    def _feasible(self, now: float, items: Sequence[WorkItem]) -> bool:
        """Iteration-demand test over every committed deadline, per SLO class."""
        seconds = self.iteration_seconds
        for rank in sorted({item.rank for item in items}):
            higher = [item for item in items if item.rank < rank]
            prefill = sum(item.prefill_tokens for item in higher)
            decode_span = max((item.decode_tokens for item in higher), default=0)
            deadlined: list[tuple[float, WorkItem]] = []
            for item in items:
                deadline = item.deadline
                if item.rank == rank and deadline is not None:
                    deadlined.append((deadline, item))
            deadlined.sort(key=lambda pair: pair[0])
            for deadline, item in deadlined:
                prefill += item.prefill_tokens
                decode_span = max(decode_span, item.decode_tokens)
                if deadline <= now:
                    continue
                if self.iterations_for(prefill, decode_span, rank) * seconds > deadline - now:
                    return False
        return True


def _runs_before(item: WorkItem, candidate: WorkItem) -> bool:
    """True when the scheduler serves ``item`` no later than ``candidate``."""
    return item.rank < candidate.rank or (
        item.rank == candidate.rank and _deadline_key(item) <= _deadline_key(candidate)
    )


def _deadline_key(item: WorkItem) -> float:
    return float("inf") if item.deadline is None else item.deadline
