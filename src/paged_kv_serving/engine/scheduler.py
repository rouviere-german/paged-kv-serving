"""Iteration-level continuous batching: the scheduling half of the primary pattern.

Static batching picks a group of requests, runs them together until the *longest* one finishes,
and only then accepts more work. Two costs follow: every sequence is padded to the longest
prompt, and a request that arrives one iteration too late waits for the whole batch. Continuous
batching instead re-decides the batch **every iteration**: a finished sequence leaves immediately,
a newly arrived one joins the very next step, and nothing is padded because the batch is a flat
list of tokens rather than a rectangle.

One iteration fills a token budget in three phases:

1. **Decode.** One token for each running sequence that has already consumed its prompt, in
   priority order. Decode goes first so a large prompt can never delay sequences that are already
   streaming -- the head-of-line blocking that static batching cannot avoid.
2. **Continue prefill.** Sequences whose prompt is only partly computed take the rest of the
   budget in chunks. A 2000-token prompt is therefore spread over several iterations instead of
   monopolising one (chunked prefill, as in Agrawal et al., "Taming Throughput-Latency Tradeoff
   in LLM Inference with Sarathi-Serve", OSDI 2024).
3. **Admit.** Waiting sequences join while budget, sequence slots and blocks remain, ordered by
   SLO class first and earliest deadline second.

In phases 2 and 3, while a higher SLO class has any sequence in the engine, prefill of the lower
classes shares an allowance of ``lower_class_prefill_share`` of the budget. Decode is never
capped. The allowance exists because iteration time grows with batch tokens: without it, a flood
of long batch prompts fills every iteration to the full budget and stretches each interactive
decode step to the cost of a full-budget forward pass.

When the KV pool cannot satisfy a running sequence, the lowest-priority *unscheduled* sequence is
preempted: its blocks are released and it returns to the waiting queue to be recomputed later.
Recomputation keeps the tokens it already produced, so preemption costs work but never output.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from paged_kv_serving.engine.blocks import BlockAllocationError
from paged_kv_serving.engine.config import EngineConfig
from paged_kv_serving.engine.kvcache import KVCacheManager
from paged_kv_serving.engine.runner import BatchEntry, ExecutionBatch
from paged_kv_serving.engine.sequence import Sequence, SequenceStatus

__all__ = ["ScheduledSequence", "Scheduler", "SchedulerOutput"]


@dataclass(frozen=True, slots=True)
class ScheduledSequence:
    """A sequence and the number of its tokens that this iteration will compute.

    ``produces_token`` is decided here, while ``num_computed_tokens`` still describes the state
    the batch was built from. Recomputing it later -- after the engine has advanced the sequence
    -- would silently answer a different question and drop the sampled token.
    """

    sequence: Sequence
    num_tokens: int
    is_prefill: bool
    produces_token: bool


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    batch: ExecutionBatch
    scheduled: tuple[ScheduledSequence, ...] = ()
    preempted: tuple[str, ...] = ()
    admitted: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.scheduled

    @property
    def num_batched_tokens(self) -> int:
        return self.batch.num_tokens


@dataclass(slots=True)
class SchedulerStats:
    iterations: int = 0
    preemptions: int = 0
    batched_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0

    @property
    def mean_batch_size(self) -> float:
        return self.batched_tokens / self.iterations if self.iterations else 0.0


@dataclass(slots=True)
class _LowerClassAllowance:
    """Prefill tokens left this iteration for classes ranked below ``top_rank``."""

    top_rank: int
    remaining: int

    def clip(self, sequence: Sequence, chunk: int) -> int:
        if sequence.request.slo_class.rank <= self.top_rank:
            return chunk
        return min(chunk, self.remaining)

    def spend(self, sequence: Sequence, tokens: int) -> None:
        if sequence.request.slo_class.rank > self.top_rank:
            self.remaining -= tokens


class Scheduler:
    """Waiting, running and preempted queues over one KV pool."""

    __slots__ = ("_running", "_waiting", "cache", "config", "stats")

    def __init__(self, config: EngineConfig, cache: KVCacheManager) -> None:
        self.config = config
        self.cache = cache
        self._waiting: list[Sequence] = []
        self._running: list[Sequence] = []
        self.stats = SchedulerStats()

    # --- queue management ----------------------------------------------------------------

    @property
    def waiting(self) -> tuple[Sequence, ...]:
        return tuple(self._waiting)

    @property
    def running(self) -> tuple[Sequence, ...]:
        return tuple(self._running)

    @property
    def queue_depth(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    def add(self, sequence: Sequence) -> None:
        """Queue a fresh sequence for admission."""
        sequence.status = SequenceStatus.WAITING
        self._waiting.append(sequence)

    def add_running(self, sequence: Sequence) -> None:
        """Insert a sequence whose KV is already resident (a fork of an existing sequence)."""
        sequence.status = SequenceStatus.RUNNING
        self._running.append(sequence)

    def remove(self, sequence: Sequence) -> None:
        """Drop a sequence from whichever queue holds it, releasing its blocks."""
        if sequence in self._running:
            self._running.remove(sequence)
        elif sequence in self._waiting:
            self._waiting.remove(sequence)
        self.cache.free(sequence)

    def drop_finished(self) -> list[Sequence]:
        """Release blocks of finished sequences and return them."""
        finished = [sequence for sequence in self._running if sequence.is_finished]
        for sequence in finished:
            self._running.remove(sequence)
            self.cache.free(sequence)
        return finished

    # --- scheduling ----------------------------------------------------------------------

    def _preempt(self, victim: Sequence) -> None:
        self._running.remove(victim)
        self.cache.free(victim)
        victim.reset_for_recompute()
        self._waiting.append(victim)
        self.stats.preemptions += 1

    def _pick_victim(self, pending: deque[Sequence]) -> Sequence:
        """Lowest-priority running sequence not yet scheduled in this iteration.

        ``pending`` is sorted by priority, so its tail is the weakest decoder; a sequence still
        prefilling holds blocks too and is not scheduled until the next phase, so it competes on
        the same key. Without it, a batch-class prompt half-way through prefill could keep its
        blocks while interactive decoders were preempted around it.
        """
        prefilling = [sequence for sequence in self._running if not sequence.is_decoding]
        candidates = [*prefilling, *([pending[-1]] if pending else [])]
        return max(candidates, key=lambda item: item.priority_key())

    def _watermark_blocks(self) -> int:
        return int(self.config.watermark * self.cache.blocks.num_blocks)

    def _max_chunk_by_memory(self, sequence: Sequence, reserve_blocks: int) -> int:
        """Largest chunk whose blocks fit, leaving ``reserve_blocks`` free afterwards."""
        available = self.cache.num_free_blocks + self.cache.evictable_blocks() - reserve_blocks
        capacity_tokens = (len(sequence.block_table) + max(0, available)) * self.cache.block_size
        return capacity_tokens - sequence.num_computed_tokens

    def schedule(self) -> SchedulerOutput:
        """Pick this iteration's batch. Never blocks; an empty batch means nothing can run."""
        budget = self.config.max_batched_tokens
        scheduled: list[ScheduledSequence] = []
        preempted: list[str] = []
        admitted: list[str] = []

        allowance = _LowerClassAllowance(
            top_rank=min(
                (item.request.slo_class.rank for item in (*self._running, *self._waiting)),
                default=0,
            ),
            remaining=self.config.lower_class_prefill_tokens,
        )
        budget = self._schedule_decodes(budget, scheduled, preempted)
        budget = self._continue_prefills(budget, scheduled, allowance)
        # Nothing new is admitted in an iteration that had to preempt. Re-admitting the victim
        # immediately would hand back the blocks that were just taken from it and preempt it
        # again next iteration; admitting somebody else instead would be a priority inversion.
        if not preempted:
            self._admit_waiting(budget, scheduled, admitted, allowance)

        self._waiting.sort(key=lambda sequence: sequence.priority_key())
        entries = tuple(self._entry(item) for item in scheduled)
        output = SchedulerOutput(
            batch=ExecutionBatch(entries),
            scheduled=tuple(scheduled),
            preempted=tuple(preempted),
            admitted=tuple(admitted),
        )
        if scheduled:
            self.stats.iterations += 1
            self.stats.batched_tokens += output.num_batched_tokens
            self.stats.prefill_tokens += sum(
                item.num_tokens for item in scheduled if item.is_prefill
            )
            self.stats.decode_tokens += sum(
                item.num_tokens for item in scheduled if not item.is_prefill
            )
        return output

    def _schedule_decodes(
        self, budget: int, scheduled: list[ScheduledSequence], preempted: list[str]
    ) -> int:
        """One token for every decoding sequence, preempting when the pool runs dry."""
        pending = deque(
            sequence
            for sequence in sorted(self._running, key=lambda item: item.priority_key())
            if sequence.is_decoding
        )
        while pending and budget > 0:
            sequence = pending[0]
            try:
                self.cache.reserve(sequence, sequence.num_tokens)
            except BlockAllocationError:
                # The lowest-priority sequence that has not been scheduled yet pays for the
                # shortage: a decoder still waiting for its token, or a sequence part-way through
                # its prompt. When only the current sequence is left it preempts itself, which
                # still makes progress because its blocks are released for the others.
                victim = self._pick_victim(pending)
                if pending and victim is pending[-1]:
                    pending.pop()
                self._preempt(victim)
                preempted.append(victim.seq_id)
                continue
            pending.popleft()
            scheduled.append(ScheduledSequence(sequence, 1, is_prefill=False, produces_token=True))
            budget -= 1
        return budget

    def _continue_prefills(
        self,
        budget: int,
        scheduled: list[ScheduledSequence],
        allowance: _LowerClassAllowance,
    ) -> int:
        """Give the rest of the budget to running sequences whose prompt is still incomplete."""
        candidates = sorted(
            (sequence for sequence in self._running if not sequence.is_decoding),
            key=lambda item: item.priority_key(),
        )
        for sequence in candidates:
            if budget <= 0:
                break
            chunk = min(budget, sequence.num_uncomputed_tokens)
            chunk = min(chunk, self._max_chunk_by_memory(sequence, reserve_blocks=0))
            chunk = allowance.clip(sequence, chunk)
            if chunk <= 0:
                continue
            try:
                self.cache.reserve(sequence, sequence.num_computed_tokens + chunk)
            except BlockAllocationError:  # pragma: no cover - guarded by _max_chunk_by_memory
                continue
            scheduled.append(
                ScheduledSequence(
                    sequence,
                    chunk,
                    is_prefill=True,
                    produces_token=sequence.num_computed_tokens + chunk == sequence.num_tokens,
                )
            )
            allowance.spend(sequence, chunk)
            budget -= chunk
        return budget

    def _admit_waiting(
        self,
        budget: int,
        scheduled: list[ScheduledSequence],
        admitted: list[str],
        allowance: _LowerClassAllowance,
    ) -> int:
        """Admit waiting sequences in priority order while budget, slots and blocks allow."""
        self._waiting.sort(key=lambda sequence: sequence.priority_key())
        watermark = self._watermark_blocks()
        queue = deque(self._waiting)
        while queue and budget > 0 and len(self._running) < self.config.max_running_seqs:
            sequence = queue[0]
            if not sequence.prefix_lookup_done:
                self.cache.attach_prefix(sequence)
                sequence.prefix_lookup_done = True
            chunk = min(budget, sequence.num_uncomputed_tokens)
            chunk = min(chunk, self._max_chunk_by_memory(sequence, reserve_blocks=watermark))
            chunk = allowance.clip(sequence, chunk)
            if chunk <= 0:
                # Not enough memory (or lower-class allowance) for even one token of the
                # highest-priority waiter. Admitting a lower-priority one instead would starve it,
                # so the phase stops here.
                break
            try:
                self.cache.reserve(sequence, sequence.num_computed_tokens + chunk)
            except BlockAllocationError:  # pragma: no cover - guarded by _max_chunk_by_memory
                break
            queue.popleft()
            self._waiting.remove(sequence)
            sequence.status = SequenceStatus.RUNNING
            self._running.append(sequence)
            scheduled.append(
                ScheduledSequence(
                    sequence,
                    chunk,
                    is_prefill=True,
                    produces_token=sequence.num_computed_tokens + chunk == sequence.num_tokens,
                )
            )
            admitted.append(sequence.seq_id)
            allowance.spend(sequence, chunk)
            budget -= chunk
        return budget

    def _entry(self, item: ScheduledSequence) -> BatchEntry:
        sequence = item.sequence
        start = sequence.num_computed_tokens
        stop = start + item.num_tokens
        return BatchEntry(
            seq_id=sequence.seq_id,
            token_ids=tuple(sequence.token_ids[start:stop]),
            positions=tuple(range(start, stop)),
            block_table=tuple(sequence.block_table),
            logits_index=item.num_tokens - 1 if item.produces_token else None,
        )
