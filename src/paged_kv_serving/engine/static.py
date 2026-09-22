"""The static-batching baseline the continuous engine is measured against.

This is what serving looks like without any of the machinery in the rest of this package, and it
is deliberately implemented over the *same* model, the same runner and the same sampler, so a
difference in the numbers is a difference in scheduling and memory management and nothing else.

Three properties define it, and each one costs something measurable:

* **Fixed batches.** Requests are taken in arrival order in groups of ``batch_size``, and a group
  starts only once its last member has arrived and the previous group has finished. A request that
  arrives one moment too late waits for a whole batch. With ``max_wait_seconds`` set the engine is
  the *dynamic* batcher real servers use instead (Triton's ``max_queue_delay``, for example): a
  batch also leaves, partly filled, once its oldest request has waited that long. That removes the
  wait-for-a-full-batch cost at low load, and is the fairer baseline for time to first token.
* **Padded prefill.** Every row of a batch runs the length of the longest prompt in it. The
  padding is real compute; the logits are read at each row's own last real token.
* **Decode to the longest.** The batch keeps running until its slowest member is done. A row that
  finished after 8 tokens is still executed while its neighbour generates 120.

Memory is reserved the contiguous way: each of the ``batch_size`` slots gets a fixed region big
enough for the longest sequence it might produce, whether or not it uses it. That reservation is
what :attr:`StaticBatchingEngine.kv_tokens_reserved` compares against the paged allocator's actual
occupancy in the benchmark.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.runner import BatchEntry, ExecutionBatch, ModelRunner
from paged_kv_serving.engine.sampling import sample_token
from paged_kv_serving.engine.sequence import FinishReason, Request
from paged_kv_serving.engine.sequence import Sequence as EngineSequence

__all__ = ["StaticBatchOutput", "StaticBatchingEngine"]

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class StaticBatchOutput:
    """Result of one request under static batching, shaped like ``RequestOutput``."""

    request_id: str
    key_id: str
    prompt_tokens: int
    token_ids: tuple[int, ...]
    arrival_time: float
    first_token_time: float
    finish_time: float

    @property
    def ttft(self) -> float:
        return self.first_token_time - self.arrival_time

    @property
    def latency(self) -> float:
        return self.finish_time - self.arrival_time

    @property
    def output_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def tpot(self) -> float:
        extra = self.output_tokens - 1
        if extra <= 0:
            return 0.0
        return (self.finish_time - self.first_token_time) / extra


class StaticBatchingEngine:
    """Fixed-size batches over a contiguous KV reservation."""

    __slots__ = (
        "batch_size",
        "block_size",
        "blocks_per_slot",
        "clock",
        "eos_token_id",
        "max_context_tokens",
        "max_wait_seconds",
        "model",
        "runner",
        "sleep",
        "total_padding_tokens",
        "total_wasted_decode_tokens",
    )

    def __init__(
        self,
        model: LanguageModel,
        *,
        batch_size: int = 8,
        max_context_tokens: int = 256,
        block_size: int = 16,
        eos_token_id: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        max_wait_seconds: float | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_wait_seconds is not None and max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be >= 0")
        if max_context_tokens > model.config.max_seq_len:
            raise ValueError("max_context_tokens exceeds the model's max_seq_len")
        self.model = model
        self.batch_size = batch_size
        self.max_context_tokens = max_context_tokens
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.clock = clock
        self.sleep = sleep
        self.max_wait_seconds = max_wait_seconds
        self.blocks_per_slot = -(-max_context_tokens // block_size)
        self.runner = ModelRunner(
            model, num_blocks=batch_size * self.blocks_per_slot, block_size=block_size
        )
        self.total_padding_tokens = 0
        self.total_wasted_decode_tokens = 0

    @property
    def kv_tokens_reserved(self) -> int:
        """Tokens the contiguous reservation occupies while a batch runs, used or not."""
        return self.batch_size * self.blocks_per_slot * self.block_size

    def _slot_table(self, slot: int) -> tuple[int, ...]:
        start = slot * self.blocks_per_slot
        return tuple(range(start, start + self.blocks_per_slot))

    def generate(self, requests: Iterable[Request]) -> list[StaticBatchOutput]:
        """Run every request and return one result per request.

        Arrival times are honoured: a batch waits for its last member (or, with a max wait, for
        its oldest member's wait to run out), which is what makes the comparison against the
        continuous engine an open-loop comparison rather than a throughput race on a pre-filled
        queue.
        """
        ordered = sorted(requests, key=lambda item: (item.arrival_time, item.request_id))
        for request in ordered:
            if request.prompt_len + request.sampling.max_new_tokens > self.max_context_tokens:
                raise ValueError(f"request {request.request_id} exceeds max_context_tokens")
        results: list[StaticBatchOutput] = []
        pending = deque(ordered)
        while pending:
            group = self._next_batch(pending)
            results.extend(self._run_batch(group))
        return results

    def _next_batch(self, pending: deque[Request]) -> list[Request]:
        """Wait until the next batch may leave, then take it off ``pending``."""
        if self.max_wait_seconds is None:
            group = [pending.popleft() for _ in range(min(self.batch_size, len(pending)))]
            self._sleep_until(max(request.arrival_time for request in group))
            return group
        # Dynamic: leave when full, or when the oldest request has waited max_wait_seconds --
        # whichever comes first, and never before the previous batch has finished.
        full_at = (
            pending[self.batch_size - 1].arrival_time
            if len(pending) >= self.batch_size
            else float("inf")
        )
        self._sleep_until(min(full_at, pending[0].arrival_time + self.max_wait_seconds))
        now = self.clock()
        group = [pending.popleft()]
        while pending and len(group) < self.batch_size and pending[0].arrival_time <= now:
            group.append(pending.popleft())
        return group

    def _sleep_until(self, moment: float) -> None:
        delay = moment - self.clock()
        if delay > 0:
            self.sleep(delay)

    def _run_batch(self, group: Sequence[Request]) -> list[StaticBatchOutput]:
        sequences = [EngineSequence.from_request(request, 0) for request in group]
        prompt_len = max(sequence.num_tokens for sequence in sequences)
        self.total_padding_tokens += sum(prompt_len - s.num_tokens for s in sequences)

        prefill = tuple(
            BatchEntry(
                seq_id=sequence.seq_id,
                token_ids=(*sequence.token_ids, *([0] * (prompt_len - sequence.num_tokens))),
                positions=tuple(range(prompt_len)),
                block_table=self._slot_table(slot),
                logits_index=sequence.num_tokens - 1,
            )
            for slot, sequence in enumerate(sequences)
        )
        logits = self.runner.execute(ExecutionBatch(prefill))
        first_token_time = self.clock()
        for sequence in sequences:
            self._advance(sequence, logits[sequence.seq_id])
            sequence.first_token_time = first_token_time

        rounds = max(sequence.request.sampling.max_new_tokens for sequence in sequences)
        for _ in range(rounds - 1):
            if all(sequence.is_finished for sequence in sequences):
                break
            decode = []
            for slot, sequence in enumerate(sequences):
                # Finished rows keep their slot and are still executed: a batch is a rectangle.
                position = min(sequence.num_tokens - 1, self.max_context_tokens - 1)
                decode.append(
                    BatchEntry(
                        seq_id=sequence.seq_id,
                        token_ids=(sequence.token_ids[position],),
                        positions=(position,),
                        block_table=self._slot_table(slot),
                        logits_index=0,
                    )
                )
                if sequence.is_finished:
                    self.total_wasted_decode_tokens += 1
            logits = self.runner.execute(ExecutionBatch(tuple(decode)))
            for sequence in sequences:
                if not sequence.is_finished:
                    self._advance(sequence, logits[sequence.seq_id])

        finish_time = self.clock()
        return [
            StaticBatchOutput(
                request_id=sequence.request.request_id,
                key_id=sequence.request.key_id,
                prompt_tokens=sequence.request.prompt_len,
                token_ids=tuple(sequence.output_token_ids),
                arrival_time=sequence.request.arrival_time,
                first_token_time=(
                    sequence.first_token_time
                    if sequence.first_token_time is not None
                    else finish_time
                ),
                finish_time=finish_time,
            )
            for sequence in sequences
        ]

    def _advance(self, sequence: EngineSequence, logits: FloatArray) -> None:
        if sequence.rng is None:  # pragma: no cover - set by Sequence.from_request
            raise RuntimeError("sequence has no generator")
        token = sample_token(logits, sequence.request.sampling, sequence.rng)
        sequence.append_token(token)
        sampling = sequence.request.sampling
        if self.eos_token_id is not None and token == self.eos_token_id and not sampling.ignore_eos:
            sequence.finish(FinishReason.STOP, self.clock())
        elif sequence.num_output_tokens >= sampling.max_new_tokens:
            sequence.finish(FinishReason.LENGTH, self.clock())
        elif sequence.num_tokens >= self.max_context_tokens:
            sequence.finish(FinishReason.CONTEXT_LIMIT, self.clock())
