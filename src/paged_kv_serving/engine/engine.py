"""The serving engine: one synchronous ``step()`` per scheduled iteration.

``step()`` is deliberately blocking and single-threaded. Everything asynchronous -- HTTP, SSE,
cancellation -- lives in :mod:`paged_kv_serving.engine.service` and drives this loop, so the
scheduling and memory logic can be tested, benchmarked and reasoned about without an event loop
anywhere in sight.

One iteration:

1. the scheduler picks a ragged batch and reserves the blocks it needs, preempting if it must;
2. the runner executes it, writing keys and values into the paged cache;
3. every slice that reached the end of its sequence samples one token with the sequence's own
   generator, newly completed blocks are offered to the prefix cache, and finished sequences give
   their blocks back.

Step 3 is also where parallel sampling forks: when the first sample of an ``n``-way request
finishes its prompt, the other ``n - 1`` samples are created from the *same* logits and share the
prompt's blocks by reference. They diverge on their first write, and copy-on-write gives each one
a private copy of the single partially filled block it is writing -- not of the prompt.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from paged_kv_serving.engine.admission import (
    AdmissionController,
    AdmissionDecision,
    LatencyModel,
    RejectionReason,
    WorkItem,
)
from paged_kv_serving.engine.blocks import BlockManager
from paged_kv_serving.engine.config import EngineConfig, SloClass
from paged_kv_serving.engine.keys import EntitlementGuard
from paged_kv_serving.engine.kvcache import KVCacheManager
from paged_kv_serving.engine.metrics import EngineMetrics, create_engine_metrics
from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.monitoring import PromptLengthMonitor
from paged_kv_serving.engine.prefix_cache import CacheScope, PrefixCache
from paged_kv_serving.engine.runner import BatchEntry, ExecutionBatch, ModelRunner
from paged_kv_serving.engine.sampling import sample_token
from paged_kv_serving.engine.scheduler import Scheduler, SchedulerOutput
from paged_kv_serving.engine.sequence import (
    FinishReason,
    Request,
    Sequence,
    SequenceStatus,
    finish_reason_for,
)

__all__ = ["Engine", "EngineExecutor", "RequestOutput", "SampleOutput", "StepOutput", "TokenEvent"]


FloatArray = npt.NDArray[np.float32]


class EngineExecutor(Protocol):
    """What the engine needs from a model runner. Implemented by :class:`ModelRunner`.

    Tests and the deterministic admission benchmark substitute an executor that consumes exactly
    the time the latency model predicts, which turns "the scheduler meets deadlines" into a claim
    that can be checked without depending on how fast the host happens to be.
    """

    def execute(self, batch: ExecutionBatch) -> dict[str, FloatArray]:
        """Run one batch and return logits for every entry that asked for them."""


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One generated token, as streamed to a client."""

    request_id: str
    sample_index: int
    token_id: int
    index: int
    finished: bool
    finish_reason: FinishReason | None = None


@dataclass(frozen=True, slots=True)
class SampleOutput:
    sample_index: int
    token_ids: tuple[int, ...]
    finish_reason: FinishReason
    preemptions: int


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """Everything known about a request once its last sample has finished."""

    request_id: str
    key_id: str
    slo_class: SloClass
    samples: tuple[SampleOutput, ...]
    prompt_tokens: int
    cached_prompt_tokens: int
    arrival_time: float
    first_token_time: float
    finish_time: float
    deadline: float | None = None

    @property
    def met_deadline(self) -> bool | None:
        """Whether a deadlined request finished in time, or None when it had no deadline."""
        return None if self.deadline is None else self.finish_time <= self.deadline

    @property
    def ttft(self) -> float:
        """Seconds from arrival to the first token, measured from arrival, not from admission."""
        return self.first_token_time - self.arrival_time

    @property
    def latency(self) -> float:
        return self.finish_time - self.arrival_time

    @property
    def output_tokens(self) -> int:
        return sum(len(sample.token_ids) for sample in self.samples)

    @property
    def tpot(self) -> float:
        """Mean seconds per output token after the first one. 0.0 for a single-token answer."""
        extra = self.output_tokens - len(self.samples)
        if extra <= 0:
            return 0.0
        return (self.finish_time - self.first_token_time) / extra

    @property
    def preemptions(self) -> int:
        return sum(sample.preemptions for sample in self.samples)


@dataclass(frozen=True, slots=True)
class StepOutput:
    iteration: int
    duration: float
    num_batched_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    num_running: int
    queue_depth: int
    tokens: tuple[TokenEvent, ...] = ()
    finished: tuple[RequestOutput, ...] = ()
    preempted: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.num_batched_tokens == 0


@dataclass(slots=True)
class _RequestState:
    request: Request
    sequences: dict[int, Sequence] = field(default_factory=dict)
    pending_forks: list[Sequence] = field(default_factory=list)
    finished: dict[int, SampleOutput] = field(default_factory=dict)
    first_token_time: float | None = None
    cached_prompt_tokens: int = 0

    @property
    def complete(self) -> bool:
        return len(self.finished) == self.request.n


class Engine:
    """Continuous-batching engine over a paged KV cache."""

    def __init__(
        self,
        model: LanguageModel,
        config: EngineConfig | None = None,
        *,
        executor: EngineExecutor | None = None,
        clock: Callable[[], float] = time.perf_counter,
        eos_token_id: int | None = None,
        latency_model: LatencyModel | None = None,
        metrics: EngineMetrics | None = None,
        prefix_cache_scope: CacheScope = "per_key",
        entitlements: EntitlementGuard | None = None,
        drift_monitor: PromptLengthMonitor | None = None,
        max_queued_sequences: int = 1024,
    ) -> None:
        self.config = config or EngineConfig()
        if self.config.max_context_tokens > model.config.max_seq_len:
            raise ValueError("max_context_tokens exceeds the model's max_seq_len")
        self.model = model
        self.clock = clock
        self.eos_token_id = eos_token_id
        self.metrics = metrics if metrics is not None else create_engine_metrics()
        self.entitlements = entitlements
        self.drift_monitor = drift_monitor
        self.runner = ModelRunner(
            model, num_blocks=self.config.num_blocks, block_size=self.config.block_size
        )
        self.executor: EngineExecutor = executor if executor is not None else self.runner
        self.blocks = BlockManager(
            num_blocks=self.config.num_blocks,
            block_size=self.config.block_size,
            store=self.runner,
        )
        self.prefix_cache = PrefixCache(
            self.blocks, enabled=self.config.enable_prefix_cache, scope=prefix_cache_scope
        )
        self.cache = KVCacheManager(self.blocks, self.prefix_cache)
        self.scheduler = Scheduler(self.config, self.cache)
        self.latency = latency_model or LatencyModel(
            intercept=1e-4,
            slope=1e-4,
            ewma_alpha=self.config.latency_ewma_alpha,
        )
        self.admission = AdmissionController(
            latency=self.latency,
            max_batched_tokens=self.config.max_batched_tokens,
            max_running_seqs=self.config.max_running_seqs,
            max_queued_sequences=max_queued_sequences,
            safety_factor=self.config.admission_safety_factor,
            lower_class_prefill_tokens=self.config.lower_class_prefill_tokens,
        )
        self._requests: dict[str, _RequestState] = {}
        self._iteration = 0
        self._deadlined = 0
        self._met_deadline = 0
        self.aborted_requests = 0

    # --- introspection -------------------------------------------------------------------

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def has_work(self) -> bool:
        return self.scheduler.has_work or any(
            state.pending_forks for state in self._requests.values()
        )

    @property
    def active_requests(self) -> int:
        return len(self._requests)

    @property
    def slo_attainment(self) -> float:
        """Fraction of finished deadlined requests that met their deadline (1.0 when none)."""
        return self._met_deadline / self._deadlined if self._deadlined else 1.0

    def snapshot(self) -> dict[str, float]:
        """Point-in-time gauges, used by ``/readyz``, the demo and the benchmarks."""
        return {
            "iteration": float(self._iteration),
            "queue_depth": float(self.scheduler.queue_depth),
            "running": float(self.scheduler.num_running),
            "kv_blocks_used_ratio": self.blocks.utilization,
            "prefix_cache_hit_ratio": self.prefix_cache.stats.hit_ratio,
            "preemptions_total": float(self.scheduler.stats.preemptions),
            "aborted_requests_total": float(self.aborted_requests),
            "slo_attainment": self.slo_attainment,
            "token_rate": self.admission.token_rate,
        }

    # --- admission -----------------------------------------------------------------------

    def _work_items(self) -> list[WorkItem]:
        """Outstanding prefill and decode of every live sequence, as admission control sees it."""
        items: list[WorkItem] = []
        for sequence in (*self.scheduler.running, *self.scheduler.waiting):
            remaining_output = max(
                0, sequence.request.sampling.max_new_tokens - sequence.num_output_tokens
            )
            # A decoding sequence has one uncomputed token, which is the decode step itself and is
            # already counted in remaining_output; anything more is prompt still to be prefilled.
            prefill = max(0, sequence.num_uncomputed_tokens - 1)
            items.append(
                WorkItem(
                    rank=sequence.request.slo_class.rank,
                    deadline=sequence.request.deadline,
                    prefill_tokens=prefill,
                    decode_tokens=remaining_output,
                )
            )
        for state in self._requests.values():
            for pending in state.pending_forks:
                items.append(
                    WorkItem(
                        rank=pending.request.slo_class.rank,
                        deadline=pending.request.deadline,
                        prefill_tokens=0,  # a fork inherits the parent's computed prompt
                        decode_tokens=pending.request.sampling.max_new_tokens,
                    )
                )
        return items

    def _reject(self, reason: RejectionReason, retry_after: float = 0.0) -> AdmissionDecision:
        self.admission.record_rejection(reason)
        self.metrics.rejections_total.labels(reason.value).inc()
        return AdmissionDecision(False, reason, retry_after_seconds=retry_after)

    def submit(self, request: Request) -> AdmissionDecision:
        """Validate, meter and admit a request, or refuse it with a reason and a Retry-After.

        Rejection happens before a single block is allocated, so a refused request costs the
        server a hash of its prompt and nothing else.
        """
        if request.request_id in self._requests:
            raise ValueError(f"request {request.request_id} is already in flight")
        sampling = request.sampling
        if request.prompt_len > self.config.max_prompt_tokens:
            return self._reject(RejectionReason.PROMPT_TOO_LONG)
        if sampling.max_new_tokens > self.config.max_new_tokens:
            return self._reject(RejectionReason.CONTEXT_TOO_LONG)
        if request.prompt_len + sampling.max_new_tokens > self.config.max_context_tokens:
            return self._reject(RejectionReason.CONTEXT_TOO_LONG)
        if self.entitlements is not None:
            reason = self.entitlements.check(
                request.key_id,
                slo_class=request.slo_class,
                prompt_tokens=request.prompt_len,
                max_new_tokens=sampling.max_new_tokens,
            )
            if reason is not None:
                return self._reject(reason, retry_after=60.0)

        candidate = WorkItem(
            rank=request.slo_class.rank,
            deadline=request.deadline,
            # Every sample of an n-way request shares one prefill and decodes in the same
            # iterations, so n multiplies neither term.
            prefill_tokens=request.prompt_len,
            decode_tokens=sampling.max_new_tokens,
        )
        decision = self.admission.decide(
            now=request.arrival_time,
            candidate=candidate,
            committed=self._work_items(),
            queued_sequences=self.scheduler.queue_depth,
        )
        if not decision.admitted:
            if self.entitlements is not None:
                self.entitlements.release(request.key_id)
            reason = decision.reason or RejectionReason.DEADLINE_INFEASIBLE
            self.metrics.rejections_total.labels(reason.value).inc()
            return decision

        state = _RequestState(request=request)
        primary = Sequence.from_request(request, 0)
        state.sequences[0] = primary
        for index in range(1, request.n):
            child = Sequence.from_request(request, index)
            state.sequences[index] = child
            state.pending_forks.append(child)
        self._requests[request.request_id] = state
        self.scheduler.add(primary)
        if self.drift_monitor is not None:
            self.drift_monitor.observe(request.prompt_len)
            self.metrics.prompt_length_psi.set(self.drift_monitor.score())
        return decision

    def abort(self, request_id: str) -> int:
        """Cancel a request and release every block it holds. Returns sequences removed.

        This is the path a disconnected SSE client takes. Freeing here rather than at the end of
        generation is what stops a client that hangs up mid-stream from pinning KV blocks until
        its ``max_new_tokens`` runs out.
        """
        state = self._requests.pop(request_id, None)
        if state is None:
            return 0
        removed = 0
        now = self.clock()
        for sequence in state.sequences.values():
            if sequence.sample_index in state.finished:
                continue
            self.scheduler.remove(sequence)
            sequence.finish(FinishReason.ABORTED, now)
            removed += 1
        state.pending_forks.clear()
        if self.entitlements is not None:
            self.entitlements.release(state.request.key_id)
        self.aborted_requests += 1
        self.metrics.aborted_requests_total.inc()
        return removed

    # --- the loop ------------------------------------------------------------------------

    def step(self) -> StepOutput:
        """Run one iteration. Returns an empty :class:`StepOutput` when nothing can run."""
        scheduled_output = self.scheduler.schedule()
        if scheduled_output.preempted:
            self.metrics.preemptions_total.inc(len(scheduled_output.preempted))
        if scheduled_output.is_empty:
            return StepOutput(
                iteration=self._iteration,
                duration=0.0,
                num_batched_tokens=0,
                num_prefill_tokens=0,
                num_decode_tokens=0,
                num_running=self.scheduler.num_running,
                queue_depth=self.scheduler.queue_depth,
                preempted=scheduled_output.preempted,
            )

        self._iteration += 1
        started = self.clock()
        logits = self.executor.execute(scheduled_output.batch)
        duration = self.clock() - started
        now = self.clock()
        self.latency.observe(scheduled_output.num_batched_tokens, duration)

        events: list[TokenEvent] = []
        for item in scheduled_output.scheduled:
            sequence = item.sequence
            sequence.num_computed_tokens += item.num_tokens
            self.cache.publish_full_blocks(sequence)
            if not item.produces_token:
                continue
            row = logits[sequence.seq_id]
            self._fork_pending(sequence, row, now, events)
            self._emit(sequence, row, now, events)

        finished = self._collect_finished(now)
        self._record_metrics(scheduled_output, duration)
        return StepOutput(
            iteration=self._iteration,
            duration=duration,
            num_batched_tokens=scheduled_output.num_batched_tokens,
            num_prefill_tokens=sum(
                item.num_tokens for item in scheduled_output.scheduled if item.is_prefill
            ),
            num_decode_tokens=sum(
                item.num_tokens for item in scheduled_output.scheduled if not item.is_prefill
            ),
            num_running=self.scheduler.num_running,
            queue_depth=self.scheduler.queue_depth,
            tokens=tuple(events),
            finished=finished,
            preempted=scheduled_output.preempted,
        )

    def _fork_pending(
        self, sequence: Sequence, logits: FloatArray, now: float, events: list[TokenEvent]
    ) -> None:
        """Create the remaining samples of an ``n``-way request from the prompt's final logits."""
        state = self._requests.get(sequence.request.request_id)
        if state is None or sequence.sample_index != 0 or sequence.num_output_tokens:
            return
        for child in state.pending_forks:
            self.cache.fork(sequence, child)
            self.scheduler.add_running(child)
            self._emit(child, logits, now, events)
        state.pending_forks.clear()

    def _emit(
        self, sequence: Sequence, logits: FloatArray, now: float, events: list[TokenEvent]
    ) -> None:
        if sequence.rng is None:  # pragma: no cover - set by Sequence.from_request
            raise RuntimeError("sequence has no generator")
        token = sample_token(logits, sequence.request.sampling, sequence.rng)
        sequence.append_token(token)
        if sequence.first_token_time is None:
            sequence.first_token_time = now
            state = self._requests.get(sequence.request.request_id)
            if state is not None and state.first_token_time is None:
                state.first_token_time = now
                self.metrics.ttft_seconds.observe(now - sequence.request.arrival_time)
        reason = finish_reason_for(
            sequence,
            token,
            eos_token_id=self.eos_token_id,
            max_context_tokens=self.config.max_context_tokens,
        )
        if reason is not None:
            sequence.finish(reason, now)
        events.append(
            TokenEvent(
                request_id=sequence.request.request_id,
                sample_index=sequence.sample_index,
                token_id=token,
                index=sequence.num_output_tokens - 1,
                finished=reason is not None,
                finish_reason=reason,
            )
        )

    def _collect_finished(self, now: float) -> tuple[RequestOutput, ...]:
        completed: list[RequestOutput] = []
        for sequence in self.scheduler.drop_finished():
            state = self._requests.get(sequence.request.request_id)
            if state is None:  # pragma: no cover - aborted between scheduling and collection
                continue
            state.finished[sequence.sample_index] = SampleOutput(
                sample_index=sequence.sample_index,
                token_ids=tuple(sequence.output_token_ids),
                finish_reason=sequence.finish_reason or FinishReason.LENGTH,
                preemptions=sequence.preemption_count,
            )
            state.cached_prompt_tokens = max(
                state.cached_prompt_tokens,
                sequence.cached_prefix_blocks * self.config.block_size,
            )
            if state.complete:
                completed.append(self._finalize(state, now))
        return tuple(completed)

    def _finalize(self, state: _RequestState, now: float) -> RequestOutput:
        request = state.request
        del self._requests[request.request_id]
        if self.entitlements is not None:
            self.entitlements.release(request.key_id)
        output = RequestOutput(
            request_id=request.request_id,
            key_id=request.key_id,
            slo_class=request.slo_class,
            samples=tuple(state.finished[index] for index in sorted(state.finished)),
            prompt_tokens=request.prompt_len,
            cached_prompt_tokens=state.cached_prompt_tokens,
            arrival_time=request.arrival_time,
            first_token_time=state.first_token_time if state.first_token_time is not None else now,
            finish_time=now,
            deadline=request.deadline,
        )
        deadline = request.deadline
        if deadline is not None:
            self._deadlined += 1
            if now <= deadline:
                self._met_deadline += 1
            self.metrics.slo_attainment.set(self.slo_attainment)
        self.metrics.request_latency_seconds.observe(output.latency)
        self.metrics.tpot_seconds.observe(output.tpot)
        return output

    def _record_metrics(self, scheduled_output: SchedulerOutput, duration: float) -> None:
        prefill = sum(item.num_tokens for item in scheduled_output.scheduled if item.is_prefill)
        decode = scheduled_output.num_batched_tokens - prefill
        self.metrics.observe_iteration(
            seconds=duration,
            batched_tokens=scheduled_output.num_batched_tokens,
            prefill_tokens=prefill,
            decode_tokens=decode,
        )
        self.metrics.observe_queues(
            queue_depth=self.scheduler.queue_depth,
            running=self.scheduler.num_running,
            blocks_used_ratio=self.blocks.utilization,
            prefix_hit_ratio=self.prefix_cache.stats.hit_ratio,
        )

    # --- convenience ---------------------------------------------------------------------

    def run_until_idle(self, max_iterations: int = 100_000) -> list[RequestOutput]:
        """Step until nothing is left to run. Used by tests, the demo and the benchmarks.

        An empty batch while sequences are still queued means the scheduler cannot make progress,
        which is a bug rather than a state to wait out, so it is raised instead of spun on.
        """
        outputs: list[RequestOutput] = []
        for _ in range(max_iterations):
            if not self.has_work:
                return outputs
            step = self.step()
            outputs.extend(step.finished)
            if step.is_empty:
                if not self.scheduler.has_work:
                    return outputs
                raise RuntimeError(
                    "the scheduler produced an empty batch with "
                    f"{self.scheduler.num_running} running and "
                    f"{self.scheduler.queue_depth} waiting sequences"
                )
        raise RuntimeError(f"engine did not drain within {max_iterations} iterations")

    def generate(self, requests: Iterable[Request]) -> list[RequestOutput]:
        """Submit every request, then drain. Refused requests simply do not appear in the result."""
        for request in requests:
            self.submit(request)
        return self.run_until_idle()

    def profile_latency(
        self, batch_sizes: tuple[int, ...] | None = None, *, repeats: int = 3, warmup: int = 1
    ) -> LatencyModel:
        """Measure this host's iteration latency and refit the admission model.

        Profiling runs on a scratch KV pool so the live cache is untouched, and uses single-
        sequence prefill batches: that is the shape the engine takes when it is prefill bound,
        which is the regime where admission control decides anything. The per-sequence overhead of
        a wide decode batch is not captured by the fit -- the EWMA correction absorbs it from the
        first real iterations onwards.
        """
        budget = self.config.max_batched_tokens
        limit = min(budget, self.model.config.max_seq_len)
        sizes = batch_sizes or tuple(sorted({1, max(1, limit // 8), max(1, limit // 2), limit}))
        blocks = -(-max(sizes) // self.config.block_size)
        scratch = ModelRunner(self.model, num_blocks=blocks, block_size=self.config.block_size)
        table = tuple(range(blocks))

        def run(size: int) -> None:
            entry = BatchEntry(
                seq_id="profile",
                token_ids=(0,) * size,
                positions=tuple(range(size)),
                block_table=table,
                logits_index=size - 1,
            )
            scratch.execute(ExecutionBatch((entry,)))

        fitted = LatencyModel.profile(
            run,
            sizes,
            repeats=repeats,
            warmup=warmup,
            clock=self.clock,
            ewma_alpha=self.config.latency_ewma_alpha,
        )
        self.latency.update(fitted.intercept, fitted.slope)
        return self.latency

    def flush_prefix_cache(self) -> int:
        """Drop every evictable prefix-cache entry. Returns the number of blocks released."""
        return self.prefix_cache.flush()

    def check_invariants(self) -> None:
        """Assert the allocator and cache invariants. Called by tests after every scenario."""
        self.cache.check_invariants()

    def live_sequences(self) -> tuple[Sequence, ...]:
        return (*self.scheduler.running, *self.scheduler.waiting)

    def sequence_status(self, request_id: str, sample_index: int = 0) -> SequenceStatus | None:
        state = self._requests.get(request_id)
        if state is None:
            return None
        sequence = state.sequences.get(sample_index)
        return sequence.status if sequence is not None else None
