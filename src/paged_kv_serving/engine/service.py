"""Asyncio wrapper around the synchronous engine loop.

The engine itself knows nothing about concurrency: it exposes ``step()`` and expects to be called
in a loop. This module supplies that loop as a background task, hands each in-flight request an
``asyncio.Queue`` of its own tokens, and guarantees the one thing an SSE endpoint must get right --
that a client hanging up mid-stream releases the KV blocks its request is holding, immediately,
instead of paying for ``max_tokens`` of generation nobody will read.

``step()`` runs in a worker thread so a multi-millisecond forward pass does not stall the event
loop and stop other clients' bytes from going out. A single lock serialises every mutation of the
engine (stepping, submitting, aborting), so exactly one thread is ever inside it.

Draining is explicit: ``drain()`` stops admitting, lets the queues empty, and only then stops the
loop. The SIGTERM handler installed by the CLI calls it, so a rolling restart finishes the
requests it already accepted instead of resetting their connections.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from types import TracebackType
from typing import TypeVar

from paged_kv_serving.core.logging import get_logger
from paged_kv_serving.engine.admission import AdmissionDecision
from paged_kv_serving.engine.engine import Engine, RequestOutput, TokenEvent
from paged_kv_serving.engine.runner import BatchEntry, ExecutionBatch
from paged_kv_serving.engine.sequence import Request

__all__ = ["EngineService", "RequestTimeoutError", "ServiceClosedError", "StreamItem"]

_log = get_logger("paged_kv_serving.service")
_T = TypeVar("_T")


class ServiceClosedError(RuntimeError):
    """The service is draining or stopped and cannot accept new work."""


class RequestTimeoutError(TimeoutError):
    """A request ran past its wall-clock limit and was aborted."""


@dataclass(frozen=True, slots=True)
class StreamItem:
    """One message on a request's stream: a token, the final summary, or a timeout marker."""

    token: TokenEvent | None = None
    output: RequestOutput | None = None
    timed_out: bool = False

    @property
    def is_final(self) -> bool:
        return self.output is not None


class EngineService:
    """Owns the engine, the stepping task and one queue per in-flight request."""

    def __init__(self, engine: Engine, *, idle_sleep: float = 0.002) -> None:
        if idle_sleep <= 0:
            raise ValueError("idle_sleep must be > 0")
        self.engine = engine
        self.idle_sleep = idle_sleep
        self._queues: dict[str, asyncio.Queue[StreamItem]] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._accepting = False
        self._stopping = False
        self._warm = False
        # Last verdict of the readiness allocator check, reported while the lock is busy.
        self._allocator_ok = True
        # Abort tasks started from a cancelled stream; held here so they cannot be collected.
        self._background: set[asyncio.Task[None]] = set()

    # --- lifecycle -----------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def accepting(self) -> bool:
        return self._accepting

    @property
    def inflight(self) -> int:
        return len(self._queues)

    @property
    def warm(self) -> bool:
        """True once a forward pass has touched the KV pool, so /readyz is not premature."""
        return self._warm

    def _warmup(self) -> None:
        entry = BatchEntry(
            seq_id="warmup", token_ids=(0,), positions=(0,), block_table=(0,), logits_index=0
        )
        self.engine.runner.execute(ExecutionBatch((entry,)))
        self.engine.runner.reset_cache()
        self._warm = True

    async def start(self) -> None:
        if self.running:
            return
        self._stopping = False
        await asyncio.to_thread(self._warmup)
        self._accepting = True
        self._task = asyncio.create_task(self._loop(), name="engine-loop")

    async def drain(self, timeout: float = 30.0) -> bool:
        """Stop admitting and wait for in-flight requests. True when the queue emptied in time."""
        self._accepting = False
        deadline = asyncio.get_running_loop().time() + timeout
        while self._queues and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(self.idle_sleep)
        return not self._queues

    async def stop(self) -> None:
        self._accepting = False
        self._stopping = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for queue in list(self._queues.values()):
            queue.put_nowait(StreamItem())
        self._queues.clear()

    async def __aenter__(self) -> EngineService:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.stop()

    # --- request handling ----------------------------------------------------------------

    async def submit(self, request: Request) -> AdmissionDecision:
        """Admit a request and open its stream, or return the rejection unchanged."""
        if not self._accepting:
            raise ServiceClosedError("the engine service is not accepting requests")
        async with self._lock:
            decision = self.engine.submit(request)
            if decision.admitted:
                self._queues[request.request_id] = asyncio.Queue()
        return decision

    async def run_exclusive(self, operation: Callable[[], _T]) -> _T:
        """Run ``operation`` while no iteration is in flight.

        The stepping loop holds the same lock around every ``step()``, so an admin operation that
        mutates the engine (flushing the prefix cache, swapping weights) can never interleave with
        a forward pass that is reading the blocks it is changing.
        """
        async with self._lock:
            return operation()

    async def abort(self, request_id: str) -> None:
        """Cancel a request and free its blocks. Safe to call for an unknown id."""
        async with self._lock:
            removed = self.engine.abort(request_id)
        queue = self._queues.pop(request_id, None)
        if queue is not None:
            queue.put_nowait(StreamItem())
        if removed:
            _log.info("request_aborted", request_id=request_id, sequences=removed)

    async def stream(
        self, request_id: str, *, timeout: float | None = None
    ) -> AsyncIterator[StreamItem]:
        """Yield this request's tokens and then its summary.

        ``timeout`` bounds the whole stream, not each token: when it runs out a single
        ``timed_out`` item is yielded and the stream ends. The ``finally`` clause is the
        cancellation path: whether the client disconnected, the time limit ran out, the server is
        shutting down or the caller broke out of the loop, an unfinished request is aborted and
        its blocks go back to the pool on the very next iteration.
        """
        queue = self._queues.get(request_id)
        if queue is None:
            raise KeyError(request_id)
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        finished = False
        try:
            while True:
                try:
                    async with asyncio.timeout_at(deadline):
                        item = await queue.get()
                except TimeoutError:
                    _log.info("request_timed_out", request_id=request_id)
                    yield StreamItem(timed_out=True)
                    return
                if item.is_final:
                    finished = True
                    yield item
                    return
                if item.token is None:
                    return
                yield item
        finally:
            self._queues.pop(request_id, None)
            if not finished:
                # A client disconnect arrives as cancellation of this generator. Under anyio (which
                # sse-starlette uses) every later await in a cancelled scope is cancelled again, so
                # awaiting the abort directly would never acquire the engine lock and the blocks
                # would stay pinned. The abort therefore runs as its own task, shielded from the
                # cancellation that triggered it.
                task = asyncio.get_running_loop().create_task(self.abort(request_id))
                self._background.add(task)
                task.add_done_callback(self._background.discard)
                await asyncio.shield(task)

    async def collect(
        self, request_id: str, *, timeout: float | None = None
    ) -> RequestOutput | None:
        """Consume a whole stream and return the final summary (None if it was cut short).

        Raises :class:`RequestTimeoutError` when ``timeout`` runs out first; the request has
        already been aborted and its blocks freed by then.
        """
        final: RequestOutput | None = None
        async for item in self.stream(request_id, timeout=timeout):
            if item.timed_out:
                raise RequestTimeoutError(request_id)
            if item.output is not None:
                final = item.output
        return final

    # --- the loop ------------------------------------------------------------------------

    async def _loop(self) -> None:
        stalled = False
        while not self._stopping:
            async with self._lock:
                has_work = self.engine.has_work
                step = await asyncio.to_thread(self.engine.step) if has_work else None
            if step is None:
                await asyncio.sleep(self.idle_sleep)
                continue
            if step.is_empty:
                # Work is queued but nothing could be scheduled. The scheduler is built so this
                # cannot happen (docs/SCHEDULER.md, liveness), but if it ever does the loop backs
                # off like an idle one instead of retrying at once with the lock held, which
                # would starve every submit, abort and admin call of the lock.
                if not stalled:
                    _log.warning(
                        "engine_stalled", running=step.num_running, queue_depth=step.queue_depth
                    )
                stalled = True
                await asyncio.sleep(self.idle_sleep)
                continue
            stalled = False
            for token in step.tokens:
                queue = self._queues.get(token.request_id)
                if queue is not None:
                    queue.put_nowait(StreamItem(token=token))
            for output in step.finished:
                queue = self._queues.pop(output.request_id, None)
                if queue is not None:
                    queue.put_nowait(StreamItem(output=output))
            await asyncio.sleep(0)

    # --- readiness -----------------------------------------------------------------------

    async def allocator_healthy(self, timeout: float = 1.0) -> bool:
        """True when the block allocator's invariants hold. Used by ``/readyz``.

        ``step()`` mutates the allocator in a worker thread, so reading it from the event loop
        without the engine lock can catch it half-way through an update and report a healthy pod
        as broken. The check therefore takes the lock like every other reader. If an iteration
        holds the lock for longer than ``timeout``, the previous verdict is reported: a slow
        iteration is not evidence of a broken allocator, and a probe must not queue up behind it.
        """
        try:
            async with asyncio.timeout(timeout):
                await self._lock.acquire()
        except TimeoutError:
            return self._allocator_ok
        try:
            self.engine.check_invariants()
            healthy = True
        except AssertionError:
            healthy = False
        finally:
            self._lock.release()
        self._allocator_ok = healthy
        return healthy

    def snapshot(self) -> dict[str, float]:
        return {**self.engine.snapshot(), "inflight": float(len(self._queues))}
