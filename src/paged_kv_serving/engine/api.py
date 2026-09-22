"""HTTP API: streaming completions, engine stats and audited admin operations.

``POST /v1/completions`` streams tokens over server-sent events. Streaming is not decoration here:
it is what makes cancellation observable. When a client closes the connection, the SSE generator
is closed, its ``finally`` aborts the request, and the KV blocks go back to the pool on the next
iteration instead of being held for the rest of ``max_tokens``.

Everything a caller may ask for is bounded before the engine sees it: the request body is a closed
schema, prompt and completion lengths are capped by configuration *and* by the calling key's own
policy, and the SLO class is an entitlement rather than a field the caller can simply set. A
request that cannot make its deadline is refused with 503 and a truthful ``Retry-After`` computed
from the current backlog, not a fixed constant.

Prompts and completions are never logged. Every log line and audit record carries lengths and a
SHA-256 of the prompt, which is enough to correlate a complaint with a request and not enough to
reconstruct what anybody asked.

Admin routes (flush the prefix cache, swap the weights) need a key whose policy grants the admin
scope, and each one appends to the hash-chained audit log before returning.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI
from fastapi import Request as HttpRequest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Tracer
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sse_starlette.sse import EventSourceResponse

from paged_kv_serving.core.app import create_app
from paged_kv_serving.core.config import Settings, get_settings
from paged_kv_serving.core.errors import AppError
from paged_kv_serving.core.logging import get_logger
from paged_kv_serving.core.security import Principal, require_api_key
from paged_kv_serving.engine.admission import RejectionReason
from paged_kv_serving.engine.audit import AuditLog
from paged_kv_serving.engine.config import EngineConfig, SloClass
from paged_kv_serving.engine.engine import Engine, RequestOutput
from paged_kv_serving.engine.keys import EntitlementGuard, KeyPolicy, KeyPolicyStore
from paged_kv_serving.engine.metrics import create_engine_metrics
from paged_kv_serving.engine.monitoring import PromptLengthMonitor
from paged_kv_serving.engine.sequence import Request as EngineRequest
from paged_kv_serving.engine.sequence import SamplingParams
from paged_kv_serving.engine.service import (
    EngineService,
    RequestTimeoutError,
    ServiceClosedError,
    StreamItem,
)
from paged_kv_serving.engine.tokenizer import ByteBPETokenizer
from paged_kv_serving.engine.weights import Checkpoint, load_checkpoint

__all__ = [
    "AdminWeightSwap",
    "CompletionBody",
    "CompletionResponse",
    "build_service",
    "create_api",
    "parse_key_policies",
]

_log = get_logger("paged_kv_serving.api")
# Module level on purpose: with postponed annotations FastAPI resolves a handler's annotations
# against module globals, so a dependency alias defined inside a factory is never seen.
PrincipalDep = Annotated[Principal, Depends(require_api_key)]
_SAFE_NAME = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
# Reasons that describe the caller's own request rather than the server's current load.
_CLIENT_ERRORS = {RejectionReason.PROMPT_TOO_LONG, RejectionReason.CONTEXT_TOO_LONG}


class CompletionBody(BaseModel):
    """Closed request schema. Unknown fields are rejected rather than ignored."""

    model_config = ConfigDict(extra="forbid")

    prompt: str | None = Field(default=None, max_length=65_536)
    prompt_token_ids: list[int] | None = Field(default=None, max_length=8_192)
    max_tokens: int = Field(default=32, ge=1, le=8_192)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    n: int = Field(default=1, ge=1, le=8)
    stream: bool = False
    slo_class: Literal["interactive", "batch"] = "interactive"
    deadline_ms: int | None = Field(default=None, ge=1, le=600_000)

    @model_validator(mode="after")
    def _exactly_one_prompt(self) -> CompletionBody:
        if (self.prompt is None) == (self.prompt_token_ids is None):
            raise ValueError("provide exactly one of prompt or prompt_token_ids")
        return self


class CompletionChoice(BaseModel):
    index: int
    text: str
    token_ids: list[int]
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    choices: list[CompletionChoice]
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    ttft_seconds: float
    latency_seconds: float


class AdminWeightSwap(BaseModel):
    """Load a checkpoint by *name* from the configured weights root, never by path."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _safe_name(self) -> AdminWeightSwap:
        if set(self.name) - _SAFE_NAME or self.name.startswith("."):
            raise ValueError("name must be a plain directory name")
        return self


def parse_key_policies(entries: Sequence[str]) -> KeyPolicyStore:
    """Build a policy store from the validated ``APP_KEY_POLICIES`` entries.

    A policy is keyed by the first twelve hex characters of the key's SHA-256, which is exactly
    the non-secret ``key_id`` the authentication layer derives, so the configuration never has to
    repeat a credential.
    """
    policies = []
    for entry in entries:
        digest, classes, per_minute, concurrent, prompt, new_tokens, admin = entry.split(":")
        policies.append(
            KeyPolicy(
                key_id=digest[:12],
                slo_classes=frozenset(SloClass.parse(name) for name in classes.split("|")),
                tokens_per_minute=int(per_minute),
                max_concurrent_requests=int(concurrent),
                max_prompt_tokens=int(prompt),
                max_new_tokens=int(new_tokens),
                admin=admin == "1",
            )
        )
    return KeyPolicyStore(policies)


def _engine_config(settings: Settings) -> EngineConfig:
    return EngineConfig(
        block_size=settings.engine_block_size,
        num_blocks=settings.engine_num_blocks,
        max_batched_tokens=settings.engine_max_batched_tokens,
        max_running_seqs=settings.engine_max_running_seqs,
        max_prompt_tokens=settings.engine_max_prompt_tokens,
        max_new_tokens=settings.engine_max_new_tokens,
        max_context_tokens=settings.engine_max_context_tokens,
        enable_prefix_cache=settings.engine_enable_prefix_cache,
    )


def build_service(
    checkpoint: Checkpoint,
    settings: Settings,
    *,
    registry: CollectorRegistry | None = None,
) -> EngineService:
    """Wire a verified checkpoint, the configured limits and the entitlements into a service."""
    store = parse_key_policies(settings.key_policies)
    engine = Engine(
        checkpoint.model,
        _engine_config(settings),
        eos_token_id=checkpoint.tokenizer.eos_token_id,
        metrics=create_engine_metrics(registry),
        prefix_cache_scope=settings.engine_prefix_cache_scope,
        entitlements=EntitlementGuard(store) if len(store) else None,
        drift_monitor=PromptLengthMonitor.learning(
            settings.drift_reference_size, window=settings.drift_window
        ),
        max_queued_sequences=settings.engine_max_queued_sequences,
    )
    if settings.engine_profile_on_start:
        engine.profile_latency()
    return EngineService(engine)


def _principal_policy(request: HttpRequest, principal: Principal) -> KeyPolicy | None:
    store: KeyPolicyStore | None = request.app.state.key_policies
    return store.get(principal.key_id) if store is not None else None


def _reject(reason: RejectionReason, retry_after: str) -> AppError:
    """Map an engine refusal to HTTP: 400 for the request itself, 403 for an entitlement the key
    does not have, 429 for the key's own limits, 503 (with a truthful Retry-After) for load."""
    if reason in _CLIENT_ERRORS:
        return AppError(400, reason.value, "The request exceeds a configured limit.")
    if reason is RejectionReason.CLASS_NOT_ENTITLED:
        return AppError(403, reason.value, "This key is not entitled to that service class.")
    if reason is RejectionReason.KEY_TOKEN_RATE or reason is RejectionReason.KEY_CONCURRENCY:
        return AppError(
            429,
            reason.value,
            "The key's rate or concurrency limit is exhausted.",
            headers={"Retry-After": retry_after},
        )
    return AppError(
        503,
        reason.value,
        "The server cannot finish this request in time; retry later.",
        headers={"Retry-After": retry_after},
    )


def create_api(
    checkpoint: Checkpoint,
    settings: Settings | None = None,
    *,
    service: EngineService | None = None,
    audit_log: AuditLog | None = None,
    registry: CollectorRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    """Build the full application: baseline hardening, completions, stats and admin.

    Pass ``service`` together with the ``registry`` it was built on to keep the HTTP and engine
    metric families on one registry, which is what makes ``/metrics`` show both.
    """
    cfg = settings if settings is not None else get_settings()
    metrics_registry = registry if registry is not None else CollectorRegistry()
    engine_service = (
        service
        if service is not None
        else build_service(checkpoint, cfg, registry=metrics_registry)
    )
    tokenizer = checkpoint.tokenizer
    audit = audit_log if audit_log is not None else AuditLog(cfg.data_dir / "admin-audit.jsonl")
    policies = parse_key_policies(cfg.key_policies)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await engine_service.start()
        try:
            yield
        finally:
            # Graceful drain: in-flight requests finish before the loop stops. SIGTERM reaches
            # uvicorn, uvicorn closes the lifespan, and this is where the engine winds down.
            await engine_service.drain(timeout=cfg.drain_timeout_seconds)
            await engine_service.stop()

    app = create_app(
        cfg,
        lifespan=lifespan,
        readiness_checks={
            "weights_warmed": lambda: engine_service.warm,
            "engine_running": lambda: engine_service.running,
            "accepting": lambda: engine_service.accepting,
            "allocator_healthy": engine_service.allocator_healthy,
        },
        registry=metrics_registry,
        tracer_provider=tracer_provider,
    )
    app.state.engine_service = engine_service
    app.state.key_policies = policies
    app.state.audit_log = audit
    app.state.tokenizer = tokenizer
    app.state.checkpoint = checkpoint

    _install_completion_routes(
        app,
        cfg=cfg,
        service=engine_service,
        checkpoint=checkpoint,
        tracer=app.state.tracer_provider.get_tracer("paged_kv_serving.engine"),
    )

    @app.get("/v1/engine/stats")
    async def engine_stats(principal: PrincipalDep) -> dict[str, float]:
        """Live gauges for an operator. Authenticated: queue depth is load information."""
        _log.debug("engine_stats", key_id=principal.key_id)
        return engine_service.snapshot()

    _install_admin_routes(app, cfg=cfg, service=engine_service, audit=audit, tokenizer=tokenizer)
    return app


def _install_completion_routes(
    app: FastAPI,
    *,
    cfg: Settings,
    service: EngineService,
    checkpoint: Checkpoint,
    tracer: Tracer,
) -> None:
    """``POST /v1/completions``: validate, admit or refuse, then answer as JSON or SSE."""
    tokenizer = checkpoint.tokenizer

    def _build_request(body: CompletionBody, principal: Principal, arrival: float) -> EngineRequest:
        token_ids = (
            tokenizer.encode(body.prompt)
            if body.prompt is not None
            else list(body.prompt_token_ids or [])
        )
        if not token_ids:
            raise AppError(400, "empty_prompt", "The prompt must contain at least one token.")
        vocab = checkpoint.model.config.vocab_size
        if any(not 0 <= token < vocab for token in token_ids):
            raise AppError(400, "invalid_token", "A prompt token is outside the vocabulary.")
        return EngineRequest(
            request_id=uuid.uuid4().hex,
            prompt_token_ids=tuple(token_ids),
            sampling=SamplingParams(
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                seed=body.seed,
            ),
            n=body.n,
            slo_class=SloClass.parse(body.slo_class),
            deadline_seconds=None if body.deadline_ms is None else body.deadline_ms / 1000.0,
            key_id=principal.key_id,
            arrival_time=arrival,
        )

    @app.post("/v1/completions")
    async def completions(
        body: CompletionBody, principal: PrincipalDep, http_request: HttpRequest
    ) -> Any:
        engine = service.engine
        arrival = engine.clock()
        engine_request = _build_request(body, principal, arrival)
        _log.info(
            "completion_requested",
            request_id=engine_request.request_id,
            key_id=principal.key_id,
            prompt_tokens=engine_request.prompt_len,
            prompt_sha256=engine_request.prompt_sha256(),
            max_tokens=body.max_tokens,
            slo_class=body.slo_class,
            stream=body.stream,
        )
        # One span per completion, child of the HTTP span. Attributes are lengths, classes and
        # outcomes only -- never prompt or completion text.
        span = tracer.start_span(
            "engine.completion",
            attributes={
                "engine.request_id": engine_request.request_id,
                "engine.key_id": principal.key_id,
                "engine.slo_class": body.slo_class,
                "engine.prompt_tokens": engine_request.prompt_len,
                "engine.max_tokens": body.max_tokens,
                "engine.n": body.n,
                "engine.stream": body.stream,
            },
        )
        handed_off = False
        try:
            with trace.use_span(span, end_on_exit=False):
                try:
                    decision = await service.submit(engine_request)
                except ServiceClosedError as exc:
                    span.set_attribute("engine.outcome", "draining")
                    raise AppError(
                        503,
                        "draining",
                        "The server is shutting down.",
                        headers={"Retry-After": "5"},
                    ) from exc
                if not decision.admitted:
                    reason = decision.reason or RejectionReason.DEADLINE_INFEASIBLE
                    span.set_attribute("engine.outcome", f"rejected:{reason.value}")
                    raise _reject(reason, decision.retry_after_header)
                timeout = cfg.request_timeout_seconds
                if body.stream:
                    handed_off = True
                    return EventSourceResponse(
                        _sse(
                            service,
                            engine_request.request_id,
                            tokenizer,
                            http_request,
                            timeout=timeout,
                            span=span,
                        ),
                        ping=15,
                    )
                try:
                    output = await service.collect(engine_request.request_id, timeout=timeout)
                except RequestTimeoutError as exc:
                    span.set_attribute("engine.outcome", "timeout")
                    raise AppError(
                        504, "request_timeout", "The request ran past the server's time limit."
                    ) from exc
                if output is None:  # pragma: no cover - only on shutdown mid-request
                    span.set_attribute("engine.outcome", "cancelled")
                    raise AppError(503, "cancelled", "The request did not complete.")
                _record_output(span, output)
        finally:
            if not handed_off:
                span.end()
        return CompletionResponse(
            id=output.request_id,
            choices=[
                CompletionChoice(
                    index=sample.sample_index,
                    text=tokenizer.decode(sample.token_ids),
                    token_ids=list(sample.token_ids),
                    finish_reason=sample.finish_reason.value,
                )
                for sample in output.samples
            ],
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.output_tokens,
            cached_prompt_tokens=output.cached_prompt_tokens,
            ttft_seconds=round(output.ttft, 6),
            latency_seconds=round(output.latency, 6),
        )


def _install_admin_routes(
    app: FastAPI,
    *,
    cfg: Settings,
    service: EngineService,
    audit: AuditLog,
    tokenizer: ByteBPETokenizer,
) -> None:
    """Flush the prefix cache and swap weights: admin scope only, every call audited."""

    def _require_admin(request: HttpRequest, principal: Principal) -> KeyPolicy:
        policy = _principal_policy(request, principal)
        if policy is None or not policy.admin:
            raise AppError(403, "forbidden", "This operation requires the admin scope.")
        return policy

    @app.post("/v1/admin/prefix-cache/flush")
    async def flush_prefix_cache(
        principal: PrincipalDep, http_request: HttpRequest
    ) -> dict[str, Any]:
        _require_admin(http_request, principal)
        released = await service.run_exclusive(service.engine.flush_prefix_cache)
        entry = audit.append(principal.key_id, "prefix_cache.flush", {"released_blocks": released})
        return {"released_blocks": released, "audit_index": entry.index, "audit_head": entry.digest}

    @app.post("/v1/admin/weights")
    async def swap_weights(
        body: AdminWeightSwap, principal: PrincipalDep, http_request: HttpRequest
    ) -> dict[str, Any]:
        _require_admin(http_request, principal)
        engine = service.engine
        directory = Path(cfg.weights_root) / body.name
        try:
            # Hash verification reads every byte, so it runs off the event loop and outside the
            # engine lock; nothing is deserialised unless the digests match.
            loaded = await asyncio.to_thread(
                load_checkpoint, directory, expect_manifest_sha256=body.manifest_sha256
            )
        except (ValueError, OSError) as exc:
            audit.append(
                principal.key_id,
                "weights.swap_rejected",
                {"name": body.name, "error": type(exc).__name__},
            )
            raise AppError(
                400, "invalid_checkpoint", "The checkpoint failed verification."
            ) from exc
        if (
            loaded.model.config != engine.model.config
            or loaded.tokenizer.merges != tokenizer.merges
        ):
            audit.append(
                principal.key_id,
                "weights.swap_rejected",
                {"name": body.name, "error": "shape_mismatch"},
            )
            raise AppError(
                409, "shape_mismatch", "The checkpoint has a different model shape or tokenizer."
            )

        def swap() -> bool:
            if engine.active_requests:
                return False
            engine.model.weights = loaded.model.weights
            engine.flush_prefix_cache()
            engine.runner.reset_cache()
            return True

        # Under the engine lock: no forward pass can be reading the old weights or the KV blocks
        # they produced while they are replaced.
        if not await service.run_exclusive(swap):
            raise AppError(
                409, "engine_busy", "Weights can only be swapped while the engine is idle."
            )
        entry = audit.append(
            principal.key_id,
            "weights.swap",
            {"name": body.name, "manifest_sha256": body.manifest_sha256},
        )
        return {"swapped": body.name, "audit_index": entry.index, "audit_head": entry.digest}


async def _sse(
    service: EngineService,
    request_id: str,
    tokenizer: ByteBPETokenizer,
    http_request: HttpRequest,
    *,
    timeout: float | None = None,
    span: Span | None = None,
) -> AsyncIterator[dict[str, str]]:
    """Yield one event per token, then a summary and ``[DONE]``.

    Breaking out of this generator -- which is what a client disconnect does -- runs the abort in
    :meth:`EngineService.stream`'s ``finally``. A request that runs past ``timeout`` gets an
    ``error`` event instead of a summary, and has already been aborted when it is sent.
    """
    outcome = "disconnected"
    try:
        async for item in service.stream(request_id, timeout=timeout):
            if await http_request.is_disconnected():
                break
            if item.timed_out:
                outcome = "timeout"
                yield {"event": "error", "data": json.dumps({"error": "request_timeout"})}
                continue
            if item.output is not None and span is not None:
                _record_output(span, item.output)
                outcome = "completed"
            for event in _sse_events(item, tokenizer):
                yield event
        yield {"event": "done", "data": "[DONE]"}
    finally:
        if span is not None:
            if outcome != "completed":
                span.set_attribute("engine.outcome", outcome)
            span.end()


def _record_output(span: Span, output: RequestOutput) -> None:
    span.set_attribute("engine.outcome", "completed")
    span.set_attribute("engine.completion_tokens", output.output_tokens)
    span.set_attribute("engine.cached_prompt_tokens", output.cached_prompt_tokens)
    span.set_attribute("engine.ttft_seconds", round(output.ttft, 6))
    span.set_attribute("engine.latency_seconds", round(output.latency, 6))
    span.set_attribute("engine.preemptions", output.preemptions)


def _sse_events(item: StreamItem, tokenizer: ByteBPETokenizer) -> list[dict[str, str]]:
    """The SSE events for one stream item: a ``token`` event or the ``summary`` event."""
    if item.token is not None:
        payload: dict[str, Any] = {
            "index": item.token.sample_index,
            "token_id": item.token.token_id,
            "text": tokenizer.decode([item.token.token_id]),
            "finished": item.token.finished,
        }
        return [{"event": "token", "data": json.dumps(payload)}]
    if item.output is not None:
        output = item.output
        payload = {
            "id": output.request_id,
            "prompt_tokens": output.prompt_tokens,
            "completion_tokens": output.output_tokens,
            "cached_prompt_tokens": output.cached_prompt_tokens,
            "ttft_seconds": round(output.ttft, 6),
            "latency_seconds": round(output.latency, 6),
            "choices": [
                {
                    "index": sample.sample_index,
                    "text": tokenizer.decode(sample.token_ids),
                    "finish_reason": sample.finish_reason.value,
                }
                for sample in output.samples
            ],
        }
        return [{"event": "summary", "data": json.dumps(payload)}]
    return []
