"""The served API: completions over JSON and SSE, entitlements, limits, cancellation, admin.

Most tests drive the real FastAPI app in process, lifespan included. The cancellation test runs
uvicorn on a loopback port instead, because httpx's in-process transport buffers a response and a
client using it can never hang up mid-stream.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry

from paged_kv_serving.core.config import Settings
from paged_kv_serving.core.logging import configure_logging
from paged_kv_serving.core.security import hash_api_key
from paged_kv_serving.core.tracing import build_tracer_provider
from paged_kv_serving.engine.admission import RejectionReason
from paged_kv_serving.engine.api import (
    AdminWeightSwap,
    CompletionBody,
    _reject,
    create_api,
    parse_key_policies,
)
from paged_kv_serving.engine.audit import AuditChainError, AuditLog
from paged_kv_serving.engine.config import EngineConfig
from paged_kv_serving.engine.engine import Engine
from paged_kv_serving.engine.keys import EntitlementGuard
from paged_kv_serving.engine.localserver import serve_in_thread
from paged_kv_serving.engine.metrics import create_engine_metrics
from paged_kv_serving.engine.model import random_weights
from paged_kv_serving.engine.runner import ExecutionBatch, ModelRunner
from paged_kv_serving.engine.sequence import Request
from paged_kv_serving.engine.service import EngineService, ServiceClosedError
from paged_kv_serving.engine.weights import Checkpoint, write_checkpoint

pytestmark = pytest.mark.integration

USER = "user-key"
BATCH_ONLY = "batch-only-key"
ADMIN = "admin-key"
STINGY = "stingy-key"
UNLISTED = "unlisted-key"
SECRET_PROMPT = "zebra quartz nebula prompt"


def settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    digests = {name: hash_api_key(name) for name in (USER, BATCH_ONLY, ADMIN, STINGY, UNLISTED)}
    values: dict[str, Any] = {
        "_env_file": None,
        "api_key_sha256": list(digests.values()),
        "key_policies": [
            f"{digests[USER]}:interactive|batch:100000:4:64:16:0:4",
            f"{digests[BATCH_ONLY]}:batch:100000:4:64:16:0",
            f"{digests[ADMIN]}:interactive|batch:100000:4:64:16:1",
            f"{digests[STINGY]}:interactive|batch:40:4:64:16:0",
        ],
        "rate_limit_per_second": 1000.0,
        "rate_limit_burst": 1000,
        "engine_block_size": 8,
        "engine_num_blocks": 64,
        "engine_max_batched_tokens": 32,
        "engine_max_running_seqs": 6,
        "engine_max_prompt_tokens": 96,
        "engine_max_new_tokens": 16,
        "engine_max_context_tokens": 128,
        "engine_profile_on_start": False,
        "weights_root": tmp_path / "weights",
        "data_dir": tmp_path / "data",
    }
    values.update(overrides)
    return Settings(**values)


class SlowExecutor:
    """The real runner, plus a fixed delay per iteration, so a request is still running later."""

    def __init__(self, runner: ModelRunner, delay: float) -> None:
        self.runner = runner
        self.delay = delay

    def execute(self, batch: ExecutionBatch) -> dict[str, np.ndarray]:
        time.sleep(self.delay)
        return self.runner.execute(batch)


def slow_service(
    checkpoint: Checkpoint, settings: Settings, registry: CollectorRegistry, delay: float
) -> EngineService:
    engine = Engine(
        checkpoint.model,
        EngineConfig(
            block_size=settings.engine_block_size,
            num_blocks=settings.engine_num_blocks,
            max_batched_tokens=settings.engine_max_batched_tokens,
            max_running_seqs=settings.engine_max_running_seqs,
            max_prompt_tokens=settings.engine_max_prompt_tokens,
            max_new_tokens=settings.engine_max_new_tokens,
            max_context_tokens=settings.engine_max_context_tokens,
        ),
        eos_token_id=None,
        metrics=create_engine_metrics(registry),
        entitlements=EntitlementGuard(parse_key_policies(settings.key_policies)),
    )
    engine.executor = SlowExecutor(engine.runner, delay)
    return EngineService(engine)


async def running_client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
            yield client


@pytest.fixture
async def api(
    checkpoint: Checkpoint, tmp_path: Path
) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    settings = settings_for(tmp_path)
    app = create_api(checkpoint, settings, audit_log=AuditLog(tmp_path / "audit.jsonl"))
    async for client in running_client(app):
        yield client, app


def headers(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


def sse_events(text: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    event = ""
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            events.append((event, line[5:].strip()))
    return events


async def test_a_completion_returns_tokens_and_usage(
    api: tuple[httpx.AsyncClient, Any],
) -> None:
    client, app = api
    response = await client.post(
        "/v1/completions", json={"prompt": "hello there", "max_tokens": 5}, headers=headers(USER)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["completion_tokens"] == 5
    assert body["prompt_tokens"] == len(app.state.tokenizer.encode("hello there"))
    assert len(body["choices"]) == 1
    assert len(body["choices"][0]["token_ids"]) == 5
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["latency_seconds"] >= body["ttft_seconds"] >= 0


async def test_parallel_samples_and_token_id_prompts_are_supported(
    api: tuple[httpx.AsyncClient, Any],
) -> None:
    client, _ = api
    response = await client.post(
        "/v1/completions",
        json={"prompt_token_ids": [5, 6, 7, 8], "max_tokens": 4, "n": 3, "temperature": 1.0},
        headers=headers(USER),
    )
    assert response.status_code == 200
    choices = response.json()["choices"]
    assert [choice["index"] for choice in choices] == [0, 1, 2]


async def test_parallel_samples_are_limited_by_the_keys_policy(
    api: tuple[httpx.AsyncClient, Any],
) -> None:
    """Review regression: n was accepted up to 8 whatever the key's policy said."""
    client, _ = api
    body = {"prompt_token_ids": [5, 6, 7, 8], "max_tokens": 4, "temperature": 1.0}
    over = await client.post("/v1/completions", json={**body, "n": 5}, headers=headers(USER))
    assert over.status_code == 400
    assert over.json()["error"]["code"] == "too_many_samples"
    batch_body = {**body, "n": 2, "slo_class": "batch"}
    none = await client.post("/v1/completions", json=batch_body, headers=headers(BATCH_ONLY))
    assert none.status_code == 400
    assert none.json()["error"]["code"] == "too_many_samples"


async def test_streaming_emits_token_events_then_a_summary_and_done(
    api: tuple[httpx.AsyncClient, Any],
) -> None:
    client, _ = api
    response = await client.post(
        "/v1/completions",
        json={"prompt": "stream me", "max_tokens": 6, "stream": True},
        headers=headers(USER),
    )
    assert response.status_code == 200
    events = sse_events(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds.count("token") == 6
    assert kinds[-2:] == ["summary", "done"]
    summary = json.loads(events[-2][1])
    assert summary["completion_tokens"] == 6
    assert events[-1][1] == "[DONE]"


async def test_the_request_body_is_a_closed_bounded_schema(
    api: tuple[httpx.AsyncClient, Any],
) -> None:
    client, _ = api
    cases = [
        {"prompt": "x", "unexpected": 1},
        {"prompt": "x", "prompt_token_ids": [1]},
        {},
        {"prompt": "x", "max_tokens": 0},
        {"prompt": "x", "n": 99},
        {"prompt": "x", "slo_class": "platinum"},
    ]
    for body in cases:
        response = await client.post("/v1/completions", json=body, headers=headers(USER))
        assert response.status_code == 422, body
    out_of_vocab = await client.post(
        "/v1/completions", json={"prompt_token_ids": [10**6]}, headers=headers(USER)
    )
    assert out_of_vocab.status_code == 400
    assert out_of_vocab.json()["error"]["code"] == "invalid_token"


async def test_authentication_and_entitlements_are_enforced(
    api: tuple[httpx.AsyncClient, Any],
) -> None:
    client, _ = api
    body = {"prompt": "hi", "max_tokens": 2}
    assert (await client.post("/v1/completions", json=body)).status_code == 401
    assert (
        await client.post("/v1/completions", json=body, headers=headers("wrong"))
    ).status_code == 401
    # A valid key with no policy entry cannot use the engine at all.
    unlisted = await client.post("/v1/completions", json=body, headers=headers(UNLISTED))
    assert unlisted.status_code == 403
    # A batch-only key cannot buy the interactive class by asking for it.
    asked = await client.post(
        "/v1/completions", json={**body, "slo_class": "interactive"}, headers=headers(BATCH_ONLY)
    )
    assert asked.status_code == 403
    assert asked.json()["error"]["code"] == "class_not_entitled"
    allowed = await client.post(
        "/v1/completions", json={**body, "slo_class": "batch"}, headers=headers(BATCH_ONLY)
    )
    assert allowed.status_code == 200


async def test_keys_without_any_policies_cannot_use_the_engine(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    """Review regression: with API keys configured and no policies at all, entitlements were
    switched off and every valid key got interactive, parallel samples and no token or
    concurrency limit."""
    app = create_api(checkpoint, settings_for(tmp_path, key_policies=[]))
    async for client in running_client(app):
        for body in (
            {"prompt": "hi", "max_tokens": 2},
            {"prompt": "hi", "max_tokens": 2, "slo_class": "batch"},
            {"prompt": "hi", "max_tokens": 2, "n": 4, "temperature": 1.0},
        ):
            response = await client.post("/v1/completions", json=body, headers=headers(USER))
            assert response.status_code == 403, body
            assert response.json()["error"]["code"] == "class_not_entitled"
        engine = app.state.engine_service.engine
        assert engine.entitlements is not None
        assert engine.active_requests == 0


async def test_prompt_and_output_ceilings_are_hard_limits(
    api: tuple[httpx.AsyncClient, Any],
) -> None:
    client, _ = api
    too_long = await client.post(
        "/v1/completions",
        json={"prompt_token_ids": [1] * 65, "max_tokens": 2},
        headers=headers(USER),
    )
    assert too_long.status_code == 400
    assert too_long.json()["error"]["code"] == "prompt_too_long"
    too_many = await client.post(
        "/v1/completions", json={"prompt": "hi", "max_tokens": 17}, headers=headers(USER)
    )
    assert too_many.status_code == 400
    assert too_many.json()["error"]["code"] == "context_too_long"


async def test_a_key_over_its_token_rate_gets_429_with_retry_after(
    api: tuple[httpx.AsyncClient, Any],
) -> None:
    client, _ = api
    body = {"prompt_token_ids": [1] * 20, "max_tokens": 10}  # 30 tokens of a 40/minute budget
    assert (
        await client.post("/v1/completions", json=body, headers=headers(STINGY))
    ).status_code == 200
    second = await client.post("/v1/completions", json=body, headers=headers(STINGY))
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "key_token_rate"
    assert int(second.headers["Retry-After"]) >= 1


async def test_an_infeasible_deadline_is_refused_early_with_retry_after(
    api: tuple[httpx.AsyncClient, Any],
) -> None:
    client, app = api
    response = await client.post(
        "/v1/completions",
        json={"prompt_token_ids": [1] * 60, "max_tokens": 16, "deadline_ms": 1},
        headers=headers(USER),
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "deadline_infeasible"
    assert int(response.headers["Retry-After"]) >= 1
    engine = app.state.engine_service.engine
    assert engine.active_requests == 0
    assert engine.blocks.num_free_blocks == engine.config.num_blocks


def test_every_rejection_reason_maps_to_a_distinct_client_signal() -> None:
    statuses = {reason: _reject(reason, "3").status_code for reason in RejectionReason}
    assert statuses[RejectionReason.PROMPT_TOO_LONG] == 400
    assert statuses[RejectionReason.CONTEXT_TOO_LONG] == 400
    assert statuses[RejectionReason.TOO_MANY_SAMPLES] == 400
    assert statuses[RejectionReason.CLASS_NOT_ENTITLED] == 403
    assert statuses[RejectionReason.KEY_CONCURRENCY] == 429
    assert statuses[RejectionReason.KEY_TOKEN_RATE] == 429
    assert statuses[RejectionReason.DEADLINE_INFEASIBLE] == 503
    assert statuses[RejectionReason.QUEUE_FULL] == 503
    assert _reject(RejectionReason.QUEUE_FULL, "3").headers == {"Retry-After": "3"}


@pytest.mark.usefixtures("restore_logging")
async def test_prompts_and_outputs_are_never_logged(checkpoint: Checkpoint, tmp_path: Path) -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    app = create_api(checkpoint, settings_for(tmp_path))
    async for client in running_client(app):
        response = await client.post(
            "/v1/completions",
            json={"prompt": SECRET_PROMPT, "max_tokens": 4},
            headers=headers(USER),
        )
        assert response.status_code == 200
        completion_text = response.json()["choices"][0]["text"]
    logged = stream.getvalue()
    assert "completion_requested" in logged
    assert "prompt_sha256" in logged
    assert SECRET_PROMPT not in logged
    assert "zebra" not in logged
    if len(completion_text) > 3:
        assert completion_text not in logged


async def test_finished_requests_reach_the_validated_telemetry_file(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    """Review gap: the telemetry data contract was never on a production path.

    Finished requests are now exported, one contract-validated row each, and the file holds
    lengths, timings and a prompt digest only.
    """
    settings = settings_for(tmp_path)
    app = create_api(checkpoint, settings)
    async for client in running_client(app):
        for index in range(3):
            response = await client.post(
                "/v1/completions",
                json={"prompt": f"{SECRET_PROMPT} {index}", "max_tokens": 3},
                headers=headers(USER),
            )
            assert response.status_code == 200
        refused = await client.post(
            "/v1/completions",
            json={"prompt": "x", "max_tokens": 2, "deadline_ms": 1},
            headers=headers(USER),
        )
        assert refused.status_code == 503
    path = settings.data_dir / "telemetry" / "requests.jsonl"
    text = path.read_text(encoding="utf-8")
    exported = [json.loads(line) for line in text.splitlines()]
    assert len(exported) == 3, "one row per finished request; refused ones never ran"
    assert {row["key_id"] for row in exported} == {hash_api_key(USER)[:12]}
    assert all(row["completion_tokens"] == 3 for row in exported)
    assert all(len(str(row["prompt_sha256"])) == 64 for row in exported)
    assert "zebra" not in text


async def test_completion_spans_carry_lengths_and_outcomes_not_text(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("test", extra_exporter=exporter)
    app = create_api(checkpoint, settings_for(tmp_path), tracer_provider=provider)
    async for client in running_client(app):
        await client.post(
            "/v1/completions",
            json={"prompt": SECRET_PROMPT, "max_tokens": 3},
            headers=headers(USER),
        )
        await client.post(
            "/v1/completions",
            json={"prompt": SECRET_PROMPT, "max_tokens": 3, "stream": True},
            headers=headers(USER),
        )
        await client.post(
            "/v1/completions",
            json={"prompt": "x", "max_tokens": 2, "deadline_ms": 1},
            headers=headers(USER),
        )
    spans = [span for span in exporter.get_finished_spans() if span.name == "engine.completion"]
    assert len(spans) == 3
    outcomes = sorted(str((span.attributes or {})["engine.outcome"]) for span in spans)
    assert outcomes == ["completed", "completed", "rejected:deadline_infeasible"]
    for span in spans:
        attributes = dict(span.attributes or {})
        assert "engine.prompt_tokens" in attributes
        assert not any("zebra" in str(value) for value in attributes.values())
    completed = [s for s in spans if (s.attributes or {})["engine.outcome"] == "completed"]
    assert all((s.attributes or {})["engine.completion_tokens"] == 3 for s in completed)


async def test_stats_metrics_and_readiness_report_the_engine(
    api: tuple[httpx.AsyncClient, Any],
) -> None:
    client, _ = api
    await client.post(
        "/v1/completions", json={"prompt": "warm", "max_tokens": 2}, headers=headers(USER)
    )
    assert (await client.get("/v1/engine/stats")).status_code == 401
    stats = (await client.get("/v1/engine/stats", headers=headers(USER))).json()
    assert {"queue_depth", "kv_blocks_used_ratio", "prefix_cache_hit_ratio", "inflight"} <= set(
        stats
    )
    readiness = (await client.get("/readyz")).json()
    assert readiness["status"] == "ready"
    metrics = (await client.get("/metrics", headers=headers(USER))).text
    for name in (
        "engine_ttft_seconds",
        "engine_tpot_seconds",
        "engine_kv_blocks_used_ratio",
        "engine_prefix_cache_hit_ratio",
        "engine_preemptions_total",
        "engine_rejections_total",
        "engine_queue_depth",
        "engine_slo_attainment",
        "engine_prompt_length_psi",
        "engine_aborted_requests_total",
    ):
        assert name in metrics, name


async def test_a_request_over_the_time_limit_is_aborted_and_frees_its_blocks(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    settings = settings_for(tmp_path, request_timeout_seconds=0.15)
    registry = CollectorRegistry()
    service = slow_service(checkpoint, settings, registry, delay=0.05)
    app = create_api(checkpoint, settings, service=service, registry=registry)
    async for client in running_client(app):
        response = await client.post(
            "/v1/completions", json={"prompt": "slow", "max_tokens": 16}, headers=headers(USER)
        )
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "request_timeout"
        streamed = await client.post(
            "/v1/completions",
            json={"prompt": "slow", "max_tokens": 16, "stream": True},
            headers=headers(USER),
        )
        events = sse_events(streamed.text)
        assert ("error", json.dumps({"error": "request_timeout"})) in events
        assert "summary" not in [kind for kind, _ in events]
        engine = service.engine
        for _ in range(100):
            if engine.active_requests == 0:
                break
            await asyncio.sleep(0.01)
        assert engine.active_requests == 0
        assert engine.aborted_requests == 2
        assert not any(sequence.block_table for sequence in engine.live_sequences())
        engine.check_invariants()


async def test_a_deadline_longer_than_the_request_timeout_is_refused(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    """Review regression: deadline_ms up to 600,000 was accepted under a 60 s hard timeout, so a
    request could be promised a deadline the server would kill it before reaching."""
    app = create_api(checkpoint, settings_for(tmp_path, request_timeout_seconds=2.0))
    async for client in running_client(app):
        body = {"prompt": "hi", "max_tokens": 2, "deadline_ms": 2_001}
        response = await client.post("/v1/completions", json=body, headers=headers(USER))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "deadline_too_long"
        engine = app.state.engine_service.engine
        assert engine.active_requests == 0
        ok = await client.post(
            "/v1/completions", json={**body, "deadline_ms": 2_000}, headers=headers(USER)
        )
        assert ok.status_code == 200


async def test_a_request_past_its_deadline_is_aborted_at_the_deadline(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    """Review regression: only the 60 s request timeout was enforced, so a request that missed
    its deadline kept running and holding KV blocks for nobody.

    Admission is fooled on purpose (the latency model believes an iteration takes 3 ms, the
    executor takes 50 ms), so the request is admitted and then overruns its 200 ms deadline.
    """
    settings = settings_for(tmp_path, request_timeout_seconds=30.0)
    registry = CollectorRegistry()
    service = slow_service(checkpoint, settings, registry, delay=0.05)
    app = create_api(checkpoint, settings, service=service, registry=registry)
    engine = service.engine
    body = {"prompt": "slow", "max_tokens": 16, "deadline_ms": 200}
    async for client in running_client(app):
        started = time.perf_counter()
        response = await client.post("/v1/completions", json=body, headers=headers(USER))
        elapsed = time.perf_counter() - started
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "deadline_exceeded"
        assert elapsed < 0.6, elapsed  # 16 tokens at 50 ms each would take about 0.85 s
        # Forget what the first request taught the latency model, so admission is fooled again.
        engine.latency.update(engine.latency.intercept, engine.latency.slope)
        streamed = await client.post(
            "/v1/completions", json={**body, "stream": True}, headers=headers(USER)
        )
        assert streamed.status_code == 200, streamed.text
        events = sse_events(streamed.text)
        assert ("error", json.dumps({"error": "deadline_exceeded"})) in events
        assert "summary" not in [kind for kind, _ in events]
        for _ in range(100):
            if engine.active_requests == 0:
                break
            await asyncio.sleep(0.01)
        assert engine.active_requests == 0
        assert engine.aborted_requests == 2
        assert not any(sequence.block_table for sequence in engine.live_sequences())


def test_hanging_up_on_a_non_streaming_request_aborts_it(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    """Review regression: only the SSE path noticed a disconnect. A client that sent
    ``stream: false`` and hung up kept its sequences and blocks until max_tokens or the timeout."""
    settings = settings_for(tmp_path)
    registry = CollectorRegistry()
    service = slow_service(checkpoint, settings, registry, delay=0.05)
    app = create_api(checkpoint, settings, service=service, registry=registry)
    engine = service.engine

    async def hang_up(base_url: str) -> None:
        async with httpx.AsyncClient(base_url=base_url, timeout=0.3) as client:
            with pytest.raises(httpx.ReadTimeout):
                await client.post(
                    "/v1/completions",
                    json={"prompt": "a long answer please", "max_tokens": 16},
                    headers=headers(USER),
                )
        # No check of active_requests here: the server may already have noticed. An aborted
        # count of one below is the proof -- a request that had finished cannot be aborted.
        for _ in range(300):
            if engine.active_requests == 0:
                break
            await asyncio.sleep(0.01)

    with serve_in_thread(app) as base_url:
        asyncio.run(hang_up(base_url))
        assert engine.active_requests == 0
        assert engine.aborted_requests == 1, "the disconnect must abort, not finish, the request"
        assert not any(sequence.block_table for sequence in engine.live_sequences())
        engine.check_invariants()


def test_hanging_up_mid_stream_aborts_the_request_and_frees_its_blocks(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    settings = settings_for(tmp_path)
    registry = CollectorRegistry()
    service = slow_service(checkpoint, settings, registry, delay=0.03)
    app = create_api(checkpoint, settings, service=service, registry=registry)
    engine = service.engine

    async def hang_up(base_url: str) -> int:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            async with client.stream(
                "POST",
                "/v1/completions",
                json={"prompt": "a long answer please", "max_tokens": 16, "stream": True},
                headers=headers(USER),
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("event: token"):
                        break
            held = sum(len(sequence.block_table) for sequence in engine.live_sequences())
            for _ in range(300):
                if engine.active_requests == 0:
                    break
                await asyncio.sleep(0.01)
            return held

    with serve_in_thread(app) as base_url:
        held_at_hangup = asyncio.run(hang_up(base_url))
        assert engine.active_requests == 0
        assert engine.aborted_requests == 1, "the disconnect must abort, not finish, the request"
        assert held_at_hangup > 0, "the request should still have owned blocks when hung up on"
        assert not any(sequence.block_table for sequence in engine.live_sequences())
        engine.check_invariants()


async def test_admin_operations_need_the_admin_scope_and_are_audited(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    app = create_api(checkpoint, settings_for(tmp_path), audit_log=audit)
    async for client in running_client(app):
        await client.post(
            "/v1/completions",
            json={"prompt_token_ids": list(range(1, 40)), "max_tokens": 2},
            headers=headers(USER),
        )
        denied = await client.post("/v1/admin/prefix-cache/flush", headers=headers(USER))
        assert denied.status_code == 403
        # Refused attempts are evidence too (review: they used to leave no trace).
        assert [(entry.action, entry.actor, entry.detail) for entry in audit.entries()] == [
            ("admin.denied", hash_api_key(USER)[:12], {"operation": "prefix_cache.flush"})
        ]
        flushed = await client.post("/v1/admin/prefix-cache/flush", headers=headers(ADMIN))
        assert flushed.status_code == 200
        assert flushed.json()["released_blocks"] > 0
    entries = audit.entries()
    # The intent is written before the cache is touched, the outcome after.
    assert [entry.action for entry in entries] == [
        "admin.denied",
        "prefix_cache.flush.requested",
        "prefix_cache.flush",
    ]
    assert {entry.actor for entry in entries[1:]} == {hash_api_key(ADMIN)[:12]}
    assert audit.verify(expect_head=flushed.json()["audit_head"]) == entries[-1].digest


async def test_an_admin_action_is_refused_when_its_audit_entry_cannot_be_written(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    """Review regression: flush and swap acted first and audited afterwards.

    With one byte of the audit log edited, a flush still emptied the cache and then failed with
    500, leaving no record of it. The intent entry is now written (after verifying the chain)
    before anything changes, so a log that cannot be appended to blocks the action instead.
    """
    settings = settings_for(tmp_path)
    digest = _second_checkpoint(checkpoint, settings.weights_root / "v2", seed=5)
    audit = AuditLog(tmp_path / "audit.jsonl")
    app = create_api(checkpoint, settings, audit_log=audit)
    body = {"prompt_token_ids": list(range(1, 40)), "max_tokens": 2}
    async for client in running_client(app):
        await client.post("/v1/completions", json=body, headers=headers(USER))
        first = await client.post("/v1/admin/prefix-cache/flush", headers=headers(ADMIN))
        assert first.status_code == 200
        released = first.json()["released_blocks"]
        await client.post("/v1/completions", json=body, headers=headers(USER))
        engine = app.state.engine_service.engine
        cached = len(engine.prefix_cache)
        assert cached > 0
        text = audit.path.read_text(encoding="utf-8")
        tampered = text.replace(
            f'"released_blocks": {released}', f'"released_blocks": {released + 1}'
        )
        assert tampered != text
        audit.path.write_text(tampered, encoding="utf-8")

        refused = await client.post("/v1/admin/prefix-cache/flush", headers=headers(ADMIN))
        assert refused.status_code == 503
        assert refused.json()["error"]["code"] == "audit_unavailable"
        assert len(engine.prefix_cache) == cached, "the cache must not be flushed unaudited"
        swap = await client.post(
            "/v1/admin/weights",
            json={"name": "v2", "manifest_sha256": digest},
            headers=headers(ADMIN),
        )
        assert swap.status_code == 503
        assert swap.json()["error"]["code"] == "audit_unavailable"
        assert app.state.checkpoint.model.weights is checkpoint.model.weights
        # A denied attempt is still refused when it cannot be recorded.
        denied = await client.post("/v1/admin/prefix-cache/flush", headers=headers(USER))
        assert denied.status_code == 403
    assert audit.path.read_text(encoding="utf-8") == tampered, "nothing appended after tamper"


async def test_the_audit_log_is_keyed_when_a_key_is_configured(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    key = "k" * 40
    settings = settings_for(tmp_path, audit_hmac_key=key)
    app = create_api(checkpoint, settings)
    async for client in running_client(app):
        flushed = await client.post("/v1/admin/prefix-cache/flush", headers=headers(ADMIN))
        assert flushed.status_code == 200
    path = settings.data_dir / "admin-audit.jsonl"
    head = flushed.json()["audit_head"]
    assert AuditLog(path, key=key.encode()).verify(expect_head=head) == head
    with pytest.raises(AuditChainError):
        AuditLog(path).verify()


def _second_checkpoint(checkpoint: Checkpoint, directory: Path, seed: int) -> str:
    config = checkpoint.model.config
    manifest = write_checkpoint(
        directory, config, random_weights(config, seed=seed), checkpoint.tokenizer
    )
    return manifest.sha256()


async def test_weight_swap_verifies_the_pinned_manifest_and_is_audited(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    settings = settings_for(tmp_path)
    digest = _second_checkpoint(checkpoint, settings.weights_root / "v2", seed=99)
    audit = AuditLog(tmp_path / "audit.jsonl")
    # The app swaps weights in place, so it gets its own copy of the shared session checkpoint.
    own = Checkpoint(
        model=type(checkpoint.model)(checkpoint.model.config, checkpoint.model.weights),
        tokenizer=checkpoint.tokenizer,
        manifest=checkpoint.manifest,
        directory=checkpoint.directory,
    )
    app = create_api(own, settings, audit_log=audit)
    async for client in running_client(app):
        wrong = await client.post(
            "/v1/admin/weights",
            json={"name": "v2", "manifest_sha256": "0" * 64},
            headers=headers(ADMIN),
        )
        assert wrong.status_code == 400
        assert own.model.weights is checkpoint.model.weights
        missing = await client.post(
            "/v1/admin/weights",
            json={"name": "nope", "manifest_sha256": digest},
            headers=headers(ADMIN),
        )
        assert missing.status_code == 400
        traversal = await client.post(
            "/v1/admin/weights",
            json={"name": "../v2", "manifest_sha256": digest},
            headers=headers(ADMIN),
        )
        assert traversal.status_code == 422
        not_admin = await client.post(
            "/v1/admin/weights",
            json={"name": "v2", "manifest_sha256": digest},
            headers=headers(USER),
        )
        assert not_admin.status_code == 403
        swapped = await client.post(
            "/v1/admin/weights",
            json={"name": "v2", "manifest_sha256": digest},
            headers=headers(ADMIN),
        )
        assert swapped.status_code == 200
        assert own.model.weights is not checkpoint.model.weights
        after = await client.post(
            "/v1/completions", json={"prompt": "hi", "max_tokens": 2}, headers=headers(USER)
        )
        assert after.status_code == 200
    actions = [entry.action for entry in audit.entries()]
    assert actions == [
        "weights.swap.requested",
        "weights.swap_rejected",  # wrong manifest digest
        "weights.swap.requested",
        "weights.swap_rejected",  # no such checkpoint
        "admin.denied",  # not an admin key
        "weights.swap.requested",
        "weights.swap",
    ]
    audit.verify()


async def test_weight_swap_refuses_a_different_shape_and_a_busy_engine(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    from paged_kv_serving.engine.weights import build_random_checkpoint

    settings = settings_for(tmp_path)
    other = build_random_checkpoint(
        settings.weights_root / "wide",
        n_layers=1,
        d_model=16,
        n_heads=2,
        d_ff=32,
        max_seq_len=192,
        tokenizer_vocab_size=320,
    )
    same_shape = _second_checkpoint(checkpoint, settings.weights_root / "v3", seed=7)
    registry = CollectorRegistry()
    service = slow_service(checkpoint, settings, registry, delay=0.03)
    audit = AuditLog(tmp_path / "audit.jsonl")
    app = create_api(checkpoint, settings, service=service, registry=registry, audit_log=audit)
    async for client in running_client(app):
        mismatch = await client.post(
            "/v1/admin/weights",
            json={"name": "wide", "manifest_sha256": other.manifest.sha256()},
            headers=headers(ADMIN),
        )
        assert mismatch.status_code == 409
        assert mismatch.json()["error"]["code"] == "shape_mismatch"

        busy_request = asyncio.create_task(
            client.post(
                "/v1/completions",
                json={"prompt": "busy", "max_tokens": 16},
                headers=headers(USER),
            )
        )
        for _ in range(200):
            if service.engine.active_requests:
                break
            await asyncio.sleep(0.005)
        busy = await client.post(
            "/v1/admin/weights",
            json={"name": "v3", "manifest_sha256": same_shape},
            headers=headers(ADMIN),
        )
        assert busy.status_code == 409
        assert busy.json()["error"]["code"] == "engine_busy"
        assert (await busy_request).status_code == 200


async def test_a_draining_service_finishes_in_flight_work_and_refuses_new_work(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    settings = settings_for(tmp_path)
    registry = CollectorRegistry()
    service = slow_service(checkpoint, settings, registry, delay=0.01)
    app = create_api(checkpoint, settings, service=service, registry=registry)
    async for client in running_client(app):
        in_flight = asyncio.create_task(
            client.post(
                "/v1/completions",
                json={"prompt": "finish me", "max_tokens": 8},
                headers=headers(USER),
            )
        )
        for _ in range(1_000):
            if service.inflight:
                break
            await asyncio.sleep(0.005)
        assert service.inflight, "the first request never reached the engine"
        drained = asyncio.create_task(service.drain(timeout=10.0))
        await asyncio.sleep(0)
        refused = await client.post(
            "/v1/completions", json={"prompt": "too late", "max_tokens": 2}, headers=headers(USER)
        )
        assert refused.status_code == 503
        assert refused.json()["error"]["code"] == "draining"
        assert (await in_flight).status_code == 200, "in-flight work must finish during a drain"
        assert await drained is True
        readiness = (await client.get("/readyz")).json()
        assert readiness["status"] != "ready"
    with pytest.raises(ServiceClosedError):
        await service.submit(Request(request_id="late", prompt_token_ids=(1, 2)))


def test_admin_swap_names_are_plain_directory_names() -> None:
    AdminWeightSwap(name="model-v2.1", manifest_sha256="a" * 64)
    for bad in ("../etc", ".hidden", "a/b", "a\\b", ""):
        with pytest.raises(ValueError, match="validation error"):
            AdminWeightSwap(name=bad, manifest_sha256="a" * 64)


def test_completion_body_requires_exactly_one_prompt() -> None:
    CompletionBody(prompt="x")
    CompletionBody(prompt_token_ids=[1])
    with pytest.raises(ValueError, match="exactly one"):
        CompletionBody()
