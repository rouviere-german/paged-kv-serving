"""The HTTP open-loop generator: planning, launching on schedule, and counting every outcome."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from paged_kv_serving.engine.http_loadgen import (
    HttpOutcome,
    HttpRequestPlan,
    HttpTrafficClass,
    plan_http_workload,
    run_http_open_loop,
    summarize_http,
)

pytestmark = pytest.mark.unit

INTERACTIVE = HttpTrafficClass("interactive", "key-a", "interactive", deadline_ms=2_000)
BATCH = HttpTrafficClass("batch", "key-b", "batch", prompt_words=(10, 20), max_tokens=(8, 16))


def test_plans_are_seeded_bounded_and_split_by_share() -> None:
    first = plan_http_workload([INTERACTIVE, BATCH], rate=50.0, count=200, seed=3)
    again = plan_http_workload([INTERACTIVE, BATCH], rate=50.0, count=200, seed=3)
    assert first == again
    assert [plan.offset for plan in first] == sorted(plan.offset for plan in first)
    batch = [plan for plan in first if plan.traffic == "batch"]
    assert 60 < len(batch) < 140
    for plan in batch:
        assert 8 <= plan.body["max_tokens"] <= 16
        assert plan.body["slo_class"] == "batch"
        assert plan.body["stream"] is True
        assert "deadline_ms" not in plan.body
    interactive = [plan for plan in first if plan.traffic == "interactive"]
    assert all(plan.body["deadline_ms"] == 2_000 for plan in interactive)


def test_a_burst_window_compresses_arrivals() -> None:
    plans = plan_http_workload(
        [INTERACTIVE], rate=10.0, count=60, seed=1, burst=(200.0, 1.0, 0.5), system_prompt="sys "
    )
    inside = [plan for plan in plans if 1.0 <= plan.offset < 1.5]
    assert len(inside) > 20
    assert all(plan.body["prompt"].startswith("sys ") for plan in plans)


def test_traffic_classes_are_validated() -> None:
    with pytest.raises(ValueError, match="share"):
        HttpTrafficClass("x", "k", share=0.0)
    with pytest.raises(ValueError, match="slo_class"):
        HttpTrafficClass("x", "k", slo_class="gold")
    with pytest.raises(ValueError, match="prompt_words"):
        HttpTrafficClass("x", "k", prompt_words=(5, 2))
    with pytest.raises(ValueError, match="max_tokens"):
        HttpTrafficClass("x", "k", max_tokens=(0, 2))
    with pytest.raises(ValueError, match="at least one"):
        plan_http_workload([], rate=1.0, count=1)


def _sse(*events: tuple[str, str]) -> bytes:
    return "".join(f"event: {name}\r\ndata: {data}\r\n\r\n" for name, data in events).encode()


async def test_requests_are_launched_on_schedule_and_every_outcome_is_kept() -> None:
    """Five requests 20 ms apart against a server that takes 200 ms each: open loop means the
    run takes about 0.3 s, not 1 s, and refusals and stream errors are rows, not gaps."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        await asyncio.sleep(0.2)
        if body["max_tokens"] == 1:
            payload = {"error": {"code": "deadline_infeasible", "message": "", "request_id": ""}}
            return httpx.Response(503, json=payload)
        if body["max_tokens"] == 2:
            return httpx.Response(
                200, content=_sse(("token", "{}"), ("error", '{"error": "request_timeout"}'))
            )
        if body["max_tokens"] == 3:
            return httpx.Response(500, content=b"not json")
        return httpx.Response(
            200,
            content=_sse(("token", "{}"), ("token", "{}"), ("summary", "{}"), ("done", "[DONE]")),
        )

    plans = [
        HttpRequestPlan(offset=index * 0.02, traffic="t", api_key="k", body={"max_tokens": tokens})
        for index, tokens in enumerate((8, 8, 1, 2, 3))
    ]
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        started = time.perf_counter()
        outcomes = await run_http_open_loop(client, plans)
        elapsed = time.perf_counter() - started
    # Serially this would take 5 x 0.2 s; open loop it takes about 0.28 s.
    assert elapsed < 0.9, "requests must not wait for each other"
    assert [outcome.ok for outcome in outcomes] == [True, True, False, False, False]
    assert outcomes[0].tokens == 2
    assert outcomes[0].ttft is not None
    assert outcomes[0].latency is not None
    # The server's 200 ms is inside the latency. asyncio times its sleeps with time.monotonic,
    # whose resolution is one 15.6 ms tick on Windows (CPython 3.12), and fires a timer up to one
    # tick early -- measured against perf_counter that is as low as 0.186 s here.
    assert outcomes[0].latency >= 0.2 - time.get_clock_info("monotonic").resolution
    assert outcomes[2].status == 503
    assert outcomes[2].error == "deadline_infeasible"
    assert outcomes[3].error == "request_timeout"
    assert outcomes[4].error == "http_500"
    summary = summarize_http(outcomes, wall_seconds=elapsed)
    row = summary["classes"]["t"]
    assert row["sent"] == 5
    assert row["completed"] == 2
    assert row["errors"] == {"deadline_infeasible": 1, "http_500": 1, "request_timeout": 1}


async def test_transport_failures_are_recorded_not_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    plans = [HttpRequestPlan(offset=0.0, traffic="t", api_key="k", body={})]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://engine"
    ) as client:
        (outcome,) = await run_http_open_loop(client, plans)
    assert outcome.error == "ConnectError"
    assert not outcome.ok
    assert outcome.ttft is None


def test_an_outcome_without_a_summary_is_not_ok() -> None:
    assert not HttpOutcome(traffic="t", status=200, scheduled=0.0, first_token=0.1).ok
