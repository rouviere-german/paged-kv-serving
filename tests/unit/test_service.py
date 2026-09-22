"""The asyncio service around the engine: loop behaviour that the API tests cannot pin down."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from paged_kv_serving.engine.config import EngineConfig
from paged_kv_serving.engine.engine import Engine, StepOutput
from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.runner import ExecutionBatch, ModelRunner
from paged_kv_serving.engine.sequence import Request, SamplingParams
from paged_kv_serving.engine.service import EngineService
from paged_kv_serving.engine.telemetry import TelemetryExporter

pytestmark = pytest.mark.unit


class _NullRunner:
    def execute(self, batch: ExecutionBatch) -> dict[str, object]:
        return {}

    def reset_cache(self) -> None:
        return None


class StuckEngine:
    """An engine that always has work and never manages to schedule any of it."""

    def __init__(self) -> None:
        self.runner = _NullRunner()
        self.steps = 0

    @property
    def has_work(self) -> bool:
        return True

    def step(self) -> StepOutput:
        self.steps += 1
        return StepOutput(
            iteration=0,
            duration=0.0,
            num_batched_tokens=0,
            num_prefill_tokens=0,
            num_decode_tokens=0,
            num_running=1,
            queue_depth=1,
        )


async def test_an_empty_iteration_with_work_queued_backs_off_instead_of_spinning() -> None:
    """Review regression: the loop used to retry an empty step at once, holding the engine lock.

    A scheduler that cannot make progress is a bug, but the loop around it must not turn the
    bug into a busy spin that starves every other coroutine of the lock.
    """
    stuck = StuckEngine()
    service = EngineService(cast(Engine, stuck), idle_sleep=0.01)
    await service.start()
    await asyncio.sleep(0.25)
    await service.stop()
    # Backing off by idle_sleep allows about 25 attempts in 0.25 s; a spin makes thousands.
    assert 1 <= stuck.steps <= 60, stuck.steps


class GatedExecutor:
    """The real runner, held inside ``execute`` until the test opens the gate."""

    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.inside = False

    def execute(self, batch: ExecutionBatch) -> dict[str, np.ndarray]:
        self.inside = True
        self.entered.set()
        try:
            self.gate.wait(timeout=10.0)
            return self.runner.execute(batch)
        finally:
            self.inside = False


async def test_the_readiness_allocator_check_never_runs_inside_an_iteration(
    tiny_model: LanguageModel, engine_config: EngineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review regression: /readyz read the allocator on the event loop while ``step()`` was
    mutating it in a worker thread, and failed spuriously under load (197 of 20,000 probes).

    The check now waits for the engine lock. While an iteration is stuck it reports the last
    verdict after its timeout instead of reading a half-updated allocator.
    """
    engine = Engine(tiny_model, engine_config)
    gated = GatedExecutor(engine.runner)
    engine.executor = gated
    inside_when_checked: list[bool] = []
    check = engine.check_invariants

    def recording_check() -> None:
        inside_when_checked.append(gated.inside)
        check()

    monkeypatch.setattr(engine, "check_invariants", recording_check)
    service = EngineService(engine, idle_sleep=0.001)
    await service.start()
    try:
        await service.submit(
            Request(
                request_id="r",
                prompt_token_ids=(1, 2, 3),
                sampling=SamplingParams(max_new_tokens=2),
            )
        )
        assert await asyncio.to_thread(gated.entered.wait, 10.0)
        # Busy: after the timeout the previous verdict is reported, without touching the allocator.
        assert await service.allocator_healthy(timeout=0.02) is True
        pending = asyncio.create_task(service.allocator_healthy())
        await asyncio.sleep(0.05)
        assert not pending.done(), "the check must wait for the iteration to finish"
        gated.gate.set()
        healthy = await pending
    finally:
        gated.gate.set()
        await service.stop()
    assert healthy is True
    assert inside_when_checked == [False], inside_when_checked


async def test_telemetry_is_written_in_batches_and_flushed_on_stop(
    tiny_model: LanguageModel, engine_config: EngineConfig, tmp_path: Path
) -> None:
    engine = Engine(tiny_model, engine_config)
    telemetry = TelemetryExporter(
        tmp_path / "requests.jsonl", engine_clock=engine.clock, batch_size=2
    )
    service = EngineService(engine, idle_sleep=0.001, telemetry=telemetry)
    await service.start()
    identifiers = [f"{index:032x}" for index in range(3)]
    try:
        for identifier in identifiers:
            request = Request(
                request_id=identifier,
                prompt_token_ids=(1, 2, 3),
                sampling=SamplingParams(max_new_tokens=2),
                key_id="tenant-a",
                arrival_time=engine.clock(),
            )
            assert (await service.submit(request)).admitted
            assert await service.collect(identifier, timeout=10.0) is not None
        for _ in range(500):
            if telemetry.written_rows >= 2:
                break
            await asyncio.sleep(0.01)
        # A full batch goes out as soon as it is complete; the remainder waits for stop().
        assert (telemetry.written_rows, telemetry.pending) == (2, 1)
        assert telemetry.rejected_batches == 0
    finally:
        await service.stop()
    assert telemetry.written_rows == 3
    lines = (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["request_id"] for line in lines) == identifiers


async def test_a_telemetry_write_failure_does_not_stop_serving(
    tiny_model: LanguageModel, engine_config: EngineConfig, tmp_path: Path
) -> None:
    blocked = tmp_path / "not-a-file"
    blocked.mkdir()  # appending to a directory fails with an OSError
    engine = Engine(tiny_model, engine_config)
    telemetry = TelemetryExporter(blocked, engine_clock=engine.clock, batch_size=1)
    service = EngineService(engine, idle_sleep=0.001, telemetry=telemetry)
    await service.start()
    try:
        for index in range(2):
            identifier = f"{index:032x}"
            request = Request(
                request_id=identifier,
                prompt_token_ids=(4, 5),
                sampling=SamplingParams(max_new_tokens=2),
                key_id="tenant-a",
                arrival_time=engine.clock(),
            )
            assert (await service.submit(request)).admitted
            assert await service.collect(identifier, timeout=10.0) is not None
        assert service.running
    finally:
        await service.stop()
    assert telemetry.written_rows == 0
