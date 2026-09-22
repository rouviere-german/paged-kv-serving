"""The asyncio service around the engine: loop behaviour that the API tests cannot pin down."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from paged_kv_serving.engine.engine import Engine, StepOutput
from paged_kv_serving.engine.runner import ExecutionBatch
from paged_kv_serving.engine.service import EngineService

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
