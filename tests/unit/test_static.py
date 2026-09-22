"""The batching baselines: static (wait for a full batch) and dynamic (leave after a max wait)."""

from __future__ import annotations

import pytest

from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.sequence import Request, SamplingParams
from paged_kv_serving.engine.static import StaticBatchingEngine

pytestmark = pytest.mark.unit


class FakeClock:
    """Time moves only when the engine sleeps, so batch dispatch times are exact."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


def request(index: int, arrival: float, max_new_tokens: int = 3) -> Request:
    return Request(
        request_id=f"r{index}",
        prompt_token_ids=tuple(range(1, 6 + index)),
        sampling=SamplingParams(max_new_tokens=max_new_tokens),
        arrival_time=arrival,
    )


def engine(model: LanguageModel, clock: FakeClock, max_wait: float | None) -> StaticBatchingEngine:
    return StaticBatchingEngine(
        model,
        batch_size=4,
        max_context_tokens=64,
        block_size=8,
        clock=clock,
        sleep=clock.sleep,
        max_wait_seconds=max_wait,
    )


def test_static_batching_waits_for_a_full_batch(tiny_model: LanguageModel) -> None:
    clock = FakeClock()
    outputs = engine(tiny_model, clock, None).generate(
        [request(0, 0.0), request(1, 0.01), request(2, 0.5)]
    )
    # Three requests are one (short) batch, and it starts only when its last member arrives.
    assert [round(output.first_token_time, 6) for output in outputs] == [0.5, 0.5, 0.5]
    assert round(outputs[0].ttft, 6) == 0.5


def test_dynamic_batching_dispatches_a_partial_batch_after_the_max_wait(
    tiny_model: LanguageModel,
) -> None:
    """Review gap: the only baseline had no batch timeout, so at low load its TTFT measured the
    wait for a batch to fill rather than anything about scheduling."""
    clock = FakeClock()
    outputs = engine(tiny_model, clock, 0.05).generate(
        [request(0, 0.0), request(1, 0.01), request(2, 0.5)]
    )
    first_tokens = {output.request_id: round(output.first_token_time, 6) for output in outputs}
    # r0 and r1 leave once r0 has waited 50 ms; r2 waits its own 50 ms.
    assert first_tokens == {"r0": 0.05, "r1": 0.05, "r2": 0.55}
    assert [len(output.token_ids) for output in outputs] == [3, 3, 3]


def test_dynamic_batching_still_dispatches_a_full_batch_at_once(tiny_model: LanguageModel) -> None:
    clock = FakeClock()
    arrivals = [request(index, 0.001 * index) for index in range(6)]
    outputs = engine(tiny_model, clock, 1.0).generate(arrivals)
    first_tokens = [round(output.first_token_time, 6) for output in outputs]
    assert first_tokens[:4] == [0.003] * 4, "a full batch does not wait for the timeout"
    # The last two arrived while the first batch ran (which takes no fake time), so they wait out
    # the max wait measured from the older of them.
    assert first_tokens[4:] == [1.004, 1.004]


def test_a_request_that_has_waited_past_the_max_wait_leaves_with_the_next_batch(
    tiny_model: LanguageModel,
) -> None:
    clock = FakeClock()
    batcher = engine(tiny_model, clock, 0.05)
    clock.now = 2.0  # the batcher was busy until long after these arrived
    outputs = batcher.generate([request(0, 0.0), request(1, 1.0)])
    assert [round(output.first_token_time, 6) for output in outputs] == [2.0, 2.0]


def test_the_max_wait_must_be_non_negative(tiny_model: LanguageModel) -> None:
    with pytest.raises(ValueError, match="max_wait_seconds"):
        engine(tiny_model, FakeClock(), -0.1)
