"""Engine-level equivalence: every logits row the engine samples from matches a dense recompute.

The runner-level properties in ``test_equivalence.py`` check one batch at a time. This file checks
the whole machine: hypothesis draws an engine configuration (block size, a pool small enough to
force preemption, a token budget small enough to force chunked prefill, a cap on running
sequences, prefix caching on or off) and a workload (shared and unshared prompts, two tenants,
greedy and sampled requests, parallel samples, arrivals spread over iterations). The engine runs
it with a recording executor, and every row it sampled from is compared with
``reference_forward`` over that sequence's final tokens.

Along the way the allocator invariants are checked after every iteration, no iteration may come
back empty while work is queued (that would be a livelock), every request must finish with
exactly the tokens it asked for, and once the prefix cache is flushed the pool must be whole
again -- no leaks, whatever happened.

The generator deliberately reaches the corners that hurt: prompts several times longer than the
token budget, pools barely larger than one maximum-length sequence (so two long prefills can hold
it between them), and up to eight parallel samples of one prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from paged_kv_serving.engine.config import EngineConfig, SloClass
from paged_kv_serving.engine.engine import Engine, RequestOutput
from paged_kv_serving.engine.equivalence import RecordingExecutor, engine_logit_gap, sequence_tokens
from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.sequence import FinishReason, Request, SamplingParams

pytestmark = pytest.mark.unit

TOLERANCE = 1e-4
MAX_PROMPT = 56
MAX_NEW = 8
MAX_CONTEXT = MAX_PROMPT + MAX_NEW


@dataclass(frozen=True)
class Scenario:
    config: EngineConfig
    requests: tuple[tuple[int, Request], ...]  # (submit at iteration, request)


@st.composite
def scenarios(draw: st.DrawFn) -> Scenario:
    block_size = draw(st.sampled_from([4, 8]))
    minimum = -(-MAX_CONTEXT // block_size)
    config = EngineConfig(
        block_size=block_size,
        # Half the draws sit within two blocks of one maximum-length sequence.
        num_blocks=draw(
            st.one_of(
                st.integers(min_value=minimum, max_value=minimum + 2),
                st.integers(min_value=minimum, max_value=minimum * 3),
            )
        ),
        max_batched_tokens=draw(st.integers(min_value=4, max_value=48)),
        max_running_seqs=draw(st.integers(min_value=1, max_value=8)),
        max_prompt_tokens=MAX_PROMPT,
        max_new_tokens=MAX_NEW,
        max_context_tokens=MAX_CONTEXT,
        enable_prefix_cache=draw(st.booleans()),
        watermark=draw(st.sampled_from([0.0, 0.05])),
        lower_class_prefill_share=draw(st.sampled_from([0.1, 0.25, 1.0])),
    )
    token = st.integers(min_value=0, max_value=63)
    shared = draw(st.lists(token, min_size=2 * block_size, max_size=2 * block_size))
    requests: list[tuple[int, Request]] = []
    for index in range(draw(st.integers(min_value=1, max_value=6))):
        # Short questions, or prompts close to the limit: several times the token budget, and two
        # of them are enough to fill a pool sized near one maximum-length sequence.
        length = draw(
            st.one_of(
                st.integers(min_value=1, max_value=16),
                st.integers(min_value=MAX_PROMPT - 16, max_value=MAX_PROMPT),
            )
        )
        tail = draw(st.lists(token, min_size=length, max_size=length))
        prompt = ([*shared, *tail] if draw(st.booleans()) else tail)[:MAX_PROMPT]
        request = Request(
            request_id=f"r{index}",
            prompt_token_ids=tuple(prompt),
            sampling=SamplingParams(
                max_new_tokens=draw(st.integers(min_value=1, max_value=MAX_NEW)),
                temperature=draw(st.sampled_from([0.0, 0.0, 1.0])),
                seed=index,
            ),
            # Every sample needs a running slot, so n never exceeds the running-set cap.
            n=min(draw(st.sampled_from([1, 1, 2, 8])), config.max_running_seqs),
            key_id=draw(st.sampled_from(["tenant-a", "tenant-b"])),
            slo_class=draw(st.sampled_from([SloClass.INTERACTIVE, SloClass.BATCH])),
        )
        requests.append((draw(st.integers(min_value=0, max_value=6)), request))
    return Scenario(config, tuple(requests))


def run(
    model: LanguageModel, scenario: Scenario
) -> tuple[Engine, RecordingExecutor, list[RequestOutput]]:
    engine = Engine(model, scenario.config)
    recorder = RecordingExecutor(engine.runner)
    engine.executor = recorder
    pending = sorted(scenario.requests, key=lambda item: item[0])
    outputs: list[RequestOutput] = []
    for iteration in range(5_000):
        while pending and pending[0][0] <= iteration:
            decision = engine.submit(pending.pop(0)[1])
            assert decision.admitted, decision
        if not engine.has_work and not pending:
            return engine, recorder, outputs
        step = engine.step()
        engine.check_invariants()
        assert not (step.is_empty and engine.has_work), "empty iteration with work queued"
        outputs.extend(step.finished)
    raise AssertionError("the engine did not drain")


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(scenario=scenarios())
def test_every_sampled_logits_row_matches_a_cache_free_recompute(
    tiny_model: LanguageModel, scenario: Scenario
) -> None:
    engine, recorder, outputs = run(tiny_model, scenario)
    requests = [request for _, request in scenario.requests]
    assert {output.request_id for output in outputs} == {r.request_id for r in requests}

    by_id = {request.request_id: request for request in requests}
    for output in outputs:
        request = by_id[output.request_id]
        assert len(output.samples) == request.n
        for sample in output.samples:
            # No EOS is configured, so every sample runs to exactly its token budget.
            assert sample.finish_reason is FinishReason.LENGTH
            assert len(sample.token_ids) == request.sampling.max_new_tokens

    tokens = sequence_tokens(
        requests, {o.request_id: [s.token_ids for s in o.samples] for o in outputs}
    )
    gap, rows = engine_logit_gap(tiny_model, recorder.records, tokens)
    assert rows >= len(requests)
    assert gap < TOLERANCE, f"max |dlogit| {gap} over {rows} rows"

    # Nothing leaks: with every request finished and the cache flushed, the pool is whole.
    engine.flush_prefix_cache()
    engine.check_invariants()
    assert engine.blocks.num_free_blocks == scenario.config.num_blocks


def test_a_cramped_pool_really_does_preempt_and_still_matches(tiny_model: LanguageModel) -> None:
    """A fixed scenario that is known to preempt, so the property above is never vacuous."""
    config = EngineConfig(
        block_size=4,
        num_blocks=12,
        max_batched_tokens=16,
        max_running_seqs=6,
        max_prompt_tokens=40,
        max_new_tokens=MAX_NEW,
        max_context_tokens=48,
        watermark=0.0,
    )
    requests = tuple(
        (
            0,
            Request(
                request_id=f"r{index}",
                prompt_token_ids=tuple((index * 7 + offset) % 64 for offset in range(20)),
                sampling=SamplingParams(max_new_tokens=MAX_NEW, seed=index),
                n=2 if index == 0 else 1,
            ),
        )
        for index in range(5)
    )
    engine, recorder, outputs = run(tiny_model, Scenario(config, requests))
    assert engine.scheduler.stats.preemptions > 0
    tokens = sequence_tokens(
        [request for _, request in requests],
        {o.request_id: [s.token_ids for s in o.samples] for o in outputs},
    )
    gap, _ = engine_logit_gap(tiny_model, recorder.records, tokens)
    assert gap < TOLERANCE
