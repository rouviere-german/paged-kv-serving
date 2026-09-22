"""End-to-end engine behaviour: the paged, continuously batched path against the naive one."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from paged_kv_serving.engine.admission import RejectionReason
from paged_kv_serving.engine.config import EngineConfig, SloClass
from paged_kv_serving.engine.engine import Engine, RequestOutput
from paged_kv_serving.engine.keys import EntitlementGuard, KeyPolicy, KeyPolicyStore
from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.sampling import sample_token
from paged_kv_serving.engine.sequence import (
    FinishReason,
    Request,
    SamplingParams,
    Sequence,
    finish_reason_for,
)

pytestmark = pytest.mark.integration


def request_for(
    index: int,
    prompt: list[int],
    *,
    max_new_tokens: int = 4,
    n: int = 1,
    temperature: float = 0.0,
    slo_class: SloClass = SloClass.INTERACTIVE,
    deadline: float | None = None,
    key_id: str = "tenant-a",
    arrival_time: float = 0.0,
) -> Request:
    return Request(
        request_id=f"req-{index}",
        prompt_token_ids=tuple(prompt),
        sampling=SamplingParams(max_new_tokens=max_new_tokens, temperature=temperature, seed=index),
        n=n,
        slo_class=slo_class,
        deadline_seconds=deadline,
        key_id=key_id,
        arrival_time=arrival_time,
    )


def greedy_reference(model: LanguageModel, prompt: list[int], steps: int) -> list[int]:
    """What a cache-free implementation would have produced, one full recompute per token."""
    tokens = list(prompt)
    produced: list[int] = []
    for _ in range(steps):
        token = int(np.argmax(model.reference_forward(tokens)[-1]))
        produced.append(token)
        tokens.append(token)
    return produced


def prompts(count: int, seed: int, low: int = 5, high: int = 40) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [
        [int(value) for value in rng.integers(1, 64, int(rng.integers(low, high)))]
        for _ in range(count)
    ]


def test_mixed_batches_match_a_naive_per_sequence_recompute(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    """The headline claim: batching, chunking and paging do not change the tokens produced."""
    engine = Engine(tiny_model, engine_config)
    batch = prompts(6, seed=1)
    requests = [
        request_for(index, prompt, max_new_tokens=3 + index % 4)
        for index, prompt in enumerate(batch)
    ]
    outputs = {output.request_id: output for output in engine.generate(requests)}
    assert len(outputs) == len(requests)
    for index, prompt in enumerate(batch):
        produced = list(outputs[f"req-{index}"].samples[0].token_ids)
        assert produced == greedy_reference(tiny_model, prompt, len(produced))
    engine.check_invariants()


def test_a_preempted_request_produces_the_same_tokens(tiny_model: LanguageModel) -> None:
    """Recomputation after preemption must cost work and nothing else."""
    roomy = EngineConfig(
        block_size=8,
        num_blocks=64,
        max_batched_tokens=32,
        max_running_seqs=6,
        max_prompt_tokens=64,
        max_new_tokens=16,
        max_context_tokens=128,
    )
    cramped = dataclasses.replace(roomy, num_blocks=17, watermark=0.0, max_running_seqs=4)
    batch = prompts(4, seed=2, low=24, high=40)
    requests = [request_for(i, p, max_new_tokens=10) for i, p in enumerate(batch)]

    relaxed = Engine(tiny_model, roomy)
    baseline = {o.request_id: o.samples[0].token_ids for o in relaxed.generate(requests)}

    pressured = Engine(tiny_model, cramped)
    under_pressure = {o.request_id: o.samples[0].token_ids for o in pressured.generate(requests)}
    assert pressured.scheduler.stats.preemptions > 0, "the tight pool should have forced a preempt"
    assert under_pressure == baseline
    pressured.check_invariants()


@pytest.mark.parametrize("prefix_cache", [False, True])
def test_two_long_prompts_that_fill_the_pool_drain_and_match_the_reference(
    tiny_model: LanguageModel, prefix_cache: bool
) -> None:
    """Review regression: a batch prompt mid-prefill plus an interactive one used to livelock.

    Between them the two prefills held every block, neither was decoding, and preemption only
    ever ran while scheduling decodes -- so every iteration came back empty until the requests
    timed out. Now the interactive prefill preempts the batch one and both finish, with the
    tokens a cache-free recompute produces.
    """
    config = EngineConfig(
        block_size=8,
        num_blocks=16,
        max_batched_tokens=32,
        max_running_seqs=6,
        max_prompt_tokens=100,
        max_new_tokens=8,
        max_context_tokens=128,
        enable_prefix_cache=prefix_cache,
    )
    engine = Engine(tiny_model, config)
    batch_prompt = [index % 64 for index in range(100)]
    interactive_prompt = [(index * 7) % 64 for index in range(100)]
    assert engine.submit(
        request_for(0, batch_prompt, max_new_tokens=8, slo_class=SloClass.BATCH, key_id="t1")
    ).admitted
    engine.step()
    assert engine.submit(request_for(1, interactive_prompt, max_new_tokens=8, key_id="t2")).admitted
    outputs: list[RequestOutput] = []
    for _ in range(500):
        if not engine.has_work:
            break
        step = engine.step()
        assert not step.is_empty, "an empty iteration while work is queued is a livelock"
        outputs.extend(step.finished)
    assert not engine.has_work
    produced = {output.request_id: list(output.samples[0].token_ids) for output in outputs}
    assert produced["req-0"] == greedy_reference(tiny_model, batch_prompt, 8)
    assert produced["req-1"] == greedy_reference(tiny_model, interactive_prompt, 8)
    assert engine.scheduler.stats.preemptions >= 1
    engine.check_invariants()


def test_parallel_sampling_shares_the_prompt_and_diverges_on_the_first_token(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    engine = Engine(tiny_model, engine_config)
    prompt = prompts(1, seed=3, low=20, high=21)[0]
    output = engine.generate([request_for(0, prompt, max_new_tokens=6, n=4, temperature=1.0)])[0]
    assert len(output.samples) == 4
    assert len({sample.token_ids for sample in output.samples}) > 1, "samples must differ"
    assert all(len(sample.token_ids) == 6 for sample in output.samples)
    engine.check_invariants()
    # Sharing is what makes this cheap: four samples cost far less than four separate prompts.
    solo = Engine(tiny_model, engine_config)
    solo.generate([request_for(1, prompt, max_new_tokens=6, n=1, temperature=1.0)])
    assert engine.scheduler.stats.prefill_tokens == solo.scheduler.stats.prefill_tokens


def test_parallel_samples_are_reproducible_from_the_seed(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    prompt = prompts(1, seed=4, low=12, high=13)[0]
    first = Engine(tiny_model, engine_config).generate(
        [request_for(0, prompt, max_new_tokens=5, n=3, temperature=1.0)]
    )[0]
    second = Engine(tiny_model, engine_config).generate(
        [request_for(0, prompt, max_new_tokens=5, n=3, temperature=1.0)]
    )[0]
    assert [s.token_ids for s in first.samples] == [s.token_ids for s in second.samples]


def test_a_shared_prompt_prefix_is_reused_and_does_not_change_the_answer(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    shared = [7] * (engine_config.block_size * 2)
    tails = prompts(4, seed=5, low=4, high=12)
    requests = [request_for(i, [*shared, *tail], max_new_tokens=4) for i, tail in enumerate(tails)]

    cached = Engine(tiny_model, engine_config)
    with_cache = {o.request_id: o.samples[0].token_ids for o in cached.generate(requests)}
    uncached = Engine(tiny_model, dataclasses.replace(engine_config, enable_prefix_cache=False))
    without = {o.request_id: o.samples[0].token_ids for o in uncached.generate(requests)}

    assert with_cache == without, "cache hits must not change the output"
    assert cached.prefix_cache.stats.hit_blocks > 0
    assert cached.scheduler.stats.prefill_tokens < uncached.scheduler.stats.prefill_tokens
    cached.check_invariants()


def test_prefix_reuse_does_not_cross_api_keys(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    shared = [3] * (engine_config.block_size * 2)
    engine = Engine(tiny_model, engine_config)
    engine.generate([request_for(0, [*shared, 1, 2], key_id="tenant-a")])
    before = engine.prefix_cache.stats.hit_blocks
    engine.generate([request_for(1, [*shared, 1, 2], key_id="tenant-b")])
    assert engine.prefix_cache.stats.hit_blocks == before, "a prefix leaked across tenants"
    engine.generate([request_for(2, [*shared, 1, 2], key_id="tenant-a")])
    assert engine.prefix_cache.stats.hit_blocks > before


def test_aborting_a_request_frees_its_blocks_immediately(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    engine = Engine(tiny_model, engine_config)
    engine.submit(request_for(0, prompts(1, seed=6, low=30, high=31)[0], max_new_tokens=16))
    engine.step()
    held = sum(len(sequence.block_table) for sequence in engine.live_sequences())
    assert held > 0
    assert engine.abort("req-0") == 1
    assert sum(len(s.block_table) for s in engine.live_sequences()) == 0
    assert engine.active_requests == 0
    assert engine.abort("req-0") == 0  # aborting twice is harmless
    engine.check_invariants()
    engine.flush_prefix_cache()
    assert engine.blocks.num_free_blocks == engine_config.num_blocks


def test_finish_reasons_cover_length_stop_and_context(tiny_model: LanguageModel) -> None:
    config = EngineConfig(
        block_size=8,
        num_blocks=32,
        max_batched_tokens=32,
        max_running_seqs=4,
        max_prompt_tokens=32,
        max_new_tokens=8,
        max_context_tokens=40,
    )
    engine = Engine(tiny_model, config)
    output = engine.generate([request_for(0, [1, 2, 3], max_new_tokens=4)])[0]
    assert output.samples[0].finish_reason is FinishReason.LENGTH

    prompt = [1, 2, 3]
    forced = int(np.argmax(tiny_model.reference_forward(prompt)[-1]))
    stopping = Engine(tiny_model, config, eos_token_id=forced)
    stopped = stopping.generate([request_for(1, prompt, max_new_tokens=8)])[0]
    assert stopped.samples[0].finish_reason is FinishReason.STOP
    assert len(stopped.samples[0].token_ids) == 1


def test_the_finish_rule_covers_stop_length_and_the_context_cap() -> None:
    """Admission normally makes the context cap unreachable; it is the guard for when it is not."""
    request = request_for(0, [1, 2, 3], max_new_tokens=8)
    seq = Sequence.from_request(request, 0)
    seq.append_token(5)
    assert (
        finish_reason_for(seq, 5, eos_token_id=None, max_context_tokens=4)
        is FinishReason.CONTEXT_LIMIT
    )
    assert finish_reason_for(seq, 5, eos_token_id=5, max_context_tokens=99) is FinishReason.STOP
    assert finish_reason_for(seq, 5, eos_token_id=None, max_context_tokens=99) is None
    for _ in range(7):
        seq.append_token(6)
    assert (
        finish_reason_for(seq, 6, eos_token_id=None, max_context_tokens=99) is FinishReason.LENGTH
    )


def test_requests_over_the_configured_limits_are_refused_before_allocation(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    engine = Engine(tiny_model, engine_config)
    long_prompt = engine.submit(request_for(0, list(range(1, engine_config.max_prompt_tokens + 5))))
    assert long_prompt.reason is RejectionReason.PROMPT_TOO_LONG
    many_tokens = engine.submit(
        request_for(1, [1, 2, 3], max_new_tokens=engine_config.max_new_tokens + 1)
    )
    assert many_tokens.reason is RejectionReason.CONTEXT_TOO_LONG
    assert engine.blocks.num_free_blocks == engine_config.num_blocks
    assert engine.active_requests == 0
    engine.submit(request_for(2, [1, 2, 3]))
    with pytest.raises(ValueError, match="already in flight"):
        engine.submit(request_for(2, [1, 2, 3]))


def test_interactive_requests_are_served_before_batch_ones(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    engine = Engine(tiny_model, dataclasses.replace(engine_config, max_running_seqs=1))
    batch = [
        request_for(index, [1, 2, 3, 4], slo_class=SloClass.BATCH, max_new_tokens=3)
        for index in range(3)
    ]
    interactive = request_for(9, [5, 6, 7, 8], slo_class=SloClass.INTERACTIVE, max_new_tokens=3)
    for request in batch:
        engine.submit(request)
    engine.submit(interactive)
    finished = [output.request_id for output in engine.run_until_idle()]
    assert finished[0] == "req-9"


def test_snapshot_and_statistics_reflect_the_run(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    engine = Engine(tiny_model, engine_config)
    engine.generate([request_for(i, p, max_new_tokens=3) for i, p in enumerate(prompts(3, 7))])
    snapshot = engine.snapshot()
    assert snapshot["iteration"] > 0
    assert snapshot["queue_depth"] == 0
    assert 0.0 <= snapshot["kv_blocks_used_ratio"] <= 1.0
    assert snapshot["token_rate"] > 0
    assert engine.slo_attainment == 1.0
    assert engine.scheduler.stats.decode_tokens > 0


def test_deadlines_are_recorded_on_the_output(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    engine = Engine(tiny_model, engine_config)
    output = engine.generate(
        [request_for(0, [1, 2, 3], deadline=60.0, arrival_time=engine.clock())]
    )[0]
    assert output.met_deadline is True
    assert output.ttft >= 0.0
    assert output.latency >= output.ttft
    assert engine.slo_attainment == 1.0
    no_deadline = Engine(tiny_model, engine_config).generate([request_for(1, [1, 2, 3])])[0]
    assert no_deadline.met_deadline is None


def test_profiling_updates_the_admission_rate(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    engine = Engine(tiny_model, engine_config)
    before = engine.admission.token_rate
    engine.profile_latency(repeats=2, warmup=1)
    assert engine.admission.token_rate != before
    assert engine.admission.token_rate > 0
    assert engine.latency.observations == 0


def test_the_context_limit_is_checked_against_the_model(tiny_model: LanguageModel) -> None:
    too_big = EngineConfig(
        block_size=8,
        num_blocks=128,
        max_batched_tokens=32,
        max_running_seqs=2,
        max_prompt_tokens=256,
        max_new_tokens=256,
        max_context_tokens=1024,
    )
    with pytest.raises(ValueError, match="exceeds the model"):
        Engine(tiny_model, too_big)


def test_parallel_samples_never_push_the_running_set_past_its_cap(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    """Review regression: forks were added to the running set whatever its size.

    A request is admitted only when every one of its samples has a running slot, so the decode
    budget admission control reserves for running sequences is never exceeded either.
    """
    config = dataclasses.replace(engine_config, max_running_seqs=4)
    engine = Engine(tiny_model, config)
    batch = prompts(4, seed=9, low=10, high=20)
    requests = [
        request_for(0, batch[0], max_new_tokens=6, n=3, temperature=1.0),
        request_for(1, batch[1], max_new_tokens=6, n=3, temperature=1.0),
        request_for(2, batch[2], max_new_tokens=6, n=1),
        request_for(3, batch[3], max_new_tokens=6, n=4, temperature=1.0),
    ]
    for request in requests:
        assert engine.submit(request).admitted
    peak = 0
    outputs: list[RequestOutput] = []
    while engine.has_work:
        step = engine.step()
        peak = max(peak, engine.scheduler.num_running)
        outputs.extend(step.finished)
    assert peak <= config.max_running_seqs
    assert sorted(len(output.samples) for output in outputs) == [1, 3, 3, 4]
    engine.check_invariants()


def test_more_samples_than_running_slots_is_refused_up_front(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    engine = Engine(tiny_model, dataclasses.replace(engine_config, max_running_seqs=2))
    decision = engine.submit(request_for(0, [1, 2, 3], n=8, temperature=1.0))
    assert decision.reason is RejectionReason.TOO_MANY_SAMPLES
    assert engine.active_requests == 0
    assert engine.blocks.num_free_blocks == engine_config.num_blocks


def sampled_reference(
    model: LanguageModel, request: Request, sample_index: int, steps: int
) -> list[int]:
    """Sample ``sample_index`` of ``request`` as a cache-free loop with its own generator.

    The generator is the one ``Sequence.from_request`` promises: seeded from the request's seed and
    the sample index alone, so it cannot depend on the other samples or on the batch.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=request.sampling.seed, spawn_key=(sample_index,))
    )
    tokens = list(request.prompt_token_ids)
    produced: list[int] = []
    for _ in range(steps):
        token = sample_token(model.reference_forward(tokens)[-1], request.sampling, rng)
        produced.append(token)
        tokens.append(token)
    return produced


@pytest.mark.parametrize("num_blocks", [64, 16])
def test_each_parallel_sample_depends_only_on_the_seed_and_its_index(
    tiny_model: LanguageModel, engine_config: EngineConfig, num_blocks: int
) -> None:
    """Review gap: a fork that reused the parent's generator passed every test.

    Sample ``i`` of an ``n``-way request must equal a standalone run seeded with
    ``SeedSequence(seed, spawn_key=(i,))`` -- with a roomy pool, and with one small enough that
    the samples are preempted and recomputed.
    """
    config = dataclasses.replace(
        engine_config, num_blocks=num_blocks, max_running_seqs=12, enable_prefix_cache=False
    )
    engine = Engine(tiny_model, config)
    batch = prompts(3, seed=21, low=18, high=30)
    requests = [
        request_for(index, prompt, max_new_tokens=12, n=4, temperature=1.0)
        for index, prompt in enumerate(batch)
    ]
    outputs = {output.request_id: output for output in engine.generate(requests)}
    if num_blocks == 16:
        assert engine.scheduler.stats.preemptions > 0, "the small pool must force preemption"
    for request in requests:
        samples = outputs[request.request_id].samples
        assert [sample.sample_index for sample in samples] == [0, 1, 2, 3]
        for sample in samples:
            expected = sampled_reference(tiny_model, request, sample.sample_index, steps=12)
            assert list(sample.token_ids) == expected, (request.request_id, sample.sample_index)
        assert len({sample.token_ids for sample in samples}) > 1, "samples should differ"
    engine.check_invariants()


def test_a_request_the_server_refuses_does_not_spend_the_keys_budget(
    tiny_model: LanguageModel, engine_config: EngineConfig
) -> None:
    """Review regression: 503 rejections charged the token bucket and were never refunded.

    A client retrying after "deadline infeasible" drained its tokens/min and then got 429 without
    ever having been served.
    """
    store = KeyPolicyStore(
        [
            KeyPolicy(
                key_id="tenant-a",
                slo_classes=frozenset({SloClass.INTERACTIVE}),
                tokens_per_minute=100,
                max_concurrent_requests=2,
            )
        ]
    )
    engine = Engine(
        tiny_model, engine_config, entitlements=EntitlementGuard(store, clock=lambda: 0.0)
    )
    prompt = list(range(1, 41))
    for attempt in range(4):
        decision = engine.submit(request_for(attempt, prompt, max_new_tokens=10, deadline=1e-6))
        assert decision.reason is RejectionReason.DEADLINE_INFEASIBLE, attempt
    assert engine.entitlements is not None
    assert engine.entitlements.inflight("tenant-a") == 0
    assert engine.submit(request_for(9, prompt, max_new_tokens=10)).admitted
    engine.run_until_idle()
    engine.check_invariants()
