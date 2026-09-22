from __future__ import annotations

import dataclasses

import pytest

from paged_kv_serving.engine.blocks import BlockManager, RecordingBlockStore
from paged_kv_serving.engine.config import EngineConfig, SloClass
from paged_kv_serving.engine.kvcache import KVCacheManager
from paged_kv_serving.engine.prefix_cache import PrefixCache
from paged_kv_serving.engine.scheduler import Scheduler
from paged_kv_serving.engine.sequence import (
    FinishReason,
    Request,
    SamplingParams,
    Sequence,
    SequenceStatus,
)

pytestmark = pytest.mark.unit

CONFIG = EngineConfig(
    block_size=4,
    num_blocks=16,
    max_batched_tokens=8,
    max_running_seqs=4,
    max_prompt_tokens=32,
    max_new_tokens=8,
    max_context_tokens=48,
)


#: A pool that holds exactly one maximum-length sequence, so two concurrent ones must fight.
TIGHT = EngineConfig(
    block_size=4,
    num_blocks=4,
    max_batched_tokens=16,
    max_running_seqs=4,
    max_prompt_tokens=8,
    max_new_tokens=8,
    max_context_tokens=16,
    watermark=0.0,
)


def build(config: EngineConfig = CONFIG) -> Scheduler:
    store = RecordingBlockStore()
    blocks = BlockManager(num_blocks=config.num_blocks, block_size=config.block_size, store=store)
    cache = KVCacheManager(blocks, PrefixCache(blocks, enabled=config.enable_prefix_cache))
    return Scheduler(config, cache)


def make(
    index: int,
    prompt_len: int,
    *,
    slo_class: SloClass = SloClass.INTERACTIVE,
    deadline: float | None = None,
    max_new_tokens: int = 4,
) -> Sequence:
    request = Request(
        request_id=f"r{index}",
        prompt_token_ids=tuple(range(1, prompt_len + 1)),
        sampling=SamplingParams(max_new_tokens=max_new_tokens, seed=index),
        slo_class=slo_class,
        deadline_seconds=deadline,
        arrival_time=0.0,
    )
    return Sequence.from_request(request, 0)


def advance(scheduler: Scheduler, output: object) -> None:
    """Apply an iteration the way the engine does: compute the chunk, then sample if it ended."""
    for item in output.scheduled:  # type: ignore[attr-defined]
        item.sequence.num_computed_tokens += item.num_tokens
        scheduler.cache.publish_full_blocks(item.sequence)
        if item.produces_token:
            item.sequence.append_token(1)


def test_a_long_prompt_is_split_across_iterations() -> None:
    scheduler = build()
    scheduler.add(make(0, 20))
    first = scheduler.schedule()
    assert first.num_batched_tokens == CONFIG.max_batched_tokens
    assert first.scheduled[0].is_prefill
    assert not first.scheduled[0].produces_token
    advance(scheduler, first)
    second = scheduler.schedule()
    assert second.num_batched_tokens == 8
    advance(scheduler, second)
    third = scheduler.schedule()
    assert third.num_batched_tokens == 4
    assert third.scheduled[0].produces_token, "the chunk reaching the prompt end yields logits"


def test_decode_tokens_are_taken_before_prefill() -> None:
    """A newly arrived long prompt must not stall sequences that are already generating."""
    scheduler = build()
    streaming = make(0, 4)
    scheduler.add(streaming)
    advance(scheduler, scheduler.schedule())  # prefill, first token sampled
    scheduler.add(make(1, 24))
    output = scheduler.schedule()
    decode = [item for item in output.scheduled if not item.is_prefill]
    prefill = [item for item in output.scheduled if item.is_prefill]
    assert [item.sequence.seq_id for item in decode] == [streaming.seq_id]
    assert prefill
    assert prefill[0].num_tokens == CONFIG.max_batched_tokens - 1
    assert output.num_batched_tokens == CONFIG.max_batched_tokens


def test_admission_order_is_slo_class_then_deadline_then_arrival() -> None:
    scheduler = build()
    batch_urgent = make(0, 4, slo_class=SloClass.BATCH, deadline=0.1)
    interactive_late = make(1, 4, deadline=9.0)
    interactive_early = make(2, 4, deadline=1.0)
    interactive_none = make(3, 4)
    for sequence in (batch_urgent, interactive_late, interactive_early, interactive_none):
        scheduler.add(sequence)
    admitted: list[str] = []
    for _ in range(4):
        output = scheduler.schedule()
        admitted.extend(output.admitted)
        advance(scheduler, output)
        for item in output.scheduled:
            item.sequence.finish(FinishReason.LENGTH, 0.0)
        scheduler.drop_finished()
    assert admitted == [
        interactive_early.seq_id,
        interactive_late.seq_id,
        interactive_none.seq_id,
        batch_urgent.seq_id,
    ]


def test_the_running_set_is_capped() -> None:
    scheduler = build(dataclasses.replace(CONFIG, max_running_seqs=2, max_batched_tokens=32))
    for index in range(4):
        scheduler.add(make(index, 2))
    output = scheduler.schedule()
    assert len(output.admitted) == 2
    assert scheduler.num_running == 2
    assert scheduler.queue_depth == 2
    assert scheduler.has_work


def test_memory_pressure_preempts_the_lowest_priority_sequence() -> None:
    scheduler = build(TIGHT)
    first = make(0, 8, deadline=1.0)
    second = make(1, 8, deadline=2.0)
    scheduler.add(first)
    scheduler.add(second)
    advance(scheduler, scheduler.schedule())
    # Two 8-token prompts fill all four blocks; the next decode token of either needs a fifth.
    assert scheduler.cache.num_free_blocks == 0
    for _ in range(4):
        output = scheduler.schedule()
        advance(scheduler, output)
        if output.preempted:
            assert output.preempted == (second.seq_id,), "the later deadline must pay"
            assert second.status is SequenceStatus.PREEMPTED
            assert second.block_table == []
            assert second.num_computed_tokens == 0
            assert second.preemption_count == 1
            # Recomputation keeps the tokens already produced.
            assert second.num_tokens > second.request.prompt_len
            scheduler.cache.check_invariants()
            return
    pytest.fail("no preemption happened under a full pool")


def test_a_preempted_sequence_is_re_admitted_and_recomputed() -> None:
    scheduler = build(TIGHT)
    scheduler.add(make(0, 8, deadline=1.0))
    scheduler.add(make(1, 8, deadline=2.0))
    seen_preemption = False
    for _ in range(40):
        output = scheduler.schedule()
        seen_preemption = seen_preemption or bool(output.preempted)
        advance(scheduler, output)
        for item in output.scheduled:
            if item.sequence.num_output_tokens >= item.sequence.request.sampling.max_new_tokens:
                item.sequence.finish(FinishReason.LENGTH, 0.0)
        scheduler.drop_finished()
        if not scheduler.has_work:
            break
    assert seen_preemption
    assert not scheduler.has_work
    assert scheduler.stats.preemptions >= 1
    scheduler.cache.check_invariants()


def test_remove_releases_blocks_from_either_queue() -> None:
    scheduler = build()
    running = make(0, 4)
    waiting = make(1, 4)
    scheduler.add(running)
    advance(scheduler, scheduler.schedule())
    scheduler.add(waiting)
    assert running.block_table
    scheduler.remove(running)
    scheduler.remove(waiting)
    assert scheduler.num_running == 0
    assert scheduler.queue_depth == 0
    assert running.block_table == []
    scheduler.cache.check_invariants()


def test_an_empty_scheduler_produces_an_empty_batch() -> None:
    scheduler = build()
    output = scheduler.schedule()
    assert output.is_empty
    assert output.num_batched_tokens == 0
    assert scheduler.stats.iterations == 0
    assert not scheduler.has_work


def test_statistics_split_prefill_from_decode() -> None:
    scheduler = build()
    scheduler.add(make(0, 6))
    advance(scheduler, scheduler.schedule())
    advance(scheduler, scheduler.schedule())
    assert scheduler.stats.prefill_tokens == 6
    assert scheduler.stats.decode_tokens == 1
    assert scheduler.stats.iterations == 2
    assert scheduler.stats.mean_batch_size == 3.5


def test_add_running_inserts_a_fork_without_re_admission() -> None:
    scheduler = build()
    parent = make(0, 4)
    scheduler.add(parent)
    advance(scheduler, scheduler.schedule())
    child = Sequence.from_request(parent.request, 1)
    child.token_ids = list(parent.token_ids)
    scheduler.cache.fork(parent, child)
    scheduler.add_running(child)
    assert child.status is SequenceStatus.RUNNING
    assert scheduler.num_running == 2
    output = scheduler.schedule()
    assert {item.sequence.seq_id for item in output.scheduled} == {parent.seq_id, child.seq_id}


def test_a_prefilling_batch_sequence_is_preempted_before_an_interactive_decoder() -> None:
    """Under memory pressure the weakest sequence pays, even if it is part-way through prefill."""
    config = EngineConfig(
        block_size=4,
        num_blocks=5,
        max_batched_tokens=8,
        max_running_seqs=4,
        max_prompt_tokens=16,
        max_new_tokens=4,
        max_context_tokens=20,
        watermark=0.0,
        lower_class_prefill_share=1.0,
    )
    scheduler = build(config)
    interactive = make(0, 4, slo_class=SloClass.INTERACTIVE, max_new_tokens=4)
    batch = make(1, 16, slo_class=SloClass.BATCH, max_new_tokens=4)
    scheduler.add(interactive)
    scheduler.add(batch)
    for _ in range(8):
        output = scheduler.schedule()
        if output.preempted:
            assert output.preempted == (batch.seq_id,), "the interactive decoder must not pay"
            assert batch.status is SequenceStatus.PREEMPTED
            assert batch.num_computed_tokens == 0
            assert interactive.status is SequenceStatus.RUNNING
            assert any(item.sequence is interactive for item in output.scheduled)
            scheduler.cache.check_invariants()
            return
        advance(scheduler, output)
    pytest.fail("the pool never ran out; the scenario no longer exercises preemption")


def test_lower_class_prefill_is_capped_while_a_higher_class_has_work() -> None:
    config = dataclasses.replace(
        CONFIG, max_batched_tokens=16, num_blocks=32, lower_class_prefill_share=0.25
    )
    scheduler = build(config)
    interactive = make(0, 4, slo_class=SloClass.INTERACTIVE)
    batch = make(1, 28, slo_class=SloClass.BATCH)
    scheduler.add(interactive)
    scheduler.add(batch)
    first = scheduler.schedule()
    chunks = {item.sequence.seq_id: item.num_tokens for item in first.scheduled}
    assert chunks[interactive.seq_id] == 4
    assert chunks[batch.seq_id] == config.lower_class_prefill_tokens == 4
    advance(scheduler, first)
    # Once no interactive work is left, the batch prompt gets the whole budget again.
    scheduler.remove(interactive)
    second = scheduler.schedule()
    (only,) = second.scheduled
    assert only.sequence is batch
    assert only.num_tokens == config.max_batched_tokens


def test_without_higher_class_work_a_lower_class_is_not_capped() -> None:
    config = dataclasses.replace(CONFIG, max_batched_tokens=16, lower_class_prefill_share=0.25)
    scheduler = build(config)
    batch = make(0, 20, slo_class=SloClass.BATCH)
    scheduler.add(batch)
    output = scheduler.schedule()
    assert output.scheduled[0].num_tokens == 16


def distinct(
    index: int,
    prompt_len: int,
    *,
    slo_class: SloClass = SloClass.INTERACTIVE,
    max_new_tokens: int = 4,
    key_id: str = "tenant",
    offset: int = 0,
) -> Sequence:
    """A sequence whose prompt tokens start at ``offset``, so prompts can differ or coincide."""
    request = Request(
        request_id=f"d{index}",
        prompt_token_ids=tuple(offset + position for position in range(prompt_len)),
        sampling=SamplingParams(max_new_tokens=max_new_tokens, seed=index),
        slo_class=slo_class,
        key_id=key_id,
    )
    return Sequence.from_request(request, 0)


def step_until_idle(scheduler: Scheduler, max_iterations: int = 400) -> int:
    """Drive the scheduler like the engine does, failing on any empty batch while work remains.

    An empty batch with work queued is a livelock: nothing that could free memory runs, so the
    next iteration is the same as this one.
    """
    for iteration in range(max_iterations):
        if not scheduler.has_work:
            return iteration
        output = scheduler.schedule()
        assert not output.is_empty, (
            f"empty batch at iteration {iteration} with "
            f"{[(s.seq_id, s.num_computed_tokens, s.num_tokens) for s in scheduler.running]} "
            f"running and {len(scheduler.waiting)} waiting; "
            f"{scheduler.cache.num_free_blocks} blocks free"
        )
        advance(scheduler, output)
        for item in output.scheduled:
            sequence = item.sequence
            if sequence.num_output_tokens >= sequence.request.sampling.max_new_tokens:
                sequence.finish(FinishReason.LENGTH, 0.0)
        scheduler.drop_finished()
        scheduler.cache.check_invariants()
    pytest.fail("the scheduler did not drain")


def test_two_prefills_that_fill_the_pool_between_them_cannot_livelock() -> None:
    """A regression from review: two long prompts, both mid-prefill, holding every block.

    Preemption used to happen only while scheduling decodes. With no decoder running, the
    interactive prompt could not get a block, the batch prompt had none to give up, and every
    later iteration was empty.
    """
    config = EngineConfig(
        block_size=8,
        num_blocks=16,
        max_batched_tokens=32,
        max_running_seqs=6,
        max_prompt_tokens=100,
        max_new_tokens=8,
        max_context_tokens=128,
        enable_prefix_cache=False,
    )
    scheduler = build(config)
    batch = distinct(0, 100, slo_class=SloClass.BATCH, max_new_tokens=8, offset=1)
    scheduler.add(batch)
    advance(scheduler, scheduler.schedule())  # the batch prompt starts alone
    interactive = distinct(1, 100, max_new_tokens=8, offset=200)
    scheduler.add(interactive)
    step_until_idle(scheduler)
    assert scheduler.stats.preemptions >= 1
    assert interactive.status is SequenceStatus.FINISHED
    assert batch.status is SequenceStatus.FINISHED


def test_a_waiting_sequence_does_not_pin_cached_blocks_while_it_waits() -> None:
    """A waiter that looked up its prefix but could not be admitted must not keep the blocks.

    Otherwise the cached blocks it matched are neither evictable nor in use, a running prefill
    that needs them cannot progress, and the waiter itself waits for that prefill forever.
    """
    config = EngineConfig(
        block_size=4,
        num_blocks=10,
        max_batched_tokens=16,
        max_running_seqs=4,
        max_prompt_tokens=32,
        max_new_tokens=4,
        max_context_tokens=40,
        watermark=0.0,
        lower_class_prefill_share=0.5,
    )
    scheduler = build(config)
    warm = distinct(0, 16, max_new_tokens=1, key_id="k", offset=1)
    scheduler.add(warm)
    step_until_idle(scheduler)  # its four full blocks are now cached and unreferenced
    assert len(scheduler.cache.prefix_cache) == 4
    long_batch = distinct(1, 32, slo_class=SloClass.BATCH, max_new_tokens=2, offset=100)
    scheduler.add(long_batch)
    advance(scheduler, scheduler.schedule())
    # The batch prompt holds 4 blocks, 2 are free and 4 are cached. Once the interactive waiter
    # arrives the batch prompt is held to an 8-token allowance -- exactly the two free blocks --
    # and the waiter matches all four cached blocks but then finds no memory for its last token.
    waiter = distinct(2, 17, max_new_tokens=2, key_id="k", offset=1)
    scheduler.add(waiter)
    step_until_idle(scheduler)
    assert waiter.status is SequenceStatus.FINISHED
    assert long_batch.status is SequenceStatus.FINISHED


def test_the_watermark_never_blocks_admission_into_an_idle_engine() -> None:
    """The watermark reserves room for running decoders; with none running it protects nothing."""
    config = EngineConfig(
        block_size=4,
        num_blocks=10,
        max_batched_tokens=16,
        max_running_seqs=4,
        max_prompt_tokens=32,
        max_new_tokens=4,
        max_context_tokens=40,
        watermark=0.5,
    )
    scheduler = build(config)
    warm = distinct(0, 25, max_new_tokens=1, key_id="k", offset=1)
    scheduler.add(warm)
    step_until_idle(scheduler)
    assert len(scheduler.cache.prefix_cache) == 6
    # Same key, same first 24 tokens: six cached blocks match, four are free, the watermark is five.
    again = distinct(1, 25, max_new_tokens=2, key_id="k", offset=1)
    scheduler.add(again)
    step_until_idle(scheduler)
    assert again.status is SequenceStatus.FINISHED
