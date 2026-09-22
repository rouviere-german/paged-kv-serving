from __future__ import annotations

import pytest

from paged_kv_serving.engine.blocks import (
    BlockAllocationError,
    BlockManager,
    InvariantError,
    RecordingBlockStore,
)
from paged_kv_serving.engine.kvcache import KVCacheManager
from paged_kv_serving.engine.prefix_cache import PrefixCache
from paged_kv_serving.engine.sequence import Request, SamplingParams, Sequence

pytestmark = pytest.mark.unit

BLOCK_SIZE = 4


def build(
    num_blocks: int = 16, *, prefix_cache: bool = True
) -> tuple[KVCacheManager, RecordingBlockStore]:
    store = RecordingBlockStore()
    blocks = BlockManager(num_blocks=num_blocks, block_size=BLOCK_SIZE, store=store)
    return KVCacheManager(blocks, PrefixCache(blocks, enabled=prefix_cache)), store


def sequence(tokens: list[int], *, key: str = "tenant-a", index: int = 0) -> Sequence:
    request = Request(
        request_id=f"r{index}",
        prompt_token_ids=tuple(tokens),
        sampling=SamplingParams(max_new_tokens=4, seed=index),
        key_id=key,
    )
    return Sequence.from_request(request, 0)


def compute(cache: KVCacheManager, seq: Sequence, chunk: int) -> None:
    cache.reserve(seq, seq.num_computed_tokens + chunk)
    seq.num_computed_tokens += chunk
    cache.publish_full_blocks(seq)


def test_reserve_allocates_only_the_blocks_a_chunk_needs() -> None:
    cache, _ = build()
    seq = sequence(list(range(10)))
    compute(cache, seq, 4)
    assert len(seq.block_table) == 1
    compute(cache, seq, 5)
    assert len(seq.block_table) == 3  # 9 tokens need three 4-token blocks
    assert cache.blocks_required(seq, 10) == 0
    cache.check_invariants()


def test_reserve_beyond_the_sequence_is_refused() -> None:
    cache, _ = build()
    seq = sequence([1, 2, 3])
    with pytest.raises(ValueError, match="beyond the tokens"):
        cache.reserve(seq, 10)


def test_a_full_pool_raises_without_changing_the_block_table() -> None:
    cache, _ = build(num_blocks=2)
    seq = sequence(list(range(12)))
    with pytest.raises(BlockAllocationError):
        cache.reserve(seq, 12)
    assert seq.block_table == []
    cache.check_invariants()


def test_attach_prefix_reuses_published_blocks_and_leaves_a_token_to_compute() -> None:
    cache, _ = build()
    first = sequence([1, 2, 3, 4, 5, 6, 7, 8])
    compute(cache, first, 8)
    assert len(cache.prefix_cache) == 2

    second = sequence([1, 2, 3, 4, 5, 6, 7, 8], index=1)
    reuse = cache.attach_prefix(second)
    # Only the first block is reused: the last token must still be run to produce logits.
    assert reuse.blocks == 1
    assert reuse.tokens == 4
    assert second.num_computed_tokens == 4
    assert second.block_table == first.block_table[:1]
    assert cache.blocks.ref_count(first.block_table[0]) == 3  # first, cache, second


def test_prefix_reuse_is_scoped_to_the_calling_key() -> None:
    cache, _ = build()
    first = sequence([1, 2, 3, 4, 5, 6], key="tenant-a")
    compute(cache, first, 6)
    other = sequence([1, 2, 3, 4, 5, 6], key="tenant-b", index=1)
    assert cache.attach_prefix(other).blocks == 0, "prefixes leaked across keys"
    same = sequence([1, 2, 3, 4, 5, 6], key="tenant-a", index=2)
    assert cache.attach_prefix(same).blocks == 1


def test_attach_prefix_is_a_no_op_without_a_cache_or_a_full_block() -> None:
    disabled, _ = build(prefix_cache=False)
    seq = sequence(list(range(9)))
    assert disabled.attach_prefix(seq).blocks == 0
    cache, _ = build()
    short = sequence([1, 2])
    assert cache.attach_prefix(short).blocks == 0
    compute(cache, short, 2)
    assert cache.publish_full_blocks(short) == 0
    with pytest.raises(InvariantError, match="before any block"):
        cache.attach_prefix(short)


def test_forking_shares_blocks_and_the_first_write_takes_a_private_copy() -> None:
    cache, store = build()
    parent = sequence([1, 2, 3, 4, 5, 6])
    compute(cache, parent, 6)
    shared_partial = parent.block_table[1]

    child = sequence([1, 2, 3, 4, 5, 6], index=1)
    cache.fork(parent, child)
    assert child.block_table == parent.block_table
    assert cache.blocks.ref_count(shared_partial) == 2
    assert store.copies == []

    child.append_token(99)
    compute(cache, child, 1)
    assert child.block_table[1] != shared_partial, "the child wrote into a shared block"
    assert store.copies == [(shared_partial, child.block_table[1])]
    assert parent.block_table[1] == shared_partial
    # Full prompt blocks are shared, not copied: forking costs one block, not a prompt.
    assert child.block_table[0] == parent.block_table[0]
    cache.check_invariants()


def test_forking_into_a_sequence_that_already_owns_blocks_is_refused() -> None:
    cache, _ = build()
    parent = sequence([1, 2, 3, 4])
    compute(cache, parent, 4)
    child = sequence([1, 2, 3, 4], index=1)
    compute(cache, child, 4)
    with pytest.raises(InvariantError, match="already owns blocks"):
        cache.fork(parent, child)


def test_free_returns_blocks_but_the_cache_keeps_its_own_reference() -> None:
    cache, _ = build()
    seq = sequence([1, 2, 3, 4, 5, 6, 7, 8])
    compute(cache, seq, 8)
    free_before = cache.num_free_blocks
    assert cache.free(seq) == 2
    assert seq.block_table == []
    assert cache.num_free_blocks == free_before, "cached blocks are retained for reuse"
    assert cache.evictable_blocks() == 2
    assert cache.prefix_cache.flush() == 2
    assert cache.num_free_blocks == free_before + 2


def test_reserve_evicts_cached_blocks_under_pressure() -> None:
    cache, _ = build(num_blocks=4)
    first = sequence([1, 2, 3, 4, 5, 6, 7, 8])
    compute(cache, first, 8)
    cache.free(first)
    assert cache.num_free_blocks == 2
    hungry = sequence(list(range(16, 32)), index=1)
    cache.reserve(hungry, 16)
    assert len(hungry.block_table) == 4
    assert len(cache.prefix_cache) == 0, "the cache should have given its blocks back"
    cache.check_invariants()


def test_can_reserve_counts_evictable_blocks() -> None:
    cache, _ = build(num_blocks=4)
    seq = sequence([1, 2, 3, 4, 5, 6, 7, 8])
    compute(cache, seq, 8)
    cache.free(seq)
    assert cache.num_free_blocks == 2
    assert cache.can_reserve(4)
    assert not cache.can_reserve(5)
    assert 0.0 < cache.utilization <= 1.0


def test_check_invariants_detects_a_cached_block_that_lost_its_reference() -> None:
    cache, _ = build()
    seq = sequence([1, 2, 3, 4])
    compute(cache, seq, 4)
    cached = cache.prefix_cache.cached_block_ids()[0]
    cache.free(seq)
    cache.blocks.free(cached)  # simulate the cache's reference going missing
    with pytest.raises(InvariantError, match="has no reference"):
        cache.check_invariants()


def test_a_reservation_needing_a_copy_and_a_new_block_is_all_or_nothing() -> None:
    """Copy-on-write and growth are counted together, so a short pool changes nothing."""
    cache, store = build(num_blocks=3)
    parent = sequence([1, 2, 3, 4, 5, 6])
    compute(cache, parent, 6)  # block 0 full (and cached), block 1 holds two tokens
    child = sequence([1, 2, 3, 4, 5, 6], index=1)
    cache.fork(parent, child)
    child.token_ids.extend([7, 8, 9])  # writing 7..9 touches shared block 1 and needs block 2
    table_before = list(child.block_table)
    refs_before = cache.blocks.snapshot()
    with pytest.raises(BlockAllocationError):
        cache.reserve(child, 9)
    assert child.block_table == table_before
    assert cache.blocks.snapshot() == refs_before
    assert store.copies == []
    cache.check_invariants()


def test_a_shared_partial_block_is_copied_before_the_first_write() -> None:
    cache, store = build(num_blocks=8)
    parent = sequence([1, 2, 3, 4, 5, 6])
    compute(cache, parent, 6)
    child = sequence([1, 2, 3, 4, 5, 6], index=1)
    cache.fork(parent, child)
    child.token_ids.extend([7, 8, 9])
    cache.reserve(child, 9)
    assert child.block_table[0] == parent.block_table[0], "the full block stays shared"
    assert child.block_table[1] != parent.block_table[1], "the written block is private"
    assert store.copies == [(parent.block_table[1], child.block_table[1])]
    assert len(child.block_table) == 3
    cache.check_invariants()
