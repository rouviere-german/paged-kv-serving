from __future__ import annotations

import pytest

from paged_kv_serving.engine.blocks import BlockManager, RecordingBlockStore
from paged_kv_serving.engine.prefix_cache import (
    PrefixCache,
    block_hash_chain,
    chain_seed,
    next_block_hash,
)

pytestmark = pytest.mark.unit


def cache(num_blocks: int = 8, scope: str = "per_key") -> tuple[PrefixCache, BlockManager]:
    blocks = BlockManager(num_blocks=num_blocks, block_size=4, store=RecordingBlockStore())
    return PrefixCache(blocks, scope=scope), blocks  # type: ignore[arg-type]


def test_the_chain_commits_to_every_block_in_front_of_a_block() -> None:
    """Two prompts sharing only block 1's tokens must not share block 1's key."""
    first = block_hash_chain([1, 1, 1, 1, 9, 9, 9, 9], block_size=4, num_blocks=2)
    second = block_hash_chain([2, 2, 2, 2, 9, 9, 9, 9], block_size=4, num_blocks=2)
    assert first[0] != second[0]
    assert first[1] != second[1], "block 1 keys collided despite different histories"
    same = block_hash_chain([1, 1, 1, 1, 9, 9, 9, 9], block_size=4, num_blocks=2)
    assert first == same


def test_scopes_never_produce_a_common_prefix() -> None:
    tokens = [3, 3, 3, 3]
    a = block_hash_chain(tokens, block_size=4, num_blocks=1, scope=b"tenant-a")
    b = block_hash_chain(tokens, block_size=4, num_blocks=1, scope=b"tenant-b")
    assert a != b


def test_hash_chain_argument_validation() -> None:
    with pytest.raises(ValueError, match="block_size must be"):
        block_hash_chain([1], block_size=0, num_blocks=0)
    with pytest.raises(ValueError, match="num_blocks must be"):
        block_hash_chain([1], block_size=1, num_blocks=-1)
    with pytest.raises(ValueError, match="not enough tokens"):
        block_hash_chain([1, 2], block_size=4, num_blocks=1)
    with pytest.raises(ValueError, match="only full blocks"):
        next_block_hash(chain_seed(), [1, 2], 4)


def test_lookup_stops_at_the_first_miss() -> None:
    prefix, blocks = cache()
    hashes = block_hash_chain([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], block_size=4, num_blocks=3)
    first, second = blocks.allocate(), blocks.allocate()
    assert prefix.register(hashes[0], first)
    assert prefix.register(hashes[2], second)  # a hole at index 1
    matched = prefix.lookup(hashes)
    assert matched == [first], "a hit behind a miss must not be used"
    assert blocks.ref_count(first) == 3  # sequence + cache + this lookup
    blocks.free(first)


def test_registering_the_same_hash_twice_keeps_the_first_block() -> None:
    prefix, blocks = cache()
    hashes = block_hash_chain([7, 7, 7, 7], block_size=4, num_blocks=1)
    first, second = blocks.allocate(), blocks.allocate()
    assert prefix.register(hashes[0], first)
    assert not prefix.register(hashes[0], second)
    assert prefix.cached_block_ids() == (first,)


def test_eviction_is_least_recently_used_and_skips_blocks_in_use() -> None:
    prefix, blocks = cache()
    ids = []
    hashes = []
    for value in range(3):
        block = blocks.allocate()
        block_hash = block_hash_chain([value] * 4, block_size=4, num_blocks=1)[0]
        prefix.register(block_hash, block)
        blocks.free(block)  # the producing sequence has gone; only the cache holds it
        ids.append(block)
        hashes.append(block_hash)
    # A live sequence picks up the oldest entry, which both refreshes it and pins it.
    prefix.lookup([hashes[0]])
    assert prefix.evict(1) == 1
    assert ids[0] in prefix.cached_block_ids(), "a block a sequence is reading was evicted"
    assert ids[1] not in prefix.cached_block_ids(), "the LRU evictable entry should have gone"
    assert set(prefix.cached_block_ids()) == {ids[2], ids[0]}
    # Even asking for more than exists only releases what nobody is reading.
    assert prefix.evict(5) == 1
    assert prefix.cached_block_ids() == (ids[0],)
    blocks.free(ids[0])
    assert prefix.evict(1) == 1
    assert prefix.cached_block_ids() == ()


def test_evict_until_free_releases_only_what_is_needed() -> None:
    prefix, blocks = cache(num_blocks=4)
    held = blocks.allocate_many(2)
    for index, block in enumerate(held):
        prefix.register(block_hash_chain([index] * 4, block_size=4, num_blocks=1)[0], block)
        blocks.free(block)
    assert blocks.num_free_blocks == 2
    assert prefix.evict_until_free(3) == 1
    assert blocks.num_free_blocks == 3
    assert prefix.evict_until_free(1) == 0
    assert prefix.evict(0) == 0


def test_a_disabled_cache_never_hits_or_registers() -> None:
    prefix, blocks = cache()
    prefix.enabled = False
    block = blocks.allocate()
    hashes = block_hash_chain([1] * 4, block_size=4, num_blocks=1)
    assert not prefix.register(hashes[0], block)
    assert prefix.lookup(hashes) == []
    assert prefix.stats.queried_blocks == 1
    assert prefix.stats.hit_ratio == 0.0


def test_statistics_and_flush() -> None:
    prefix, blocks = cache()
    hashes = block_hash_chain([1, 1, 1, 1, 2, 2, 2, 2], block_size=4, num_blocks=2)
    for block_hash in hashes:
        block = blocks.allocate()
        prefix.register(block_hash, block)
        blocks.free(block)
    assert prefix.lookup(hashes) == list(prefix.cached_block_ids())
    assert prefix.stats.hit_ratio == 1.0
    # A prompt that shares nothing drags the ratio down: the counter is over queried blocks.
    assert prefix.lookup(block_hash_chain([9] * 8, block_size=4, num_blocks=2)) == []
    assert prefix.stats.hit_ratio == 0.5
    for block in prefix.cached_block_ids():
        blocks.free(block)
    assert prefix.flush() == 2
    assert len(prefix) == 0
    stats = prefix.stats.as_dict()
    assert stats["registered_blocks"] == 2
    assert stats["evicted_blocks"] == 2


def test_scope_bytes_depends_on_the_configured_scope() -> None:
    per_key, _ = cache()
    shared, _ = cache(scope="global")
    assert per_key.scope_bytes("alice") != per_key.scope_bytes("bob")
    assert shared.scope_bytes("alice") == shared.scope_bytes("bob")
