from __future__ import annotations

import pytest

from paged_kv_serving.engine.blocks import (
    BlockAllocationError,
    BlockManager,
    InvariantError,
    RecordingBlockStore,
)

pytestmark = pytest.mark.unit


def manager(num_blocks: int = 8, block_size: int = 4) -> tuple[BlockManager, RecordingBlockStore]:
    store = RecordingBlockStore()
    return BlockManager(num_blocks=num_blocks, block_size=block_size, store=store), store


def test_allocation_is_reference_counted_and_returns_blocks_to_the_free_list() -> None:
    blocks, _ = manager()
    first = blocks.allocate()
    assert blocks.ref_count(first) == 1
    assert blocks.num_free_blocks == 7
    blocks.incref(first)
    assert blocks.ref_count(first) == 2
    assert blocks.free(first) == 1
    assert blocks.num_free_blocks == 7  # still owned by the second holder
    assert blocks.free(first) == 0
    assert blocks.num_free_blocks == 8
    blocks.check_invariants()


def test_allocate_many_is_all_or_nothing() -> None:
    blocks, _ = manager(num_blocks=4)
    with pytest.raises(BlockAllocationError, match="need 5 blocks"):
        blocks.allocate_many(5)
    assert blocks.num_free_blocks == 4
    assert len(blocks.allocate_many(4)) == 4
    with pytest.raises(BlockAllocationError):
        blocks.allocate()
    blocks.check_invariants()


def test_copy_on_write_gives_the_writer_a_private_block() -> None:
    blocks, store = manager()
    shared = blocks.allocate()
    blocks.incref(shared)  # a second sequence now reads the same block
    private = blocks.prepare_for_write(shared)
    assert private != shared
    assert store.copies == [(shared, private)]
    # The other holder still sees the original, and each side owns exactly one block.
    assert blocks.ref_count(shared) == 1
    assert blocks.ref_count(private) == 1
    blocks.check_invariants()


def test_an_unshared_block_is_written_in_place_without_a_copy() -> None:
    blocks, store = manager()
    block = blocks.allocate()
    assert blocks.prepare_for_write(block) == block
    assert store.copies == []


def test_copy_on_write_of_a_three_way_share_costs_one_block_per_writer() -> None:
    blocks, store = manager()
    shared = blocks.allocate()
    blocks.incref(shared)
    blocks.incref(shared)
    first = blocks.prepare_for_write(shared)
    second = blocks.prepare_for_write(shared)
    third = blocks.prepare_for_write(shared)  # last owner writes in place
    assert third == shared
    assert len({first, second, third}) == 3
    assert len(store.copies) == 2
    blocks.check_invariants()


def test_double_free_and_out_of_range_ids_are_refused() -> None:
    blocks, _ = manager()
    block = blocks.allocate()
    blocks.free(block)
    with pytest.raises(InvariantError, match="freed twice"):
        blocks.free(block)
    with pytest.raises(InvariantError, match="outside the pool"):
        blocks.ref_count(99)
    with pytest.raises(InvariantError, match="cannot gain a reference"):
        blocks.incref(block)
    with pytest.raises(InvariantError, match="cannot be written"):
        blocks.prepare_for_write(block)


def test_invariant_check_detects_a_corrupted_free_list() -> None:
    blocks, _ = manager(num_blocks=4)
    block = blocks.allocate()
    blocks._free.append(block)
    with pytest.raises(InvariantError):
        blocks.check_invariants()


def test_utilization_tracks_referenced_blocks() -> None:
    blocks, _ = manager(num_blocks=4)
    assert blocks.utilization == 0.0
    held = blocks.allocate_many(2)
    assert blocks.utilization == 0.5
    assert blocks.num_used_blocks == 2
    blocks.free_many(held)
    assert blocks.utilization == 0.0
    assert blocks.snapshot() == (0, 0, 0, 0)


@pytest.mark.parametrize(("num_blocks", "block_size"), [(0, 4), (4, 0)])
def test_degenerate_pools_are_refused(num_blocks: int, block_size: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        BlockManager(num_blocks=num_blocks, block_size=block_size, store=RecordingBlockStore())


def test_a_block_store_must_not_copy_a_block_onto_itself() -> None:
    store = RecordingBlockStore()
    with pytest.raises(InvariantError, match="onto itself"):
        store.copy_block(1, 1)


def test_allocate_many_rejects_a_negative_count() -> None:
    blocks, _ = manager()
    with pytest.raises(ValueError, match="count must be"):
        blocks.allocate_many(-1)
    assert blocks.allocate_many(0) == []
