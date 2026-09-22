"""Physical KV block allocation: free list, reference counts and copy-on-write.

The KV cache is one fixed-size pool of ``num_blocks`` blocks, each holding ``block_size`` tokens
of keys and values for every layer and head. A sequence owns a *block table* -- an ordered list of
physical block ids -- so its logical token positions are contiguous while its physical memory is
not. That is what removes external fragmentation: a sequence never needs a contiguous run, so the
only waste is at most ``block_size - 1`` tokens in its last block.

Blocks are reference counted, so several sequences can share the same physical block (a cached
prompt prefix, or the prompt of a request sampled ``n`` times). Sharing is safe only while nobody
writes, so a write to a block whose reference count is above one first copies it
(:meth:`BlockManager.prepare_for_write`). Sharing plus copy-on-write is the whole reason a request
sampled four ways costs one copy of the prompt instead of four.

Terminology and the block-table idea follow Kwon et al., "Efficient Memory Management for Large
Language Model Serving with PagedAttention" (SOSP 2023); the implementation here is independent
and deliberately small.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

__all__ = [
    "BlockAllocationError",
    "BlockManager",
    "BlockStore",
    "InvariantError",
    "RecordingBlockStore",
]


class BlockAllocationError(RuntimeError):
    """No free block is available. The caller must free, evict or preempt before retrying."""


class InvariantError(AssertionError):
    """An allocator invariant is broken (a leak, a negative count or a corrupted free list)."""


@runtime_checkable
class BlockStore(Protocol):
    """The physical bytes behind block ids. The allocator only ever asks for a block copy."""

    def copy_block(self, source: int, destination: int) -> None:
        """Copy every byte of block ``source`` over block ``destination``."""


class RecordingBlockStore:
    """A block store that records copies instead of moving bytes. Used by allocator tests."""

    def __init__(self) -> None:
        self.copies: list[tuple[int, int]] = []

    def copy_block(self, source: int, destination: int) -> None:
        if source == destination:
            raise InvariantError("copy-on-write must not copy a block onto itself")
        self.copies.append((source, destination))


class BlockManager:
    """Reference-counted allocator over a fixed pool of KV blocks.

    Allocation is FIFO over the free list, so a freed block is not handed straight back out. That
    keeps the ``RecordingBlockStore`` history readable in tests and spreads reuse across the pool.
    """

    __slots__ = ("_free", "_num_blocks", "_refs", "_store", "block_size")

    def __init__(self, *, num_blocks: int, block_size: int, store: BlockStore) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self._num_blocks = num_blocks
        self.block_size = block_size
        self._store = store
        self._refs = [0] * num_blocks
        self._free: deque[int] = deque(range(num_blocks))

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self._num_blocks - len(self._free)

    @property
    def utilization(self) -> float:
        """Fraction of the pool currently referenced by at least one owner."""
        return self.num_used_blocks / self._num_blocks

    def ref_count(self, block_id: int) -> int:
        return self._refs[self._checked(block_id)]

    def _checked(self, block_id: int) -> int:
        if not 0 <= block_id < self._num_blocks:
            raise InvariantError(f"block id {block_id} is outside the pool")
        return block_id

    def allocate(self) -> int:
        """Take one free block with a reference count of 1."""
        if not self._free:
            raise BlockAllocationError("the KV block pool is exhausted")
        block_id = self._free.popleft()
        self._refs[block_id] = 1
        return block_id

    def allocate_many(self, count: int) -> list[int]:
        """Allocate ``count`` blocks, or none at all if the pool cannot satisfy the whole run."""
        if count < 0:
            raise ValueError("count must be >= 0")
        if count > len(self._free):
            raise BlockAllocationError(
                f"need {count} blocks, {len(self._free)} free in a pool of {self._num_blocks}"
            )
        return [self.allocate() for _ in range(count)]

    def incref(self, block_id: int) -> int:
        """Add an owner to a live block and return the new reference count."""
        block_id = self._checked(block_id)
        if self._refs[block_id] == 0:
            raise InvariantError(f"block {block_id} is free and cannot gain a reference")
        self._refs[block_id] += 1
        return self._refs[block_id]

    def free(self, block_id: int) -> int:
        """Drop one owner. The block returns to the free list when the last owner leaves."""
        block_id = self._checked(block_id)
        if self._refs[block_id] == 0:
            raise InvariantError(f"block {block_id} was freed twice")
        self._refs[block_id] -= 1
        if self._refs[block_id] == 0:
            self._free.append(block_id)
        return self._refs[block_id]

    def free_many(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            self.free(block_id)

    def prepare_for_write(self, block_id: int) -> int:
        """Return the block to write into, copying first when the block is shared.

        With a single owner the block is returned unchanged. With several owners a fresh block
        receives a byte copy, the caller's reference to the original is dropped, and the new block
        id is returned -- the caller must store it in its block table. Every other owner keeps
        seeing the untouched original, which is what makes a shared prefix safe to keep.
        """
        block_id = self._checked(block_id)
        count = self._refs[block_id]
        if count == 0:
            raise InvariantError(f"block {block_id} is free and cannot be written")
        if count == 1:
            return block_id
        destination = self.allocate()
        self._store.copy_block(block_id, destination)
        self.free(block_id)
        return destination

    def check_invariants(self) -> None:
        """Raise :class:`InvariantError` on a leak, a double free or a corrupted free list.

        Checked: no negative counts, the free list holds exactly the zero-count blocks with no
        duplicates, and free plus referenced accounts for every block in the pool.
        """
        free_set = set(self._free)
        if len(free_set) != len(self._free):
            raise InvariantError("the free list contains duplicate block ids")
        for block_id, count in enumerate(self._refs):
            if count < 0:
                raise InvariantError(f"block {block_id} has a negative reference count")
            if (count == 0) != (block_id in free_set):
                raise InvariantError(
                    f"block {block_id} has count {count} but "
                    f"{'is' if block_id in free_set else 'is not'} on the free list"
                )
        referenced = sum(1 for count in self._refs if count > 0)
        if referenced + len(self._free) != self._num_blocks:
            raise InvariantError(
                f"{referenced} referenced + {len(self._free)} free != {self._num_blocks} blocks"
            )

    def snapshot(self) -> tuple[int, ...]:
        """Reference count of every block, for assertions and debugging."""
        return tuple(self._refs)
