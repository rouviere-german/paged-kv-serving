"""Sequence-level view of the paged cache: block tables, prefix reuse and copy-on-write.

This is the layer the scheduler talks to. It turns "sequence S now needs its first 96 tokens
resident" into free-list operations, prefix-cache lookups, evictions and copy-on-write, and it is
the only place that mutates a sequence's ``block_table``.

Two invariants hold at every return from this module and are asserted by the stateful tests:

* a block is reachable from a live sequence's block table, or from the prefix cache, or is free --
  never two of those without a matching reference count, and never none of them;
* a block registered in the prefix cache is immutable. Only full blocks are registered, a full
  block is never written again by the sequence that produced it, and any other sequence that
  wants to write into a shared block gets a private copy first.
"""

from __future__ import annotations

from dataclasses import dataclass

from paged_kv_serving.engine.blocks import BlockAllocationError, BlockManager, InvariantError
from paged_kv_serving.engine.prefix_cache import PrefixCache, chain_seed, next_block_hash
from paged_kv_serving.engine.sequence import Sequence

__all__ = ["KVCacheManager", "PrefixReuse"]


@dataclass(frozen=True, slots=True)
class PrefixReuse:
    """What a prefix-cache lookup saved for one sequence."""

    blocks: int
    tokens: int
    queried_blocks: int


class KVCacheManager:
    """Block tables, prefix reuse and eviction for every live sequence."""

    __slots__ = ("block_size", "blocks", "prefix_cache")

    def __init__(self, blocks: BlockManager, prefix_cache: PrefixCache) -> None:
        self.blocks = blocks
        self.prefix_cache = prefix_cache
        self.block_size = blocks.block_size

    # --- capacity ------------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return self.blocks.num_free_blocks

    @property
    def utilization(self) -> float:
        return self.blocks.utilization

    def evictable_blocks(self) -> int:
        """Cached blocks no live sequence is reading, and so available under memory pressure."""
        return sum(
            1
            for block_id in self.prefix_cache.cached_block_ids()
            if self.blocks.ref_count(block_id) == 1
        )

    def blocks_required(self, sequence: Sequence, up_to_tokens: int) -> int:
        """Extra blocks this sequence needs before ``up_to_tokens`` can be written."""
        needed = -(-up_to_tokens // self.block_size)
        return max(0, needed - len(sequence.block_table))

    def can_reserve(self, count: int) -> bool:
        """True when ``count`` blocks are available, counting evictable cached blocks."""
        return count <= self.blocks.num_free_blocks + self.evictable_blocks()

    # --- prefix reuse --------------------------------------------------------------------

    def attach_prefix(self, sequence: Sequence) -> PrefixReuse:
        """Adopt cached blocks for the longest matching prefix of a sequence about to prefill.

        At least one token is always left to compute: a request whose every block is cached still
        has to run its final token to get logits, and a zero-token iteration would produce no next
        token at all. Sharing is scoped by the caller's key id (see :mod:`prefix_cache`).
        """
        if sequence.num_computed_tokens or sequence.block_table:
            raise InvariantError("attach_prefix runs before any block is allocated")
        if not self.prefix_cache.enabled:
            return PrefixReuse(blocks=0, tokens=0, queried_blocks=0)
        reusable = (sequence.num_tokens - 1) // self.block_size
        if reusable < 1:
            return PrefixReuse(blocks=0, tokens=0, queried_blocks=0)
        scope = self.prefix_cache.scope_bytes(sequence.request.key_id)
        parent = chain_seed(scope)
        hashes: list[bytes] = []
        for index in range(reusable):
            chunk = sequence.token_ids[index * self.block_size : (index + 1) * self.block_size]
            parent = next_block_hash(parent, chunk, self.block_size)
            hashes.append(parent)
        matched = self.prefix_cache.lookup(hashes)
        sequence.block_table.extend(matched)
        sequence.block_hashes.extend(hashes[: len(matched)])
        sequence.cached_prefix_blocks = len(matched)
        sequence.num_computed_tokens = len(matched) * self.block_size
        return PrefixReuse(
            blocks=len(matched), tokens=len(matched) * self.block_size, queried_blocks=reusable
        )

    def publish_full_blocks(self, sequence: Sequence) -> int:
        """Offer every newly completed block of a sequence to the prefix cache.

        Only blocks entirely covered by ``num_computed_tokens`` are offered, so their contents can
        no longer change. Returns the number of blocks the cache accepted.
        """
        if not self.prefix_cache.enabled:
            return 0
        complete = sequence.num_computed_tokens // self.block_size
        scope = self.prefix_cache.scope_bytes(sequence.request.key_id)
        published = 0
        while len(sequence.block_hashes) < complete:
            index = len(sequence.block_hashes)
            parent = sequence.block_hashes[-1] if sequence.block_hashes else chain_seed(scope)
            chunk = sequence.token_ids[index * self.block_size : (index + 1) * self.block_size]
            block_hash = next_block_hash(parent, chunk, self.block_size)
            sequence.block_hashes.append(block_hash)
            if self.prefix_cache.register(block_hash, sequence.block_table[index]):
                published += 1
        return published

    # --- allocation ----------------------------------------------------------------------

    def reserve(self, sequence: Sequence, up_to_tokens: int) -> None:
        """Make positions ``[num_computed_tokens, up_to_tokens)`` writable for this sequence.

        Allocates the missing blocks (evicting cached ones if the free list is short) and applies
        copy-on-write to every block in the write range that is shared with anybody else. Raises
        :class:`BlockAllocationError` when the pool cannot cover the request, leaving the block
        table and every reference count as they were (prefix-cache entries evicted while trying
        stay evicted; they held no live sequence's data).
        """
        if up_to_tokens > sequence.num_tokens:
            raise ValueError("cannot reserve beyond the tokens the sequence has")
        extra = self.blocks_required(sequence, up_to_tokens)
        first_write = sequence.num_computed_tokens // self.block_size
        last_write = -(-up_to_tokens // self.block_size)
        # Shared blocks already in the table that this write would touch. Each needs one fresh
        # block for its private copy, so they are counted *before* anything is allocated: a
        # reservation either succeeds completely or leaves the table and the pool untouched.
        shared = [
            index
            for index in range(first_write, min(last_write, len(sequence.block_table)))
            if self.blocks.ref_count(sequence.block_table[index]) > 1
        ]
        needed = extra + len(shared)
        if needed:
            self.prefix_cache.evict_until_free(needed)
            if self.blocks.num_free_blocks < needed:
                raise BlockAllocationError(
                    f"need {needed} blocks, {self.blocks.num_free_blocks} free"
                )
        for index in shared:
            sequence.block_table[index] = self.blocks.prepare_for_write(sequence.block_table[index])
        if extra:
            sequence.block_table.extend(self.blocks.allocate_many(extra))

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Point ``child`` at ``parent``'s blocks, sharing rather than copying the prompt.

        Both sequences keep reading the same physical blocks until one of them writes, at which
        point :meth:`reserve` gives the writer a private copy of the single block it is writing.
        """
        if child.block_table:
            raise InvariantError("fork target already owns blocks")
        for block_id in parent.block_table:
            self.blocks.incref(block_id)
        child.block_table.extend(parent.block_table)
        child.block_hashes.extend(parent.block_hashes)
        child.num_computed_tokens = parent.num_computed_tokens
        child.cached_prefix_blocks = parent.cached_prefix_blocks

    def free(self, sequence: Sequence) -> int:
        """Release every block of a sequence. Returns how many blocks it held."""
        count = len(sequence.block_table)
        self.blocks.free_many(sequence.block_table)
        sequence.block_table.clear()
        sequence.block_hashes.clear()
        sequence.cached_prefix_blocks = 0
        return count

    def check_invariants(self) -> None:
        """Allocator invariants plus 'every cached block is alive'."""
        self.blocks.check_invariants()
        for block_id in self.prefix_cache.cached_block_ids():
            if self.blocks.ref_count(block_id) < 1:
                raise InvariantError(f"cached block {block_id} has no reference")
