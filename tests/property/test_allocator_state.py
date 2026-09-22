"""Stateful property tests for the paged allocator.

A memory manager is exactly the kind of component where the bug is never in the operation you are
looking at -- it is in the tenth operation after a fork, an eviction and a preemption. So instead
of scripted scenarios, hypothesis drives the real :class:`KVCacheManager` through arbitrary
interleavings of admit, prefill, decode, fork, preempt, free and evict, and four invariants are
checked after *every* step:

1. the allocator is internally consistent (free list, reference counts, totals);
2. every sequence can still read back every token it has computed, at the right slot -- this is
   the one that catches a missed copy-on-write, a wrong slot index or a block handed out twice;
3. a block registered in the prefix cache never changes content;
4. nothing leaks: once every sequence is freed and the cache is flushed, the pool is whole again.

A shadow store records what would be written where, so invariant 2 is checked against the block
tables the manager actually produced rather than against a second implementation of them.
"""

from __future__ import annotations

import contextlib

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from paged_kv_serving.engine.blocks import BlockAllocationError, BlockManager
from paged_kv_serving.engine.kvcache import KVCacheManager
from paged_kv_serving.engine.prefix_cache import PrefixCache
from paged_kv_serving.engine.sequence import Request, SamplingParams, Sequence

pytestmark = pytest.mark.unit

BLOCK_SIZE = 4
NUM_BLOCKS = 24
KEYS = ("tenant-a", "tenant-b")


class ShadowStore:
    """Remembers what each block holds, so copy-on-write can be checked byte for byte."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.cells: list[list[int | None]] = [[None] * block_size for _ in range(num_blocks)]

    def copy_block(self, source: int, destination: int) -> None:
        self.cells[destination] = list(self.cells[source])

    def write(self, block_id: int, offset: int, token: int) -> None:
        self.cells[block_id][offset] = token

    def snapshot(self, block_id: int) -> tuple[int | None, ...]:
        return tuple(self.cells[block_id])


class PagedCacheMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.store = ShadowStore(NUM_BLOCKS, BLOCK_SIZE)
        self.blocks = BlockManager(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, store=self.store)
        self.cache = PrefixCache(self.blocks, scope="per_key")
        self.kv = KVCacheManager(self.blocks, self.cache)
        self.sequences: dict[str, Sequence] = {}
        self.cached_snapshots: dict[bytes, tuple[int | None, ...]] = {}
        self.counter = 0

    # --- helpers -------------------------------------------------------------------------

    def _pick(self, index: int) -> Sequence:
        keys = sorted(self.sequences)
        return self.sequences[keys[index % len(keys)]]

    def _write_chunk(self, sequence: Sequence, chunk: int) -> None:
        """Mirror exactly what the model runner does: reserve, then write each token's slot."""
        self.kv.reserve(sequence, sequence.num_computed_tokens + chunk)
        for position in range(sequence.num_computed_tokens, sequence.num_computed_tokens + chunk):
            block = sequence.block_table[position // BLOCK_SIZE]
            self.store.write(block, position % BLOCK_SIZE, sequence.token_ids[position])
        sequence.num_computed_tokens += chunk
        self.kv.publish_full_blocks(sequence)
        self._record_cached()

    def _record_cached(self) -> None:
        for block_hash, block_id in zip(
            self.cache._entries.keys(),
            self.cache.cached_block_ids(),
            strict=True,
        ):
            self.cached_snapshots.setdefault(block_hash, self.store.snapshot(block_id))

    # --- rules ---------------------------------------------------------------------------

    @rule(
        prompt=st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=14),
        key=st.sampled_from(KEYS),
        shared_prefix=st.booleans(),
    )
    def admit(self, prompt: list[int], key: str, shared_prefix: bool) -> None:
        if len(self.sequences) >= 6:
            return
        tokens = [1, 2, 3, 4, *prompt] if shared_prefix else prompt
        self.counter += 1
        request = Request(
            request_id=f"r{self.counter}",
            prompt_token_ids=tuple(tokens),
            sampling=SamplingParams(max_new_tokens=4, seed=self.counter),
            key_id=key,
        )
        sequence = Sequence.from_request(request, 0)
        self.kv.attach_prefix(sequence)
        self.sequences[sequence.seq_id] = sequence

    @precondition(lambda self: bool(self.sequences))
    @rule(
        index=st.integers(min_value=0, max_value=100), chunk=st.integers(min_value=1, max_value=6)
    )
    def prefill_chunk(self, index: int, chunk: int) -> None:
        sequence = self._pick(index)
        remaining = sequence.num_uncomputed_tokens
        if remaining < 1:
            return
        # A full pool is fine: a scheduler would preempt; the allocator just must not corrupt.
        with contextlib.suppress(BlockAllocationError):
            self._write_chunk(sequence, min(chunk, remaining))

    @precondition(lambda self: bool(self.sequences))
    @rule(
        index=st.integers(min_value=0, max_value=100), token=st.integers(min_value=1, max_value=12)
    )
    def decode(self, index: int, token: int) -> None:
        sequence = self._pick(index)
        if sequence.num_uncomputed_tokens != 0 or sequence.num_output_tokens >= 6:
            return
        sequence.append_token(token)
        try:
            self._write_chunk(sequence, 1)
        except BlockAllocationError:
            sequence.token_ids.pop()

    @precondition(lambda self: bool(self.sequences))
    @rule(index=st.integers(min_value=0, max_value=100))
    def fork(self, index: int) -> None:
        parent = self._pick(index)
        if parent.num_computed_tokens < 1 or len(self.sequences) >= 8:
            return
        self.counter += 1
        child = Sequence.from_request(parent.request, self.counter)
        child.token_ids = list(parent.token_ids)
        child.seq_id = f"{parent.seq_id}-fork{self.counter}"
        self.kv.fork(parent, child)
        self.sequences[child.seq_id] = child

    @precondition(lambda self: bool(self.sequences))
    @rule(index=st.integers(min_value=0, max_value=100))
    def preempt(self, index: int) -> None:
        sequence = self._pick(index)
        self.kv.free(sequence)
        sequence.reset_for_recompute()

    @precondition(lambda self: bool(self.sequences))
    @rule(index=st.integers(min_value=0, max_value=100))
    def finish(self, index: int) -> None:
        sequence = self._pick(index)
        self.kv.free(sequence)
        del self.sequences[sequence.seq_id]

    @rule(count=st.integers(min_value=1, max_value=4))
    def evict(self, count: int) -> None:
        released = self.cache.evict(count)
        assert released <= count

    # --- invariants ----------------------------------------------------------------------

    @invariant()
    def allocator_is_consistent(self) -> None:
        self.kv.check_invariants()

    @invariant()
    def sequences_read_back_their_own_tokens(self) -> None:
        for sequence in self.sequences.values():
            for position in range(sequence.num_computed_tokens):
                block = sequence.block_table[position // BLOCK_SIZE]
                stored = self.store.cells[block][position % BLOCK_SIZE]
                assert stored == sequence.token_ids[position], (
                    f"{sequence.seq_id} position {position}: block {block} holds {stored}, "
                    f"expected {sequence.token_ids[position]}"
                )

    @invariant()
    def cached_blocks_are_never_mutated(self) -> None:
        entries = dict(
            zip(
                self.cache._entries.keys(),
                self.cache.cached_block_ids(),
                strict=True,
            )
        )
        for block_hash, block_id in entries.items():
            expected = self.cached_snapshots.get(block_hash)
            if expected is not None:
                assert self.store.snapshot(block_id) == expected, (
                    f"cached block {block_id} changed after registration"
                )

    @invariant()
    def every_sequence_has_blocks_for_its_computed_tokens(self) -> None:
        for sequence in self.sequences.values():
            needed = -(-sequence.num_computed_tokens // BLOCK_SIZE)
            assert len(sequence.block_table) >= needed

    def teardown(self) -> None:
        for sequence in self.sequences.values():
            self.kv.free(sequence)
        self.cache.flush()
        self.blocks.check_invariants()
        assert self.blocks.num_free_blocks == NUM_BLOCKS, "blocks leaked"


PagedCacheTest = PagedCacheMachine.TestCase
PagedCacheTest.settings = settings(
    max_examples=40,
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
