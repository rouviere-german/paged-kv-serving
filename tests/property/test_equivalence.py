"""The central correctness property: paging and batching must not change the answer.

Every claim this repository makes about throughput is only interesting if the engine computes the
same thing a naive implementation would. Hypothesis generates the token sequence, the chunk
boundaries and the physical block order, and each example is checked against
``LanguageModel.reference_forward`` -- a dense, cache-free recompute of the whole sequence.

The three axes are deliberate, because each one is a different way to get it wrong:

* **chunking** exercises the online-softmax recurrence and the position bookkeeping of chunked
  prefill (a chunk boundary in the middle of a block is the interesting case);
* **block order** exercises the block table itself: reversed, shuffled and sparse tables must all
  give the same result as a contiguous one, which is false the moment a slot index is computed
  from a logical position instead of through the table;
* **batch composition** exercises the runner's flattening: a sequence must produce the same logits
  whether it is alone in a batch or interleaved with others of different lengths.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hypothesis_budget import hypothesis_examples
from paged_kv_serving.engine.equivalence import max_logit_gap, paged_chunked_logits, split_sizes
from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.runner import BatchEntry, ExecutionBatch, ModelRunner

pytestmark = pytest.mark.unit

#: The claim in the README. Observed gaps on the development machine are around 2e-6.
TOLERANCE = 1e-4
BLOCK_SIZE = 8


def tokens(draw: st.DrawFn, vocab: int, low: int = 1, high: int = 40) -> list[int]:
    length = draw(st.integers(min_value=low, max_value=high))
    return draw(
        st.lists(st.integers(min_value=0, max_value=vocab - 1), min_size=length, max_size=length)
    )


@st.composite
def token_sequence(draw: st.DrawFn) -> list[int]:
    return tokens(draw, vocab=64)


@settings(max_examples=hypothesis_examples(25), deadline=None)
@given(sequence=token_sequence(), chunk=st.integers(min_value=1, max_value=13))
def test_chunked_prefill_matches_a_cache_free_recompute(
    tiny_model: LanguageModel, sequence: list[int], chunk: int
) -> None:
    gap = max_logit_gap(
        tiny_model, sequence, block_size=BLOCK_SIZE, chunks=split_sizes(len(sequence), chunk)
    )
    assert gap < TOLERANCE, f"chunk={chunk} len={len(sequence)} gap={gap}"


@settings(max_examples=hypothesis_examples(25), deadline=None)
@given(sequence=token_sequence(), seed=st.integers(min_value=0, max_value=10_000))
def test_a_scattered_block_table_gives_the_same_logits(
    tiny_model: LanguageModel, sequence: list[int], seed: int
) -> None:
    needed = -(-len(sequence) // BLOCK_SIZE)
    rng = np.random.default_rng(seed)
    # Deliberately sparse and shuffled: the physical order must be irrelevant.
    order = [int(value) for value in rng.permutation(needed * 2 + 3)[:needed]]
    gap = max_logit_gap(tiny_model, sequence, block_size=BLOCK_SIZE, block_order=order)
    assert gap < TOLERANCE, f"order={order} gap={gap}"


@settings(max_examples=hypothesis_examples(20), deadline=None)
@given(
    sequence=token_sequence(),
    split=st.integers(min_value=1, max_value=20),
    tile=st.integers(min_value=1, max_value=6),
)
def test_tile_size_does_not_change_the_result(
    tiny_model: LanguageModel, sequence: list[int], split: int, tile: int
) -> None:
    """The online-softmax recurrence must be tile-size independent, including one block per step."""
    reference = tiny_model.reference_forward(sequence)[-1]
    blocks = -(-len(sequence) // BLOCK_SIZE)
    runner = ModelRunner(tiny_model, num_blocks=blocks, block_size=BLOCK_SIZE, tile_blocks=tile)
    sizes = split_sizes(len(sequence), split)
    start = 0
    logits = None
    for size in sizes:
        stop = start + size
        entry = BatchEntry(
            seq_id="s",
            token_ids=tuple(sequence[start:stop]),
            positions=tuple(range(start, stop)),
            block_table=tuple(range(blocks)),
            logits_index=size - 1 if stop == len(sequence) else None,
        )
        result = runner.execute(ExecutionBatch((entry,)))
        logits = result.get("s", logits)
        start = stop
    assert logits is not None
    assert float(np.max(np.abs(logits - reference))) < TOLERANCE


@settings(max_examples=hypothesis_examples(15), deadline=None)
@given(
    sequences=st.lists(token_sequence(), min_size=2, max_size=4),
    decode_first=st.booleans(),
)
def test_batch_composition_does_not_change_a_sequences_logits(
    tiny_model: LanguageModel, sequences: list[list[int]], decode_first: bool
) -> None:
    """One ragged batch of several sequences must agree with each sequence run on its own."""
    block_size = BLOCK_SIZE
    per_sequence_blocks = [-(-len(tokens) // block_size) for tokens in sequences]
    total_blocks = sum(per_sequence_blocks)
    runner = ModelRunner(tiny_model, num_blocks=total_blocks, block_size=block_size)
    entries = []
    offset = 0
    order = list(range(len(sequences)))
    if decode_first:
        order.reverse()
    for index in order:
        tokens = sequences[index]
        start = sum(per_sequence_blocks[:index])
        entries.append(
            BatchEntry(
                seq_id=f"s{index}",
                token_ids=tuple(tokens),
                positions=tuple(range(len(tokens))),
                block_table=tuple(range(start, start + per_sequence_blocks[index])),
                logits_index=len(tokens) - 1,
            )
        )
        offset += len(tokens)
    batched = runner.execute(ExecutionBatch(tuple(entries)))
    for index, tokens in enumerate(sequences):
        alone = paged_chunked_logits(tiny_model, tokens, block_size=block_size)
        gap = float(np.max(np.abs(batched[f"s{index}"] - alone)))
        assert gap < TOLERANCE, f"sequence {index} gap={gap}"


@settings(max_examples=hypothesis_examples(15), deadline=None)
@given(
    prefix=st.lists(st.integers(min_value=0, max_value=63), min_size=8, max_size=16),
    suffix_a=st.lists(st.integers(min_value=0, max_value=63), min_size=1, max_size=12),
    suffix_b=st.lists(st.integers(min_value=0, max_value=63), min_size=1, max_size=12),
)
def test_sharing_a_prefix_block_does_not_change_either_continuation(
    tiny_model: LanguageModel, prefix: list[int], suffix_a: list[int], suffix_b: list[int]
) -> None:
    """Two sequences reading the same physical prefix blocks still get their own answers."""
    block_size = BLOCK_SIZE
    shared_blocks = len(prefix) // block_size
    prefix = prefix[: shared_blocks * block_size]
    if shared_blocks < 1:
        pytest.skip("needs at least one full shared block")
    sequence_a, sequence_b = [*prefix, *suffix_a], [*prefix, *suffix_b]
    extra = max(
        -(-len(sequence_a) // block_size) - shared_blocks,
        -(-len(sequence_b) // block_size) - shared_blocks,
    )
    runner = ModelRunner(
        tiny_model, num_blocks=shared_blocks + 2 * max(extra, 1), block_size=block_size
    )
    shared_table = tuple(range(shared_blocks))
    # Compute the shared prefix once.
    runner.execute(
        ExecutionBatch(
            (
                BatchEntry(
                    seq_id="prefix",
                    token_ids=tuple(prefix),
                    positions=tuple(range(len(prefix))),
                    block_table=shared_table,
                ),
            )
        )
    )
    for name, sequence, base in (
        ("a", sequence_a, shared_blocks),
        ("b", sequence_b, shared_blocks + max(extra, 1)),
    ):
        own = -(-len(sequence) // block_size) - shared_blocks
        table = shared_table + tuple(range(base, base + max(own, 0)))
        result = runner.execute(
            ExecutionBatch(
                (
                    BatchEntry(
                        seq_id=name,
                        token_ids=tuple(sequence[len(prefix) :]),
                        positions=tuple(range(len(prefix), len(sequence))),
                        block_table=table,
                        logits_index=len(sequence) - len(prefix) - 1,
                    ),
                )
            )
        )
        reference = tiny_model.reference_forward(sequence)[-1]
        gap = float(np.max(np.abs(result[name] - reference)))
        assert gap < TOLERANCE, f"continuation {name} gap={gap}"
