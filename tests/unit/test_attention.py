from __future__ import annotations

import numpy as np
import pytest

from paged_kv_serving.engine.attention import paged_attention
from paged_kv_serving.engine.model import dense_causal_attention

pytestmark = pytest.mark.unit

HEADS, DIM, BLOCK = 2, 8, 4


def build_cache(length: int, blocks: int, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    keys = np.zeros((blocks, BLOCK, HEADS, DIM), dtype=np.float32)
    values = np.zeros_like(keys)
    flat_k = rng.standard_normal((length, HEADS, DIM)).astype(np.float32)
    flat_v = rng.standard_normal((length, HEADS, DIM)).astype(np.float32)
    keys.reshape(-1, HEADS, DIM)[:length] = flat_k
    values.reshape(-1, HEADS, DIM)[:length] = flat_v
    return keys, values, flat_k, flat_v


@pytest.mark.parametrize("tile_blocks", [1, 2, 3, 8])
def test_paged_attention_matches_dense_attention(tile_blocks: int) -> None:
    length = 11
    blocks = -(-length // BLOCK)
    keys, values, flat_k, flat_v = build_cache(length, blocks)
    rng = np.random.default_rng(1)
    queries = rng.standard_normal((length, HEADS, DIM)).astype(np.float32)
    scale = DIM**-0.5
    expected = dense_causal_attention(queries, flat_k, flat_v, scale)
    got = paged_attention(
        queries,
        np.arange(length, dtype=np.int64),
        keys,
        values,
        np.arange(blocks, dtype=np.int64),
        length,
        scale,
        tile_blocks=tile_blocks,
    )
    assert np.max(np.abs(got - expected)) < 1e-5


def test_a_single_decode_query_attends_to_the_whole_context() -> None:
    length = 9
    blocks = -(-length // BLOCK)
    keys, values, flat_k, flat_v = build_cache(length, blocks, seed=2)
    rng = np.random.default_rng(3)
    query = rng.standard_normal((1, HEADS, DIM)).astype(np.float32)
    scale = DIM**-0.5
    scores = np.einsum("qhd,khd->hqk", query, flat_k) * scale
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    expected = np.einsum("hqk,khd->qhd", weights, flat_v)
    got = paged_attention(
        query,
        np.asarray([length - 1], dtype=np.int64),
        keys,
        values,
        np.arange(blocks, dtype=np.int64),
        length,
        scale,
        tile_blocks=1,
    )
    assert np.max(np.abs(got - expected)) < 1e-5


def test_the_block_table_order_selects_the_keys() -> None:
    """Swapping two entries of the block table must change what a masked query can see.

    Attention over an unmasked context is permutation invariant -- keys and values move together --
    so the query sits at position 3, where the causal mask admits only the first block. Which
    physical block that is comes from the block table and nowhere else.
    """
    length = 8
    keys, values, _, _ = build_cache(length, 2, seed=4)
    query = np.ones((1, HEADS, DIM), dtype=np.float32)
    args = (np.asarray([3], dtype=np.int64), keys, values)
    forward = paged_attention(query, *args, np.asarray([0, 1], dtype=np.int64), length, 0.5)
    swapped = paged_attention(query, *args, np.asarray([1, 0], dtype=np.int64), length, 0.5)
    assert not np.allclose(forward, swapped)


def test_a_query_below_the_context_sees_only_its_own_prefix() -> None:
    length = 12
    blocks = -(-length // BLOCK)
    keys, values, flat_k, flat_v = build_cache(length, blocks, seed=5)
    rng = np.random.default_rng(6)
    query = rng.standard_normal((1, HEADS, DIM)).astype(np.float32)
    scale = DIM**-0.5
    # Position 3 must ignore keys 4..11 even though they are present in the cache.
    got = paged_attention(
        query,
        np.asarray([3], dtype=np.int64),
        keys,
        values,
        np.arange(blocks, dtype=np.int64),
        length,
        scale,
        tile_blocks=1,
    )
    truncated = dense_causal_attention(
        np.concatenate([np.zeros((3, HEADS, DIM), dtype=np.float32), query]),
        flat_k[:4],
        flat_v[:4],
        scale,
    )[-1]
    assert np.max(np.abs(got[0] - truncated)) < 1e-5


def test_partial_last_block_is_masked_off() -> None:
    length = 5  # one full block plus one token
    keys, values, _flat_k, _flat_v = build_cache(length, 2, seed=7)
    # Poison the unused slots: a correct kernel never reads them.
    keys.reshape(-1, HEADS, DIM)[length:] = 1e6
    values.reshape(-1, HEADS, DIM)[length:] = 1e6
    rng = np.random.default_rng(8)
    query = rng.standard_normal((1, HEADS, DIM)).astype(np.float32)
    got = paged_attention(
        query,
        np.asarray([length - 1], dtype=np.int64),
        keys,
        values,
        np.asarray([0, 1], dtype=np.int64),
        length,
        DIM**-0.5,
    )
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got)) < 100.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_len": 0}, "context_len must be >= 1"),
        ({"context_len": 99}, "exceeds the blocks"),
        ({"tile_blocks": 0}, "tile_blocks must be >= 1"),
    ],
)
def test_invalid_arguments_are_refused(kwargs: dict[str, int], message: str) -> None:
    keys, values, _, _ = build_cache(4, 1)
    query = np.zeros((1, HEADS, DIM), dtype=np.float32)
    call = {
        "queries": query,
        "query_positions": np.asarray([0], dtype=np.int64),
        "key_cache": keys,
        "value_cache": values,
        "block_table": np.asarray([0], dtype=np.int64),
        "context_len": 4,
        "scale": 0.5,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        paged_attention(**call)  # type: ignore[arg-type]


def test_shape_mismatches_are_refused() -> None:
    keys, values, _, _ = build_cache(4, 1)
    with pytest.raises(ValueError, match="num_queries, num_heads, head_dim"):
        paged_attention(
            np.zeros((1, DIM), dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            keys,
            values,
            np.asarray([0], dtype=np.int64),
            4,
            0.5,
        )
    with pytest.raises(ValueError, match="one position per query"):
        paged_attention(
            np.zeros((2, HEADS, DIM), dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            keys,
            values,
            np.asarray([0], dtype=np.int64),
            4,
            0.5,
        )
    with pytest.raises(ValueError, match="same shape"):
        paged_attention(
            np.zeros((1, HEADS, DIM), dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            keys,
            values[:, :1],
            np.asarray([0], dtype=np.int64),
            4,
            0.5,
        )
