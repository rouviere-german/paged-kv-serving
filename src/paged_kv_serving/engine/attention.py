"""Paged attention: one softmax over keys and values scattered across non-contiguous blocks.

A sequence's keys and values live in whatever blocks the allocator handed out, in the order given
by its block table. Materialising *all* of them into one contiguous array per sequence would undo
the point of paging, so the kernel walks the block table in tiles and folds each tile into a
running result with the online-softmax recurrence (Milakov and Gimelshein, "Online normalizer
calculation for softmax", 2018; the same recurrence that makes FlashAttention possible):

    m_new = max(m, max(scores_tile))
    l     = l * exp(m - m_new) + sum(exp(scores_tile - m_new))
    acc   = acc * exp(m - m_new) + exp(scores_tile - m_new) @ V_tile

After the last tile ``acc / l`` equals a single softmax over all the scores, but only one tile of
scores is ever in memory.

``tile_blocks`` sets how many blocks are gathered per step. One block per step is the strict form
and is what the property tests drive; the engine uses a larger tile because in NumPy the cost of
a step is Python and BLAS-call overhead rather than arithmetic, and a real kernel tiles well above
its page size for the same reason. The recurrence runs whenever a sequence is longer than one
tile, which on any realistic context it is.

Masking is by absolute position, so the same kernel serves a one-token decode step and a
mid-prompt prefill chunk whose queries sit above thousands of already-cached keys. Scores are
formed with batched ``matmul`` over the head axis rather than ``einsum``: for arrays this small,
``einsum``'s contraction-path bookkeeping costs more than the multiplication it plans.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["DEFAULT_TILE_BLOCKS", "paged_attention"]

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

#: Blocks gathered per online-softmax step when the caller does not choose.
DEFAULT_TILE_BLOCKS = 8


def paged_attention(  # noqa: PLR0917 - kernel-style signature: the arrays it reads, in order
    queries: FloatArray,
    query_positions: IntArray,
    key_cache: FloatArray,
    value_cache: FloatArray,
    block_table: IntArray,
    context_len: int,
    scale: float,
    *,
    tile_blocks: int = DEFAULT_TILE_BLOCKS,
) -> FloatArray:
    """Causal attention of ``queries`` over the first ``context_len`` cached tokens.

    Args:
        queries: ``(num_queries, num_heads, head_dim)``.
        query_positions: absolute position of each query, used for the causal mask.
        key_cache, value_cache: ``(num_blocks, block_size, num_heads, head_dim)`` pools.
        block_table: physical block ids of this sequence, in logical order.
        context_len: number of valid tokens, counted from logical position zero.
        scale: ``1 / sqrt(head_dim)``.
        tile_blocks: blocks folded into the running softmax per step.

    Returns:
        ``(num_queries, num_heads, head_dim)``.
    """
    if queries.ndim != 3:
        raise ValueError("queries must have shape (num_queries, num_heads, head_dim)")
    if queries.shape[0] != query_positions.shape[0]:
        raise ValueError("one position per query is required")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key and value caches must have the same shape")
    if tile_blocks < 1:
        raise ValueError("tile_blocks must be >= 1")
    block_size = key_cache.shape[1]
    if context_len < 1:
        raise ValueError("context_len must be >= 1")
    num_used_blocks = -(-context_len // block_size)
    if num_used_blocks > block_table.shape[0]:
        raise ValueError("context_len exceeds the blocks listed in the block table")

    num_queries, num_heads, head_dim = queries.shape
    running_max = np.full((num_queries, num_heads), -np.inf, dtype=np.float32)
    running_sum = np.zeros((num_queries, num_heads), dtype=np.float32)
    accumulator = np.zeros((num_heads, num_queries, head_dim), dtype=np.float32)
    positions = query_positions.reshape(num_queries, 1)
    # (heads, queries, dim) once, so every tile is a batched GEMM over the head axis.
    queries_by_head = np.ascontiguousarray(queries.transpose(1, 0, 2))

    for first in range(0, num_used_blocks, tile_blocks):
        last = min(first + tile_blocks, num_used_blocks)
        ids = block_table[first:last]
        start = first * block_size
        valid = min((last - first) * block_size, context_len - start)
        # The gather is the block table doing its job: physically scattered blocks become one
        # logically contiguous tile, and nothing outside this sequence's table is reachable.
        keys = key_cache[ids].reshape(-1, num_heads, head_dim)[:valid]
        values = value_cache[ids].reshape(-1, num_heads, head_dim)[:valid]

        # (heads, queries, valid)
        scores = np.matmul(queries_by_head, keys.transpose(1, 2, 0)) * np.float32(scale)
        key_positions = start + np.arange(valid, dtype=np.int64)
        masked = key_positions.reshape(1, valid) > positions
        scores = np.where(masked[None, :, :], np.float32(-np.inf), scores)

        tile_max = scores.max(axis=-1).transpose(1, 0)
        new_max = np.maximum(running_max, tile_max)
        # A row is all -inf only when every key in this tile is masked out and nothing has been
        # accumulated yet. Substituting 0 for the shift there keeps exp(-inf - 0) = 0 instead of
        # producing exp(-inf + inf) = nan.
        shift = np.where(np.isfinite(new_max), new_max, np.float32(0.0))
        rescale = np.exp(running_max - shift)
        probabilities = np.exp(scores - shift.transpose(1, 0)[:, :, None])

        running_sum = running_sum * rescale + probabilities.sum(axis=-1).transpose(1, 0)
        accumulator *= rescale.transpose(1, 0)[:, :, None]
        accumulator += np.matmul(probabilities, values.transpose(1, 0, 2))
        running_max = new_max

    denominator = np.where(running_sum > 0.0, running_sum, np.float32(1.0))
    return (accumulator.transpose(1, 0, 2) / denominator[:, :, None]).astype(np.float32)
