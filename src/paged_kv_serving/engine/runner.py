"""Batched forward passes over the paged KV cache.

One iteration of the engine is one call to :meth:`ModelRunner.execute`. The batch it receives is
deliberately ragged: a 48-token prefill chunk of one request sits next to single decode tokens of
six others. Everything that does not depend on a sequence's history -- embedding, the fused QKV
projection, the output projection, the feed-forward network, the final projection to logits -- is
run once over the flattened token axis, so those stay large matrix multiplications. Only
attention has to know where one sequence ends and the next begins, and that is the one place the
per-sequence block table is consulted.

The cache itself is a single pre-allocated pool per layer, shaped
``(num_blocks, block_size, num_heads, head_dim)``, and is written through a flat slot index
computed from each token's block table entry and its offset inside the block. The runner is also
the :class:`~paged_kv_serving.engine.blocks.BlockStore` behind copy-on-write: it is the only
component that knows how to copy the bytes of one block onto another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from paged_kv_serving.engine.attention import DEFAULT_TILE_BLOCKS, paged_attention
from paged_kv_serving.engine.model import LanguageModel

__all__ = ["BatchEntry", "ExecutionBatch", "ModelRunner"]

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class BatchEntry:
    """One sequence's slice of an iteration.

    ``token_ids`` are the tokens to run now, ``positions`` their absolute positions, and
    ``context_len`` the number of valid cached tokens once this slice has been written (that is,
    ``positions[-1] + 1``). ``logits_index`` selects the one token inside the slice whose logits
    the caller wants back, or None when the slice only fills the cache. Continuous batching always
    asks for the last token; the static baseline asks for the last *real* token of a padded row.
    """

    seq_id: str
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    block_table: tuple[int, ...]
    logits_index: int | None = None

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("a batch entry must run at least one token")
        if len(self.token_ids) != len(self.positions):
            raise ValueError("token_ids and positions must have the same length")
        if self.logits_index is not None and not 0 <= self.logits_index < len(self.token_ids):
            raise ValueError("logits_index must point inside the slice")

    @property
    def needs_logits(self) -> bool:
        return self.logits_index is not None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def context_len(self) -> int:
        return self.positions[-1] + 1


@dataclass(frozen=True, slots=True)
class ExecutionBatch:
    """The set of slices scheduled for one iteration."""

    entries: tuple[BatchEntry, ...]

    @property
    def num_tokens(self) -> int:
        return sum(entry.num_tokens for entry in self.entries)

    @property
    def num_sequences(self) -> int:
        return len(self.entries)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(entry.num_tokens for entry in self.entries if entry.num_tokens > 1)


class ModelRunner:
    """Owns the KV pool and executes ragged batches against it."""

    __slots__ = ("_keys", "_values", "block_size", "model", "num_blocks", "tile_blocks")

    def __init__(
        self,
        model: LanguageModel,
        *,
        num_blocks: int,
        block_size: int,
        tile_blocks: int = DEFAULT_TILE_BLOCKS,
    ) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        if tile_blocks < 1:
            raise ValueError("tile_blocks must be >= 1")
        config = model.config
        self.model = model
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.tile_blocks = tile_blocks
        shape = (config.n_layers, num_blocks, block_size, config.n_heads, config.head_dim)
        self._keys: FloatArray = np.zeros(shape, dtype=np.float32)
        self._values: FloatArray = np.zeros(shape, dtype=np.float32)

    @property
    def cache_bytes(self) -> int:
        return int(self._keys.nbytes + self._values.nbytes)

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def block_view(self, layer: int) -> tuple[FloatArray, FloatArray]:
        """Key and value pools of one layer. Exposed for tests and the block-integrity check."""
        return self._keys[layer], self._values[layer]

    def copy_block(self, source: int, destination: int) -> None:
        """Copy-on-write byte copy of one block across every layer."""
        if source == destination:
            raise ValueError("copy-on-write must not copy a block onto itself")
        self._keys[:, destination] = self._keys[:, source]
        self._values[:, destination] = self._values[:, source]

    def block_fingerprint(self, block_id: int) -> bytes:
        """Hash of a block's bytes, used to assert that cached blocks are never mutated."""
        import hashlib  # noqa: PLC0415 - only needed by integrity checks

        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(self._keys[:, block_id]).tobytes())
        digest.update(np.ascontiguousarray(self._values[:, block_id]).tobytes())
        return digest.digest()

    def reset_cache(self) -> None:
        """Zero the whole pool. Only safe when no sequence holds a block."""
        self._keys.fill(0.0)
        self._values.fill(0.0)

    def _slots(self, entry: BatchEntry) -> list[int]:
        block_size = self.block_size
        slots = []
        for position in entry.positions:
            block_index = position // block_size
            if block_index >= len(entry.block_table):
                raise ValueError(
                    f"sequence {entry.seq_id}: position {position} has no block in the table"
                )
            slots.append(entry.block_table[block_index] * block_size + position % block_size)
        return slots

    def execute(self, batch: ExecutionBatch) -> dict[str, FloatArray]:
        """Run one iteration and return logits for the entries that asked for them.

        Keys and values for every token in the batch are written into the pool as a side effect,
        so the caller must have allocated the blocks listed in each entry's block table first.
        """
        if not batch.entries:
            return {}
        config = self.model.config
        tokens = np.fromiter(
            (token for entry in batch.entries for token in entry.token_ids),
            dtype=np.int64,
            count=batch.num_tokens,
        )
        if np.any(tokens < 0) or np.any(tokens >= config.vocab_size):
            raise ValueError("token id outside the vocabulary")
        positions = np.fromiter(
            (position for entry in batch.entries for position in entry.positions),
            dtype=np.int64,
            count=batch.num_tokens,
        )
        if np.any(positions >= config.max_seq_len):
            raise ValueError("position beyond max_seq_len")
        slots = np.asarray(
            [slot for entry in batch.entries for slot in self._slots(entry)], dtype=np.int64
        )
        block_tables = [np.asarray(entry.block_table, dtype=np.int64) for entry in batch.entries]

        hidden = self.model.embed(tokens)
        total = batch.num_tokens
        scale = self.model.attention_scale
        for index, layer in enumerate(self.model.weights.layers):
            queries, keys, values = self.model.project_qkv(hidden, layer, positions)
            key_pool, value_pool = self._keys[index], self._values[index]
            key_pool.reshape(-1, config.n_heads, config.head_dim)[slots] = keys
            value_pool.reshape(-1, config.n_heads, config.head_dim)[slots] = values

            attended = np.empty_like(queries)
            offset = 0
            for entry, table in zip(batch.entries, block_tables, strict=True):
                stop = offset + entry.num_tokens
                attended[offset:stop] = paged_attention(
                    queries[offset:stop],
                    positions[offset:stop],
                    key_pool,
                    value_pool,
                    table,
                    entry.context_len,
                    scale,
                    tile_blocks=self.tile_blocks,
                )
                offset = stop
            hidden = hidden + attended.reshape(total, config.d_model) @ layer.out_proj
            hidden = (hidden + self.model.feed_forward(hidden, layer)).astype(np.float32)

        wanted: list[int] = []
        names: list[str] = []
        offset = 0
        for entry in batch.entries:
            if entry.logits_index is not None:
                wanted.append(offset + entry.logits_index)
                names.append(entry.seq_id)
            offset += entry.num_tokens
        if not wanted:
            return {}
        logits = self.model.logits_from_hidden(hidden[wanted])
        return dict(zip(names, logits, strict=True))
