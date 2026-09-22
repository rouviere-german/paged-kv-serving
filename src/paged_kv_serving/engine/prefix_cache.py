"""Block-level prefix cache keyed by a hash chain over full blocks.

Two requests that start with the same tokens produce the same keys and values for those tokens,
so the KV of a shared prefix only has to be computed once. The cache keys each *full* block by

    h_0 = H(scope)                      h_i = H(h_{i-1} || block_size || tokens of block i)

so a key commits to the entire token history in front of it, not just to the block's own tokens.
A lookup therefore cannot match block 3 of one prompt against block 3 of a different prompt that
happens to share those 16 tokens. Partial blocks are never cached: their contents still change.

Reuse is scoped. With ``scope="per_key"`` the chain starts from the caller's key id, so prefixes
are shared only inside one tenant. Global reuse would leak a timing oracle -- a caller could learn
that *somebody else* has already sent a given prompt by measuring its own time to first token --
and that is a poor trade for a cache that is most useful on a tenant's own system prompt anyway
(see docs/adr/0003-per-key-prefix-cache-scope.md).

Entries are evicted least-recently-used, and only when the cache holds the block's last reference:
a block that some live sequence is reading is never taken away from it.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from paged_kv_serving.engine.blocks import BlockManager

__all__ = [
    "CacheScope",
    "PrefixCache",
    "PrefixCacheStats",
    "block_hash_chain",
    "chain_seed",
    "next_block_hash",
]

CacheScope = Literal["per_key", "global"]

_DOMAIN = b"paged-kv-serving/prefix-v1"
_DIGEST_BYTES = 16


def chain_seed(scope: bytes = b"") -> bytes:
    """Start of a hash chain. Distinct scopes can never produce a common prefix."""
    return hashlib.blake2b(_DOMAIN + b"|" + scope, digest_size=_DIGEST_BYTES).digest()


def next_block_hash(parent: bytes, token_ids: Sequence[int], block_size: int) -> bytes:
    """Extend a chain by one full block of exactly ``block_size`` tokens."""
    if len(token_ids) != block_size:
        raise ValueError("only full blocks are hashed and cached")
    payload = b"".join(int(token).to_bytes(4, "little", signed=False) for token in token_ids)
    return hashlib.blake2b(
        parent + block_size.to_bytes(4, "little") + payload, digest_size=_DIGEST_BYTES
    ).digest()


def block_hash_chain(
    token_ids: Sequence[int], *, block_size: int, num_blocks: int, scope: bytes = b""
) -> list[bytes]:
    """Return the chained hash of each of the first ``num_blocks`` full blocks of ``token_ids``.

    Raises if fewer than ``num_blocks`` full blocks are available, because a caller asking for a
    hash of tokens that do not exist yet would silently key a block by a shorter prefix.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if num_blocks < 0:
        raise ValueError("num_blocks must be >= 0")
    if num_blocks * block_size > len(token_ids):
        raise ValueError("not enough tokens for the requested number of full blocks")
    parent = chain_seed(scope)
    hashes: list[bytes] = []
    for index in range(num_blocks):
        parent = next_block_hash(
            parent, token_ids[index * block_size : (index + 1) * block_size], block_size
        )
        hashes.append(parent)
    return hashes


@dataclass(slots=True)
class PrefixCacheStats:
    """Counters over the lifetime of the cache."""

    queried_blocks: int = 0
    hit_blocks: int = 0
    registered_blocks: int = 0
    evicted_blocks: int = 0

    @property
    def hit_ratio(self) -> float:
        return self.hit_blocks / self.queried_blocks if self.queried_blocks else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "queried_blocks": float(self.queried_blocks),
            "hit_blocks": float(self.hit_blocks),
            "registered_blocks": float(self.registered_blocks),
            "evicted_blocks": float(self.evicted_blocks),
            "hit_ratio": self.hit_ratio,
        }


class PrefixCache:
    """LRU map from chained block hash to physical block id, holding one reference per entry."""

    __slots__ = ("_entries", "_manager", "enabled", "scope", "stats")

    def __init__(
        self, manager: BlockManager, *, enabled: bool = True, scope: CacheScope = "per_key"
    ) -> None:
        self._manager = manager
        self._entries: OrderedDict[bytes, int] = OrderedDict()
        self.enabled = enabled
        self.scope = scope
        self.stats = PrefixCacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    def scope_bytes(self, key_id: str) -> bytes:
        """The chain seed for a caller: its key id, or a constant for a global cache."""
        return key_id.encode("utf-8") if self.scope == "per_key" else b"shared"

    def lookup(self, hashes: Sequence[bytes]) -> list[int]:
        """Return block ids for the longest matching *prefix* of ``hashes``, increfed.

        Matching stops at the first miss: block 5 is only usable when blocks 0..4 were also hits,
        since a sequence's block table has to be contiguous from position zero. Every returned
        block gains a reference that the caller owns and must eventually free.
        """
        self.stats.queried_blocks += len(hashes)
        if not self.enabled:
            return []
        matched: list[int] = []
        for block_hash in hashes:
            block_id = self._entries.get(block_hash)
            if block_id is None:
                break
            self._entries.move_to_end(block_hash)
            self._manager.incref(block_id)
            matched.append(block_id)
        self.stats.hit_blocks += len(matched)
        return matched

    def register(self, block_hash: bytes, block_id: int) -> bool:
        """Cache a newly filled block. Returns False when the hash is already cached.

        The cache takes its own reference, so the block survives the sequence that produced it.
        """
        if not self.enabled or block_hash in self._entries:
            return False
        self._manager.incref(block_id)
        self._entries[block_hash] = block_id
        self.stats.registered_blocks += 1
        return True

    def evict(self, count: int = 1) -> int:
        """Evict up to ``count`` least-recently-used entries the cache holds alone.

        Entries whose block is still referenced by a live sequence are skipped, not dropped: the
        sequence is reading from it. Returns the number of blocks actually released.
        """
        if count <= 0:
            return 0
        released = 0
        for block_hash in list(self._entries):
            if released >= count:
                break
            block_id = self._entries[block_hash]
            if self._manager.ref_count(block_id) > 1:
                continue
            del self._entries[block_hash]
            self._manager.free(block_id)
            released += 1
        self.stats.evicted_blocks += released
        return released

    def evict_until_free(self, needed: int) -> int:
        """Evict just enough entries for ``needed`` free blocks. Returns blocks released."""
        shortfall = needed - self._manager.num_free_blocks
        return self.evict(shortfall) if shortfall > 0 else 0

    def flush(self) -> int:
        """Drop every evictable entry. Returns the number of blocks released."""
        return self.evict(len(self._entries))

    def cached_block_ids(self) -> tuple[int, ...]:
        """Block ids currently cached, in least-recently-used order."""
        return tuple(self._entries.values())
