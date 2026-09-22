"""A byte-level BPE tokenizer trained in-repo, with no model or vocabulary download.

Byte level means every possible input already has a representation: ids 0..255 are the raw bytes,
so encoding never fails and never needs an unknown token, whatever the language or the mojibake.
Merges are learned on top (Sennrich, Haddow and Birch, "Neural Machine Translation of Rare Words
with Subword Units", 2016, applied to bytes as in GPT-2), and the vocabulary is serialised as
plain JSON -- a list of merge pairs and a table of special tokens. Nothing in the format can
execute code on load, which is the same rule the weights follow.

Training splits text into whitespace-delimited chunks first, so no merge can ever span a space.
That bounds the work per token and keeps the learned vocabulary interpretable. Ties between
equally frequent pairs are broken by the pair itself, so training is deterministic and two runs
over the same corpus produce byte-identical vocabularies.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from paged_kv_serving.core.fileio import atomic_write_text

__all__ = ["BOS_TOKEN", "EOS_TOKEN", "PAD_TOKEN", "ByteBPETokenizer"]

# Special token strings, not credentials (ruff's S105 and bandit's B105 match the word TOKEN).
BOS_TOKEN = "<|bos|>"  # noqa: S105  # nosec B105
EOS_TOKEN = "<|eos|>"  # noqa: S105  # nosec B105
PAD_TOKEN = "<|pad|>"  # noqa: S105  # nosec B105
_DEFAULT_SPECIALS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)
_BYTE_TOKENS = 256
_CHUNK = re.compile(rb"\s*\S+|\s+")
_SCHEMA_VERSION = 1


class ByteBPETokenizer:
    """Byte fallback plus learned merges. Encoding is total; decoding is lossless."""

    __slots__ = ("_ranks", "_specials", "_vocab", "merges")

    def __init__(
        self,
        merges: Sequence[tuple[int, int]] = (),
        specials: Mapping[str, int] | None = None,
    ) -> None:
        special_map = dict(specials) if specials is not None else _default_specials()
        for name, token_id in special_map.items():
            if not _BYTE_TOKENS <= token_id < _BYTE_TOKENS + len(special_map):
                raise ValueError(f"special token {name} has an out-of-range id {token_id}")
        if len(set(special_map.values())) != len(special_map):
            raise ValueError("special token ids must be unique")
        self._specials = special_map
        self.merges = tuple((int(a), int(b)) for a, b in merges)
        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        base = _BYTE_TOKENS + len(special_map)
        vocab: list[bytes] = [bytes([index]) for index in range(_BYTE_TOKENS)]
        vocab.extend(b"" for _ in special_map)
        for left, right in self.merges:
            if left >= len(vocab) or right >= len(vocab):
                raise ValueError("a merge refers to a token that does not exist yet")
            vocab.append(vocab[left] + vocab[right])
        self._vocab = vocab
        if len(vocab) != base + len(self.merges):  # pragma: no cover - guarded above
            raise ValueError("inconsistent vocabulary")

    # --- vocabulary ----------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def specials(self) -> Mapping[str, int]:
        return dict(self._specials)

    @property
    def eos_token_id(self) -> int:
        return self._specials[EOS_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self._specials[BOS_TOKEN]

    def token_bytes(self, token_id: int) -> bytes:
        if not 0 <= token_id < len(self._vocab):
            raise ValueError(f"token id {token_id} is outside the vocabulary")
        return self._vocab[token_id]

    # --- encoding ------------------------------------------------------------------------

    def _merge_chunk(self, ids: list[int]) -> list[int]:
        """Apply merges to one chunk, always taking the lowest-ranked applicable pair."""
        while len(ids) > 1:
            best_rank: int | None = None
            best_index = -1
            for index in range(len(ids) - 1):
                rank = self._ranks.get((ids[index], ids[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, index
            if best_rank is None:
                break
            merged = _BYTE_TOKENS + len(self._specials) + best_rank
            ids[best_index : best_index + 2] = [merged]
        return ids

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to token ids. Any string encodes; no input is ever rejected."""
        ids: list[int] = [self.bos_token_id] if add_bos else []
        for match in _CHUNK.finditer(text.encode("utf-8")):
            ids.extend(self._merge_chunk(list(match.group())))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        """Decode ids back to text. Invalid UTF-8 from a partial stream is replaced, not raised."""
        special_ids = set(self._specials.values())
        payload = b"".join(
            self.token_bytes(int(token_id))
            for token_id in token_ids
            if not (skip_special and int(token_id) in special_ids)
        )
        return payload.decode("utf-8", errors="replace")

    # --- training ------------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        *,
        vocab_size: int,
        specials: Sequence[str] = _DEFAULT_SPECIALS,
    ) -> ByteBPETokenizer:
        """Learn merges until the vocabulary reaches ``vocab_size``.

        Training stops early if no pair occurs more than once, so a tiny corpus produces a small
        vocabulary rather than merges that only fit noise.
        """
        special_map = {name: _BYTE_TOKENS + index for index, name in enumerate(specials)}
        base = _BYTE_TOKENS + len(special_map)
        if vocab_size < base:
            raise ValueError(f"vocab_size must be at least {base} for byte and special tokens")
        chunks: Counter[bytes] = Counter()
        for text in corpus:
            for match in _CHUNK.finditer(text.encode("utf-8")):
                chunks[match.group()] += 1
        if not chunks:
            raise ValueError("the training corpus is empty")
        words: list[tuple[list[int], int]] = [
            (list(chunk), count) for chunk, count in sorted(chunks.items())
        ]
        merges: list[tuple[int, int]] = []
        while base + len(merges) < vocab_size:
            pairs: Counter[tuple[int, int]] = Counter()
            for ids, count in words:
                for index in range(len(ids) - 1):
                    pairs[ids[index], ids[index + 1]] += count
            if not pairs:
                break
            # Deterministic: highest count, then the smallest pair.
            best = min(pairs.items(), key=lambda item: (-item[1], item[0]))[0]
            if pairs[best] < 2:
                break
            new_id = base + len(merges)
            merges.append(best)
            for ids, _ in words:
                _apply_merge(ids, best, new_id)
        return cls(merges, special_map)

    # --- serialisation -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "specials": dict(self._specials),
            "merges": [[left, right] for left, right in self.merges],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ByteBPETokenizer:
        version = data.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported tokenizer schema version {version!r}")
        merges = [(int(pair[0]), int(pair[1])) for pair in data["merges"]]
        specials = {str(name): int(value) for name, value in data["specials"].items()}
        return cls(merges, specials)

    def save(self, path: Path) -> None:
        atomic_write_text(path, json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> ByteBPETokenizer:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _default_specials() -> dict[str, int]:
    return {name: _BYTE_TOKENS + index for index, name in enumerate(_DEFAULT_SPECIALS)}


def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> None:
    index = 0
    while index < len(ids) - 1:
        if (ids[index], ids[index + 1]) == pair:
            ids[index : index + 2] = [new_id]
        else:
            index += 1
