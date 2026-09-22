"""Compare the paged, chunked path against a cache-free recompute.

This is the repository's central correctness claim, so it lives in the package rather than in the
tests: the demo prints it, the benchmark tracks it against a threshold, and the property tests
drive it with generated token sequences and generated chunk splits.

:func:`paged_chunked_logits` runs a sequence through the real :class:`ModelRunner` -- real block
tables, real slot writes, real online-softmax attention over blocks -- split into arbitrary chunks
whose boundaries need not line up with block boundaries. :meth:`LanguageModel.reference_forward`
runs the same tokens in one dense pass with no cache at all. The two use different reduction
orders, so they agree to float32 rounding, and :func:`max_logit_gap` reports by how much.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.runner import BatchEntry, ExecutionBatch, ModelRunner
from paged_kv_serving.engine.sequence import Request, new_sequence_id

__all__ = [
    "LogitRecord",
    "RecordingExecutor",
    "engine_logit_gap",
    "max_logit_gap",
    "paged_chunked_logits",
    "sequence_tokens",
    "split_sizes",
]

FloatArray = npt.NDArray[np.float32]


def split_sizes(total: int, chunk: int) -> list[int]:
    """Split ``total`` tokens into chunks of at most ``chunk``, the last one possibly shorter."""
    if total < 1 or chunk < 1:
        raise ValueError("total and chunk must be >= 1")
    sizes = [chunk] * (total // chunk)
    if total % chunk:
        sizes.append(total % chunk)
    return sizes


def paged_chunked_logits(
    model: LanguageModel,
    token_ids: Sequence[int],
    *,
    block_size: int = 16,
    chunks: Sequence[int] | None = None,
    block_order: Sequence[int] | None = None,
) -> FloatArray:
    """Logits at the last position, computed through the paged cache in ``chunks``.

    ``block_order`` lets a caller hand out deliberately scattered physical blocks -- reversed,
    shuffled, interleaved with another sequence's -- which is the situation a block table exists
    to handle and the one a contiguous cache silently gets away with.
    """
    tokens = list(token_ids)
    if not tokens:
        raise ValueError("token_ids must not be empty")
    sizes = list(chunks) if chunks is not None else [len(tokens)]
    if sum(sizes) != len(tokens) or any(size < 1 for size in sizes):
        raise ValueError("chunks must be positive and sum to the number of tokens")
    num_blocks = -(-len(tokens) // block_size)
    table = tuple(block_order) if block_order is not None else tuple(range(num_blocks))
    if len(table) < num_blocks or len(set(table)) != len(table):
        raise ValueError("block_order must list at least one distinct block per block of tokens")
    runner = ModelRunner(model, num_blocks=max(table) + 1, block_size=block_size)

    start = 0
    logits: FloatArray | None = None
    for size in sizes:
        stop = start + size
        entry = BatchEntry(
            seq_id="equiv",
            token_ids=tuple(tokens[start:stop]),
            positions=tuple(range(start, stop)),
            block_table=table,
            logits_index=size - 1 if stop == len(tokens) else None,
        )
        result = runner.execute(ExecutionBatch((entry,)))
        if "equiv" in result:
            logits = result["equiv"]
        start = stop
    if logits is None:  # pragma: no cover - the last chunk always asks for logits
        raise RuntimeError("no chunk produced logits")
    return logits


def max_logit_gap(
    model: LanguageModel,
    token_ids: Sequence[int],
    *,
    block_size: int = 16,
    chunks: Sequence[int] | None = None,
    block_order: Sequence[int] | None = None,
) -> float:
    """Largest absolute difference between the paged logits and the cache-free reference."""
    paged = paged_chunked_logits(
        model, token_ids, block_size=block_size, chunks=chunks, block_order=block_order
    )
    reference = model.reference_forward(list(token_ids))[-1]
    return float(np.max(np.abs(paged - reference)))


@dataclass(frozen=True, slots=True)
class LogitRecord:
    """Logits the engine produced for one sequence at one context length."""

    seq_id: str
    context_len: int
    logits: FloatArray


@dataclass(slots=True)
class RecordingExecutor:
    """Executes on the engine's own runner and keeps a copy of every logits row it returns.

    Installed as ``engine.executor`` after construction, it sees exactly the batches the
    scheduler built -- mixed prefill and decode, chunk boundaries, recomputation after
    preemption, reads through shared and cached blocks -- so the rows it records are what the
    engine actually sampled from.
    """

    runner: ModelRunner
    records: list[LogitRecord] = field(default_factory=list)

    def execute(self, batch: ExecutionBatch) -> dict[str, FloatArray]:
        result = self.runner.execute(batch)
        for entry in batch.entries:
            row = result.get(entry.seq_id)
            if row is not None:
                self.records.append(LogitRecord(entry.seq_id, entry.context_len, row.copy()))
        return result


def sequence_tokens(
    requests: Iterable[Request], outputs: Mapping[str, Sequence[Sequence[int]]]
) -> dict[str, list[int]]:
    """Full token list (prompt plus output) of every finished sample, keyed by sequence id.

    ``outputs`` maps a request id to the output tokens of each of its samples in order.
    """
    tokens: dict[str, list[int]] = {}
    for request in requests:
        samples = outputs.get(request.request_id)
        if samples is None:
            continue
        for index, produced in enumerate(samples):
            tokens[new_sequence_id(request.request_id, index)] = [
                *request.prompt_token_ids,
                *produced,
            ]
    return tokens


def engine_logit_gap(
    model: LanguageModel, records: Iterable[LogitRecord], tokens: Mapping[str, Sequence[int]]
) -> tuple[float, int]:
    """Largest gap between recorded engine logits and a cache-free recompute, and rows checked.

    Row ``context_len - 1`` of ``reference_forward(full_tokens)`` is the reference for a record:
    the dense pass is causal, so one recompute per sequence covers every step it took. Records of
    sequences with no finished output (aborted requests) are skipped.
    """
    references: dict[str, FloatArray] = {}
    worst = 0.0
    checked = 0
    for record in records:
        full = tokens.get(record.seq_id)
        if full is None:
            continue
        if record.context_len > len(full):
            raise ValueError(f"{record.seq_id}: record beyond the sequence it belongs to")
        reference = references.get(record.seq_id)
        if reference is None:
            reference = model.reference_forward(list(full))
            references[record.seq_id] = reference
        gap = float(np.max(np.abs(reference[record.context_len - 1] - record.logits)))
        worst = max(worst, gap)
        checked += 1
    return worst, checked
