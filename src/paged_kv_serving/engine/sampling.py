"""Token sampling: greedy, temperature and nucleus (top-p), with a per-request generator.

Every sequence carries its own ``numpy.random.Generator`` seeded from the request, and sampling
draws exactly one uniform per token whatever the vocabulary size. Two consequences matter for a
serving engine:

* output depends only on ``(seed, sample index)``, never on which other requests happened to be
  in the same batch, so a reproducibility complaint can actually be reproduced;
* replaying a request after a preemption-and-recompute cannot silently diverge, because the
  recomputed prefill consumes no randomness at all.

Nucleus sampling follows Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020):
keep the shortest set of most probable tokens whose mass reaches ``top_p``, renormalise, sample.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from paged_kv_serving.engine.sequence import SamplingParams

__all__ = ["greedy_token", "sample_token", "top_p_filter"]

FloatArray = npt.NDArray[np.float32]


def greedy_token(logits: FloatArray) -> int:
    """Highest-scoring token id; ties resolve to the lowest id, so it is fully deterministic."""
    return int(np.argmax(logits))


def top_p_filter(probabilities: FloatArray, top_p: float) -> FloatArray:
    """Zero every token outside the nucleus and renormalise.

    The token that crosses the threshold is kept, so the nucleus always holds at least one token
    even when a single token already carries more than ``top_p`` of the mass.
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_p == 1.0:
        return probabilities
    order = np.argsort(-probabilities, kind="stable")
    ordered = probabilities[order]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    # Number of tokens up to and including the one that reaches top_p.
    keep = int(np.searchsorted(cumulative, top_p, side="left")) + 1
    filtered = np.zeros_like(probabilities)
    filtered[order[:keep]] = ordered[:keep]
    total = float(filtered.sum())
    if total <= 0.0:  # pragma: no cover - only reachable with a degenerate distribution
        return probabilities
    return (filtered / total).astype(np.float32)


def sample_token(logits: FloatArray, params: SamplingParams, rng: np.random.Generator) -> int:
    """Draw one token id from ``logits`` under ``params``.

    Greedy decoding consumes no randomness, which keeps a greedy request bit-identical whether or
    not it was ever preempted.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a 1-D array over the vocabulary")
    if params.greedy:
        return greedy_token(logits)
    scaled = (logits.astype(np.float32) / np.float32(params.temperature)).astype(np.float32)
    shifted = scaled - np.max(scaled)
    exponentiated = np.exp(shifted.astype(np.float64))
    probabilities = (exponentiated / exponentiated.sum()).astype(np.float32)
    probabilities = top_p_filter(probabilities, params.top_p)
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    # One uniform draw per token, independent of the vocabulary size.
    draw = float(rng.random()) * float(cumulative[-1])
    return int(np.searchsorted(cumulative, draw, side="right").clip(0, probabilities.size - 1))
