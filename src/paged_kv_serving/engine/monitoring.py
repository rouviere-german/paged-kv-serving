"""Input drift monitoring for a serving engine.

The model here is fixed and random, so accuracy drift is meaningless. What is not meaningless is
*workload* drift: the distribution of prompt lengths drives block demand, prefix-cache hit rate
and the latency model that admission control depends on. A deployment whose prompts quietly grow
from 40 to 400 tokens keeps returning 200s while its admission decisions become fiction.

The monitor keeps a reference sample of prompt lengths and a rolling window of recent ones, and
reports the Population Stability Index between them. The reference is either supplied (a sample
from the capacity-planning workload) or learned: :meth:`PromptLengthMonitor.learning` freezes the
first ``reference_size`` prompts a server sees as its reference, which is what the API wires up,
because "the traffic this deployment was sized for" is the traffic it saw first (the PSI
implementation is the shared one in :mod:`paged_kv_serving.ml.drift`). The usual reading applies:
below 0.1 stable, 0.1 to 0.2 moderate, above 0.2 a shift worth alerting on.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from paged_kv_serving.ml.drift import psi

__all__ = ["PromptLengthMonitor"]


class PromptLengthMonitor:
    """Rolling PSI of observed prompt lengths against a fixed reference sample."""

    __slots__ = ("_learn", "_recent", "_reference", "bins", "min_samples", "threshold")

    def __init__(
        self,
        reference: Iterable[int],
        *,
        window: int = 512,
        bins: int = 10,
        threshold: float = 0.2,
        min_samples: int = 32,
    ) -> None:
        reference_list = [int(value) for value in reference]
        if len(reference_list) < 2:
            raise ValueError("the reference needs at least two prompt lengths")
        if window < 2 or bins < 2 or min_samples < 2:
            raise ValueError("window, bins and min_samples must be >= 2")
        self._reference = reference_list
        self._learn = 0
        self._recent: deque[int] = deque(maxlen=window)
        self.bins = bins
        self.threshold = threshold
        self.min_samples = min_samples

    @classmethod
    def learning(
        cls,
        reference_size: int,
        *,
        window: int = 512,
        bins: int = 10,
        threshold: float = 0.2,
        min_samples: int = 32,
    ) -> PromptLengthMonitor:
        """A monitor whose reference is the first ``reference_size`` prompts it observes."""
        if reference_size < 2:
            raise ValueError("the reference needs at least two prompt lengths")
        monitor = cls(
            [0, 0], window=window, bins=bins, threshold=threshold, min_samples=min_samples
        )
        monitor._reference = []
        monitor._learn = reference_size
        return monitor

    def __len__(self) -> int:
        return len(self._recent)

    @property
    def reference_size(self) -> int:
        return len(self._reference)

    @property
    def reference_ready(self) -> bool:
        """False while a learning monitor is still collecting its reference."""
        return len(self._reference) >= max(2, self._learn)

    def observe(self, prompt_length: int) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be >= 0")
        if not self.reference_ready:
            self._reference.append(int(prompt_length))
            return
        self._recent.append(int(prompt_length))

    def score(self) -> float:
        """PSI of the rolling window, or 0.0 until the reference and ``min_samples`` exist."""
        if not self.reference_ready or len(self._recent) < self.min_samples:
            return 0.0
        return psi(self._reference, list(self._recent), bins=self.bins)

    def drifted(self) -> bool:
        return self.score() > self.threshold
