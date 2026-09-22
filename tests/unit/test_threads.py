"""BLAS thread pinning."""

from __future__ import annotations

import pytest

from paged_kv_serving.engine.threads import blas_thread_counts, limit_blas_threads

pytestmark = pytest.mark.unit


def test_blas_is_pinned_to_one_thread_for_the_suite() -> None:
    counts = limit_blas_threads(1)
    assert counts
    assert set(counts.values()) == {1}
    assert blas_thread_counts() == counts


def test_none_leaves_blas_alone_and_zero_is_refused() -> None:
    assert limit_blas_threads(None) == blas_thread_counts()
    with pytest.raises(ValueError, match="threads must be"):
        limit_blas_threads(0)
