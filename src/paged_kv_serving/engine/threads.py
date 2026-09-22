"""Pin the BLAS thread pool, because this engine's matrices are too small to share.

Every matrix multiplication here is tiny: a decode iteration multiplies a few dozen rows by a
128 x 384 projection. A multithreaded BLAS wakes its worker threads for each call anyway, and on
a small matrix the wake-up and join cost more than the arithmetic. On the development machine
(6 cores, OpenBLAS via numpy) a single-threaded pool cut the profiled per-iteration intercept
from about 13 ms to about 1 ms and the per-token slope by an order of magnitude
(docs/adr/0006-single-threaded-blas.md). Serving concurrency comes from batching inside one
iteration, not from BLAS threads.

The limit is applied through ``threadpoolctl`` at process start by the CLI, the benchmarks and
the test suite, and is configurable (``APP_ENGINE_BLAS_THREADS``) for anyone running a model large
enough for threads to pay off.
"""

from __future__ import annotations

from typing import Any

# Imported for its side effect: numpy loads the BLAS library, and threadpoolctl can only limit a
# library that is already loaded.
import numpy  # noqa: F401
from threadpoolctl import threadpool_info, threadpool_limits

__all__ = ["blas_thread_counts", "limit_blas_threads"]

_LIMITERS: list[Any] = []


def limit_blas_threads(threads: int | None = 1) -> dict[str, int]:
    """Limit every loaded BLAS library to ``threads`` threads. ``None`` leaves them alone.

    Returns the resulting thread count per library, for logging and for the benchmark header.
    The limiter object is kept alive for the life of the process: the limit is process-wide on
    purpose, since the engine runs one forward pass at a time.
    """
    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be >= 1")
        _LIMITERS.append(threadpool_limits(limits=threads, user_api="blas"))
    return blas_thread_counts()


def blas_thread_counts() -> dict[str, int]:
    """Current thread count of each loaded BLAS library, keyed by library name."""
    return {
        str(entry.get("internal_api", "blas")): int(entry.get("num_threads", 0))
        for entry in threadpool_info()
        if entry.get("user_api") == "blas"
    }
