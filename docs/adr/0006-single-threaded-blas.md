# 6. Pin BLAS to one thread per process

- Status: accepted
- Date: 2026-09-22

## Context

The engine is float32 NumPy, and every matrix multiplication is small: a decode iteration
multiplies a few dozen rows by a 128 x 384 projection. NumPy's bundled OpenBLAS starts one thread
per core by default (six on the development machine) and wakes them for every call.

Profiling the demo workload with and without a thread limit, on the same machine and checkpoint:

| BLAS threads | fitted iteration intercept | fitted per-token slope | open-loop run (32 requests) |
|---|---:|---:|---:|
| 6 (default) | 13.3 ms | 0.55 ms | 1.37 s |
| 1 | 1.1 ms | 0.04 ms | 0.72 s |

With the default pool, thread wake-up and join cost more than the arithmetic, and the static
baseline (a few large padded batches) looked faster than continuous batching (many small ones)
for reasons that had nothing to do with scheduling.

## Decision

`engine/threads.py` limits every loaded BLAS library through `threadpoolctl`. The CLI applies
`APP_ENGINE_BLAS_THREADS` (default 1) before loading the checkpoint; the demo, both benchmarks and
the test suite pin to one thread. Serving concurrency comes from batching inside an iteration,
not from BLAS threads.

## Consequences

- Both schedulers in the comparison run under the same limit, so the ratios in the README
  compare scheduling, not thread pools. Benchmark results record the limit in their `machine`
  block.
- A larger model, where one matrix multiplication is big enough to split, would want more
  threads; the setting stays configurable, and an empty value leaves BLAS alone.
- `threadpoolctl` is one more runtime dependency (pure Python, no transitive dependencies).
