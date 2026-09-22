# Scheduler and memory manager

How one iteration of `Engine.step()` is built, what the KV pool looks like while it runs, and which
invariants the tests hold the code to. File references are under `src/paged_kv_serving/engine/`.

## One bookkeeping rule

Every sequence (`sequence.py`) carries `token_ids` (prompt plus everything sampled so far) and
`num_computed_tokens` (the leading tokens whose keys and values are in the KV cache). An iteration
runs `token_ids[num_computed : num_computed + chunk]` at those absolute positions, and produces a
next token exactly when the chunk reaches the end of `token_ids`.

Prefill (a chunk of many), chunked prefill (a chunk of some), decode (a chunk of one) and
recomputation after preemption (a chunk over prompt *and* previously generated tokens) are all the
same operation with different chunk sizes. That is why the scheduler can mix them in one batch.

## The KV pool

```
runner._keys / runner._values : float32[n_layers, num_blocks, block_size, n_heads, head_dim]

sequence A  block_table = [7, 2, 9]        tokens 0..15 -> block 7, 16..31 -> block 2, 32.. -> block 9
sequence B  block_table = [7, 2, 4]        shares A's first 32 tokens (prefix cache or fork)

BlockManager: ref_count[7] = 3 (A, B, prefix cache)   ref_count[9] = 1   free list = FIFO deque
```

- **Allocation** (`blocks.py`) is a FIFO free list with reference counts. `allocate_many` is all or
  nothing. `check_invariants` verifies that the free list holds exactly the zero-count blocks with
  no duplicates and that free plus referenced covers the pool (the leak checker).
- **Copy-on-write**: `prepare_for_write` on a block with more than one owner allocates a fresh
  block, asks the `BlockStore` (the runner) to copy the bytes across all layers, drops the
  caller's reference to the original and returns the new id. `KVCacheManager.reserve` counts the
  copies and the new blocks a write needs *before* touching anything, so a reservation either
  succeeds completely or leaves every table and reference count unchanged.
- **Prefix cache** (`prefix_cache.py`): each *full* block is keyed by
  `h_i = BLAKE2b(h_{i-1} || block_size || tokens of block i)` with `h_0` derived from the caller's
  key id (ADR 0003). The cache holds one reference per entry. Lookup returns the longest matching
  *prefix* of a new prompt's chain (always leaving at least one token to compute, so the request
  still gets logits). Eviction is LRU and skips entries a live sequence is reading.
- **Fork** (`kvcache.py`): the samples of an `n > 1` request share the prompt's blocks by
  reference; each gets a private copy of the partially filled last block on its first write.

## Paged attention

`attention.py` computes causal attention of a slice's queries over the first `context_len` cached
tokens by walking the block table in tiles of `tile_blocks` blocks and folding each tile into a
running result with the online-softmax recurrence (Milakov and Gimelshein, 2018). Masking is by
absolute position, so the same kernel serves a single decode query and a mid-prompt prefill chunk.
The runner (`runner.py`) runs everything that does not depend on history (embedding, QKV,
output projection, feed-forward, logits) once over the flattened token axis of the whole ragged
batch; only attention consults per-sequence block tables.

## One iteration

```mermaid
flowchart TD
    start([schedule]) --> d{running sequence<br/>with one uncomputed token?}
    d -- yes, in priority order --> r[reserve 1 token<br/>COW if the last block is shared]
    r -- pool exhausted --> v[preempt lowest-priority<br/>unscheduled sequence]
    v --> d
    r -- ok --> d
    d -- none left / budget 0 --> o[one pass over running prefills and waiters<br/>class, then deadline, then arrival]
    o --> k{next in order}
    k -- running prefill --> c[next chunk = min budget, remaining, memory, allowance<br/>no block at all: preempt the lowest-priority<br/>running prefill behind it]
    k -- waiter, admission open --> w[admit with its first chunk if a running slot<br/>forks included, memory and allowance remain;<br/>otherwise admission closes for this iteration]
    c --> k
    w --> k
    k -- budget 0 / end --> x([ExecutionBatch])
    x --> run[runner.execute: write K/V into slots, paged attention, logits for slices that end]
    run --> s[sample with the sequence's own RNG<br/>fork pending samples on the first token]
    s --> pub[publish newly full blocks to the prefix cache]
    pub --> fin[free finished sequences, record metrics]
```

1. **Decode first.** Every running sequence that has consumed its prompt gets one token, in
   priority order (SLO class, then earliest deadline, then arrival). If a reservation fails, the
   lowest-priority running sequence *not yet scheduled in this iteration* is preempted -- a decoder
   still waiting or a sequence part-way through prefill -- and the reservation is retried. A
   sequence that is the last one standing preempts itself, which releases its blocks for others.
2. **Prefill, in one priority order.** Running sequences whose prompt is incomplete and waiting
   sequences share one pass ordered by class, deadline and arrival -- the order admission control
   assumes. A running prompt takes its next chunk (Sarathi-Serve style chunked prefill, Agrawal et
   al., OSDI 2024), limited by the budget, free plus evictable blocks and the allowance; if it
   cannot get a single block it preempts the lowest-priority running prefill behind it. A waiter is
   admitted with its first chunk while a running slot (counting the slots its future forks will
   need), blocks above a small watermark and allowance remain. Admission closes at the first waiter
   that cannot be admitted, so a lower-priority request never jumps a higher-priority one waiting
   for memory, and for the rest of any iteration that preempted; running prefills behind it still
   continue, because they already hold blocks and finishing them is what frees memory.

   Continuing every running prefill before admitting anybody (the original order) let a prompt with
   a loose deadline that happened to start first take the whole budget ahead of a tight one that
   arrived an iteration later -- admitted on a promise the scheduler then broke (review
   counterexample, now `tests/integration/test_deadlines.py::test_an_earlier_deadline_arriving_mid_prefill_is_served_first`).

While any higher-class sequence is running or waiting, all lower-class prefill in phase 2 shares an
allowance of `lower_class_prefill_share * max_batched_tokens` tokens (ADR 0005). Decode is never
capped.

**Liveness.** No iteration is empty while work is queued. A decoder or a running prefill that
cannot get a block preempts somebody behind it; a waiter that is not admitted releases the cached
blocks it matched, so it pins nothing; and the watermark is ignored while nothing is running.
`EngineConfig` guarantees that one maximum-length sequence fits the pool, so the highest-priority
sequence can always run. Two long prompts that each held part of the pool used to leave every
later iteration empty until their requests timed out; the service loop now also backs off (and
logs `engine_stalled`) instead of spinning with the lock held if an empty iteration ever happens.

**Parallel samples.** The first sample of an `n`-way request is admitted only when `n` running
slots are free, and holds `n - 1` of them until it forks, so the running set -- and with it the
decode budget admission control reserves -- never exceeds `max_running_seqs`. A request with `n`
above that cap is refused up front (400 `too_many_samples`).

## Admission control (before step 1)

`Engine.submit` rejects, before a single block is allocated: prompts, `max_new_tokens` or `n` over
the engine limits (400), keys not entitled to the requested SLO class (403), keys over their
concurrency or tokens-per-minute limits (429; an `n`-way request counts `n` sequences and is
charged `prompt + n * max_new_tokens`), a full queue (503) and requests whose deadline the
iteration-counting test says cannot be met (503 with a `Retry-After` computed from the backlog).
A request the server refuses for load gets its token charge back. The test and its pessimism are
described in ADR 0004.

## Cancellation, timeouts and drain

- `Engine.abort` removes every sample of a request from whichever queue holds it and frees its
  blocks immediately. The SSE path calls it from the stream generator's `finally`, as a separate
  task shielded from the cancellation that triggered it (`service.py`): under anyio every await in
  a cancelled scope is cancelled again, so awaiting the abort inline would never take the engine
  lock and the blocks would stay pinned.
- `APP_REQUEST_TIMEOUT_SECONDS` bounds the wall time of every request, streaming or not; a
  request that runs out is aborted the same way (504, or an SSE `error` event). A request with
  `deadline_ms` is aborted at its deadline instead (`deadline_exceeded`), and `deadline_ms` may not
  exceed the timeout.
- A non-streaming request races its collector against a disconnect watcher, so a client that
  sends `stream: false` and hangs up is aborted too. Both paths close the engine stream explicitly;
  leaving it to garbage collection delayed the abort.
- On shutdown the lifespan calls `EngineService.drain`: stop admitting (new requests get 503
  `draining`), wait for in-flight requests up to `APP_DRAIN_TIMEOUT_SECONDS`, then stop the loop.

## Invariants and where they are checked

| Invariant | Checked by |
|---|---|
| Reference counts never negative; free list = zero-count blocks, no duplicates; nothing leaks | `BlockManager.check_invariants` after every step in the engine property test and the stateful allocator machine |
| Every sequence reads back exactly the tokens it computed, at the right slots (catches a missed copy-on-write or a block handed out twice) | `tests/property/test_allocator_state.py` (hypothesis stateful) |
| A block registered in the prefix cache never changes content | same stateful machine |
| Paged, chunked, batched, shared, preempted execution gives the logits of a dense recompute | `tests/property/test_engine_equivalence.py`, `tests/property/test_equivalence.py` |
| A preempted request produces the same tokens | `tests/integration/test_engine.py` |
| No iteration is empty while work is queued | `tests/property/test_engine_equivalence.py` (asserted after every step), `tests/unit/test_scheduler.py` |
| The running set never exceeds `max_running_seqs`, forks included | `tests/integration/test_engine.py`, `tests/integration/test_adversarial.py` |
| Sample `i` of an `n`-way request depends only on `(seed, i)`, with or without preemption | `tests/integration/test_engine.py` |
| With an exact latency model and a fake clock, no admitted request misses its deadline -- observed on the benchmark workloads and in a hypothesis search with prompts several times the budget, not proven; the bound does not model preemption or KV pressure (ADR 0004) | `tests/integration/test_deadlines.py`, `bench/bench_serving.py` |
