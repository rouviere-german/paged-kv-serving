# paged-kv-serving

**A language-model server, built from scratch, that answers far more requests inside their time
limit than the textbook approach -- on the same machine, the same model and the same memory
budget.**

When a language model serves many people at once, the thing that breaks first is usually not the
arithmetic. It is scheduling and memory. The simple design groups requests into fixed batches and
reserves a worst-case slab of memory for each one, so a two-word question waits behind someone
else's long answer, most of the reserved memory is never used, and when a burst arrives the server
accepts everything and then times out on all of it. The bill is real: paid-for compute spent on
padding, a customer staring at a spinner and leaving, an on-call page because a queue grew for ten
minutes before anyone noticed, and -- if prompt caching is shared carelessly between customers --
one tenant able to learn what another tenant recently asked.

This repository implements the mechanisms production serving engines use to avoid all of that
(paged KV cache, continuous batching, chunked prefill, prefix caching, deadline-aware admission
control), tests them against a slow-but-obviously-correct reference, and measures them against two
baselines that share the same model code. Everything runs offline on a CPU. No GPU, no download,
no paid API key.

Two things to know before reading the numbers. The model's weights are random on purpose: the claim
is that this engine computes *exactly* what a naive recomputation computes, and how much better it
*schedules* the same work -- not that the generated text means anything. And because the engine is
float32 NumPy on one CPU thread, absolute speed is dominated by Python overhead, so only ratios
measured between schedulers on the same machine are claimed.

## What this does

- **Hands out memory in small pages instead of one big slab per request.** A request's growing
  attention state (its "KV cache" -- the model's working memory for that one request) is stored in
  fixed-size pages, so the server holds what is in use rather than what might be needed.
  Measured: 93% of the memory it has allocated holds live data, against 33% for the usual
  one-slab-per-request design.
- **Starts and finishes each request on its own schedule.** Requests join and leave the work in
  progress every iteration instead of moving in lockstep groups, so a short question is not stuck
  behind a long answer.
- **Reuses work it has already done.** When many requests begin with the same text (a shared system
  prompt), the computation for that shared opening is stored once and reused. Measured: first
  response 60% faster, 77% less prompt computation.
- **Refuses early and honestly instead of timing out later.** If a request cannot finish inside the
  deadline the caller asked for, it is rejected at arrival with a retry hint, before it takes any
  memory. In the overload sweep, none of the 882 admitted requests missed its deadline; a server
  that accepted everything missed 188 deadlines out of 300 at the heaviest load.
- **Proves it still computes the right answer.** All of the above is only worth having if the output
  is unchanged. Every number the engine produces is compared against a plain, cache-free
  recomputation of the same request: largest difference measured, 0.0000032.

Terms used below, once each: *prefill* is reading the prompt; *decode* is producing one output token
at a time; *TTFT* is time to first token (how long the caller waits before anything appears); *p99*
means the slowest 1 request in 100; *goodput* counts only the requests that finished before their
deadline.

## Architecture

```mermaid
flowchart TB
    subgraph http["HTTP (engine/api.py, engine/service.py)"]
        client[client] -->|"POST /v1/completions<br/>X-API-Key"| mw["baseline middleware<br/>auth, rate limit, body limit, CORS"]
        mw --> body["closed schema<br/>tokenize, bound lengths"]
        body --> submit["EngineService.submit<br/>(engine lock)"]
        submit -->|503 + Retry-After / 429 / 403 / 400| client
        loop["stepping loop<br/>step() in a worker thread"] -->|token events| sse["SSE stream<br/>timeout, disconnect = abort"]
        sse --> client
    end
    subgraph engine["Engine (engine/engine.py)"]
        adm["admission<br/>entitlements, limits,<br/>iteration-counting deadline test"]
        sched["Scheduler<br/>1 decode, 2 prefill + admit in one priority order<br/>preempt lowest priority"]
        kv["KVCacheManager<br/>block tables, COW, prefix reuse"]
        bm["BlockManager<br/>free list, ref counts"]
        pc["PrefixCache<br/>hash chain, per-key scope, LRU"]
        run["ModelRunner<br/>ragged batch, paged attention<br/>K/V pool: layers x blocks x block_size x heads x dim"]
        samp["sampler<br/>greedy / temperature / top-p<br/>per-sequence RNG"]
        lat["LatencyModel<br/>profiled L(n), EWMA"]
    end
    submit --> adm
    adm --> sched
    loop --> sched
    sched --> kv
    kv --> bm
    kv --> pc
    sched -->|ExecutionBatch| run
    run -->|logits| samp
    run -. measured time .-> lat
    lat -. worst-case iteration cost .-> adm
    ckpt[("checkpoint<br/>safetensors + JSON + manifest")] -->|"SHA-256 verified"| run
    admin["/v1/admin/* (admin scope)"] --> audit[("hash-chained audit log")]
    engine -.-> obs["Prometheus metrics, OTel span per request,<br/>JSON logs with lengths and hashes only"]
```

### How a request flows

1. **It arrives** at `POST /v1/completions` with an API key. The key is checked against a stored
   hash, the body against a closed schema, and the prompt is turned into tokens.
2. **It is priced before it is accepted.** The server charges the key for the worst case the request
   could cost and checks the key's limits (how many requests at once, how many tokens per minute,
   how long a prompt, which service class). Over a limit, the caller gets a clear refusal now.
3. **It is checked against its own deadline.** The server keeps a profiled model of how long one
   iteration costs, counts how many iterations this request and everything already promised would
   need, and refuses with `503` plus a `Retry-After` if the deadline cannot be met. Nothing has
   been allocated yet.
4. **It joins the waiting queue**, ordered by service class first (interactive before batch), then
   by earliest deadline, then by arrival time.
5. **The engine takes one step.** Each step first gives every in-flight request its single next
   token, then spends whatever remains of a fixed token budget on reading prompts -- a long prompt
   in pieces across several steps, so it never freezes the requests already streaming.
6. **Memory is handed out a page at a time.** Pages that hold text shared with an earlier request
   are reused rather than recomputed. If the pool runs dry, the lowest-priority request that has
   not run yet this step gives up its pages and is redone later; it still produces the same tokens.
7. **One forward pass runs the whole mixed batch.** Attention walks each request's own list of
   pages; everything else runs once over the flattened batch. Each request that reached the end of
   its text samples exactly one token with its own random generator.
8. **The token is streamed out** over SSE. Finished requests release their pages; newly completed
   pages are published for reuse. If the caller hangs up, the request is aborted and its pages are
   freed immediately.

The scheduling and memory rules, including the liveness argument, are written out step by step in
[docs/SCHEDULER.md](docs/SCHEDULER.md).

## Results

Every number here comes from a benchmark script in `bench/` and is recorded as JSON in
`bench/results/`. One machine: Windows 10, Intel Core i5-8500 (6 cores), CPython 3.12.14, NumPy
2.5.3, OpenBLAS pinned to one thread ([ADR 0006](docs/adr/0006-single-threaded-blas.md)). The model
is the 820,352-parameter demo checkpoint (4 layers, `d_model` 128, vocabulary 512).
`bench/results/serving.json` embeds a run manifest with the checkpoint seed, package versions and
the SHA-256 of every checkpoint file.

| What was measured | Result | Command |
|---|---|---|
| On-time requests per second vs static batching, steady state | **2.38x** (1.11-3.91 over 3 seeds) | `uv run python bench/bench_serving.py --check` |
| On-time requests per second vs a 50 ms max-wait dynamic batcher, steady state | **1.70x** (1.01-2.59) | same |
| p99 time to first token vs static batching, steady state | **31x lower** (0.05 s against 1.56 s) | same |
| Output tokens per second vs either baseline, at and above capacity | **1.9-2.1x** | same |
| Share of allocated KV memory holding live tokens | **93.3%** against 33.5% | same |
| First-response time with prompt reuse on | **60.3% faster**, 76.5% fewer prompt tokens computed | same |
| Admitted requests that missed their deadline | **0 of 882** | same |
| Largest disagreement with a cache-free recompute | **3.248e-06** (CI fails above 1e-4) | `uv run python bench/bench_baseline.py --check` |

What the first row means in plain terms: given the same arrivals, the same model and the same
memory, this engine answered 2.38 times as many requests within their promised time as the standard
fixed-batch design. What the third row means: the unluckiest 1 request in 100 waited 0.05 s for its
first token here, against 1.56 s with fixed batching -- the difference between "instant" and
"noticeably broken".

### Continuous vs static and dynamic batching

`uv run python bench/bench_serving.py --check` (511.7 s for the whole sweep). The three schedulers
get the same KV memory (4,096 tokens: either the paged pool, or 16 batch slots of 256 tokens), the
same model and sampler, and the same Poisson arrival schedule at each load point. The **static**
batcher starts a batch only when 16 requests are waiting; the **dynamic** batcher also starts a
partial batch once its oldest request has waited 50 ms (a max queue delay, as in Triton). Load is
relative to capacity measured first (71.24 requests/s, continuous engine saturated). Every load
point runs 500 requests for each of three workload seeds (17, 18, 19); a request is interactive with
probability 0.5 (242 of 500 on average) and then carries a 1 s deadline; latency is measured from
the scheduled arrival, not from when the client got around to sending it (open loop). Values are
means over the three seeds, with the min-max over seeds in parentheses; times are in seconds.

| load | scheduler | completed | refused | output tokens/s | TTFT p50 | TTFT p99 | latency p99 | SLO attainment | goodput/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5x | continuous | 499.7 | 0.3 | 654 | 0.009 | 0.05 (0.03-0.07) | 0.20 | 1.00 (1.00-1.00) | 17.4 (17.3-17.7) |
| 0.5x | static | 500.0 | 0.0 | 600 | 0.801 | 1.56 (0.85-2.40) | 1.79 | 0.58 (0.30-0.92) | 9.5 (4.4-15.6) |
| 0.5x | dynamic | 500.0 | 0.0 | 609 | 0.481 | 1.28 (0.55-1.90) | 1.51 | 0.72 (0.43-1.00) | 11.9 (6.7-17.1) |
| 1x | continuous | 500.0 | 0.0 | 1202 | 0.042 | 1.07 (0.45-1.94) | 1.33 | 1.00 (1.00-1.00) | 32.1 (30.0-33.7) |
| 1x | static | 500.0 | 0.0 | 626 | 3.856 | 7.52 (7.03-7.92) | 7.70 | 0.09 (0.07-0.11) | 1.5 (1.3-1.7) |
| 1x | dynamic | 500.0 | 0.0 | 601 | 3.758 | 8.24 (6.44-10.63) | 8.46 | 0.10 (0.06-0.13) | 1.7 (0.8-2.5) |
| 1.5x | continuous | 499.3 | 0.7 | 1257 | 0.229 | 3.21 (2.79-3.53) | 3.60 | 1.00 (1.00-1.00) | 33.6 (31.7-37.2) |
| 1.5x | static | 500.0 | 0.0 | 629 | 5.089 | 9.68 (9.00-10.16) | 9.87 | 0.08 (0.07-0.08) | 1.3 (1.2-1.4) |
| 1.5x | dynamic | 500.0 | 0.0 | 629 | 4.987 | 9.70 (8.93-10.52) | 9.88 | 0.08 (0.07-0.09) | 1.4 (1.2-1.6) |
| 3x | continuous | 419.0 | 81.0 | 1340 | 2.313 | 3.23 (3.02-3.44) | 3.63 | 1.00 (1.00-1.00) | 34.4 (33.0-37.0) |
| 3x | static | 500.0 | 0.0 | 644 | 5.852 | 11.59 (11.07-12.35) | 11.80 | 0.07 (0.07-0.08) | 1.3 (1.1-1.4) |
| 3x | dynamic | 500.0 | 0.0 | 644 | 5.861 | 11.59 (11.25-11.77) | 11.77 | 0.07 (0.06-0.08) | 1.2 (1.1-1.4) |

"SLO attainment 0.58" in row two means 58% of the deadline-carrying requests were answered in time;
the continuous engine's 1.00 means all of them were, at every load point.

| continuous / baseline | 0.5x | 1x | 1.5x | 3x |
|---|---:|---:|---:|---:|
| goodput vs static | 2.38 (1.11-3.91) | 21.2 (17.7-25.5) | **26.0** (23.6-27.3) | 27.2 (22.9-29.8) |
| goodput vs dynamic | 1.70 (1.01-2.59) | 23.1 (13.7-37.3) | **24.0** (21.8-26.7) | 28.4 (23.6-34.0) |
| output tokens/s vs static | 1.09 (1.02-1.16) | 1.92 (1.89-1.96) | 2.00 (1.97-2.05) | 2.08 (2.03-2.15) |
| output tokens/s vs dynamic | 1.07 (1.01-1.12) | 2.02 (1.84-2.24) | 2.00 (1.90-2.06) | 2.08 (2.06-2.10) |
| TTFT p99, static / continuous | 31.1 (21.1-45.5) | 9.62 (4.08-15.59) | 3.03 (2.88-3.22) | 3.60 (3.30-3.83) |
| TTFT p99, dynamic / continuous | 24.6 (17.1-36.1) | 9.67 (5.48-14.29) | 3.03 (2.73-3.19) | 3.60 (3.43-3.73) |

How to read it -- there are two regimes, and the honest headline is the first one:

- **0.5x load is the steady-state comparison.** Both baselines are near, not past, their own
  capacity. Continuous batching completes 2.38x the on-time requests per second of static batching
  and 1.70x that of dynamic batching, with a wide spread across seeds (1.11-3.91 and 1.01-2.59,
  because the baselines sit right at their limit), and cuts TTFT p99 by 31x and 25x. The 50 ms max
  wait fixes much of static batching's time-to-first-token cost (TTFT p50 0.48 s against 0.80 s)
  and none of its throughput cost: padding every prompt to the longest in the batch and decoding
  until the longest answer finishes are unchanged.
- **1x and above is sustained overload for the baselines**, so their ratios are inflated. They
  deliver 600-645 output tokens/s against the continuous engine's 1,200-1,340 -- about half -- so
  from 1x load their queues grow for the whole 500-request run and only 7-10% of interactive
  requests meet the 1 s deadline. The 21-28x goodput ratios there measure that overload and grow
  with run length. The steady-state throughput ratio, 1.9-2.1x, is the number to trust. The
  continuous engine meets every interactive deadline at every load point, refusing requests at
  arrival instead (0.7 of 500 at 1.5x, 81 of 500 at 3x, on average).

The spec's target -- goodput at least 1.5x static at 1.5x load -- is met against either baseline. CI
fails if the mean 1.5x goodput ratio drops below 1.5 against either baseline, if the 3x one drops
below 1.5, or if the 1.5x throughput ratio drops below 1.3.

An earlier version of this benchmark ran 48 requests per load point with one seed and reported
goodput ratios of 1.39-3.42 and "static batching misses 31% of interactive deadlines from 1x load".
With only 48 requests the overloaded baselines' queues never grew far, p99 was simply the maximum,
and a rerun measured 2.60 instead of 3.42 at 1x and 17% instead of 31% missed. Those numbers are
superseded by the table above.

### Prefix cache

Same command. For each of the three seeds: 20 requests sharing a 128-token system prompt plus a
16-48-token user turn, one key, well below capacity so TTFT measures prompt reading rather than
queueing. Means over the seeds:

| | cache off | cache on |
|---|---:|---:|
| TTFT p50 (s) | 0.01123 | 0.00446 |
| prefill tokens computed | 3179 | 747 |
| block hit ratio | 0.0 | 0.804 |

TTFT p50 reduction **60.3%** (59.5%-61.4% across seeds; target at least 30%); prompt tokens computed
down 76.5%. In plain terms: the shared opening of a prompt is read once instead of twenty times, so
the median caller sees the first token in 4.5 ms instead of 11.2 ms. The token counts are
deterministic per seed; the TTFT reduction moves with timing noise (single runs of the earlier
benchmark measured 61.9% and 71.6%).

### KV memory

Same command. 24 requests submitted at once, prefix cache off:

| | value |
|---|---:|
| paged: tokens resident / tokens in allocated blocks, mean over iterations | 0.9331 (min 0.8889) |
| static: tokens in a batch at its end / reserved slot memory | 0.3348 |
| paged peak KV tokens allocated | 1392 |
| static KV tokens reserved | 4096 |

The static figure is measured at the batch's *end*, when every row is as long as it will ever get --
the most favourable moment for static batching, and still a third. Read the last two rows together:
paging served the same workload from 1,392 tokens of memory where reserving a worst-case region per
slot needed 4,096. On real hardware that difference is how many requests fit on one accelerator at
once.

### Admission control (fake clock, exact latency model)

Same command. The executor consumes exactly the time the latency model predicts (2e-4 s per token,
about 45 requests/s of capacity), so the only thing under test is the admission decision. 300
requests per load level, deadlines 0.75/1.5/3 s, run twice: with admission, and with admission
effectively off (a vanishing safety factor).

| load | admitted | refused at arrival | admitted that missed | on time (admission) | on time (accept all) | missed (accept all) |
|---|---:|---:|---:|---:|---:|---:|
| 0.5x | 287 | 13 | 0 | 287 | 300 | 0 |
| 1x | 277 | 23 | 0 | 277 | 300 | 0 |
| 2x | 186 | 114 | 0 | 186 | 196 | 104 |
| 4x | 132 | 168 | 0 | 132 | 112 | 188 |

No admitted request missed its deadline at any load (882 admitted of 1,200). That is an observation
on these workloads and on a property-based search with prompts several times the token budget
(`tests/integration/test_deadlines.py`), not a proof: the bound does not model preemption or
KV-memory pressure ([ADR 0004](docs/adr/0004-iteration-counting-admission.md)). It also depends on
the scheduler prefilling in the order the bound assumes -- an earlier version continued every
running prefill first, and a counterexample found in review (an earlier-deadline request arriving
while a loose-deadline prompt was mid-prefill) was admitted and then missed; that case is now a
test.

The test is deliberately pessimistic: every iteration is costed at a full token budget, so at low
load it refuses requests that would have made it, and at 2x load accepting everything completes
slightly more on time. What it buys is the 4x row. Accepting everything there misses 188 of 300
deadlines -- 188 callers who waited and got nothing -- while every one of those requests held memory
the whole time. Refusing 168 at arrival, immediately and with a retry hint, is the better failure.

### Correctness and micro-benchmarks

`uv run python bench/bench_baseline.py --check`:

| benchmark | runs | p50 (us) | p95 (us) | other |
|---|---:|---:|---:|---|
| block_allocate_free_16 | 5000 | 5.4 | 5.5 | |
| copy_on_write_block | 5000 | 1.8 | 2.0 | |
| prefix_hash_chain_512_tokens | 2000 | 138.7 | 151.6 | |
| paged_attention_decode_256ctx | 2000 | 129.4 | 171.9 | |
| sample_top_p_512_vocab | 5000 | 77.7 | 98.6 | |
| admission_decide_64_committed | 2000 | 80.3 | 94.9 | |
| engine_iteration_128_tokens | 20 | 12330.9 | 13426.5 | |
| engine_throughput_24_requests | 1 | | | 288 tokens in 0.198 s, 1453 tokens/s |
| paged_vs_reference_equivalence | 4 | | | max abs logit gap 2.444e-06 |
| engine_vs_reference_equivalence | 1 | | | max abs logit gap 3.248e-06 over 264 rows, 7 preemptions, 30 prefix-hit blocks |

The two equivalence rows are the point of the whole repository. They take every output the engine
sampled from -- mixed prompt-and-decode batches, prompts split at awkward boundaries inside a page,
pages reused from the cache, requests that were kicked out and redone, parallel samples that forked
from a shared prompt -- and compare it with a dense, cache-free recomputation of the same sequence.
The largest disagreement is 3.248e-06, which is float32 rounding noise. CI fails if either gap
exceeds 1e-4, or if the scenario stops producing preemptions and cache hits (a passing test that
stopped exercising the hard paths would be worse than a failing one).

## How it works

Each mechanism is a small module you can read in one sitting, and each one has something that
checks it. Files are under `src/paged_kv_serving/engine/` unless noted.

| Mechanism | Where | How it is checked |
|---|---|---|
| Paged KV cache: fixed-size blocks, block tables, FIFO free list, reference counts, copy-on-write, leak checker | `blocks.py`, `kvcache.py` | property-based state machine over admit/prefill/decode/fork/preempt/free/evict |
| Paged attention over non-contiguous blocks with an online softmax | `attention.py`, `runner.py` | property tests against dense attention |
| Iteration-level continuous batching: decode first, then running prefills and new admissions in one (class, deadline, arrival) order, chunked; no empty iteration while work is queued | `scheduler.py` | unit tests, engine-level logit equivalence, liveness asserted after every generated step |
| Preemption by recomputation of the lowest-priority unscheduled sequence | `scheduler.py` | token-for-token and logit-for-logit equivalence under a cramped pool |
| Block prefix cache keyed by `hash(parent_hash, block tokens)`, per-key scope, LRU eviction of unreferenced blocks | `prefix_cache.py` | chain, scope and eviction unit tests; TTFT benchmark |
| Parallel sampling (`n`) sharing the prompt's blocks; each sample counted against running slots, concurrency and tokens/min | `engine.py` | shared prefill counted, per-sample RNG independence, running-set cap |
| Deadline-aware admission: iteration-counting feasibility test, profiled latency model with EWMA correction, early 503 with `Retry-After` | `admission.py` | fake-clock simulation: zero admitted misses |
| Lower-class prefill allowance protecting interactive decode steps from batch floods | `scheduler.py` | adversarial workload tests |
| Per-key entitlements: SLO classes, tokens/minute, concurrent sequences, prompt, output and `n` ceilings; fail closed with no policies | `keys.py` | unit, API and adversarial tests |
| FastAPI `/v1/completions` with SSE streaming, request timeouts and deadline aborts, cancellation on disconnect (streaming or not), graceful drain | `api.py`, `service.py` | API tests, including a real loopback server |
| Static and dynamic (max queue delay) batching baselines over the same runner and sampler | `static.py` | fake-clock dispatch tests; used by the benchmarks and the demo |
| Request telemetry: one data-contract-validated JSONL row per finished request | `telemetry.py`, `ml/contracts.py` | contract, rotation and API tests |
| Audited admin operations: intent logged before acting, optional HMAC chain, head published to logs | `audit.py`, `api.py` | tamper, keyed-chain and refused-action tests |
| Open-loop load generators (in-process and HTTP) free of coordinated omission | `loadgen.py`, `http_loadgen.py` | schedule-not-completion tests |
| safetensors checkpoint with a SHA-256 manifest, in-repo byte-level BPE tokenizer | `weights.py`, `tokenizer.py` | tamper and pin tests |

### One bookkeeping rule makes the rest possible

Every sequence carries its tokens and a count of how many of them already have keys and values in
the cache. An iteration runs the next *chunk* of tokens at their absolute positions. Reading a
prompt, reading part of a prompt, producing one token, and redoing a preempted request are then the
same operation at four different chunk sizes -- which is exactly why the scheduler can mix them
freely in one batch and why one attention kernel serves all of them.

### The decisions worth arguing about

- **Preempt by recomputation, not by swapping.** When the pool runs dry, the victim's pages are
  freed and its work is redone later, rather than copied to a slower memory tier. Here the "fast"
  tier is already host RAM, so a swap tier would be a second copy of the same kind of memory, plus
  its own allocator, its own full-swap-space policy and an extra state in every queue. The cost is
  repeated prefill under sustained pressure; the prefix cache absorbs much of it, because the
  victim's own full pages are usually still cached.
  [ADR 0002](docs/adr/0002-preemption-by-recomputation.md)
- **Scope prefix-cache reuse to the calling key by default.** Sharing the cache across tenants
  would raise the hit rate and turn time-to-first-token into an oracle: send a guessed prompt
  prefix, time your own response, and learn whether somebody else recently sent it. That is a
  disclosure when the prefix is a document or a reset link. The chain is seeded with the caller's
  key id, so two keys can never produce a common prefix; a `global` scope exists for single-tenant
  deployments that want the hit rate and accept the channel.
  [ADR 0003](docs/adr/0003-per-key-prefix-cache-scope.md)
- **Admit by counting iterations at their worst-case cost, not by dividing tokens by a rate.** The
  obvious model -- work ahead divided by a token rate -- is wrong twice over for an LLM engine.
  Decode is serial per request and parallel across requests, so 20 output tokens need 20 iterations
  however idle the machine is, and twenty such requests still need 20 iterations between them. And
  iteration cost depends on prompts that arrive *after* admission. Counting iterations and costing
  each at a full budget is pessimistic by construction -- it refuses some requests that would have
  made it -- and it is what makes "no admitted request missed its deadline" hold on every workload
  measured. [ADR 0004](docs/adr/0004-iteration-counting-admission.md)
- **Cap lower-class prefill while a higher class has work.** Scheduling interactive work first is
  not enough: iteration time grows with the tokens in the batch, and whatever budget interactive
  work leaves over gets filled with batch prompts, so every interactive decode step pays for a
  full-budget forward pass. Flooding the engine with maximum-length batch prompts pushed the
  victim's interactive latency from about 13 ms to about 400 ms per request. Lower classes together
  now get at most 25% of the budget for prefill while a higher class has work; decode is never
  capped. [ADR 0005](docs/adr/0005-lower-class-prefill-share.md)
- **Pin BLAS to one thread.** Every matrix multiply here is small (a few dozen rows by a 128x384
  projection), and OpenBLAS's default one-thread-per-core pool spent more time waking and joining
  threads than doing arithmetic: a fitted iteration intercept of 13.3 ms on six threads against
  1.1 ms on one, and an open-loop run of 1.37 s against 0.72 s on the same checkpoint. With
  the default pool the static baseline looked faster than continuous batching for reasons that had
  nothing to do with scheduling. Serving concurrency comes from batching inside an iteration, not
  from BLAS threads. [ADR 0006](docs/adr/0006-single-threaded-blas.md)

### How it is tested

`uv run python tasks.py test` runs the suite offline (non-loopback sockets are blocked) with
combined line and branch coverage of 96.23% -- 97.34% of lines and 91.60% of branches -- against a
floor of 85%. Property tests take their example counts from the active profile
(`tests/hypothesis_budget.py`), and CI runs five times the local budget, derandomised. The core of
the suite is property-based rather than example-based:

- `tests/property/test_engine_equivalence.py` generates engine configurations (block size, pools
  within two blocks of one maximum-length sequence or larger, budgets small enough to force
  chunking, prefix cache on or off, allowance shares) and workloads (shared prefixes, two tenants,
  two classes, greedy and sampled, `n` up to 8, prompts several times the token budget, staggered
  arrivals). Every sampled logits row must match a dense recompute within 1e-4, no iteration may be
  empty while work is queued, every request must finish with its exact token count, and the pool
  must be whole afterwards.
- `tests/property/test_allocator_state.py` drives the real `KVCacheManager` through admit, prefill,
  decode, fork, preempt, finish and evict in generated orders, checking after every step that
  reference counts are consistent, every sequence reads back its own tokens, cached blocks never
  change, and nothing leaks.
- `tests/property/test_equivalence.py` covers chunk splits, scattered block tables, tile sizes,
  batch composition and shared prefixes at the runner level.
- `tests/integration/test_deadlines.py` and `tests/integration/test_adversarial.py` cover admission
  (including a search for admitted misses) and multi-tenant isolation on a fake clock;
  `tests/integration/test_api.py` covers the served API, including a real loopback server for
  disconnects, deadlines, audit tampering and telemetry.

Each fix from an independent review comes with a regression test that failed before the fix; the
test name says what it guards and its docstring says what used to happen. Every claim in this
README maps to a test, bench script or CI job in [CLAIMS.md](CLAIMS.md), and
`scripts/check_baseline.py` fails if any of those references stops existing.

### Observability

- Prometheus (`/metrics`, authenticated by default): `engine_ttft_seconds`, `engine_tpot_seconds`,
  `engine_request_latency_seconds`, `engine_iteration_seconds`, `engine_batched_tokens`,
  `engine_kv_blocks_used_ratio`, `engine_prefix_cache_hit_ratio`, `engine_preemptions_total`,
  `engine_rejections_total{reason}`, `engine_aborted_requests_total`, `engine_queue_depth`,
  `engine_running_sequences`, `engine_slo_attainment`, `engine_prompt_length_psi`, plus HTTP
  metrics by route template.
- OpenTelemetry: a server span per HTTP request and an `engine.completion` child span per completion
  with prompt and completion lengths, class, outcome, TTFT and preemptions.
- `/healthz` (liveness) and `/readyz` (weights loaded and warmed, engine loop running, accepting,
  allocator invariants hold -- checked under the engine lock, never mid-iteration). SIGTERM drains:
  new requests get 503, in-flight ones finish.
- Workload drift: population stability index of recent prompt lengths against the first prompts the
  server saw.

## Security

A multi-tenant inference server is a shared memory pool, a shared queue and a shared cache behind
one door. The defences below exist because each of those is a way for one customer to hurt another,
or for an attacker to change what the server says. Threats, controls and tests are mapped in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

- **Only the right callers get in, and only to what they paid for.** API keys are held as SHA-256
  digests and compared without an early exit. Each key has a policy: which service classes it may
  ask for (asking for "interactive" without the entitlement is a 403, not a free priority upgrade),
  tokens per minute charged for the worst case at admission (`prompt + n * max_new_tokens`, refunded
  if the server refuses the request for load), how many sequences it may run at once (an `n`-way
  request counts `n`), and ceilings on prompt, output and `n`. A valid key with no policy cannot
  call the engine at all, and with no policies configured every key is denied.
- **No single request can exhaust the machine.** The request body is a closed schema; prompt
  tokens, `max_new_tokens`, `n` and total context are bounded by the engine and by the key before
  one page is allocated. A wall-clock limit per request -- or the caller's own `deadline_ms`, which
  may not exceed it -- aborts and frees pages. A client that hangs up, streaming or not, has its
  request aborted and its pages freed immediately.
- **One noisy tenant cannot ruin another tenant's latency.** A key sending maximum-length prompts,
  cache-thrashing unique prompts, maximum-`max_tokens` requests or eight-sample requests does not
  push another key's interactive p99 past its SLO. That is tested with a simulated executor; the
  lower-class prefill allowance, the per-key limits and the class entitlement are each shown to be
  part of why.
- **The server will not run weights it cannot verify.** safetensors only, loaded after every file's
  SHA-256 is checked against a manifest that must cover the weights and the tokenizer; the manifest
  digest itself can be pinned, so a swapped checkpoint is refused at start-up. No pickle-family
  serialization anywhere in `src/`, enforced by a test and a semgrep rule.
- **Prompts are not written down.** Prompts and completions are never logged or traced -- lengths
  and a prompt SHA-256 only. Prefix-cache reuse is per key by default, which closes the
  cross-tenant timing channel described above
  ([ADR 0003](docs/adr/0003-per-key-prefix-cache-scope.md)).
- **Privileged actions leave a trail that cannot be quietly edited.** Cache flush and weight swap
  need the admin scope and load checkpoints by plain name under a fixed root with a required
  manifest digest. The intent is appended to a hash-chained audit log *before* anything changes, and
  the action is refused (503) if it cannot be recorded; refused attempts are recorded too. With
  `APP_AUDIT_HMAC_KEY` the chain is an HMAC, so it cannot be rebuilt by whoever can write the file,
  and every append logs the new head -- the anchor `paged-kv-serving audit verify --expect-head`
  needs to detect truncation.
- **Telemetry cannot grow a prompt column.** One row per finished request (lengths, timings, a
  prompt SHA-256, never text) is validated against a strict data contract before it is written to
  `APP_DATA_DIR/telemetry/requests.jsonl`; a batch that breaks the contract is dropped whole and
  reported without values.
- **Supply chain and container.** Locked dependencies, pip-audit, bandit, semgrep, gitleaks,
  hadolint, trivy and a CycloneDX SBOM in CI; non-root, read-only image with capabilities dropped.

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12. No GPU, no model download, no API
key, no network.

```bash
uv sync --frozen
uv run python tasks.py demo      # offline end-to-end demo (about 10 s)
uv run python tasks.py test      # pytest with coverage
uv run python tasks.py bench     # micro-benchmarks + serving comparison (about 9 minutes)
uv run python tasks.py check     # lint, typecheck, test, audit, baseline
```

The demo builds a checkpoint, checks logit equivalence through the real scheduler, measures the
prefix cache, refuses an infeasible burst, cancels a request, compares continuous and static
batching on one arrival schedule, then starts the server on a loopback port, streams over SSE, hangs
up mid-stream, and sends an open-loop burst of interactive and batch traffic from two keys. What it
prints (from one run on the machine described above, 7.9 s; timings vary between runs, and the
served section runs client and server in one process, so its HTTP latencies include contention for
the interpreter lock and are illustrative only):

```
equivalence: single-prompt max |dlogit| 2.21e-06; engine max |dlogit| 2.95e-06 over 165 rows with 1 preemptions
prefix cache: hit ratio 0.783, prefill tokens saved 75.1%, TTFT p50 reduction 54.4%
admission: 60 offered, 32 admitted, 28 refused early, 100% of admitted met the 0.302s deadline
cancellation: 6 blocks held -> 0 after abort; pool whole after cache flush: True
server: healthz 200, readyz ready, no key 401, batch key asking for interactive 403, hang-up aborted 1 request(s), sequence blocks after hang-up 0, audit verified True
http burst [batch]: sent 32, completed 18, errors {'key_concurrency': 14}, TTFT p50/p99 0.4774/1.1469s
http burst [interactive]: sent 28, completed 28, errors {}, TTFT p50/p99 0.2849/1.0995s

continuous vs static batching (same arrivals, same model):
label       completed  tokens_per_second  ttft_p50  ttft_p99  latency_p99  on_time  goodput_per_second
continuous  48         1117               0.0282    0.176     0.2568       24       41.13
static      48         619.9              0.3174    0.6176    0.7123       24       22.82
```

### Serving it yourself

```bash
uv run paged-kv-serving build-checkpoint --out var/checkpoint     # prints manifest_sha256
uv run paged-kv-serving generate-api-key                          # prints api_key + digest
export APP_API_KEY_SHA256=<digest>
export APP_KEY_POLICIES="<digest>:interactive|batch:60000:8:512:128:1"
export APP_CHECKPOINT_MANIFEST_SHA256=<manifest_sha256>           # refuse a swapped checkpoint
uv run paged-kv-serving serve
curl -N localhost:8000/v1/completions -H "X-API-Key: <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Why reject early?", "max_tokens": 16, "stream": true, "deadline_ms": 2000}'
```

A key policy is
`<sha256>:<classes>:<tokens/min>:<max concurrent sequences>:<max prompt>:<max new>:<admin>[:<max n>]`;
an `n`-way request counts `n` sequences against the concurrency cap and is charged
`prompt + n * max_new_tokens`, and `n` above the key's limit (default 1) is a 400. Every setting is
listed in [.env.example](.env.example). `docker compose up --build` runs the server and an open-loop
load generator against it; the generator reads raw keys from `LOADGEN_INTERACTIVE_API_KEY` and
`LOADGEN_BATCH_API_KEY` in your shell or `.env` (the compose file documents the variables).

## Non-goals

What this repository deliberately does not do or claim:

- **Quality.** The model is random-init and never trained. The spec's optional PyTorch `train`
  extra for producing a toy checkpoint was cut: it adds a heavy dependency and nothing the
  equivalence claims need.
- **Absolute throughput.** Float32 NumPy on one CPU thread; no GPU kernels, no fused attention, no
  quantisation, no CUDA graphs. Only ratios on the same machine are claimed.
- **Swap-based preemption.** Preempted sequences are recomputed, never swapped to a second memory
  tier ([ADR 0002](docs/adr/0002-preemption-by-recomputation.md)).
- **Tenant fair queuing.** Isolation comes from per-key limits, SLO classes and the lower-class
  allowance; keys of the same class are not weighted against each other.
- **Distributed serving.** One process, one host; no tensor or pipeline parallelism, no
  disaggregated prefill, no multi-replica routing.
- **Speculative decoding, beam search, stop strings, logprobs.** Not implemented.
- **A tighter or proven admission bound.** Admission costs every iteration at its worst case and
  ignores KV memory, preemption and the running-set cap in its bound
  ([ADR 0004](docs/adr/0004-iteration-counting-admission.md)). "No admitted request missed its
  deadline" is observed on the benchmark workloads and in a property-based search with an exact
  latency model on a fake clock; it is not a proof, and is not guaranteed under memory pressure that
  forces preemption.
- **Batched attention.** Attention runs as a Python loop over the sequences of a batch (paged,
  tiled, online softmax); everything else runs once over the flattened ragged batch.
- **Telemetry shipping.** Telemetry is a local JSONL file with one rotation; shipping it anywhere is
  the deployment's job. `ml/registry.py` and `ml/serialization.py` come from the portfolio baseline
  and are used only by `scripts/check_baseline.py`.
- **Bisectable early history.** The first commits were made in one burst and several of them do not
  pass the test suite on their own (only their final state was checked); commits since are made
  after the suite passes. The history was not rewritten.
- **Hugging Face `tokenizers`.** The spec listed it; the tokenizer here is a byte-level BPE trained
  in-repo and stored as plain JSON, which keeps the checkpoint format free of any executable or
  library-specific serialisation.
- **HTTP-level performance numbers.** The HTTP path is tested and demonstrated, but only the engine
  is benchmarked; the in-process demo shares one interpreter lock between client and server.
- **Containers verified locally.** The Dockerfile and compose file are built and smoke-tested in CI
  only.

## Repository layout

```
src/paged_kv_serving/
  engine/   blocks, kvcache, prefix_cache, attention, runner, model, scheduler, admission,
            engine, sequence, sampling, keys, static, loadgen, http_loadgen, simulation,
            equivalence, service, api, weights, tokenizer, audit, metrics, monitoring, demo
  core/     settings, redacting logs, tracing, metrics, hardened FastAPI app, rate limiting
  ml/       request data contract, PSI/KS drift, manifests, registry, pickle-free serialisation
bench/      bench_baseline.py, bench_serving.py, thresholds, results/
docs/       SCHEDULER.md, THREAT_MODEL.md, adr/
tests/      unit/, integration/, property/
```

## References

- W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with
  PagedAttention", SOSP 2023 (block tables, copy-on-write, recomputation and swapping).
- G.-I. Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models",
  OSDI 2022 (iteration-level scheduling).
- A. Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve",
  OSDI 2024 (chunked prefill under a token budget).
- M. Milakov and N. Gimelshein, "Online normalizer calculation for softmax", 2018.
- C. L. Liu and J. W. Layland, "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time
  Environment", JACM 1973 (EDF, the basis of the admission test).
- G. Tene, "How NOT to Measure Latency" (coordinated omission).

The implementation is original; the papers are cited for the ideas.

## License

MIT, see [LICENSE](LICENSE).
