# paged-kv-serving

A from-scratch CPU inference engine for a tiny transformer: a paged KV cache with reference-counted
copy-on-write blocks, iteration-level continuous batching with chunked prefill and preemption,
hash-chained block prefix caching, and deadline-aware admission control, served over SSE and
measured against static and dynamic (max-wait) batching baselines that share the same model code.

Everything runs offline on a CPU in float32 NumPy. The model has random weights on purpose: the
claim is that the engine computes exactly what a naive recompute computes, and how much faster it
schedules the same work, not that the text means anything.

## Problem

Serving an autoregressive model is a scheduling and memory problem before it is a compute problem.

- **Static batching** runs a fixed group until its longest member finishes. Short requests wait
  for long ones, every prompt is padded to the longest in its batch, and a request that arrives
  just after a batch starts waits for the whole batch (head-of-line blocking).
- **Contiguous KV caches** reserve a maximum-length region per sequence. Most of it is never used,
  and what is used fragments.
- **Servers that accept every request** under a burst do not serve more; they turn every request
  into a timeout while still spending memory on answers nobody will wait for.

The mechanisms that fix these (PagedAttention, continuous batching, chunked prefill, prefix
caching, admission control) are what LLM serving engines are built from. This repository
implements them directly, tests them against a dense reference, and measures them.

## What is implemented

| Mechanism | Where | How it is checked |
|---|---|---|
| Paged KV cache: fixed-size blocks, block tables, FIFO free list, reference counts, copy-on-write, leak checker | `engine/blocks.py`, `engine/kvcache.py` | hypothesis stateful machine over admit/prefill/decode/fork/preempt/free/evict |
| Paged attention over non-contiguous blocks with an online softmax | `engine/attention.py`, `engine/runner.py` | property tests against dense attention |
| Iteration-level continuous batching: decode first, then running prefills and new admissions in one (SLO class, deadline, arrival) order, chunked; no empty iteration while work is queued | `engine/scheduler.py` | unit tests, engine-level logit equivalence, liveness asserted after every generated step |
| Preemption by recomputation of the lowest-priority unscheduled sequence | `engine/scheduler.py` | token-for-token and logit-for-logit equivalence under a cramped pool |
| Block prefix cache keyed by `hash(parent_hash, block tokens)`, per-key scope, LRU eviction of unreferenced blocks | `engine/prefix_cache.py` | chain, scope and eviction unit tests; TTFT benchmark |
| Parallel sampling (`n`) sharing the prompt's blocks; each sample counted against running slots, concurrency and tokens/min | `engine/engine.py` | shared prefill counted, per-sample RNG independence, running-set cap |
| Deadline-aware admission: iteration-counting feasibility test, profiled latency model with EWMA correction, early 503 with `Retry-After` | `engine/admission.py` | fake-clock simulation: zero admitted misses |
| Lower-class prefill allowance protecting interactive decode steps from batch floods | `engine/scheduler.py` | adversarial workload tests |
| Per-key entitlements: SLO classes, tokens/minute, concurrent sequences, prompt, output and `n` ceilings; fail closed with no policies | `engine/keys.py` | unit, API and adversarial tests |
| FastAPI `/v1/completions` with SSE streaming, request timeouts and deadline aborts, cancellation on disconnect (streaming or not), graceful drain | `engine/api.py`, `engine/service.py` | API tests, including a real loopback server |
| Static and dynamic (max queue delay) batching baselines over the same runner and sampler | `engine/static.py` | fake-clock dispatch tests; used by the benchmarks and the demo |
| Request telemetry: one data-contract-validated JSONL row per finished request | `engine/telemetry.py`, `ml/contracts.py` | contract, rotation and API tests |
| Audited admin operations: intent logged before acting, optional HMAC chain, head published to logs | `engine/audit.py`, `engine/api.py` | tamper, keyed-chain and refused-action tests |
| Open-loop load generators (in-process and HTTP) free of coordinated omission | `engine/loadgen.py`, `engine/http_loadgen.py` | schedule-not-completion tests |
| safetensors checkpoint with a SHA-256 manifest, in-repo byte-level BPE tokenizer | `engine/weights.py`, `engine/tokenizer.py` | tamper and pin tests |

The scheduling and memory logic is described step by step in [docs/SCHEDULER.md](docs/SCHEDULER.md).

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

One iteration: the scheduler builds a ragged batch under a token budget (decode tokens of running
sequences first, then prefill chunks of running prompts and new admissions in one priority order),
reserving blocks and preempting if the pool runs dry; the runner writes each token's keys and
values into its block-table slot and runs paged attention per sequence; every slice that reaches
the end of its sequence samples one token with that sequence's own generator; newly full blocks
are published to the prefix cache; finished sequences release their blocks. Benchmarks drive the
same engine without HTTP.

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12. No GPU, no model download, no API
key.

```bash
uv sync --frozen
uv run python tasks.py demo      # offline end-to-end demo (about 10 s)
uv run python tasks.py test      # pytest with coverage
uv run python tasks.py bench     # micro-benchmarks + serving comparison (about 9 minutes)
uv run python tasks.py check     # lint, typecheck, test, audit, baseline
```

The demo builds a checkpoint, checks logit equivalence through the real scheduler, measures the
prefix cache, refuses an infeasible burst, cancels a request, compares continuous and static
batching on one arrival schedule, then starts the server on a loopback port, streams over SSE,
hangs up mid-stream and sends an open-loop burst of interactive and batch traffic from two keys.

Serving it yourself:

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
`prompt + n * max_new_tokens`, and `n` above the key's limit (default 1) is a 400.
Every setting is listed in [.env.example](.env.example). `docker compose up --build` runs the
server and an open-loop load generator against it; the generator reads raw keys from
`LOADGEN_INTERACTIVE_API_KEY` and `LOADGEN_BATCH_API_KEY` in your shell or `.env` (the compose
file documents the variables).

## Results

All numbers below were produced on one machine: Windows 10, Intel Core i5-8500 (6 cores), CPython
3.12.14, NumPy 2.5.3, OpenBLAS pinned to one thread (docs/adr/0006-single-threaded-blas.md). The
model is the 820,352-parameter demo checkpoint (4 layers, `d_model` 128, vocabulary 512). Python
and NumPy overhead dominate absolute speed, so only ratios between schedulers on the same machine
are claimed. Full JSON output is in `bench/results/`; `serving.json` embeds a run
manifest with the checkpoint seed, package versions and the SHA-256 of every checkpoint file.

### Continuous vs static and dynamic batching

`uv run python bench/bench_serving.py --check` (511.7 s). The three engines get the same KV memory
(4,096 tokens: the paged pool, or 16 batch slots of 256 tokens), the same model and sampler, and
the same Poisson arrival schedule at each load point. The **static** batcher starts a batch only
when 16 requests are waiting; the **dynamic** batcher also starts a partial batch once its oldest
request has waited 50 ms (a max queue delay, as in Triton). Load is relative to capacity measured
first (71.24 requests/s, continuous engine saturated). Every load point runs 500 requests for each
of three workload seeds (17, 18, 19); a request is interactive with probability 0.5 (242 of 500 on
average) and then has a 1 s deadline; latency is measured from the scheduled arrival (open loop).
Values are means over the three seeds, with the min-max over seeds in parentheses; times are in
seconds.

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

| continuous / baseline | 0.5x | 1x | 1.5x | 3x |
|---|---:|---:|---:|---:|
| goodput vs static | 2.38 (1.11-3.91) | 21.2 (17.7-25.5) | **26.0** (23.6-27.3) | 27.2 (22.9-29.8) |
| goodput vs dynamic | 1.70 (1.01-2.59) | 23.1 (13.7-37.3) | **24.0** (21.8-26.7) | 28.4 (23.6-34.0) |
| output tokens/s vs static | 1.09 (1.02-1.16) | 1.92 (1.89-1.96) | 2.00 (1.97-2.05) | 2.08 (2.03-2.15) |
| output tokens/s vs dynamic | 1.07 (1.01-1.12) | 2.02 (1.84-2.24) | 2.00 (1.90-2.06) | 2.08 (2.06-2.10) |
| TTFT p99, static / continuous | 31.1 (21.1-45.5) | 9.62 (4.08-15.59) | 3.03 (2.88-3.22) | 3.60 (3.30-3.83) |
| TTFT p99, dynamic / continuous | 24.6 (17.1-36.1) | 9.67 (5.48-14.29) | 3.03 (2.73-3.19) | 3.60 (3.43-3.73) |

How to read it -- there are two regimes:

- **0.5x load** is the steady-state comparison: both baselines are near, not past, their own
  capacity. Continuous batching completes 2.38x the on-time requests per second of static
  batching and 1.70x that of dynamic batching, with a wide spread across seeds (1.11-3.91 and
  1.01-2.59, because the baselines sit right at their limit), and cuts TTFT p99 by 31x and 25x.
  The 50 ms max wait fixes much of static batching's time-to-first-token cost (TTFT p50 0.48 s
  against 0.80 s) and none of its throughput cost: padding to the longest prompt and decoding to
  the longest answer are unchanged.
- **1x and above** is sustained overload for the baselines. They deliver 600-645 output tokens/s
  against the continuous engine's 1,200-1,340, about half, so from 1x load their queues grow for
  the whole 500-request run and only 7-10% of interactive requests meet the 1 s deadline. The
  21-28x goodput ratios there measure that overload and grow with run length; the steady-state
  figure is the throughput ratio, 1.9-2.1x. The continuous engine meets every interactive
  deadline at every load point, refusing requests at arrival instead (0.7 of 500 at 1.5x, 81 of
  500 at 3x, on average).

The spec's target, goodput at least 1.5x static at 1.5x load, is met against either baseline.
CI fails if the mean 1.5x goodput ratio drops below 1.5 against either baseline, if the 3x one
drops below 1.5, or if the 1.5x throughput ratio drops below 1.3.

An earlier version of this benchmark ran 48 requests per load point with one seed and reported
goodput ratios of 1.39-3.42 and "static batching misses 31% of interactive deadlines from 1x
load"; with 48 requests the overloaded baselines' queues never grew far, p99 was simply the
maximum, and a rerun measured 2.60 instead of 3.42 at 1x and 17% instead of 31% missed. Those
numbers are superseded by the table above.

### Prefix cache

Same command. For each of the three seeds: 20 requests sharing a 128-token system prompt plus a
16-48-token user turn, one key, well below capacity so TTFT measures prefill rather than
queueing. Means over the seeds:

| | cache off | cache on |
|---|---:|---:|
| TTFT p50 (s) | 0.01123 | 0.00446 |
| prefill tokens computed | 3179 | 747 |
| block hit ratio | 0.0 | 0.804 |

TTFT p50 reduction **60.3%** (59.5%-61.4% across seeds; target at least 30%); prefill tokens saved
76.5%. The token counts are deterministic per seed; the TTFT reduction moves with timing noise
(single runs of the earlier benchmark measured 61.9% and 71.6%).

### KV memory

Same command. 24 requests submitted at once, prefix cache off:

| | value |
|---|---:|
| paged: tokens resident / tokens in allocated blocks, mean over iterations | 0.9331 (min 0.8889) |
| static: tokens in a batch at its end / reserved slot memory | 0.3348 |
| paged peak KV tokens allocated | 1392 |
| static KV tokens reserved | 4096 |

### Admission control (fake clock, exact latency model)

Same command. The executor consumes exactly the time the latency model predicts (2e-4 s per
token, about 45 requests/s of capacity), so the only thing measured is the admission test. 300
requests per load level, deadlines 0.75/1.5/3 s, run with admission and with admission effectively
off (a vanishing safety factor):

| load | admitted | refused at arrival | admitted that missed | on time (admission) | on time (accept all) | missed (accept all) |
|---|---:|---:|---:|---:|---:|---:|
| 0.5x | 287 | 13 | 0 | 287 | 300 | 0 |
| 1x | 277 | 23 | 0 | 277 | 300 | 0 |
| 2x | 186 | 114 | 0 | 186 | 196 | 104 |
| 4x | 132 | 168 | 0 | 132 | 112 | 188 |

No admitted request missed its deadline at any load (882 admitted of 1,200). That is an
observation on these workloads and on a hypothesis search with prompts several times the token
budget (`tests/integration/test_deadlines.py`), not a proof: the bound does not model preemption or
KV-memory pressure (ADR 0004). It also depends on the scheduler prefilling in the order the bound
assumes -- an earlier version continued every running prefill first, and a counterexample found in
review (an earlier-deadline request arriving while a loose-deadline prompt was mid-prefill) was
admitted and missed; it is now a test. The test is deliberately pessimistic (every iteration
costed at a full budget): at low load it refuses requests that would have made it, and at 2x load
accepting everything completes slightly more on time. What it buys is that overload produces
immediate, retryable refusals instead of requests that hold KV blocks and then time out: at 4x,
accept-all misses 188 of 300 deadlines.

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

The equivalence rows compare every logits row the engine sampled from (mixed prefill/decode
batches, chunk boundaries inside blocks, prefix-cache hits, recomputation after preemption,
forked samples) with a dense, cache-free recompute of the same sequence. CI fails if either gap
exceeds 1e-4 or if the scenario stops producing preemptions and prefix hits.

### Demo excerpt

`uv run python tasks.py demo` (same machine, 7.9 s; timings vary between runs, and the served
section runs client and server in one process, so its HTTP latencies include GIL contention and
are illustrative only; the admission deadline is set to 40 profiled worst-case iterations):

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

## Security

Threats, controls and tests are in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md); every claim here
maps to evidence in [CLAIMS.md](CLAIMS.md).

- **Authentication and entitlements.** API keys are held as SHA-256 digests. Each key has a policy:
  allowed SLO classes (asking for a class you are not entitled to is a 403, not a priority
  upgrade), tokens per minute charged for the worst case at admission (`prompt + n *
  max_new_tokens`, refunded if the server refuses the request for load), concurrent sequences
  (an `n`-way request takes `n`), prompt, output and `n` ceilings. A valid key without a policy
  cannot call the engine, and with no policies configured every key is denied.
- **Hard limits.** Closed request schema; prompt tokens, `max_new_tokens`, `n` and total context
  are bounded by the engine and by the key, before any block is allocated. A wall-clock limit per
  request (or its own `deadline_ms`, which may not exceed it) aborts and frees blocks; a client
  that hangs up, streaming or not, has its request aborted and its blocks freed.
- **Adversarial tenants.** One key sending maximum-length prompts, cache-thrashing unique prompts,
  maximum-`max_tokens` requests or eight-sample requests does not push another key's interactive
  p99 past its SLO (tested with a simulated executor; the lower-class prefill allowance, per-key
  limits and SLO entitlement are each part of that).
- **Weights.** safetensors only, loaded after every file's SHA-256 is checked against the
  manifest, which must cover the weights and the tokenizer; the manifest digest can be pinned. No
  pickle anywhere in `src/`.
- **Privacy.** Prompts and completions are never logged or traced: lengths and a prompt SHA-256
  only. Prefix-cache reuse is per key by default, closing the cross-tenant TTFT side channel
  (ADR 0003).
- **Admin.** Cache flush and weight swap need the admin scope and load checkpoints by name under
  a fixed root with a required manifest digest. The intent is appended to a hash-chained audit log
  before anything changes, and the action is refused (503) if it cannot be recorded; refused
  attempts are recorded too. With `APP_AUDIT_HMAC_KEY` the chain is an HMAC, so it cannot be
  rebuilt without the key; every append logs the new head, the anchor
  `paged-kv-serving audit verify --expect-head` needs to detect truncation.
- **Telemetry.** One row per finished request (lengths, timings, a prompt SHA-256, never text) is
  validated against a strict data contract before it is written to
  `APP_DATA_DIR/telemetry/requests.jsonl`.
- **Supply chain and container.** Locked dependencies, pip-audit, bandit, semgrep, gitleaks,
  hadolint, trivy and a CycloneDX SBOM in CI; non-root, read-only image with capabilities dropped.

## Observability

- Prometheus (`/metrics`, authenticated by default): `engine_ttft_seconds`, `engine_tpot_seconds`,
  `engine_request_latency_seconds`, `engine_iteration_seconds`, `engine_batched_tokens`,
  `engine_kv_blocks_used_ratio`, `engine_prefix_cache_hit_ratio`, `engine_preemptions_total`,
  `engine_rejections_total{reason}`, `engine_aborted_requests_total`, `engine_queue_depth`,
  `engine_running_sequences`, `engine_slo_attainment`, `engine_prompt_length_psi`, plus HTTP
  metrics by route template.
- OpenTelemetry: a server span per HTTP request and an `engine.completion` child span per
  completion with prompt and completion lengths, class, outcome, TTFT and preemptions.
- `/healthz` (liveness) and `/readyz` (weights loaded and warmed, engine loop running, accepting,
  allocator invariants hold -- checked under the engine lock, never mid-iteration). SIGTERM
  drains: new requests get 503, in-flight ones finish.
- Workload drift: PSI of recent prompt lengths against the first prompts the server saw.

## Testing

`uv run python tasks.py test` runs 529 tests (528 pass; one template-only check is skipped) with
branch coverage 96.23% (floor 85%). Property tests take their example counts from the active
hypothesis profile (`tests/hypothesis_budget.py`): CI's `ci` profile runs five times the local
budget, derandomised. The core of it is property-based:

- `tests/property/test_engine_equivalence.py`: hypothesis draws engine configurations (block size,
  pools within two blocks of one maximum-length sequence or larger, budgets small enough to chunk,
  prefix cache on or off, allowance shares) and workloads (shared prefixes, two tenants, two
  classes, greedy and sampled, `n` up to 8, prompts several times the token budget, staggered
  arrivals); every sampled logits row must match a dense recompute within 1e-4, no iteration may
  be empty while work is queued, every request must finish with its exact token count, and the
  pool must be whole afterwards.
- `tests/property/test_allocator_state.py`: a hypothesis state machine drives the real
  `KVCacheManager` through admit, prefill, decode, fork, preempt, finish and evict, checking after
  every step that reference counts are consistent, every sequence reads back its own tokens,
  cached blocks never change, and nothing leaks.
- `tests/property/test_equivalence.py`: chunk splits, scattered block tables, tile sizes, batch
  composition and shared prefixes at the runner level.
- `tests/integration/test_deadlines.py` and `tests/integration/test_adversarial.py`: admission
  (including a hypothesis search for admitted misses) and multi-tenant isolation on a fake clock;
  `tests/integration/test_api.py`: the served API, including a real loopback server for
  disconnects, deadlines, audit tampering and telemetry.

Each fix from an independent review comes with a regression test that failed before the fix; its
name says what it guards and its docstring says what used to happen.

## Non-goals

- **Quality.** The model is random-init and never trained. The spec's optional PyTorch `train`
  extra for producing a toy checkpoint was cut: it adds a heavy dependency and nothing the
  equivalence claims need.
- **Absolute throughput.** Float32 NumPy on one CPU thread; no GPU kernels, no fused attention, no
  quantisation, no CUDA graphs. Only ratios on the same machine are claimed.
- **Swap-based preemption.** Preempted sequences are recomputed, never swapped to a second memory
  tier (ADR 0002).
- **Tenant fair queuing.** Isolation comes from per-key limits, SLO classes and the lower-class
  allowance; keys of the same class are not weighted against each other.
- **Distributed serving.** One process, one host; no tensor or pipeline parallelism, no
  disaggregated prefill, no multi-replica routing.
- **Speculative decoding, beam search, stop strings, logprobs.** Not implemented.
- **A tighter or proven admission bound.** Admission costs every iteration at its worst case and
  ignores KV memory, preemption and the running-set cap in its bound (ADR 0004). "No admitted
  request missed its deadline" is observed on the benchmark workloads and in a hypothesis search
  with an exact latency model on a fake clock; it is not a proof, and is not guaranteed under
  memory pressure that forces preemption.
- **Batched attention.** Attention runs as a Python loop over the sequences of a batch (paged,
  tiled, online softmax); everything else runs once over the flattened ragged batch.
- **Telemetry shipping.** Telemetry is a local JSONL file with one rotation; shipping it anywhere is
  the deployment's job. `ml/registry.py` and `ml/serialization.py` come from the portfolio
  baseline and are used only by `scripts/check_baseline.py`.
- **Bisectable early history.** The first commits were made in one burst and several of them do not
  pass the test suite on their own (only their final state was checked); commits since are made
  after the suite passes. The history was not rewritten.
- **Hugging Face `tokenizers`.** The spec listed it; the tokenizer here is a byte-level BPE trained
  in-repo and stored as plain JSON, which keeps the checkpoint format free of any executable or
  library-specific serialisation.
- **HTTP-level performance numbers.** The HTTP path is tested and demonstrated, but only the
  engine is benchmarked; the in-process demo shares one GIL between client and server.
- **Containers verified locally.** The Dockerfile and compose file are built and smoke-tested in
  CI only.

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
