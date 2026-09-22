# Model card: paged-kv-serving toy decoder

Format based on Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). The model in this
repository exists to exercise a serving engine. It is **not trained** and produces text that
carries no meaning; every number below is about the engine's numerical behaviour, not about
language quality.

## Model details

- **Name and version:** `paged-kv-serving` random-init decoder, checkpoint schema version 1.
- **Type:** decoder-only transformer: pre-norm RMSNorm, rotary position embeddings, multi-head
  causal attention, SwiGLU feed-forward, output projection tied to the token embedding
  (`engine/model.py`).
- **Size:** the demo and benchmark checkpoint has 4 layers, `d_model` 128, 4 heads, `d_ff` 320,
  a 512-token vocabulary and 820,352 float32 parameters (`build_random_checkpoint(seed=3)`);
  `paged-kv-serving build-checkpoint` builds the same shape with `max_seq_len` 1024.
- **Weights:** `numpy.random.default_rng(seed)` normal initialisation, scaled by `d**-0.5`. No
  training run, no training data, no download.
- **Tokenizer:** byte-level BPE trained in-repo on the short original corpus in
  `engine/corpus.py`; ids 0-255 are raw bytes, so every input encodes. Stored as JSON.
- **Owner:** Rouviere German. **Date:** 2026-09-22. **License:** MIT.
- **Artifact format:** `weights.safetensors` + `tokenizer.json` + `manifest.json`. The manifest
  records the architecture, parameter count, tokenizer vocabulary size and the SHA-256 and byte
  size of every other file; `load_checkpoint` verifies all of them before parsing anything, and
  the manifest's own digest can be pinned (`APP_CHECKPOINT_MANIFEST_SHA256`). Never pickle.

## Intended use

- **Primary use:** a deterministic function of the input for studying serving-engine mechanics --
  paged KV caching, continuous batching, prefix caching, preemption and admission control -- and
  for checking that the optimised path computes exactly what a naive recompute computes.
- **Intended users:** engineers reading or extending the engine.
- **Out-of-scope uses:** anything that consumes the generated text. The output is noise from
  random weights.

## Evaluation

The claim is numerical equivalence, measured as the largest absolute difference between logits
from the paged engine and from `LanguageModel.reference_forward` (dense attention, no cache).

| Check | Result | Source |
|---|---:|---|
| One prompt, arbitrary chunk splits and a reversed block table (4 prompts) | max abs gap 2.44e-06 | `uv run python bench/bench_baseline.py` |
| Whole workload through the scheduler: 16 requests, a 20-block pool (8 preemptions), prefix-cache hits (35 blocks), greedy and sampled, `n = 2` forks; 264 logits rows | max abs gap 3.19e-06 | same |
| Hypothesis-generated engine configurations and workloads | every row below 1e-4 | `tests/property/test_engine_equivalence.py` |

The tolerance used everywhere is 1e-4; the observed gaps are float32 rounding from a different
reduction order (online softmax over blocks against one softmax over the whole row).

## Request data contract

Request telemetry (one row per finished request: ids, key id, prompt SHA-256, class, finish reason,
token counts, TTFT, latency, arrival time) is validated by the strict pandera schema
`RequestRecordSchema` (`ml/contracts.py`). Unknown columns fail validation, so a column holding
prompt or completion text cannot be added by accident; failures are summarised as column, check
and count without the offending values. Cross-column checks: cached prompt tokens never exceed the
prompt, TTFT never exceeds latency.

## Monitoring

The weights never change unless an admin swaps them, so accuracy drift is meaningless here.
Workload drift is not: prompt lengths drive block demand, the prefix-cache hit rate and the latency
model admission control depends on. `PromptLengthMonitor` freezes the first
`APP_DRIFT_REFERENCE_SIZE` prompt lengths a server sees as its reference and exports the
Population Stability Index of the last `APP_DRIFT_WINDOW` prompts against it as
`engine_prompt_length_psi` (below 0.1 stable, above 0.2 worth an alert). With 10 bins the PSI of
two samples from the same distribution has a noise floor of roughly 9/n per sample, which is why
both windows default to 256 prompts.

## Limitations

- No quality evaluation exists or is meaningful.
- Only float32 NumPy on CPU; no quantisation, no GPU kernels.
- The tokenizer's merges come from a few paragraphs of text; most real text encodes close to one
  token per byte.
