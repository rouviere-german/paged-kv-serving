# 5. Cap lower-class prefill per iteration while a higher class has work

- Status: accepted
- Date: 2026-09-22

## Context

The scheduler fills each iteration's token budget with decode tokens first and then prefill in
priority order, so interactive work is always scheduled before batch work. That ordering is not
enough on its own. Iteration time grows with the number of tokens in the batch, and whatever
budget interactive work leaves over is filled with batch prefill. Under a flood of long batch
prompts every iteration becomes a full-budget iteration, and every interactive decode step --
one token of useful work -- takes as long as a 64- or 128-token forward pass.

The adversarial test that floods the engine with maximum-length batch prompts from one key
showed exactly that: the victim's interactive latency rose from about 13 ms per request to about
400 ms, and because each request stayed in flight longer, the victim's own per-key concurrency
cap started refusing its requests.

## Decision

`EngineConfig.lower_class_prefill_share` (default 0.25). In every iteration where a sequence of a
higher SLO class is running or waiting, prefill of all lower classes together may take at most
`share * max_batched_tokens` tokens. Decode tokens are never capped, and when no higher-class work
exists the lower class gets the whole budget.

Admission control uses the allowance as the guaranteed prefill rate of the lower classes
(`AdmissionController.prefill_tokens_per_iteration_for`), unconditionally, because higher-class
work can arrive after a batch request was admitted.

## Consequences

- Under the long-prompt flood the victim keeps all 24 requests, meets every deadline, and its p99
  stays under a quarter of the SLO; with the allowance switched off its p99 is more than twice as
  high (`tests/integration/test_adversarial.py::test_the_lower_class_prefill_share_is_what_protects_interactive_latency`).
- Batch prompts take longer to prefill while interactive traffic is present, and admission makes
  batch deadlines correspondingly more conservative.
- The share is a static knob. An adaptive share driven by measured interactive TPOT against its
  target would use capacity better; it is not implemented.
