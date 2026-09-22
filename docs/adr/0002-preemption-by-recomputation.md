# 2. Preempt by recomputation, not by swapping KV blocks out

- Status: accepted
- Date: 2026-09-22

## Context

When the KV pool cannot hold the next decode token of every running sequence, something has to
give up its blocks. PagedAttention-style engines have two options (Kwon et al., SOSP 2023):

- **swap**: copy the victim's blocks to a second, slower memory tier (host RAM behind GPU memory)
  and copy them back when it is rescheduled;
- **recompute**: free the blocks and, when the victim is rescheduled, run its prompt *and the
  tokens it already generated* through prefill again.

Here the "fast" tier is already host RAM (a NumPy array), so a swap tier would be a second copy
of the same kind of memory. Swapping also needs its own allocator, a policy for when the swap
space is full, and an extra state (swapped) in every queue.

## Decision

Preemption always recomputes. `Scheduler._preempt` releases the victim's blocks and puts it back
on the waiting queue with `num_computed_tokens = 0` but **with its generated tokens kept**
(`Sequence.reset_for_recompute`). On re-admission the prefix cache is consulted again, so the full
blocks of the victim's own prompt are often still cached and only the tail is recomputed.

The victim is the lowest-priority running sequence that has not been scheduled yet in the
iteration -- a decoder still waiting for its token or a sequence part-way through prefill
(`Scheduler._pick_victim`). Nothing new is admitted in an iteration that preempted, so a victim is
not immediately re-admitted into the blocks it just gave up.

## Consequences

- Preemption costs compute (the recomputed tokens) and never output: the sampled tokens are part
  of the recomputed text, and greedy decoding consumes no randomness, so a preempted request
  produces exactly what it would have without preemption. This is tested token-for-token
  (`tests/integration/test_engine.py::test_a_preempted_request_produces_the_same_tokens`) and
  logit-for-logit across generated workloads
  (`tests/property/test_engine_equivalence.py::test_every_sampled_logits_row_matches_a_cache_free_recompute`).
- A long sequence preempted late pays for a long recompute. With a GPU and PCIe-attached host
  memory, swapping would often be cheaper for long contexts; that trade-off is not measured here.
- There is one fewer memory tier to size, monitor and secure.
