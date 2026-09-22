# 4. Admit by counting iterations at their worst-case cost

- Status: accepted
- Date: 2026-09-22

## Context

A request with a deadline should be refused *at arrival* when it cannot finish in time: an early
503 with a truthful `Retry-After` is useful to a client, a timeout after 20 seconds of queueing is
not. That needs a prediction of when the request would finish, and the obvious model -- tokens of
work ahead divided by a token rate -- is wrong for an LLM engine in two ways:

1. **Decode is serial per request and parallel across requests.** Every iteration advances every
   running sequence by one token, so a request asking for 20 tokens needs 20 iterations however
   idle the machine is, and twenty such requests still need 20 iterations between them. A fluid
   token-rate model predicts that an idle machine finishes 20 tokens in the time of two iterations.
2. **Iteration cost depends on work the scheduler has not seen yet.** The token budget of every
   iteration can be filled by prompts that arrive *after* a request was admitted, including
   lower-priority ones. A guarantee computed from today's batch sizes can be broken by tomorrow's
   arrivals.

## Decision

`AdmissionController` counts iterations, not tokens, and costs every iteration at its worst case:

    iterations(D) = ceil(prefill tokens ahead of D / guaranteed prefill tokens per iteration)
                    + the largest number of output tokens still owed by any request due by D

    admit iff, for every class and every committed deadline D:
        iterations(D) * L(max_batched_tokens) * safety_factor <= D - now

- "Ahead of D" follows the scheduler's order: all work of higher classes plus same-class work due
  no later than D.
- The guaranteed prefill rate is the budget left after decode for the top class, and the
  lower-class allowance for lower classes (ADR 0005), so a later interactive arrival cannot slow
  admitted batch work below what admission assumed.
- `L(n) = intercept + slope * n` is fitted by least squares at start-up (`Engine.profile_latency`)
  and multiplied by an EWMA of measured-over-predicted iteration times.
- Deadlines already passed are excluded from the test but kept in the demand, so a late request
  still delays the ones behind it without making every new request infeasible.

## Consequences

- With an exact latency model and a fake clock, no admitted request misses its deadline: 0 misses
  among the 877 requests admitted out of 1,200 offered at 0.5x-4x load (`bench/bench_serving.py`,
  admission table; the simulation is deterministic), and
  `tests/integration/test_deadlines.py::test_no_admitted_request_misses_its_deadline_under_heavy_overload`.
- The price is pessimism. Costing a two-token decode iteration like a full 128-token one refuses
  requests that would have made it: in the same benchmark, accepting everything completes 197
  on time at 2x load against 187 with admission, and at 0.5x load admission refuses 13 requests
  whose deadlines were tight but, as it turned out, feasible. What admission buys is the other
  side: at 4x load accept-all lets 187 of 300 requests miss their deadline (each one holding KV
  blocks until it does), while admission refuses 174 of them at arrival and misses none.
- The bound ignores two scheduler limits: KV memory (a preemption can delay a request) and the
  running-set cap. Both are covered only by the safety factor; the simulated tests size the pool
  so that neither binds.
- A tighter bound would cost decode-only iterations at `L(running sequences)`. It is not used,
  because lower-priority prefill can join any iteration and make it a full one.
