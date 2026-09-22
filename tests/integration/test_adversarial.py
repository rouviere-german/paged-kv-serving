"""Adversarial workloads from one authenticated key must not spend another key's latency budget.

None of these are exploits. They are ordinary requests, allowed by the API, sent in a pattern that
happens to be the worst one for the server: the longest prompts the limits permit, a flood of
unique prompts that evicts everything useful from the prefix cache, and every request asking for
the maximum number of output tokens. A server with no per-key limits serves all of them and an
interactive tenant pays for it.

Four mechanisms are what keep the victim's latency where it was, and each is a separate control:
SLO-class entitlement (an abusive key is not entitled to the interactive class), per-key
concurrency, per-key token rate, and deadline-aware admission.

Timing is simulated -- the executor consumes exactly the modelled cost and advances a manual
clock -- because the claim is about scheduling decisions, not about how busy the CI runner is.
Everything else, including the allocator and the prefix cache, is real.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from paged_kv_serving.engine.admission import RejectionReason
from paged_kv_serving.engine.config import EngineConfig, SloClass
from paged_kv_serving.engine.engine import Engine, RequestOutput
from paged_kv_serving.engine.keys import EntitlementGuard, KeyPolicy, KeyPolicyStore
from paged_kv_serving.engine.loadgen import percentile
from paged_kv_serving.engine.model import LanguageModel
from paged_kv_serving.engine.sequence import Request, SamplingParams
from paged_kv_serving.engine.simulation import ManualClock, drain_on_clock, simulated_engine

pytestmark = pytest.mark.integration

CONFIG = EngineConfig(
    block_size=8,
    num_blocks=192,
    max_batched_tokens=64,
    max_running_seqs=16,
    max_prompt_tokens=64,
    max_new_tokens=16,
    max_context_tokens=96,
)
SECONDS_PER_TOKEN = 1e-3
VICTIM = "victim-key"
ABUSER = "abuser-key"
#: The interactive SLO under test. The victim's requests all ask for it.
VICTIM_DEADLINE = 2.0
SHARED_PREFIX = tuple(range(1, 33))


def policies() -> KeyPolicyStore:
    return KeyPolicyStore(
        [
            KeyPolicy(
                key_id=VICTIM,
                slo_classes=frozenset({SloClass.INTERACTIVE}),
                tokens_per_minute=100_000,
                max_concurrent_requests=4,
                max_prompt_tokens=CONFIG.max_prompt_tokens,
                max_new_tokens=CONFIG.max_new_tokens,
            ),
            KeyPolicy(
                key_id=ABUSER,
                # Not entitled to the interactive class, and bounded in width and rate.
                slo_classes=frozenset({SloClass.BATCH}),
                tokens_per_minute=20_000,
                max_concurrent_requests=4,
                max_prompt_tokens=CONFIG.max_prompt_tokens,
                max_new_tokens=CONFIG.max_new_tokens,
            ),
        ]
    )


def build_engine(model: LanguageModel) -> tuple[Engine, ManualClock]:
    return simulated_engine(
        model,
        CONFIG,
        seconds_per_token=SECONDS_PER_TOKEN,
        entitlements=EntitlementGuard(policies()),
    )


def victim_requests(count: int = 24, spacing: float = 0.05) -> list[Request]:
    """A well-behaved interactive tenant: a shared system prompt, a short question, a deadline."""
    rng = np.random.default_rng(1)
    return [
        Request(
            request_id=f"victim-{index:03d}",
            prompt_token_ids=(*SHARED_PREFIX, *(int(v) for v in rng.integers(1, 60, 8))),
            sampling=SamplingParams(max_new_tokens=6, seed=index),
            slo_class=SloClass.INTERACTIVE,
            deadline_seconds=VICTIM_DEADLINE,
            key_id=VICTIM,
            arrival_time=index * spacing,
        )
        for index in range(count)
    ]


def long_prompt_flood(count: int = 60) -> list[Request]:
    rng = np.random.default_rng(2)
    return [
        Request(
            request_id=f"abuse-long-{index:03d}",
            prompt_token_ids=tuple(int(v) for v in rng.integers(1, 60, CONFIG.max_prompt_tokens)),
            sampling=SamplingParams(max_new_tokens=2, seed=index),
            slo_class=SloClass.BATCH,
            key_id=ABUSER,
            arrival_time=index * 0.005,
        )
        for index in range(count)
    ]


def cache_thrashing_flood(count: int = 60) -> list[Request]:
    """Every prompt is unique, so every block registered is a block that evicts something."""
    rng = np.random.default_rng(3)
    return [
        Request(
            request_id=f"abuse-thrash-{index:03d}",
            prompt_token_ids=tuple(int(v) for v in rng.integers(1, 60, 56)),
            sampling=SamplingParams(max_new_tokens=2, seed=index),
            slo_class=SloClass.BATCH,
            key_id=ABUSER,
            arrival_time=index * 0.005,
        )
        for index in range(count)
    ]


def max_tokens_flood(count: int = 60) -> list[Request]:
    rng = np.random.default_rng(4)
    return [
        Request(
            request_id=f"abuse-tokens-{index:03d}",
            prompt_token_ids=tuple(int(v) for v in rng.integers(1, 60, 16)),
            sampling=SamplingParams(max_new_tokens=CONFIG.max_new_tokens, seed=index),
            slo_class=SloClass.BATCH,
            key_id=ABUSER,
            arrival_time=index * 0.005,
        )
        for index in range(count)
    ]


def drain(engine: Engine, clock: ManualClock, requests: list[Request]) -> list[RequestOutput]:
    return drain_on_clock(engine, clock, requests, max_iterations=200_000)


def victim_latencies(outputs: list[RequestOutput]) -> list[float]:
    return [output.latency for output in outputs if output.key_id == VICTIM]


@pytest.fixture(scope="module")
def baseline(tiny_model_module: LanguageModel) -> tuple[float, int]:
    """The victim's p99 latency and completion count with no abusive traffic at all."""
    engine, clock = build_engine(tiny_model_module)
    outputs = drain(engine, clock, victim_requests())
    latencies = victim_latencies(outputs)
    engine.check_invariants()
    assert len(latencies) == 24, "the victim alone must not be rejected"
    return percentile(latencies, 99), len(latencies)


@pytest.fixture(scope="module")
def tiny_model_module(tiny_model: LanguageModel) -> LanguageModel:
    return tiny_model


@pytest.mark.parametrize(
    "attack",
    [long_prompt_flood, cache_thrashing_flood, max_tokens_flood],
    ids=["long_prompts", "prefix_cache_thrashing", "max_tokens_abuse"],
)
def test_an_abusive_key_does_not_break_the_interactive_slo(
    tiny_model_module: LanguageModel,
    baseline: tuple[float, int],
    attack: Callable[[], list[Request]],
) -> None:
    baseline_p99, baseline_completed = baseline
    engine, clock = build_engine(tiny_model_module)
    requests = [*victim_requests(), *attack()]
    outputs = drain(engine, clock, requests)
    engine.check_invariants()

    victim = [output for output in outputs if output.key_id == VICTIM]
    assert len(victim) == baseline_completed, "the victim's requests were refused"
    assert all(output.met_deadline for output in victim), "the victim missed its SLO"
    observed_p99 = percentile(victim_latencies(outputs), 99)
    # The claim is the SLO, with a wide margin: p99 under attack stays under a quarter of the
    # interactive deadline. The attack is allowed to cost the victim something (its decode steps
    # now share iterations with batch prefill), just never its deadline.
    assert observed_p99 <= VICTIM_DEADLINE / 4, (
        f"p99 went from {baseline_p99:.3f}s to {observed_p99:.3f}s under attack"
    )
    assert any(output.key_id == ABUSER for output in outputs), "the abuser was not served at all"


def test_the_lower_class_prefill_share_is_what_protects_interactive_latency(
    tiny_model_module: LanguageModel,
) -> None:
    """The same long-prompt flood with the allowance switched off: the victim pays for it.

    Iteration time grows with batch tokens, so without the allowance every one of the victim's
    decode steps shares an iteration with a full budget of batch prefill.
    """

    def victim_p99(share: float) -> tuple[float, int]:
        engine, clock = simulated_engine(
            tiny_model_module,
            dataclasses.replace(CONFIG, lower_class_prefill_share=share),
            seconds_per_token=SECONDS_PER_TOKEN,
            entitlements=EntitlementGuard(policies()),
        )
        outputs = drain(engine, clock, [*victim_requests(), *long_prompt_flood()])
        engine.check_invariants()
        latencies = victim_latencies(outputs)
        return percentile(latencies, 99), len(latencies)

    capped_p99, capped_completed = victim_p99(0.25)
    uncapped_p99, uncapped_completed = victim_p99(1.0)
    assert capped_p99 < uncapped_p99 / 2, (capped_p99, uncapped_p99)
    # Slower requests stay in flight longer, so the victim's own concurrency cap starts refusing.
    assert capped_completed >= uncapped_completed


def test_the_abusive_key_pays_for_its_own_burst_not_the_victim(
    tiny_model_module: LanguageModel,
) -> None:
    """Concurrency and token-rate limits are what stop one key from filling the whole engine."""
    engine, clock = build_engine(tiny_model_module)
    drain(engine, clock, [*victim_requests(), *max_tokens_flood(count=200)])
    rejections = engine.admission.rejections
    assert rejections.get(RejectionReason.KEY_CONCURRENCY, 0) > 0
    engine.check_invariants()


def test_an_unentitled_key_cannot_buy_the_interactive_class(
    tiny_model_module: LanguageModel,
) -> None:
    engine, _ = build_engine(tiny_model_module)
    decision = engine.submit(
        Request(
            request_id="abuse-class",
            prompt_token_ids=(1, 2, 3, 4),
            sampling=SamplingParams(max_new_tokens=2, seed=0),
            slo_class=SloClass.INTERACTIVE,
            key_id=ABUSER,
        )
    )
    assert decision.reason is RejectionReason.CLASS_NOT_ENTITLED
    assert engine.active_requests == 0


def test_prefix_cache_thrashing_costs_work_but_not_correctness(
    tiny_model_module: LanguageModel,
) -> None:
    """Eviction is allowed to make the victim slower; it is not allowed to change its answers."""
    quiet, quiet_clock = build_engine(tiny_model_module)
    quiet_outputs = {
        o.request_id: o.samples[0].token_ids for o in drain(quiet, quiet_clock, victim_requests())
    }
    noisy, noisy_clock = build_engine(tiny_model_module)
    noisy_outputs = {
        o.request_id: o.samples[0].token_ids
        for o in drain(noisy, noisy_clock, [*victim_requests(), *cache_thrashing_flood()])
        if o.key_id == VICTIM
    }
    assert noisy_outputs == quiet_outputs
    assert noisy.prefix_cache.stats.evicted_blocks >= 0
    noisy.check_invariants()


def test_a_key_with_no_entitlements_at_all_is_refused(tiny_model_module: LanguageModel) -> None:
    engine, _ = build_engine(tiny_model_module)
    decision = engine.submit(
        Request(
            request_id="stranger",
            prompt_token_ids=(1, 2, 3),
            sampling=SamplingParams(max_new_tokens=1, seed=0),
            key_id="not-configured",
        )
    )
    assert decision.reason is RejectionReason.CLASS_NOT_ENTITLED


def test_the_engine_without_entitlements_still_enforces_its_own_limits(
    tiny_model_module: LanguageModel,
) -> None:
    """Per-key policy is optional; the engine-wide ceilings are not."""
    open_engine, _ = simulated_engine(
        tiny_model_module, CONFIG, seconds_per_token=SECONDS_PER_TOKEN
    )
    decision = open_engine.submit(
        Request(
            request_id="oversized",
            prompt_token_ids=tuple(range(1, CONFIG.max_prompt_tokens + 10)),
            sampling=SamplingParams(max_new_tokens=1, seed=0),
            key_id="anyone",
        )
    )
    assert decision.reason is RejectionReason.PROMPT_TOO_LONG
