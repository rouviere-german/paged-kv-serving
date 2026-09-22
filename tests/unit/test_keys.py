from __future__ import annotations

import pytest

from paged_kv_serving.engine.admission import RejectionReason
from paged_kv_serving.engine.config import SloClass
from paged_kv_serving.engine.keys import EntitlementGuard, KeyPolicy, KeyPolicyStore

pytestmark = pytest.mark.unit


def policy(key_id: str = "abc123def456", **kwargs: object) -> KeyPolicy:
    defaults: dict[str, object] = {
        "slo_classes": frozenset({SloClass.INTERACTIVE, SloClass.BATCH}),
        "tokens_per_minute": 600,
        "max_concurrent_requests": 2,
        "max_prompt_tokens": 64,
        "max_new_tokens": 32,
    }
    return KeyPolicy(key_id=key_id, **{**defaults, **kwargs})  # type: ignore[arg-type]


def guard(*policies: KeyPolicy, clock_values: list[float] | None = None) -> EntitlementGuard:
    times = clock_values if clock_values is not None else [0.0]

    def clock() -> float:
        return times[0]

    return EntitlementGuard(KeyPolicyStore(policies), clock=clock)


def test_a_key_without_a_policy_is_denied() -> None:
    entitlements = guard()
    assert (
        entitlements.check("unknown", slo_class=SloClass.BATCH, prompt_tokens=1, max_new_tokens=1)
        is RejectionReason.CLASS_NOT_ENTITLED
    )


def test_slo_class_is_an_entitlement_not_a_request_field() -> None:
    entitlements = guard(policy(slo_classes=frozenset({SloClass.BATCH})))
    assert (
        entitlements.check(
            "abc123def456", slo_class=SloClass.INTERACTIVE, prompt_tokens=1, max_new_tokens=1
        )
        is RejectionReason.CLASS_NOT_ENTITLED
    )
    assert (
        entitlements.check(
            "abc123def456", slo_class=SloClass.BATCH, prompt_tokens=1, max_new_tokens=1
        )
        is None
    )


def test_per_request_ceilings_are_enforced() -> None:
    entitlements = guard(policy())
    assert (
        entitlements.check(
            "abc123def456", slo_class=SloClass.BATCH, prompt_tokens=65, max_new_tokens=1
        )
        is RejectionReason.PROMPT_TOO_LONG
    )
    assert (
        entitlements.check(
            "abc123def456", slo_class=SloClass.BATCH, prompt_tokens=1, max_new_tokens=33
        )
        is RejectionReason.CONTEXT_TOO_LONG
    )


def test_concurrency_is_capped_and_released() -> None:
    entitlements = guard(policy(max_concurrent_requests=2))
    for _ in range(2):
        assert (
            entitlements.check(
                "abc123def456", slo_class=SloClass.BATCH, prompt_tokens=1, max_new_tokens=1
            )
            is None
        )
    assert entitlements.inflight("abc123def456") == 2
    assert (
        entitlements.check(
            "abc123def456", slo_class=SloClass.BATCH, prompt_tokens=1, max_new_tokens=1
        )
        is RejectionReason.KEY_CONCURRENCY
    )
    entitlements.release("abc123def456")
    assert entitlements.inflight("abc123def456") == 1
    assert (
        entitlements.check(
            "abc123def456", slo_class=SloClass.BATCH, prompt_tokens=1, max_new_tokens=1
        )
        is None
    )
    entitlements.release("abc123def456")
    entitlements.release("abc123def456")
    assert entitlements.snapshot() == {}


def test_tokens_are_charged_for_the_worst_case_and_refill_over_time() -> None:
    """A key cannot reserve cheaply and generate expensively: max_new_tokens is charged up front."""
    now = [0.0]
    entitlements = guard(
        policy(tokens_per_minute=120, max_concurrent_requests=100), clock_values=now
    )
    # The bucket holds one minute of allowance: 120 tokens.
    assert (
        entitlements.check(
            "abc123def456", slo_class=SloClass.BATCH, prompt_tokens=10, max_new_tokens=30
        )
        is None
    )
    assert (
        entitlements.check(
            "abc123def456", slo_class=SloClass.BATCH, prompt_tokens=40, max_new_tokens=32
        )
        is None
    )
    assert (
        entitlements.check(
            "abc123def456", slo_class=SloClass.BATCH, prompt_tokens=10, max_new_tokens=30
        )
        is RejectionReason.KEY_TOKEN_RATE
    )
    now[0] = 60.0  # a minute later the bucket has refilled
    assert (
        entitlements.check(
            "abc123def456", slo_class=SloClass.BATCH, prompt_tokens=10, max_new_tokens=30
        )
        is None
    )


def test_one_key_cannot_spend_another_keys_allowance() -> None:
    first, second = policy("aaaaaaaaaaaa", tokens_per_minute=60), policy("bbbbbbbbbbbb")
    entitlements = guard(first, second)
    assert (
        entitlements.check(
            "aaaaaaaaaaaa", slo_class=SloClass.BATCH, prompt_tokens=30, max_new_tokens=30
        )
        is None
    )
    assert (
        entitlements.check(
            "aaaaaaaaaaaa", slo_class=SloClass.BATCH, prompt_tokens=1, max_new_tokens=1
        )
        is RejectionReason.KEY_TOKEN_RATE
    )
    assert (
        entitlements.check(
            "bbbbbbbbbbbb", slo_class=SloClass.BATCH, prompt_tokens=30, max_new_tokens=30
        )
        is None
    )


def test_the_store_rejects_duplicates_and_reports_admin_keys() -> None:
    with pytest.raises(ValueError, match="duplicate policy"):
        KeyPolicyStore([policy(), policy()])
    store = KeyPolicyStore([policy("aaaaaaaaaaaa", admin=True), policy("bbbbbbbbbbbb")])
    assert store.admin_key_ids() == frozenset({"aaaaaaaaaaaa"})
    assert len(store) == 2
    assert store.get("missing") is None


def test_a_default_policy_never_grants_the_admin_scope() -> None:
    store = KeyPolicyStore([], default=policy("template", admin=True))
    derived = store.get("someone-else")
    assert derived is not None
    assert derived.admin is False
    assert derived.key_id == "someone-else"
    assert derived.allows(SloClass.BATCH)


def test_policy_validation() -> None:
    for kwargs, message in (
        ({"key_id": ""}, "key_id"),
        ({"slo_classes": frozenset()}, "at least one SLO class"),
        ({"tokens_per_minute": 0}, "tokens_per_minute"),
        ({"max_concurrent_requests": 0}, "max_concurrent_requests"),
        ({"max_prompt_tokens": 0}, "max_prompt_tokens"),
        ({"max_new_tokens": 0}, "max_prompt_tokens and max_new_tokens"),
    ):
        with pytest.raises(ValueError, match=message):
            policy(**kwargs)


def test_slo_class_parsing() -> None:
    assert SloClass.parse("interactive") is SloClass.INTERACTIVE
    assert SloClass.parse("batch").rank == 1
    with pytest.raises(ValueError, match="unknown SLO class"):
        SloClass.parse("urgent")
