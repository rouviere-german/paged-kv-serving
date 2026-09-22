"""Per-key entitlements: SLO class, token rate, concurrency and per-request ceilings.

Authentication says *who* is calling; this module says what that caller may ask the engine for.
Without it, one key can take the whole server: a flood of 512-token prompts, a handful of requests
each asking for the maximum number of new tokens, or simply more concurrent streams than the KV
pool can hold. All three are ordinary load from an authenticated client, so they have to be bounded
per key rather than detected.

Three independent limits, because they fail differently:

* ``max_concurrent_requests`` bounds how much of the *KV pool* one key can occupy at once. It
  counts sequences, so an ``n``-way request takes ``n`` of it: every sample holds its own blocks
  and its own running slot;
* ``tokens_per_minute`` bounds sustained work with a token bucket, charged at admission for the
  worst case (the prompt plus ``n`` times the full ``max_new_tokens``) so a key cannot reserve
  cheaply and then generate expensively;
* ``max_prompt_tokens``, ``max_new_tokens`` and ``max_n`` bound a single request.

``slo_classes`` is an entitlement, not a hint: a key that is not entitled to the interactive class
cannot reach the front of the queue by asking for it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from paged_kv_serving.core.ratelimit import TokenBucket
from paged_kv_serving.engine.admission import RejectionReason
from paged_kv_serving.engine.config import SloClass

__all__ = ["EntitlementGuard", "KeyPolicy", "KeyPolicyStore"]


@dataclass(frozen=True, slots=True)
class KeyPolicy:
    """What one API key is allowed to ask for."""

    key_id: str
    slo_classes: frozenset[SloClass] = frozenset({SloClass.BATCH})
    tokens_per_minute: int = 6_000
    max_concurrent_requests: int = 4
    max_prompt_tokens: int = 512
    max_new_tokens: int = 128
    admin: bool = False
    # Parallel samples per request. Opt-in: one request with n samples costs n sequences.
    max_n: int = 1

    def __post_init__(self) -> None:
        if not self.key_id:
            raise ValueError("key_id must not be empty")
        if not self.slo_classes:
            raise ValueError("a policy must grant at least one SLO class")
        if self.tokens_per_minute < 1:
            raise ValueError("tokens_per_minute must be >= 1")
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be >= 1")
        if self.max_prompt_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("max_prompt_tokens and max_new_tokens must be >= 1")
        if self.max_n < 1:
            raise ValueError("max_n must be >= 1")

    def allows(self, slo_class: SloClass) -> bool:
        return slo_class in self.slo_classes

    def ceiling_violation(
        self, prompt_tokens: int, max_new_tokens: int, n: int
    ) -> RejectionReason | None:
        """The per-request ceiling a request breaks, if any."""
        if prompt_tokens > self.max_prompt_tokens:
            return RejectionReason.PROMPT_TOO_LONG
        if max_new_tokens > self.max_new_tokens:
            return RejectionReason.CONTEXT_TOO_LONG
        if n > self.max_n:
            return RejectionReason.TOO_MANY_SAMPLES
        return None


class KeyPolicyStore:
    """Immutable map from key id to policy, with a default for keys that have none."""

    __slots__ = ("_default", "_policies")

    def __init__(self, policies: Iterable[KeyPolicy], *, default: KeyPolicy | None = None) -> None:
        self._policies: dict[str, KeyPolicy] = {}
        for policy in policies:
            if policy.key_id in self._policies:
                raise ValueError(f"duplicate policy for key id {policy.key_id}")
            self._policies[policy.key_id] = policy
        self._default = default

    def __len__(self) -> int:
        return len(self._policies)

    def get(self, key_id: str) -> KeyPolicy | None:
        """The policy for ``key_id``, the configured default, or None (which denies)."""
        policy = self._policies.get(key_id)
        if policy is not None:
            return policy
        if self._default is None:
            return None
        return KeyPolicy(
            key_id=key_id,
            slo_classes=self._default.slo_classes,
            tokens_per_minute=self._default.tokens_per_minute,
            max_concurrent_requests=self._default.max_concurrent_requests,
            max_prompt_tokens=self._default.max_prompt_tokens,
            max_new_tokens=self._default.max_new_tokens,
            admin=False,
            max_n=self._default.max_n,
        )

    def admin_key_ids(self) -> frozenset[str]:
        return frozenset(key for key, policy in self._policies.items() if policy.admin)


@dataclass(slots=True)
class EntitlementGuard:
    """Live per-key state: in-flight requests and a token-per-minute bucket."""

    store: KeyPolicyStore
    clock: Callable[[], float] = time.monotonic
    _inflight: dict[str, int] = field(default_factory=dict)
    _buckets: dict[str, TokenBucket] = field(default_factory=dict)

    def inflight(self, key_id: str) -> int:
        return self._inflight.get(key_id, 0)

    def _bucket(self, policy: KeyPolicy) -> TokenBucket:
        bucket = self._buckets.get(policy.key_id)
        if bucket is None:
            # Capacity of one minute's allowance: a key may burst a minute of work, then refills.
            bucket = TokenBucket(
                rate=policy.tokens_per_minute / 60.0,
                capacity=float(policy.tokens_per_minute),
                now=self.clock(),
            )
            self._buckets[policy.key_id] = bucket
        return bucket

    def check(
        self,
        key_id: str,
        *,
        slo_class: SloClass,
        prompt_tokens: int,
        max_new_tokens: int,
        n: int = 1,
    ) -> RejectionReason | None:
        """Return the reason a request must be refused, or None when it may be admitted.

        On success the key's in-flight count grows by ``n`` (one per sample) and the worst-case
        token cost -- the prompt, shared by the samples, plus ``n * max_new_tokens`` -- is
        charged; call :meth:`release` with the same ``n`` when the request finishes or is aborted.
        """
        policy = self.store.get(key_id)
        if policy is None or not policy.allows(slo_class):
            return RejectionReason.CLASS_NOT_ENTITLED
        ceiling = policy.ceiling_violation(prompt_tokens, max_new_tokens, n)
        if ceiling is not None:
            return ceiling
        if self.inflight(key_id) + n > policy.max_concurrent_requests:
            return RejectionReason.KEY_CONCURRENCY
        cost = float(prompt_tokens + n * max_new_tokens)
        allowed, _ = self._bucket(policy).try_acquire(self.clock(), cost)
        if not allowed:
            return RejectionReason.KEY_TOKEN_RATE
        self._inflight[key_id] = self.inflight(key_id) + n
        return None

    def release(self, key_id: str, n: int = 1) -> None:
        """Give back a request's ``n`` in-flight slots once it has run (finished or aborted).

        Tokens are not refunded: the key reserved them, and a request that ran -- even one the
        client abandoned -- spent the server's time.
        """
        remaining = self.inflight(key_id) - n
        if remaining > 0:
            self._inflight[key_id] = remaining
        else:
            self._inflight.pop(key_id, None)

    def refund(self, key_id: str, *, prompt_tokens: int, max_new_tokens: int, n: int = 1) -> None:
        """Undo a successful :meth:`check` for a request the *server* then refused.

        Admission control runs after the key's own limits, so a request turned away for load or an
        infeasible deadline has already been charged. It never ran, so the slots and the whole
        charge come back; otherwise a client retrying politely after a 503 would drain its own
        tokens-per-minute and end up with a 429 without ever having been served.
        """
        self.release(key_id, n)
        bucket = self._buckets.get(key_id)
        if bucket is not None:
            bucket.refund(float(prompt_tokens + n * max_new_tokens))

    def snapshot(self) -> Mapping[str, int]:
        return dict(self._inflight)
