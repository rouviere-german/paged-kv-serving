# 3. Scope prefix-cache reuse to the calling API key by default

- Status: accepted
- Date: 2026-09-22

## Context

The prefix cache keys every full KV block by a hash chain over the block's tokens and every block
in front of it (`engine/prefix_cache.py`). Two requests with the same first 128 tokens share the
physical blocks of those tokens, which is what cuts time to first token on a shared system prompt.

If the chain is shared across tenants, the cache becomes a timing oracle: a caller can send a
guessed prompt prefix and measure its own TTFT. A fast answer means *somebody else* recently sent
that prefix. For a system prompt that is merely inefficient to hide; for a prompt that contains a
document, an e-mail or a password reset link, it is a disclosure. Timing side channels through
shared prompt and KV caches in LLM serving have been demonstrated in published security research
since 2024; the mechanism needs nothing more than the TTFT a client already observes.

## Decision

The chain seed is the caller's key id (`PrefixCache.scope_bytes`): with the default
`APP_ENGINE_PREFIX_CACHE_SCOPE=per_key`, `h_0 = H(domain || key_id)`, so chains from two keys can
never produce a common prefix and a lookup can only hit blocks the same key produced. A
`global` scope exists for single-tenant deployments that want the higher hit rate and accept the
side channel.

## Consequences

- Reuse inside a tenant is unaffected: the shared-system-prompt benchmark (one key) measures a
  76.7% cut in prefill tokens (`bench/bench_serving.py`).
- Two tenants with the same system prompt each pay for it once. Across many small tenants this
  lowers the hit rate compared with a global cache.
- Isolation is by key id, not by user. Several users behind one API key share a scope.
- Tested: `tests/integration/test_engine.py::test_prefix_reuse_does_not_cross_api_keys` and
  `tests/unit/test_prefix_cache.py::test_scopes_never_produce_a_common_prefix`.
