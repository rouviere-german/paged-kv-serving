# Security policy

## Supported versions

Only the latest commit on `main` is supported. This is a portfolio project, not a product with
release branches.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting ("Report a vulnerability" under the repository's Security tab). Include:

- the affected commit and component;
- steps to reproduce, ideally as a failing test;
- the impact you expect.

You can expect an acknowledgement within 7 days and a fix or a written assessment within 30 days.
Credit is given in the fix commit unless you ask otherwise.

## Scope

In scope: the code in this repository, its container image and its CI configuration.
Out of scope: denial of service by traffic volume, findings that need a compromised host, and
issues in third-party dependencies that are already publicly tracked (please report those
upstream).

## How this repository is hardened

- Threats, controls and the tests that prove them: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- Every security claim is linked to evidence in [CLAIMS.md](CLAIMS.md).
- API keys are stored as SHA-256 digests; each key has a policy (SLO classes, tokens per minute,
  concurrent sequences, prompt and output ceilings, parallel samples `n`, admin scope). A key
  without a policy cannot call the engine, including when no policies are configured at all. An
  `n`-way request counts `n` sequences and is charged `prompt + n * max_new_tokens`.
- Weights are loaded only from safetensors after every file's SHA-256 has been checked against the
  checkpoint manifest, which must list the weights and the tokenizer by plain file name; the
  manifest digest can be pinned, and the admin weight swap requires it.
  Nothing in `src/` uses pickle or a pickle-compatible format.
- Prompts and completions are never logged or put in trace spans: lengths and a SHA-256 of the
  prompt only.
- Prefix-cache reuse is scoped to the calling key by default, closing the cross-tenant
  time-to-first-token side channel (docs/adr/0003-per-key-prefix-cache-scope.md).
- Admin operations (prefix-cache flush, weight swap) need the admin scope. The intent is appended
  to a hash-chained audit log (HMAC-SHA256 with `APP_AUDIT_HMAC_KEY`) before anything changes, and
  the action is refused if it cannot be recorded; refused attempts are recorded too. Every append
  logs the new chain head, which `paged-kv-serving audit verify --expect-head` checks against.
- CI runs ruff, mypy --strict, pytest with a coverage floor, bandit, semgrep (registry packs plus
  `.semgrep/rules.yml`), pip-audit, gitleaks, hadolint, a trivy image scan and a syft CycloneDX
  SBOM.
- Secrets come only from the environment; only `.env.example` is committed.
