# Contributing

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv can install it).

```bash
uv sync --frozen
uv run python tasks.py check      # lint, typecheck, test, audit, baseline
```

If the repository sits on a small or slow drive, keep the virtual environment elsewhere:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/<repo>"
uv sync --frozen
```

## Tasks

`tasks.py` is the cross-platform task runner; the Makefile only delegates to it.

| Target | What it runs |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `format` | `ruff format` and `ruff check --fix` |
| `typecheck` | `mypy --strict` over src, tests, scripts, bench and tasks.py |
| `test` | pytest with branch coverage; fails below 85% |
| `bench` | `bench/bench_baseline.py` and `bench/bench_serving.py`, each against its thresholds file |
| `demo` | the offline demo (no API key) |
| `audit` | bandit and pip-audit |
| `baseline` | `scripts/check_baseline.py` |

## Rules for changes

- **Tests first for behavior.** Mark tests `unit` or `integration`. Use hypothesis for properties
  (invariants, round trips, oracles) rather than hand-picked examples alone.
- **Offline by default.** Tests, demo and bench must not touch the network or download a model;
  `tests/conftest.py` blocks non-loopback sockets. Checkpoints are built from a seed
  (`build_random_checkpoint`), never fetched.
- **Equivalence first.** A change to the scheduler, allocator, runner or attention kernel must keep
  `tests/property/test_engine_equivalence.py` and `tests/property/test_allocator_state.py` green;
  they are the merge gate for the primary pattern (CI job `correctness`).
- **Timing claims on a fake clock.** Claims about scheduling decisions (deadlines, isolation) are
  tested with `engine/simulation.py`, not with a stopwatch.
- **Evidence for claims.** A new README claim needs a row in `CLAIMS.md` pointing to a test,
  bench script or CI job. Performance numbers must be copied from real `bench/` output together
  with the command that produced them.
- **Threat model.** New entry points, tools or artifacts need a STRIDE row in
  `docs/THREAT_MODEL.md` with a control and a test.
- **Decisions.** Record significant trade-offs as ADRs in `docs/adr/`.
- **No pickle.** Weights are safetensors, everything else JSON (or npz without pickling).
- **No secrets in git.** Configuration goes in `.env` (ignored); document new settings in
  `.env.example` (a test enforces that it lists every setting).
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`).
