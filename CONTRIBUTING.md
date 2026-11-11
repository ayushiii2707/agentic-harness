# Contributing

Thank you for helping. A few rules keep the project trustworthy.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker pull python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159  # sandbox tests
```

## Before you open a pull request

```bash
ruff check src tests scripts experiments && ruff format --check src tests scripts experiments
mypy src
python -m pytest -q
```

CI runs the same commands on Python 3.10–3.13, plus the Docker sandbox tests and a clean-install check.

## Rules

- **Tests come with every change.**
  - A bug fix comes with a test that fails before the fix.
  - Prefer extending an existing table-driven test over adding a near-duplicate.
- **No network in the default test run.**
  - Provider tests use synthetic response bodies (`respx`).
  - Live checks belong in `scripts/` and must say what they cost.
- **Never present mocked or simulated results as real model behaviour.** Simulated trials are flagged
  `simulated` in the store.
- **Strategies are versioned.** Changing a strategy's wording without bumping its version disables it, because its
  stored evidence would describe a different prompt.
- **A strategy must keep the user's intent.** New kinds of transformation go through
  `transform.apply_strategy` and its preservation checks; a guard is never bypassed.
- **Never weaken the sandbox, the budget reservation or the privacy defaults** without a decision record in
  `docs/DECISIONS.md`.
- **Credentials never go into commits.** API keys belong in environment variables or in the git-ignored local
  settings file described in `env.example`.
- Record significant decisions in `docs/DECISIONS.md` and progress in `docs/DEVLOG.md`.

## Research changes

The Phase 12 experiment is pre-registered (`experiments/preregistration_v1.md`). Change the protocol only by adding
a dated amendment that explains why, before collecting the affected data.
