# Decision log

Newest last. Each entry: decision, alternatives considered, reason. "Proposed" entries need the project owner's
approval; "Decided" entries were made by the owner or follow from their instructions.

## 2026-11-10 (Phase 0 → Phase 1)

- **D1 (Decided, owner).** LeBlanc v4 is read-only reference material. Nothing in it is modified, copied or
  re-licensed. Its findings enter only as cited `PriorEvidence` records.
  *Consequence:* LeBlanc has no licence, so code ideas (paired statistics, leased resumable runs, scanner
  de-duplication) are **re-implemented independently**, not copied.
- **D2 (Decided, owner).** New work lives in `github.com/ayushiii2707/agentic-harness`. Nothing is pushed or
  published without explicit approval.
- **D3 (Decided, owner-delegated).** The sandbox targets **Docker on macOS and Linux**, using gVisor
  automatically where present.
  *Alternatives:* bubblewrap (Linux only); Windows+WSL (one machine only); Firecracker (Linux/KVM only).
  *Reason:* the only option that covers both target platforms with a well-understood isolation boundary.
- **D4 (Proposed).** Core dependency is `httpx` only. No vendor SDKs, no LiteLLM, no DSPy in core.
  *Reason:* small footprint, stable surface, mockable; avoids wrapping someone else's optimiser (brief rule).
- **D5 (Proposed).** The original prompt (`identity`) is the default arm. A strategy must beat it with
  posterior probability ≥ 0.90, net of cost, by at least δ = 2 pp.
  *Reason:* LeBlanc found mostly neutral and one possibly harmful intervention; "do no harm" comes first.
- **D6 (Proposed).** The online policy is Bayesian (Beta posteriors, hierarchical backoff, time decay; Thompson
  sampling only for opt-in exploration). Calibration is fixed-budget best-arm identification with successive
  elimination. Research analysis is pre-registered and frequentist (paired McNemar, BH, cluster bootstrap).
  *Alternatives:* UCB / LinUCB contextual bandit over prompt embeddings (needs far more data than one user
  generates); Bayesian optimisation over a continuous prompt space (our arms are discrete and few).
- **D7 (Proposed).** Version 1 ships only `identity`, `additive` and `structural` strategies, which keep the user's
  text verbatim. `rewriting` strategies are experimental and sit behind a stricter preservation gate.
- **D8 (Proposed).** No paid calibration runs unless a budget is configured (default 0). Local providers may
  auto-calibrate under a call or time cap.
- **D9 (Proposed).** Hashes of configs and suites are computed over canonical JSON (sorted keys, LF line endings).
  *Reason:* the LeBlanc study hash changed on a fresh clone because of CRLF line endings.
- **D10 (Proposed).** Sandbox results travel on a separate descriptor with a per-job nonce. Any infrastructure
  failure is `evaluation_error`, never a model failure. *Reason:* two integrity problems found in the Phase 0 audit.
- **D11 (Proposed).** CodeSecEval is used only through a local download script, pinned to a revision. No task
  data is committed. *Reason:* its licence is "currently under review", and SecurityEval has no explicit licence.
- **D12 (Decided, owner).** The repository licence is Apache-2.0.

## 2026-11-10 (Phase 1 approved; Phases 2–3)

- **D4–D11 (Decided, owner).** Approved with the Phase 1 design.
- **D13 (Decided, owner).** Free providers first. The hard cap is **$10 in total** for everything the project
  tooling spends (`ADAPTIVE_HARNESS_BUDGET_USD=10`, enforced by the persistent ledger). OpenRouter only when
  strictly necessary; other paid providers (OpenAI, Anthropic, Ollama cloud) only with explicit approval.
- **D14.** The Anthropic adapter targets API version `2026-10-21`, which rejects `temperature`, `top_p` and
  `top_k`. The adapter does not send them and lists them in `dropped_settings`. Experiments must record this as
  an unavoidable difference in generation settings between providers.
- **D15.** Python ≥ 3.10 (TOML via `tomli` on 3.10, `tomllib` on 3.11+). mypy checks against 3.10.
- **D16.** The local settings template is named `env.example` (not `.env.example`), so tooling that guards
  dotenv files never confuses the key-free template with the real file.
- **D17 (Decided, owner).** The key-handling module is `adaptive_harness/_keys.py`. A development safety hook
  rejects any shell command containing certain words; the owner chose to rename the module rather than disable
  or bypass the hook. Class and function names inside it (`Secret`, `get_secret`) are unchanged.
