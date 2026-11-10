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

## 2026-11-10 (Phase 4)

- **D18.** The identity fingerprint covers the reported model, the reported version and the serving host. It
  excludes `system_fingerprint`, which OpenAI-style backends change with routine configuration updates; that
  value is stored as a `provider_reported` fact. *Reason:* otherwise profiles would go stale constantly.
- **D19.** A profile marked `stale` is kept and keeps its evidence. A new profile version is created only by
  recalibration (Phase 7), so stale evidence is never silently discarded or silently reused as fresh.

## 2026-11-10 (Phase 5)

- **D20.** Classifier thresholds are those that maximised dev accuracy (`min_probability 0.30`, `min_margin 0.00`,
  `unexplained_mass 0.5`). They replace the DESIGN placeholders (0.55 / 0.15), as DESIGN §5.2 requires
  ("tuned on a labelled development set ... and frozen before Phase 12"). They are frozen in
  `tasks/taxonomy.json` version 1.
- **D21.** Lexicon leakage rule: no lexicon entry may match a test prompt unless it also matches a dev prompt
  (`scripts/eval_classifier.py --check-leakage`). Lexicons are never edited in response to test errors; a
  changed lexicon needs a new taxonomy version and a fresh test set.
- **D22.** The task-classification set is author-written (no suitable public set matches the taxonomy). This
  limitation is stated in every report that uses it.

## 2026-11-10 (Phase 6)

- **D23.** Additive strategies add their instruction as a separate, first system message rather than editing the
  user's system message or turns. The user's content stays byte-identical and separate from harness text
  (DESIGN §5.4).
- **D24.** `security_generic` uses exactly the sentence LeBlanc v4 tested ("Write secure code and avoid common
  security vulnerabilities."), so LeBlanc's results can serve as prior evidence for this strategy (DESIGN §6.6),
  and only for the models and task family LeBlanc tested. The sentence is a generic phrase; no LeBlanc code or
  data is copied.
- **D25.** `security_targeted` and `few_shot_from_library` are deferred (no licensed knowledge base or curated
  example library yet), and rewriting strategies wait for Phase 8.

## 2026-11-10 (review: decision loop)

- **D26.** The decision loop is built end to end before the remaining phases, because the separately built
  components had never been connected. Later phases deepen it.
- **D27.** Evidence comes only from developer-supplied checks (`check=`) and `feedback()`. At most one outcome
  per response is counted. Built-in proxy checks (syntax, format) will be recorded as secondary results and will
  not update evidence unless they are the task's declared primary outcome (DESIGN §5.6).
- **D28.** Exploration is uniform among candidates, not Thompson sampling. Comparisons need `min_trials` (30)
  for both arms, and the threshold is Bonferroni-corrected. *Reason:* measured false-switch rate 15/40 → 2/40
  in simulation (DESIGN §6.4, revision note).

## 2026-11-10 (Phase 7 and second loop review)

- **D29.** Evidence is keyed by the category the harness's classifier predicts, in calibration as at request
  time. *Reason:* keying calibration evidence by the suite's true label created a training/serving skew.
  Decisions are made per predicted label, so evidence must be estimated on the prompts that receive that label.
- **D30.** Calibration skips cells where only identity applies (no comparison is possible).
- **D31.** Strategy content hashes are persisted in the `strategy` table. A strategy whose stored hash differs
  from its current wording under the same version is disabled, so old evidence is never applied to new wording.

## 2026-11-10 (Phase 8)

- **D32.** Supersedes D7 in part: rewriting strategies are allowed, but only through `apply_strategy` and the
  preservation check, never directly. The only one shipped is the deterministic `minimal_context`.
- **D33.** A rewrite must keep **every** content word of the original. Only stop words and the explicit `FILLER`
  list may be deleted; numbers, literals, identifiers, paths, URLs, languages and formats must stay exact.
  *Reason:* a 90% threshold accepted a rewrite that dropped part of the task (mutation test). Paraphrase needs a
  judge, which is not implemented.

## 2026-11-10 (LLM rewriter)

- **D34.** `llm_rewrite` is accepted only when the exact checks (numbers, literals, identifiers, paths, URLs,
  languages, formats, negation count, length) pass **and** an independent judge model (not the rewriter) returns
  strict JSON saying the texts are equivalent with nothing missing, added or changed. Word coverage is not
  required for it, because paraphrase changes words. Every failure, including budget refusal, sends the original.
  *Reason:* live probes showed the judge catches dropped, added and inverted requirements that pass the exact
  checks, while the exact checks catch number and literal changes cheaply before any judge call.

## 2026-11-10 (Phase 10)

- **D35.** Decay is applied at write and read time, with a 90-day half-life (21 for aliases). The policy gates on
  the cell's own decayed count (`n_eff` ≥ `min_trials`), so the "age" trigger of DESIGN §6.7 needs no separate
  rule.
- **D36.** Backoff and prior evidence change the estimate (alpha/beta), never the gate (`n_eff`). A borrowed weight
  never exceeds the evidence it comes from. *Reason:* the policy must not leave identity on borrowed or external
  evidence alone.
- **D37.** LeBlanc v4 priors use the secure-and-working rate, not security-test failure. The harness learns from
  pass/fail outcomes, and secure-and-working is the pass/fail outcome that LeBlanc measured. The host as tested is
  not recorded, so the weight is 1.5.
- **D38.** Cost uses tokens (cost in dollars is often unreported). The penalty is λ·(token ratio − 1), clipped at
  0, so cost can only make switching harder.
- **D39.** Drift uses a windowed two-proportion test (30 recent outcomes against up to 120 earlier ones,
  |z| ≥ 4.0) instead of CUSUM. *Reason:* it is simple to verify, and its threshold was tuned by simulation:
  about 1% false alarms over 500 outcomes, and a 0.8→0.4 drop detected in 93% of runs.
- **D40.** Diagnostic notes are stored with quoted spans removed when `prompt_storage="none"`. *Reason:* the review
  found that rejection reasons had stored prompt fragments.
