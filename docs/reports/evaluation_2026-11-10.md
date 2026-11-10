# Independent evaluation: adaptive-prompt-harness at commit 14d8f46 (2026-11-10)

## How this evaluation was done

**Source of truth.** The master prompt is no longer at `~/Downloads/`. Its full text (2,413 lines) was recovered
from this session's transcript, where it was read at Phase 0, and every requirement below is checked against it.

**gstack, used externally.** gstack's skills are **not installed** in this session, and none was executed. Running
them needs gstack's own binaries and telemetry (`~/.claude/skills/gstack/bin/...`), which installing would bring
in. Their public definitions (`review`, `plan-eng-review`, `qa-only`, `test-audit`, `cso`) were read from GitHub,
and their methods were applied by hand:

- **`/review`:** every finding cites the code; uncapped spend or a trust-boundary bypass is forced to Critical;
  confidence is scored 1–10; a "sideways sweep" checks other call sites for the same pattern.
- **`/plan-eng-review`:** scope challenge and architecture review.
- **`/qa-only`:** report only, with replayable commands and expected vs. actual results.
- **`/test-audit`:** test value cards.
- **`/cso`:** attacker, boundary and impact, then a challenge. No independent second agent was used: this is a
  sequential challenge.

**What was run** (nothing in the repository was modified):
- The full test suite on an isolated `git archive HEAD` copy.
- ruff and mypy.
- 28 adversarial QA probes (throwaway scripts outside the repository, run against an isolated copy).
- A wheel build from the working folder.
- Earlier this session: the live runs on Groq, Gemini and Ollama.

The repository was verified unchanged afterwards: HEAD `14d8f46`, the same two pre-existing untracked files, no
stash.

Severity: **Critical / High / Medium / Low**. Every finding is labelled **Confirmed defect**, **Potential risk**,
**Design recommendation** or **Unverified assumption**.

---

## 1. Overall assessment

The core loop is engineered with unusual care for a project at this stage:
- Evidence comes only from real checks or feedback.
- Missing feedback is never counted.
- Strategy switches need 30 outcomes per arm and a corrected posterior test.
- Every transformation goes through one guard that falls back to the original prompt.
- The sandbox is properly isolated.
- 256 tests pass, including 45 real-Docker tests, and the static checks are clean.

On the happy path, the loop does what it claims, in simulation and on real free-tier models.

It is **not yet reliable enough for real use or release**:
- **Money.** The budget cap can be bypassed on paid providers that do not report cost (F1).
- **Learning switches off.** One routine provider-side change disables learning permanently under the default
  configuration (F2).
- **Defaults.** Out of the box the harness never learns anything (F3). The central promise of the master prompt
  (automatic discovery after connecting a model) is not the default behaviour.
- **Evidence integrity.** Concurrency (F4) and profile-version attribution (F5) can corrupt evidence.
- **Unproven.** No strategy has yet been shown to *improve* any model, and Phase 12 (the actual research
  validation) has not started.

**Maturity:** a solid research prototype at about 60–65% of the master prompt. Its learning loop is correct but
fragile, and it is not ready to be packaged.

## 2. Architecture and engineering quality

**Sound:**
- The request path matches master prompt Part 4 step for step (`api.py:420-522`, `525-626`).
- Deterministic services everywhere, and LLM calls only in three opt-in places, as Part 12 asks ("do not turn
  every module into an autonomous agent").
- Provider differences stay inside the adapters (Part 13.1).
- Strategies are pluggable, versioned and content-hashed (Part 7).
- A forward-only SQLite schema with migrations (Part 11.3).

**Weak:**
- **`api.py` (737 lines) is the real orchestrator.** It holds planning, recording, evaluation, drift and profile
  logic. `orchestrator.py` only holds the background work. The master prompt's "Orchestration" component exists
  only nominally. *(Design recommendation, Medium.)*
- **The trial row does not store the profile that served it.** So `feedback()` and drift must reconstruct it,
  which causes F5. *(Design recommendation, Medium.)*
- **Evidence is keyed by (profile, predicted category, strategy) only.** Task complexity and prompt
  characteristics (Part 6) are computed but never used by the policy. *(Design gap, Medium.)*
- **Phase 12 condition C cannot be run.** Condition C ("model-specific, not task-specific") has no selectable
  policy mode; only the hidden model-global backoff prior pools across categories. *(Design recommendation,
  Medium.)*

## 3. Orchestration and execution loop

Verified working (evidence: `tests/unit/test_system_loop.py`, probes P25–P28, live runs):

| Behaviour | Evidence |
|---|---|
| Cold start sends the original prompt | test_system_loop |
| No exploration and no evidence without a check (P27: 0/50 explored at rate 1.0) | probe P27 |
| Evidence is learned and used after a restart | test_system_loop; clean-install run |
| Sequential feedback is counted once; a crashing check is never counted | test_system_loop; probes P4a/P4b |
| Drift → stale → identity → recalibration → profile version 2 | test_system_loop |
| A broken store or check never loses the response | test_orchestration |
| The store is unavailable → runs stateless and the call still works | probe P25 |
| Calibration resumes without splitting tasks across categories | live qwen3.8 run (184 trials, 0 split) |

What breaks under realistic conditions is listed in section 6 (F2, F4, F5, F10, F12). No infinite loop was found.
Auto-calibration is guarded by "one run at a time" and `min_interval_s`, and each calibration call is bounded by
`max_calls`.

## 4. Functional completeness against the master prompt

See the requirement table in section 11. Summary: Phases 0–11 are delivered with gaps, and Phases 12–13 are not
started. The most significant gaps:
- Calibration is not automatic by default (F3).
- Suites exist for only 4 of the 15 categories.
- No few-shot or example-driven strategies.
- No model-assisted evaluation of responses.
- No security-analysis evaluator.
- Tool calling is one-way (F11).
- No CI.

## 5. Code quality and maintainability

**Strengths:**
- Strict typing (mypy `--strict`, clean).
- Small modules with clear docstrings.
- Explicit error hierarchy.
- Bounded HTTP retries.
- Secrets held in a masking type.
- Diagnostic text scrubbed by privacy level.

**Weaknesses:**
- A broad `except Exception` safety net in `_plan` and `_finish`. It is correct for "never lose a response", but
  failures surface only as log warnings: nothing counts them in `profile()`, so a silently broken store looks
  like a quiet model (risk, Low).
- About 37 test references to private internals.
- Several in-memory caches are unbounded (R2).
- Configuration precedence is surprising (F8).
- Raw `sqlite3` errors inside `Store.transaction()` are not wrapped as `StoreError` (F16).

## 6. Findings

### F1 — Budget cap bypassed on paid providers that do not report cost
- **Severity:** Critical (forced by gstack's uncapped-spend rule; real spend is bounded only by `max_calls`).
- **Label:** Confirmed defect. **Confidence:** 9/10.
- **Where:**
  - `calibration/calibrator.py:118`: allows `max_cost_per_call_usd=0`.
  - `calibration/calibrator.py:287`: `cost = r.result.cost_usd if r.result.cost_usd is not None else self.max_cost`.
  - `scripts/calibrate.py:31`: the default is `0.0`.
  - `auxiliary.py:53-59`: same pattern; `RewriterConfig` and `ClassifierConfig` accept 0.
- **What happens:** with a per-call reservation of 0, a paid provider that reports no cost has every call
  committed at $0, so the ledger never reaches the cap.
- **When:** OpenAI chat (no `usage.cost`), DeepSeek, or any paid OpenAI-compatible endpoint, with the script's
  default flag or a cap explicitly set to 0.
- **Evidence:** P1 made 50 real calls with ledger spend $0.0 against a $0 budget. P2 made 5 paid rewriter and
  judge calls with ledger spend $0.0.
- **Why it matters:** Part 8.2 says "never unexpectedly consume large amounts of paid API credit". It is also the
  rule behind your $10 cap.
- **Fix:** for non-local adapters, refuse a per-call cap of ≤ 0 everywhere, as `Orchestrator.allowed()` already
  does. Better still, estimate cost from token usage and a price table when the provider reports none.

### F2 — One provider-side fingerprint change permanently disables learning
- **Severity:** High. **Label:** Confirmed defect. **Confidence:** 8/10.
- **Where:**
  - `models/registry.py:68-71`: a fingerprint change marks profiles stale.
  - `providers/openai_compat.py:216-217`: OpenRouter's `provider` field goes into the fingerprint as
    `serving_host`.
  - `api.py:487`: a stale profile disables exploitation.
  - `config.py:119`: auto-calibration is off by default.
- **What happens:** a change of serving host (or any fingerprint field) marks the profile stale. A stale profile
  never exploits, and with default settings nothing ever recalibrates it.
- **When:** OpenRouter routing the same model to a different host. The trigger mechanism is confirmed; how often
  OpenRouter does this is unverified (U1).
- **Evidence:** P3, with 400 checked calls and random host alternation. The result was
  `identity: no current profile evidence for this model`, one stale profile and one `fingerprint` event. The same
  scenario without host changes chooses `role_framing` (test_system_loop).
- **Why it matters:** this silently turns the product into a pass-through on a major integration target.
- **Fix:** record a host change as a fact, not as a model change. Only a reported model version should create a
  new profile. Also let a stale profile recover without manual action.

### F3 — The default configuration never learns
- **Severity:** High. **Label:** Confirmed behaviour, a requirement deviation. **Confidence:** 10/10.
- **Where:** `config.py:86` (`exploration_rate=0.0`) and `config.py:119` (`auto=False`).
- **What happens:** with defaults, every call sends identity forever, however many checks are supplied.
- **Evidence:** P7, with 300 checked calls and a strategy truly better by 65 points. Only `identity` was ever
  sent.
- **Why it matters:** master prompt Part 8.1 says calibration "should be automatic", and §16.1 says `generate()`
  "should automatically trigger the appropriate optimisation workflow". The conservative default is defensible
  for paid providers (D44), but it is applied to local models too, where it costs nothing.
- **Fix:** default to auto-calibration for local providers. For paid ones, ask once at construction, or require
  an explicit budget, which the harness then honours (after F1).

### F4 — Concurrent outcome reports are double-counted
- **Severity:** Medium. **Label:** Confirmed defect. **Confidence:** 9/10.
- **Where:** `api.py:595-623`. The `already_counted` SELECT, the INSERT and `evidence.record` are separate
  statements, not one transaction.
- **When:** a background check and `feedback()` for the same response, or two `feedback()` calls, from different
  threads. This is typical in a web server.
- **Evidence:** P4c double-counted 48 of 50 runs; P4 (background check + feedback) 30 of 30. Sequential use is
  correct (P4a, P4b).
- **Why it matters:** inflated evidence violates Part 11.1 and rule 14.
- **Fix:** do the check, the insert and the update in one `BEGIN IMMEDIATE` transaction, or add a unique partial
  index on `evaluation(trial_id) WHERE status IN ('success','failure','partial')`.

### F5 — `feedback()` credits the latest profile, not the one that served the response
- **Severity:** Medium. **Label:** Confirmed defect. **Confidence:** 9/10.
- **Where:** `api.py:345-351` uses `ORDER BY profile_version DESC LIMIT 1`.
- **Evidence:** P5: a response served by profile 1 had its feedback counted into profile 2, after recalibration.
- **Why it matters:** an outcome from a stale or old model version contaminates the fresh profile.
- **Fix:** store `profile_id` on the trial row and use it.

### F6 — Calibration cost is also booked as user spend
- **Severity:** Medium. **Label:** Confirmed defect. **Confidence:** 9/10.
- **Where:** `api.py:727-730` records user spend for every trial, including calibration trials, which the
  calibrator also commits as harness spend.
- **Evidence:** P6, with no user calls made: harness $0.000633 and user $0.000633.
- **Why it matters:** cost reporting (Part 15) is wrong. The cap logic is unaffected (it uses harness scope
  only).
- **Fix:** skip `record_user_spend` when `plan.experiment_id` is set.

### F7 — Overhead grows linearly with prompt size; the "O5 met" claim is overstated
- **Severity:** Medium. **Label:** Confirmed defect, including a documentation accuracy problem.
  **Confidence:** 9/10.
- **Where:**
  - `tasks/classifier.py:97-105`: one regex search per lexicon entry per category.
  - Claims at `docs/DESIGN.md:384-385`, `docs/DEVLOG.md:454` and `docs/reports/phase11_and_limits.md:15`.
- **Evidence:** P19b median `analyze()` time by prompt size:

  | prompt size | time |
  |---|---|
  | 200 characters | 1.4 ms |
  | 1 KB | 7.2 ms |
  | 2 KB | 14.6 ms |
  | 10 KB | 79 ms |

  It was 2.9 s at 400 KB. The benchmark that "met O5" used a one-line prompt.
- **Why it matters:** the target (< 5 ms) is missed for ordinary prompts that include code. The docs state a
  result as general when it holds only for short prompts.
- **Fix:** compile one alternation per category (or use Aho-Corasick), cap the analysed length, and restate O5
  with the size dependence.

### F8 — Environment and `.env` settings override explicit code configuration
- **Severity:** Medium. **Label:** Confirmed defect (for the environment) plus a potential risk (for `.env`).
  **Confidence:** 8/10.
- **Where:**
  - `config.py:184-193`: `from_env` applies on top of the config passed in code.
  - `_keys.py` `find_env_file`: walks from the working directory to the filesystem root.
- **Evidence:** P23. Code said `total_usd=0.0`, `ADAPTIVE_HARNESS_BUDGET_USD=5` was set, and the effective value
  was 5.0. The `.env` route is the same function; that was established by code reading only, with no `.env`
  fixture planted.
- **Attacker model (cso):** a `.env` in a cloned repo or an ancestor folder can raise the spend cap, switch
  `prompt_storage` to `full` (storing every prompt), or redirect the database. Counter-evidence: base URLs and
  keys cannot be redirected this way (checked: no environment setting controls base URLs).
- **Fix:** explicit configuration wins. Read `.env` only from the working directory or an explicit path. Never
  let the environment *raise* a budget above the value set in code.

### F9 — Hallucinated imports are never counted as model failures
- **Severity:** Medium. **Label:** Confirmed behaviour; a measurement-validity risk. **Confidence:** 8/10.
- **Where:** `sandbox/runner.py:85-86` (`ModuleNotFoundError` becomes `env_error`) and
  `evaluation/builtin.py:180-182` (`env_error` becomes `evaluation_error`, not counted).
- **Evidence:** P8: an answer importing a non-existent `fastmathx` gives `evaluation_error`, excluded from the
  evidence.
- **Why it matters:** a strategy that makes a model invent packages is never penalised, so success rates are
  biased upwards. This is exactly the kind of bias Part 18 must avoid.
- **Fix:** keep an allow-list of modules the sandbox image provides. Any other missing module is a model
  failure, or at least a separately reported counted status.

### F10 — `close()` blocks until an automatic calibration finishes
- **Severity:** Medium. **Label:** Confirmed defect. **Confidence:** 9/10.
- **Where:** `orchestrator.py:61-72`: `close()` calls `flush()`, which joins the calibration thread with no
  timeout.
- **Evidence:** P22: `close()` took 6.3 s with 20 mock calls of 0.3 s each. With real models this is minutes.
- **Fix:** add a cancel flag that the calibrator checks between calls, and give `close()` a timeout.

### F11 — Tool calling is one-way
- **Severity:** Medium. **Label:** Confirmed limitation. **Confidence:** 10/10.
- **Where:** `types.py:15`: `_ROLES = ("system","user","assistant")`.
- **Evidence:** P24: a `role="tool"` message raises `ValueError`.
- **Why it matters:** tool calls can be received but their results can't be sent back, so an agent loop with
  tools cannot run through the harness. The Phase 11 requirement "support tool-calling behaviour where feasible"
  is only partly met.
- **Fix:** add tool roles with call ids to `Message` and map them per adapter.

### F12 — A stream that the consumer abandons is never recorded
- **Severity:** Low. **Label:** Confirmed. **Confidence:** 9/10.
- **Where:** `api.py:320-328`.
- **Evidence:** P9: after a `break` on the first event, 0 trials were recorded.
- **Fix:** use a `try/finally` that records a partial or cancelled trial.

### F13 — A wheel built from the working folder ships a stale module
- **Severity:** Low. **Label:** Confirmed. **Confidence:** 10/10.
- **Evidence:** the working-folder wheel contains the old untracked leftover module next to `_keys.py`.
- **Fix:** delete the two leftover files. Build releases only from a clean checkout in CI.

### F14 — The judge's independence check compares names only
- **Severity:** Low. **Label:** Potential risk. **Confidence:** 7/10.
- **Where:** `transform/llm_rewrite.py:117` and `config.py:162`.
- **What happens:** the same model through two providers (e.g. `groq` and `openrouter` serving
  `openai/gpt-oss-20b`) passes as "independent".
- **Fix:** normalise model ids (vendor prefix) before comparing, as the priors already do.

### F15 — Empty or whitespace prompts are sent without validation
- **Severity:** Low. **Label:** Confirmed. **Confidence:** 10/10.
- **Evidence:** P26.

### F16 — Raw SQLite errors at start-up are not caught
- **Severity:** Low. **Label:** Potential risk, not reproduced. **Confidence:** 5/10 (medium confidence; verify
  this is actually an issue).
- **Where:** `store/db.py` `transaction()` re-raises raw `sqlite3.Error`. `api.py:164` catches only `StoreError`.
- **What could happen:** a locked or full disk during `load_priors` or `sync_with_store` would raise from the
  constructor instead of falling back to stateless mode.

### Potential risks (not defects today)
- **R1 — Calibration elimination is not corrected for repeated testing.** The 0.95 elimination test repeats after
  every batch, which inflates the chance of dropping an arm that is not worse. This errs toward identity, so it
  causes missed gains, not harm. (`calibrator.py` `_eliminate`)
- **R2 — Unbounded in-memory caches.** `LlmRewrite._rewrites` and `_verdicts`, and `LlmFallbackClassifier._cache`,
  grow without limit in long-running processes.
- **R3 — Auto-calibration shares the user's rate limits.** Groq's tokens-per-minute limit was hit during this
  session's live calibration; concurrent user calls would see 429 errors and retries.
- **R4 — The sandbox nonce can be read by the code under test.** A deliberately adversarial solution can read the
  nonce by frame introspection and forge a pass. This is documented. It is fine for non-adversarial model output,
  but not safe as a security boundary for scoring.
- **R5 — The LLM classifier's "confidence" is a constant.** It is `min_probability`, not a measurement, yet the
  trace shows it as confidence.
- **R6 — Prompts can steer their own category.** A prompt can influence the LLM classifier's label. Impact is
  bounded: strategies keep the user's text, and the preservation guard still applies.
- **R7 — Online learning is slow.** It needs 30 own outcomes per (category × strategy) arm. The 11 categories
  without suites can only learn from user checks.

### Unverified assumptions (could not be tested here)
- **U1.** How often OpenRouter switches `provider` for one model. Only one live call was made.
- **U2.** OpenAI and Anthropic chat: never completed (no credit). DeepSeek, vLLM and LM Studio were never run
  live.
- **U3.** Tool-call parsing: synthetic bodies only.
- **U4.** Python 3.10–3.12: never executed (all runs on 3.13.5).
- **U5.** That any strategy *helps* any model. None has been shown: qwen3.8-27b only had two harmful strategies
  eliminated (minimum detectable effect 20 pp), and gpt-oss-20b is at ceiling.
- **U6.** That calibration on synthetic suites transfers to developers' real prompts.

## 7. Testing: coverage and results

**Results** (isolated `git archive` copy of `14d8f46`, Docker 29.7.2, Python 3.13.5):
- 256 passed, 0 failed, 0 skipped: 156 unit, 53 contract, 45 real-Docker sandbox.
- ruff clean; mypy `--strict` clean (56 files).

**Quality (test-audit):**
- 171 test functions, with no assertion-free test and no source-grep test.
- Strong areas: mutation tests on the preservation check, a sandbox abuse suite (fork bomb, infinite loop,
  network, filesystem), injection tests, and the end-to-end lifecycle.

**Gaps:**
- None of F1–F12 is covered by a test.
- No concurrency tests.
- No tests for a paid provider without cost reporting.
- No fingerprint-flapping test.
- No input-size or performance regression test.
- No `live`-marked tests: the marker is defined but unused, so live claims rest on one-off scripts.
- No CI.
- No 3.10–3.12 runs.

Proposed tests, each matching a defect and failing before its fix: F1 (paid, no cost, cap 0), F2 (alternating
`serving_host`), F4 (two threads), F5 (feedback after `next_profile`), F6 (scopes), F7 (2 KB prompt < 5 ms),
F10 (`close` timeout).

## 8. Reliability and hallucination risks

**Model outputs treated as verified?**
- The rewriter's output is checked by exact rules plus a judge (good).
- The LLM classifier's label is accepted without verification, with a fixed "confidence" (R5).
- Equivalence verdicts are cached, not re-checked.

**Can it report success when something failed?**
- A background evaluation that fails to store leaves `evaluation_status="pending"` forever, with only a log line.
- Hallucinated imports are excluded rather than failed (F9).

**Do fallbacks mask failures?**
- By design, every planning failure sends identity, with a note in the trace. That is correct for availability,
  but the failures are not surfaced in aggregate.
- A stale profile looks the same as "no evidence" (F2).

**Documentation accuracy:**
- One overstated claim, O5 (F7).
- The earlier reports were otherwise consistent with the measured runs. Spend, test counts and the live results
  were rechecked here.

## 9. Security and scalability

**Security (cso, sequential challenge):**
- Keys are masked in errors, logs and repr; key-shaped strings are scrubbed. Account IDs have been masked since
  D48.
- Keys and base URLs cannot be redirected by environment or `.env` settings.
- The sandbox has no network, a read-only root, no capabilities, an unprivileged user, and limits on processes
  and memory.
- Prompt storage is off by default.
- Open issues: F1 (spend), F8 (environment and `.env` precedence), R4 (sandbox scoring is not
  adversary-proof), F14.

**Scalability:**
- One SQLite file in WAL mode, with a process-wide lock. Fine for one process; contention would grow with many
  writers.
- `arms()` reads every arm row of the profile on each call.
- Classifier cost is linear in prompt size (F7).
- Single-threaded background queue, unbounded caches (R2).
- Calibration runs serially.

## 10. Missing functionality and technical debt

**Missing:**
- Phase 12 (experiments A–D, held-out generalisation, statistics) and Phase 13 (CI, changelog, contribution
  guide, API reference, release checklist).
- Few-shot, example-driven, context-rich and goal-oriented strategies.
- Model-assisted response evaluation (Part 10.3).
- Security-analysis evaluator (Phase 9).
- Tracking of structured-output validity and tool-call success (Part 10.5).
- Suites for 11 of 15 categories.
- Use of complexity and prompt characteristics in the policy (Part 6).
- Multi-turn tool use (F11).

**Debt:**
- The orchestration lives in `api.py`.
- No `profile_id` on trials.
- Configuration precedence.
- The leftover duplicate key module.
- Unused tables: `task_category`; `prior_evidence` is now used.

**Future problems:**
- Evidence integrity under concurrency (F4).
- Version and profile churn on routed providers (F2).
- Phase 12 condition C is not expressible.

## 11. Requirement-by-requirement comparison (master prompt)

| Part / requirement | Status | Evidence / gap |
|---|---|---|
| P4 Workflow (identify → understand → retrieve → select → transform → check → send → evaluate → learn → persist) | **Met** | `api.py` `_plan` / `_finish`; test_system_loop |
| P5 Model identity: provider, id, version, fingerprint; verified vs. reported vs. inferred | **Mostly met** | `models/`; facts carry provenance; capability metadata depends on the provider. F2: host changes are treated as model changes |
| P5 Profile tied to precise version; re-validate on change | **Met, but over-sensitive** | F2 |
| P6 15 categories, extensible taxonomy, requirement extraction, uncertainty → `general` | **Met** | taxonomy.json; test accuracy 0.688 (0.766 with the opt-in LLM fallback) |
| P6 Learn the relation with complexity and prompt characteristics | **Not met** | computed, not used by the policy |
| P7 Modular strategies, each with id, description, use, categories, limitations, version | **Met** | `strategies/` |
| P7 Strategy set (15 listed) | **Partial** | 9 built-in plus the opt-in `llm_rewrite`; no few-shot, example-driven, context-rich or goal-oriented |
| P7 "Evaluation history" per strategy | **Met** | `arm_stats`, `trial` |
| P8.1 Automatic calibration on first connect | **Partial** | implemented, but off by default (F3) |
| P8.2 Cost-aware, resumable, reproducible, configurable, persistent, efficient | **Mostly met** | elimination plus resume verified live. F1 breaks cost-awareness. No token or time limits (only calls and dollars) |
| P8.3 LeBlanc as prior evidence, with conditions and no hard-coding | **Met** | `learning/priors.py`; tests show priors alone never switch |
| P9.1 Semantic preservation | **Met (strong)** | mutation tests; exact checks plus judge for paraphrase |
| P9.2 Meaningful transformations (simplify, reorder, examples…) | **Partial** | mostly additive; reordering and simplifying only through the opt-in LLM rewriter; no examples |
| P9.3 Traceability | **Met** | `trial.trace_json`, hashes, privacy levels |
| P9.4 Safe fallback | **Met** | every failure sends the original prompt |
| P10.1 Deterministic evaluators | **Met** | syntax, JSON, fields, reference, unit tests (sandbox), constraints |
| P10.3 Model-assisted evaluation | **Not met** | only the rewrite-equivalence judge |
| P10.4 Optional human feedback | **Met** | `feedback()`; F4 and F5 affect it |
| P10.5 Operational metrics, kept separate | **Mostly met** | latency, tokens, cost, retries, finish reason, errors; no structured-output or tool-call success metric |
| P10.6 Separate evaluation states; missing ≠ success | **Met** | `EvalStatus`; probes P27, P28 |
| P11 Continuous learning, change detection, recalibration | **Met in mechanism** | decay, drift, `next_profile`; F2 and F3 limit it in practice |
| P11.1 Evidence-based (sample size, confidence, correlated responses) | **Mostly met** | `min_trials`, Bonferroni, exact posterior. Repeated responses to the same task are treated as independent online; calibration pairs them |
| P11.3 Persistence, profile versioning, migrations | **Met** | migrations 0001–0002, profile versions |
| P12 Agentic but no unnecessary agents; deterministic services | **Met** | — |
| P13 Provider-agnostic, adapters, local inference | **Met** | 11 provider names; live on Groq, Gemini, OpenRouter, Ollama; OpenAI and Anthropic not completed (U2) |
| P13.1 Interface for streaming, usage, errors, tools, structured output | **Partial** | tools are one-way (F11); `response_format` only as a pass-through |
| P14 Security items 1–20 | **Mostly met** | gaps: #16 spend (F1, F8); #20 logs scrubbed; R4 |
| P15 Measure overhead, caching, background work | **Partial** | profile lookup cached and fast; F7 |
| P16 Standalone package, simple API, typed, documented, clean install | **Mostly met** | clean wheel install verified; no CI, API reference or changelog; F13 |
| P17 Related work | **Met** | `docs/RELATED_WORK.md` (Phase 1) |
| P18 Experimental validation A–D, held-out, statistics | **Not started** | Phase 12; condition C not expressible |
| P19 Phases and approval gates | **Met** | gates were followed |
| P20 Rules 10, 11, 13 (mock vs. real, no fabrication, missing ≠ success) | **Met** | simulated results flagged and stored as such |
| P20 Rule 9 (paid experiments within budget) | **At risk** | F1 |
| P20 Rule 14 (no updates on insufficient evidence) | **At risk** | F4, F5 |
| P20 Rule 21 (never skip tests) | **Met** | — |

## 12. Maturity assessment

**Genuinely working (shown by tests, probes or live runs):**
- The end-to-end request path, and learning from checked outcomes, on the happy path.
- The preservation guard and safe fallback.
- The sandbox.
- Budgeted, resumable calibration on free and local models.
- Decay and drift in simulation.
- Persistence across restarts.
- Privacy scrubbing.
- Providers: Groq, Gemini, OpenRouter and Ollama, live.

**Partially working:**
- Automatic learning: off by default (F3), and switched off by provider-side changes (F2).
- Tool calling: one-way.
- The overhead target: only for short prompts.
- Cost accounting: F1, F6.
- Concurrency safety: F4.
- The LLM helpers: they work live, but are opt-in.

**Not working or not built:**
- Phase 12 validation.
- Model-assisted evaluation, few-shot strategies, multi-turn tools.
- CI and release material.

**Cannot yet be verified:** U1–U6. Above all, whether adaptive prompting improves outcomes for any model.

**Is the orchestration reliable enough?** For a single-threaded, local or free-tier research workload: yes. For a
paid provider, a multi-threaded server, or OpenRouter: **no**, until F1, F2, F4 and F5 are fixed.

**Is the architecture right for the remaining phases?** Yes, with two structural changes first:
1. Store the serving `profile_id` on each trial and make outcome recording atomic (F4, F5).
2. Add a policy mode for "model-only" evidence, so Phase 12 can run condition C.

Moving the request path out of `api.py` into the orchestrator would make Phase 12's controlled conditions easier
to wire. It is recommended, not required.

**Must be addressed before Phase 12:**
- F1 and F2 (they corrupt costs and silently disable learning).
- F4 and F5 (they corrupt evidence).
- F9 (it biases a key metric).
- F6 (its cost reports would feed Phase 12's efficiency metrics).
- F7's corrected claim.

Without these, experimental results would not be trustworthy.

**Distance from public release:**
- Beyond the fixes above, it needs: CI on 3.10–3.13; a changelog, contribution guide, API reference and release
  checklist; removing the leftover files (F13); fixing configuration precedence (F8); a stream cancellation path
  (F12); a `close()` timeout (F10).
- Above all, Phase 12 evidence that the product does what it claims.

Estimate: roughly two to three focused phases of work. That is a judgement, not a measurement.
