# Development log

## Current status
- **Phase 0, repository audit of LeBlanc v4:** complete; approved 2026-11-10.
- **Phase 1, research and architectural design:** complete; approved 2026-11-10 (licence Apache-2.0, $10 hard
  cap, free providers first).
- **Phase 2, foundation:** done; pushed to `feat/foundation-and-providers`.
- **Phase 3, provider adapters:** done. Live-verified on Groq, Gemini and OpenRouter; OpenAI and Anthropic
  reached but their accounts have no credit; local Ollama live-verified with `qwen2.5vl:3b`.
- **Phase 4, model intelligence:** done; pushed.
- **Phase 5, task understanding:** done; pushed. Held-out test accuracy 0.688 (see
  `docs/reports/phase5_classifier.md`).
- **Phase 6, strategy library:** done; pushed.
- **Decision loop (root-cause fix, 2026-11-10):** implemented, simulated, and live-checked on Groq. See below.
  **Awaiting review.**
- Not implemented yet: calibration (Phase 7), sandboxed and format evaluators (Phase 9), evidence decay, backoff,
  priors and drift (Phase 10), and the full preservation check for rewriting strategies (Phase 8).

## 2026-11-09 to 2026-11-10: Phase 0
- Audited LeBlanc v4 at commit `51522f2` (read-only clone; database queried read-only; recomputation on a copy).
- Results: 12/12 unit tests pass. Selfcheck 11/15 on macOS; the 4 failures are expected (no API key; no
  `wsl.exe`, so no sandbox, and no code was executed). Every headline number reproduced from the database.
- Findings carried into the design:
  - study-hash fragility (line endings) → D9;
  - sandbox result forgery and infrastructure-failure-as-model-failure → D10;
  - oracle-vs-retrieved warning gap → `security_targeted` marked unvalidated;
  - preambles raise total tokens by up to +44% → the cost term in the policy;
  - most effects "unclear" at about 135 pairs per model → power planning in DESIGN §13.

## 2026-11-10: Phase 1
- Literature and tool review. See `RELATED_WORK.md`.

## 2026-11-10: Phases 2 and 3
**Built:**
- **Core:**
  - `types.py`: provider-neutral `Message`, `CompletionRequest`, `CompletionResult`, `Usage`, `FinishReason`,
    `StreamEvent`. `dropped_settings` and `simulated` are first-class fields.
  - `errors.py`: the normalised hierarchy.
  - `config.py`: code, environment, `.env` and TOML. Safe defaults: budget 0, prompts stored as hashes.
  - `_keys.py`: masked key values, redaction of registered and key-shaped strings. (Renamed from its first
    name because a plugin safety hook blocks every shell command containing that word; see D17.)
  - `_logging.py`: a redacting filter.
  - `hashing.py`: canonical JSON hashes.
  - `redaction.py`.
- **Store:** `store/db.py` plus `migrations/0001_init.sql`, the full evidence schema from DESIGN §8. The store
  refuses databases written by newer versions.
- **Budget:** `budget.py`, a persistent reserve/commit/release ledger. Open reservations count against the cap,
  checked inside an immediate transaction; tested with 40 concurrent threads.
- **API:** `api.py`.
  - `AdaptiveHarness.generate`, `chat`, `agenerate`, `achat` and `stream`, all on the identity path.
  - Every call is recorded as a trial, respecting `prompt_storage`.
  - A store failure never blocks a call.
- **Providers:**
  - `_http.py`: bounded retries with a wait budget, `Retry-After` / Gemini `RetryInfo` / Groq "try again in"
    handling, error mapping, SSE parsing.
  - `openai_compat.py` + `presets.py`: OpenAI, Groq, OpenRouter, DeepSeek, Ollama, Ollama cloud, vLLM,
    LM Studio, and custom endpoints.
  - `anthropic.py`: Messages API, version `2026-10-21`.
  - `gemini.py`: Generative Language API, key in a header.
  - `mock.py`, `registry.py`.
- **Script:** `scripts/smoke_providers.py`, a live smoke test. Free targets only unless `--allow-paid`; paid calls
  reserve from the persistent ledger first.

**Tests:** 77 offline tests pass (unit + contract); `ruff check` and `ruff format` are clean; `mypy --strict`
is clean on 22 source files. Contract fixtures are synthetic bodies in each provider's documented format.

**Live verification (2026-11-10):**
- Groq `openai/gpt-oss-20b`: ok. 78 input / 58 output tokens, 48 of them reasoning; `system_fingerprint`
  captured; context window 131,072 from `/models`.
- Gemini `gemini-3.5-flash-lite`: ok. `models.get` gives a 1,048,576 context window, version
  `3.5-flash-lite-07-2026`, thinking supported.
- Ollama cloud: model listing works; chat returns **HTTP 402 "add usage credits"**, so it is not free for this
  account and is reclassified as paid.
- Paid checks, approved by the owner with a limit of $0.01 per call (each call reserved $0.005 first):
  - OpenRouter `openai/gpt-oss-20b`: ok. Served by DeepInfra; cost **$0.0000066** as reported by OpenRouter.
  - OpenAI `gpt-4o-mini`: key accepted, but the account has **no credits** (HTTP 429). Nothing charged.
  - Anthropic `claude-haiku-5-5`: key accepted, but the **credit balance is too low** (HTTP 400). Nothing
    charged.
  - This showed that Anthropic signals an empty balance with HTTP 400, which had been classified as an invalid
    request. Fixed: billing wording on 400/429 now maps to `QuotaExceededError`, with tests using the real
    (key-free) wording.
  - Total paid spend so far: **$0.0000066** of the $10 cap.
- Finding: the three LeBlanc v4 models are reachable without OpenRouter. `openai/gpt-oss-20b` (Groq, free) and
  `gemini-3.5-flash-lite` (Gemini, free) are callable; `deepseek-v4.1-flash:1017` is listed on Ollama cloud but
  needs credits.

**Issues met:**
- macOS sets the `hidden` flag on files created under `~/Desktop` in some setups, and Python 3.13 skips hidden
  `.pth` files, so the editable install was silently not importable. `chflags nohidden` fixes it only
  temporarily (the flag came back), so pytest now imports from `src/` directly (`pythonpath = ["src"]` in
  `pyproject.toml`) and the smoke script adds `src/` to its path. To be documented in CONTRIBUTING (Phase 13).
- Reasoning models spend output tokens on hidden reasoning, so a 16-token cap returned empty answers. The smoke
  test uses 512. This matters for experiment design: `max_tokens` caps must be set per model family and
  recorded.

## 2026-11-10: Phase 4 (model intelligence)
**Built:**
- `models/identity.py`:
  - `ModelKey(provider, model_id, deployment)`; `local:<host>` for local servers.
  - Alias detection (`latest`).
  - `resolved_version()`: the version the provider reports in its responses.
  - `fingerprint_of()`: a hash of the reported model, version and serving host. `system_fingerprint` is
    excluded (D18).
- `models/registry.py`, `ModelRegistry`:
  - `observe()`: upserts the model row and makes sure a profile exists. A new resolved version creates a new
    model row with a `cold` profile and marks the previous version's profiles `stale`, with a `fingerprint`
    change event. A changed serving host on the same version also marks the profile stale.
  - `record_facts()` / `facts()`: metadata with provenance (`verified`, `provider_reported`, `inferred`,
    `observed`). An unchanged value is not stored twice.
  - `profile()`: the latest profile, with its facts and change events.
- `api.py`:
  - every successful call is observed and its trial linked to the model row (`trial.model_ref`);
  - `harness.profile()` reports the profile and makes no network call;
  - `harness.model_info()` makes one explicit metadata request and records the facts.

**Tests:** 86 unique offline tests pass (6 new); ruff and mypy --strict are clean.

**Live check (Groq, free):** one call through `AdaptiveHarness`. Result: model key `groq / openai/gpt-oss-20b /
hosted`, a `cold` profile, and facts with provenance: context window 131,072 (verified), `owned_by` and
`system_fingerprint` (provider_reported), deployment (inferred).

**Not done in Phase 4, by design:** creating new profile versions on recalibration (Phase 7), and staleness by age
or outcome drift (Phase 10).

## 2026-11-10: Phase 5 (task understanding)
**Built:**
- `tasks/taxonomy.json` (version 1): 15 categories plus `general`. `api_development` and
  `security_sensitive_code` are children of `code_generation`. Holds weighted phrases and patterns per category
  and the frozen thresholds.
- `tasks/classifier.py`: a deterministic weighted-lexicon classifier. Its distribution includes an explicit
  "unexplained" mass assigned to `general`, and it falls back to `general` below the confidence or margin
  thresholds. A parent or child runner-up does not count against the margin.
- `tasks/requirements.py`: extracts constraint sentences, negative instructions, numbers with units, quoted
  literals, identifiers, paths, URLs, languages and output formats.
- `tasks/understanding.py`: `TaskAnalyzer.analyze()` returns a `TaskSpec` with the classification, requirements,
  a low/medium/high complexity estimate, and the evaluator kinds that could score the task (named only; the
  evaluators themselves arrive in Phase 9).
- `api.py`: every call is labelled (category and confidence in the trace and in `trial.category_id`). The
  request is never changed, and a labelling failure never blocks the call.

**Evaluation** (`scripts/eval_classifier.py`, data in `data/task_classification/`):
- 320 author-written prompts, 20 per category, split before tuning: 192 dev / 128 test.
- **Leakage control:** the first lexicon draft contained 60 entries that matched test prompts but no dev prompt.
  All 60 were removed mechanically (`--check-leakage` now passes) **before** tuning or testing.
- Thresholds tuned on dev only (grid search): `min_probability 0.30`, `min_margin 0.00`,
  `unexplained_mass 0.5`. Dev accuracy 0.927.
- **Held-out test:**
  - accuracy **0.688**; parent-accepted accuracy 0.703;
  - coverage 0.812; accuracy when not `general` 0.779;
  - of 40 errors, 17 fell back to `general` (conservative) and 23 chose another specific category.
- The dev/test gap shows that the hand-written lexicons overfit. The lexicons were **not** adjusted after seeing
  the test errors.

**What this means for later phases:**
- Category labels are wrong about 30% of the time on this set, and likely more often on real prompts (same
  author for prompts, labels and lexicons).
- So the policy (Phase 10) keeps identity as the default and backs off to parent and model-level evidence, as
  designed.
- Experiments must report results by *predicted* category and examine misclassification.

## 2026-11-10: Phase 6 (strategy library)
**Built:**
- `strategies/base.py`:
  - `Strategy` (pure `transform(messages, TaskSpec) -> TransformResult`) and `StrategyInfo` (id, version, kind,
    description, intended use, limitations, categories, `general_safe`, evidence note);
  - `content_hash` over the metadata and every template, so changed wording needs a new version.
- `strategies/builtin.py`: eight v1 strategies.
  - `identity`.
  - Additive: `concise_directive`, `role_framing`, `stepwise_plan`, `output_schema`, `security_generic`.
  - Structural: `structured_sections`, `requirements_checklist`.

  Additive strategies put the harness instruction first, as its own system message. Structural ones wrap or
  append around the user's text. **None paraphrases the user's text.**
- `strategies/registry.py`: lookup by id and version; `candidates(spec)` returns the applicable strategies with
  identity first, and only `general_safe` ones when the category is `general`. Re-registering an id and version
  with different content is refused.
- The policy that chooses among candidates is Phase 10; the transformation engine and preservation check are
  Phase 8. Nothing calls the strategies on the request path yet.

**Deferred, with reasons:**
- `security_targeted` needs a CWE knowledge base. LeBlanc's cannot be reused (no licence) and writing a new one
  is a separate piece of work.
- `few_shot_from_library` needs a curated example library kept separate from every evaluation split.
- `minimal_context` and `llm_rewrite` are rewriting strategies; they come after Phase 8's preservation gate (D7).

**Tests:**
- Every built-in strategy is applied to every one of the 320 dataset prompts where it is applicable. Checked
  each time: the user's text appears verbatim, the user's own system message is unchanged, the last message is
  still from the user, and everything the strategy added is visible.
- Also tested: identity is a no-op, candidate rules, unique ids and hashes, and refusal of changed content under
  an existing version.
- 109 unique offline tests pass; ruff and mypy --strict are clean.

**Observation (local Ollama):** the first call to `qwen2.5vl:3b` timed out three times at the 120 s default while
the model loaded cold on an 8 GB Mac, with a large download running at the same time. A direct call then took
23 s, and the harness call 10 s. The default was not changed; the docs tell users to raise
`HTTPConfig.timeout_s` for large local models.

## 2026-11-10: Review: the decision loop (root-cause fix)

**Most significant limitation found:** the components did not form a working system.
- `generate()` hard-coded `identity`, and nothing outside the strategies package used the strategy library.
- No code wrote or read the `evaluation` or `arm_stats` tables.
- So the harness could not observe outcomes, accumulate evidence or choose anything. Every later phase would
  have been built on untested interfaces.
- How this was confirmed: a code audit, and a clean wheel install outside the repo that ran the quickstart
  against Groq. Every call went out unchanged.

**Fix:** the loop is now implemented end to end, as the thinnest correct version of DESIGN §6.4:
- `evaluation/`: typed outcomes; `run_check()` for developer-supplied checks; `feedback_result()`.
- `learning/evidence.py`: Beta evidence per (profile, category, strategy@version) in `arm_stats`.
- `learning/policy.py`: the exploit and explore rule.
- `config.PolicyConfig`: `exploration_rate` (default 0), `min_gain`, `confidence`, `min_trials`.
- `api.py`:
  - classify → candidates → choose → transform, with a verbatim guard (`_preserves`) and fallback to the
    original prompt on any problem → call → record (original **and** sent prompt, per privacy setting) →
    evaluate if `check=` is given → update evidence (once per response);
  - `feedback()`; `profile()["evidence"]`.

Outcome signals are only developer checks and feedback. No proxy metric (such as "the code parses") updates the
evidence, in line with LeBlanc Finding C.

**A defect found by measurement, and fixed:**
- The first rule (Thompson exploration, a 0.90 threshold with no correction, no minimum sample) switched away
  from identity in **15 of 40** simulated runs where no strategy was better.
- Causes: identity was starved of samples; the rule was re-checked on every call; and several challengers each
  had a chance to look lucky.
- Revised rule: uniform exploration, `min_trials = 30` for both arms, Bonferroni-corrected threshold.
- Result: **2 of 40** false switches, and **39 of 40** runs found a truly better strategy (0.9 vs 0.3), with 400
  calls per run.
- A larger run (100 runs × 400 calls, same settings): **3 of 100** false switches, and **98 of 100** runs found
  the better strategy. See `scripts/simulate_policy.py` (simulated; not model evidence).

**Tests:**
- Unit: checks, evidence, policy thresholds, exploration gating.
- End to end with a simulated model: the harness finds the better strategy, then sends it, with the user text
  verbatim. With nothing better, it keeps the original prompt. With no checks, nothing is learned. A crashing
  check is not counted. A failing or paraphrasing strategy falls back to the original prompt. `optimize=False`
  never changes the prompt.
- 120 unique offline tests pass; ruff and mypy --strict are clean.

**Live check (Groq `openai/gpt-oss-20b`, free, 16 calls, exploration 1.0):**
- Strategies were applied to real calls (5 of 16 sent a transformed prompt). All 16 answers passed their checks,
  and 16 trials, 16 evaluations and 7 evidence rows were stored.
- All answers being correct means these questions are at ceiling for this model, so this run says nothing about
  which strategy is better. It only verifies the mechanism.
- 8 of the 16 math questions were classified `general` and 2 `multi_step`. This is the Phase 5 weakness showing
  live; `general` correctly limited the choice to general-safe strategies.

**Phase order:** this work brings forward the minimal parts of Phases 8 (verbatim guard), 9 (typed outcomes and
developer checks), 10 (evidence and policy) and 11 (orchestration). The later phases deepen each part instead of
connecting them for the first time.

## Known issues / open questions
- OpenAI and Anthropic chat calls cannot be completed until those accounts have credit. Their adapters are
  contract-tested, and their auth, model listing and error paths are live-checked.
- Dataset licences for IFEval, GSM8K, HumanEval and MBPP must be verified before Phase 12 (CodeSecEval: local only).
- Python 3.10–3.12 compatibility is checked by mypy (`python_version = 3.10`) but tests have only run on 3.13;
  CI across versions comes in Phase 13.

## Remaining phases
Phases 4–13 as listed in `DESIGN.md` §14, each gated on owner approval.

## 2026-11-10: Phase 9 (evaluators and Docker sandbox)
- `sandbox/`: a Docker sandbox (no network, read-only root, tmpfs, uid 65534, no capabilities, pids/memory/CPU
  limits, image pinned by digest). A trusted runner executes the model's code in a separate child process that
  reports through a private pipe tagged with a per-job nonce. Infrastructure problems are `infra_error`.
- Self-test (`python -m adaptive_harness.sandbox`): **9 of 9** pass against real Docker. Network blocked, host
  files invisible, no keys in the environment, read-only root, unprivileged user, infinite loop stopped, memory
  bomb killed, fork bomb contained, printed fake result ignored.
- `evaluation/builtin.py`: `PythonSyntax`, `JsonOutput` (weak, structural), `ReferenceAnswer`, `UnitTests`
  (sandboxed). Any of them can be passed as `check=`.
- Tests: 146 pass, including 16 against real Docker (skipped, not faked, when Docker is unavailable).
- Live: Groq `gpt-oss-20b` wrote `is_prime`; the code was run in the sandbox against unit tests (success), and
  the outcome was recorded as evidence.
- Not done: gVisor auto-detection; secondary (non-primary) evaluations on the request path.
