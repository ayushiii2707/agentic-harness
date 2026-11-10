# Pre-registration: Phase 12 experiment, version 1

**Status:** written and committed on 2026-11-10, **before any data for this experiment was collected**. The commit
that adds this file is the reference point. Every later change is listed under "Amendments", with its date and its
reason. Analyses that are not listed here are reported as exploratory.

## 1. Question

Does choosing prompting strategies per model and per task category (the harness, condition **D**) change
held-out task success compared with:
- **A**, the original prompt;
- **B**, one fixed "best practice" strategy;
- **B+**, the single best fixed strategy learned from data;
- **C**, per-model selection that ignores the task category?

And at what token cost? "No effect" is an acceptable, reportable answer (DESIGN §2, O1–O2).

## 2. Conditions

All conditions use the same harness code path (`AdaptiveHarness.run_trial`): the same task analysis,
transformation guard (a strategy that cannot be applied safely falls back to the original prompt, and this is
recorded), model call, evaluator and trial record. Only the strategy differs.

| Condition | Strategy used for a held-out task |
|---|---|
| **A** | `identity`: the original prompt |
| **B** | `structured_sections`, fixed a priori (DESIGN §13), for every model and task |
| **B+** | one strategy for every model and task: the result of the harness's decision rule (§5) on calibration outcomes pooled over **all models and all calibration suites** |
| **C** | per model: the decision rule on that model's calibration outcomes pooled over **all calibration suites** (the task category is ignored) |
| **D** | per model and task: exactly what `generate()` would send with exploration off. The harness's policy runs on that model's calibration evidence, keyed by the category its classifier predicts for the task |

- **B, B+ and C** apply their strategy to every task, whatever its category.
- **D** respects the harness's own rules: if the category is uncertain (`general`), only strategies marked
  general-safe may be used.
- If a selection rule finds no strategy that beats identity, that condition *is* identity for that model. This
  is a legitimate outcome, not a failure.

## 3. Models (free tiers or local; no paid calls)

| Model | Provider | Why |
|---|---|---|
| `openai/gpt-oss-20b` | Groq | reasoning model; near ceiling on the suites in pilot work (Phase 7) |
| `qwen/qwen3.8-27b` | Groq | mid-range; off ceiling on the instruction suite (Phase 11) |
| `gemini-3.5-flash-lite` | Gemini | a different model family |

If a model becomes unavailable during collection, its results are reported as incomplete; no model is replaced
after data collection starts.

Generation settings:
- `max_tokens` is 2000 for gpt-oss-20b (it reasons before answering) and 1000 for the others.
- Temperature and top_p are the provider defaults (not sent), recorded per trial.
- No seed.

## 4. Tasks and splits (suite version 2)

The tasks are the harness's own generated suites with deterministic evaluators. Splits come from disjoint seeds;
held-out prompts never appear in calibration, and the runner checks this.

| Suite | Calibration (C, D, B+ learn here) | Held-out test (all reported numbers) | Evaluator |
|---|---|---|---|
| `math` | first 35 tasks | first 60 tasks | numeric reference answer |
| `structured_output` | first 35 tasks | first 60 tasks | exact JSON fields |
| `instructions` | first 35 tasks | first 60 tasks | exact constraint check |
| `code_generation` | **none: held-out category** | all 30 tasks | unit tests in the Docker sandbox |

- **Code generation** is never calibrated. It tests how each condition behaves on an unseen category: C and B+
  transfer a strategy blindly, and D has no cell evidence. Hallucinated imports count as failures (D56).
- **No validation split.** This deviates from DESIGN §13, deliberately: no threshold or hyperparameter is tuned in
  Phase 12. All decision-rule settings are the harness defaults frozen at the commit of this file
  (`min_trials` 30, `min_gain` 0.02, confidence 0.90 with Bonferroni, cost weight 0.1). B+ is selected on
  calibration data, which is disjoint from the test data.

## 5. Calibration design and selection rule

**Calibration.** Every calibration task is run **once** under each of 7 strategies, a full paired design with no
elimination:
- identity, concise_directive, role_framing, stepwise_plan, output_schema, structured_sections,
  requirements_checklist.

`security_generic` (code only) and the rewriting strategies (`minimal_context`, which rarely applies, and the
opt-in `llm_rewrite`) are excluded.

**Selection.** Selection uses the harness's own decision rule (`Policy.choose`, exploit mode). A strategy
replaces identity only if:
- both arms have at least 30 outcomes;
- P(strategy beats identity by at least 2 pp plus its token-cost penalty) is at least 1 − 0.10/k.

For each condition:
- **D:** the per-cell evidence the harness stores (model × predicted category).
- **C:** the same outcomes pooled over categories, per model, with all 6 non-identity arms as candidates.
- **B+:** pooled over models too.

The selections are written to `selections.json` before any held-out call.

## 6. Held-out runs

- Every held-out task is run **2 times** (repetitions 1 and 2) under each *distinct* strategy among A, B, B+, C
  and D for that task.
- When two conditions pick the same strategy for a task, they share that trial. Their outputs come from the same
  distribution; the sharing is recorded.
- Calls run in a fixed, seeded order.

## 7. Outcomes

- **Primary outcome:** task success (1/0) from the suite's deterministic evaluator.
- **Excluded from the primary analysis, counted under reliability:** `evaluation_error` results (for example,
  Docker down) and calls that fail after retries.
- **Secondary outcomes:**
  - input, output and reasoning tokens;
  - latency;
  - the transformation fallback rate (the chosen strategy could not be applied, so identity was sent);
  - API errors;
  - evaluation errors;
  - fingerprint changes during collection.

## 8. Analysis plan

The analysis is done per model. The unit of pairing is (task, repetition).

**Primary family** (in-distribution held-out tasks: math, structured_output and instructions together), 6
contrasts:
- **D − A** (primary);
- D − B, D − C, C − A, B − A, B+ − A.

For each contrast:
- the success-rate difference in percentage points;
- an exact McNemar test on discordant pairs (two-sided);
- a 95% CI from a cluster bootstrap that resamples tasks with their repetitions (10,000 resamples, seed 12);
- Benjamini–Hochberg at q = 0.05 within the model's family.

**Decision bands** (fixed now):

| Label | Rule |
|---|---|
| **improves** | BH-adjusted p < 0.05 **and** CI entirely above 0 |
| **harms** | BH-adjusted p < 0.05 **and** CI entirely below 0 |
| **non-inferior** (O1, for D − A) | lower CI bound > −5 pp |
| **inconclusive** | otherwise |

**Secondary family** (descriptive; BH applied within it): the same contrasts per suite, and on the held-out
category (code generation).

**Cost:** mean total tokens per task per condition, and the ratio to A.

**Minimum detectable effect:** reported per model for D − A, from the observed discordance rate (80% power,
alpha 0.05).

**Missing data:**
- A (task, repetition) pair enters a contrast only if both arms have a valid outcome.
- Numbers of excluded pairs are reported.
- No imputation.

## 9. Integrity

- The experiment uses its own databases (`.harness/experiments/`), separate from any other run.
- Prompts and outputs are stored in full for traceability; these are our own synthetic tasks.
- A per-trial CSV (no response text) and the analysis report are committed. The report refuses to describe as
  model results any trial marked simulated.
- **Pilot:** a pilot on at most 10 calibration tasks may run first, to check the plumbing (rate limits, token
  limits, parsing). Its data is discarded, not analysed, and it may only change operational settings
  (`max_tokens`, pauses), recorded as amendments.

**Collection window:** at most 3 days from the first main call. After that, the analysis runs on complete pairs
only, and the report states which cells are incomplete.

**Spending:** $0 real spend; free tiers only. Hosted providers need a positive per-call reservation (D53). It is
set to $0.001 against a bookkeeping budget in the experiment database; the user's $10 cap is untouched.

## 10. Known limitations, stated in advance

- **Tasks.** The tasks are generated, not natural user prompts, so transfer to real use is not shown.
- **Ceiling.** gpt-oss-20b is near ceiling on math and extraction, so small effects there are unmeasurable.
- **Calibration size.** With 35 calibration tasks per suite, the decision rule's 30-outcome gate leaves little
  room: some D cells (e.g. instruction prompts labelled `general`) will be identity by construction.
- **Power.** About 180 in-distribution tasks × 2 repetitions per model detect differences of about 8–12 pp,
  depending on discordance. Smaller effects will read as inconclusive.
- **Classifier.** Only the default deterministic classifier is used (not the opt-in LLM fallback).

## Amendments

1. **2026-11-10, pilot (plumbing only, data discarded).**
   - What ran: 3 calibration tasks per suite × 3 strategies, plus 2 code tasks, on each model (87 trials).
   - What it checked: every call finished normally, with no truncation (largest output: 921 of 2000 tokens for
     gpt-oss-20b, 284 of 1000 for qwen3.8-27b, 256 of 1000 for gemini-3.5-flash-lite); no API or evaluation
     errors; the Docker sandbox ran the code tests.
   - **No operational setting was changed.**
2. **2026-11-10, gpt-oss-20b moved to OpenRouter (before any held-out call).**
   - **Why:** after 325 of 735 calibration calls, Groq's free tier stopped gpt-oss-20b at its daily token limit
     (200,000 tokens). At that rate it needed about 4–5 more days, beyond the 3-day window, and the B+ selection
     (which pools all models) blocked every model's held-out runs.
   - **Change,** approved by the owner:
     - gpt-oss-20b runs on OpenRouter, pinned to one host, DeepInfra bf16, with fallbacks to other hosts
       disabled. OpenRouter serves this model from 11 hosts at fp4, fp8 and bf16, so pinning keeps the serving
       setup fixed.
     - It is paid: a hard $1.00 budget in this model's experiment database, and a $0.0005 reservation per call
       (the worst case at 2000 output tokens is about $0.0003).
     - The 325 Groq calibration trials are **discarded**, so all of this model's data comes from one provider and
       one host. They stay in their own database, unused.
   - **Unchanged:** the model, generation settings, tasks, conditions and analysis. qwen3.8-27b and Gemini are
     unaffected.
3. **2026-11-11, gemini-3.5-flash-lite moved to OpenRouter (before any held-out call).**
   - **Why:** the Gemini free tier's daily request quota (about 500 requests) stopped calibration at 451 of 735
     calls. Finishing on the free tier needed 3–4 more days, beyond the 3-day window.
   - **Change,** approved by the owner after a corrected cost estimate:
     - Model `google/gemini-3.5-flash-lite` on OpenRouter, pinned to Google AI Studio's flex tier (Google's own
       backend, the one the free tier also uses), with fallbacks to other hosts disabled.
     - Paid: a hard $1.50 budget, a $0.0015 reservation per call, and an expected total of about $0.70.
     - The 451 free-tier calibration trials are **discarded**, so all of this model's data comes from one route.
   - **Possible difference:** OpenRouter's defaults for this model (for example, thinking) may differ from the
     native API's. All analysed Gemini data uses the same route, so conditions stay comparable.
   - **Unchanged:** tasks, conditions, generation settings that are sent (`max_tokens` 1000) and the analysis.
