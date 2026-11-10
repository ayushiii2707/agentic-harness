# Review of Phases 1–10 (2026-11-10)

**Question:** Does the loop work end to end? Does the harness learn from real outcomes only, and fail safe?
Does every claim in the docs match the code?

## 1. What was run

| Check | How | Result |
|---|---|---|
| Unit, contract and real-Docker sandbox tests | `pytest` on an isolated `git archive HEAD` tree | **229 pass** (134 unit, 50 contract, 45 sandbox) |
| Static checks | `ruff check`, `ruff format --check`, `mypy --strict` (53 files) | clean |
| Whole lifecycle in one test | `tests/unit/test_system_loop.py` (new) | passes |
| Clean install | wheel built from the isolated tree, installed into a fresh venv, lifecycle script run with `python -I` | same behaviour as in the repo |
| Policy false-switch rate | `scripts/simulate_policy.py`, 40 × 400 calls (simulated) | 0/40 false switches; true effect found in 39/40 |
| Drift threshold | `scripts/simulate_drift.py` (simulated) | z = 4.0: 4/399 false alarms over 500 outcomes; 93/100 detections |
| Calibration → request path | `scripts/loop_audit.py` (simulated) | decision matched the calibrated best for its own cell in 200/200 held-out prompts |
| Live loop, real model | `scripts/live_loop_check.py groq openai/gpt-oss-20b --calls 16` (free tier) | 16/16 calls evaluated; evidence written per (category, strategy) |
| Live rewriter + judge | Groq gpt-oss-20b rewrites, Gemini 3.5 flash-lite judges (free tier) | 3/3 faithful rewrites sent; 3/3 damaged rewrites rejected by the judge |
| Local model | Ollama `qwen2.5-coder:7b` | answer correct, check passed (113 s cold load) |

The lifecycle test runs the following sequence on one store, across restarts:
1. Cold start: the original prompt is sent.
2. Unchecked calls never explore and create no evidence.
3. Checked calls explore; 400 outcomes are recorded.
4. After a restart, the better strategy is used, and the user's text is still sent verbatim.
5. Unchecked calls change nothing.
6. Feedback is counted once; a crashing check is never counted.
7. The model changes: drift, then stale, then identity is used.
8. Recalibration: profile version 2 is created, with no inherited evidence, and $0 spent.

Spend: **$0.00** in this review. All live calls used free tiers or a local model. The cumulative harness spend
stays at $0.0000066 of the $10 cap.

## 2. Defect found and fixed

**Prompt fragments in stored traces (privacy).**
- When a rewrite was rejected, the reason quoted parts of the prompt, such as a number, a path or a negative
  sentence.
- With the default `prompt_storage="none"`, these reasons were still written to `trial.trace_json` and to the
  warning log.
- Quoted spans are now replaced before storage, and in logs unless `log_content` is on.
- Regression test: `test_stored_rejection_notes_never_quote_the_prompt`. Decision D40.

## 3. Phase by phase: what is verified

| Phase | Verified behaviour | Evidence |
|---|---|---|
| 1–2 Foundation, store, budget | forward-only migrations; reserve/commit/release ledger; caps hold across processes; user calls recorded, never blocked | `test_store_and_budget.py` |
| 3 Providers | 9 OpenAI-compatible presets, Anthropic and Gemini; errors classified, billing errors typed; dropped settings recorded | contract tests; earlier live smoke checks |
| 4 Model intelligence | fingerprint or version change marks profiles stale; facts keep provenance | `test_model_registry.py` |
| 5 Task understanding | classifier with a `general` fallback; requirements extraction | `test_tasks.py`; `reports/phase5_classifier.md` |
| 6 Strategies | versioned, content-hashed; changed wording without a new version is disabled | `test_strategies.py` |
| 7 Calibration | paired elimination per predicted-label cell; budget reserved per call; resumable | `test_calibration.py`; earlier live run |
| 8 Transformation | one engine; verbatim rule; mutation-tested preservation; LLM rewrite needs exact checks plus an independent judge | `test_transform.py`, `test_llm_rewrite.py`; live judge probes |
| 9 Evaluators, sandbox | only `check=` and `feedback()` count; sandbox: no network, read-only, non-root; infrastructure errors never counted | `test_evaluators.py`, `tests/sandbox/` |
| 10 Adaptation | decay, capped backoff, priors (estimate only, never the gate), drift → stale → new profile version, cost penalty | `test_learning_adaptation.py`, `test_system_loop.py` |

## 4. Doc claims checked against code

These were out of date and are now corrected:
- README status;
- DEVLOG status list;
- DESIGN §6.4 ("cost term not applied yet");
- DATA_FLOWS ("later phase" for calibration and the judge/rewriter);
- the Phase 8 DEVLOG limit on paraphrase.

DESIGN §6.2, §6.6 and §6.7 now have "as implemented" notes, wherever the implementation differs from the
original plan:
- the drift test is windowed instead of CUSUM;
- age is handled by decay and the `n_eff` gate;
- priors use the secure-and-working rate.

## 5. Known limits (stated, not hidden)

**Classification.** 59 of the 100 held-out math prompts are labelled `general`, so they get identity. This is
safe, but nothing is learned for them.

**What learning can show.**
- The harness can only learn as much as the checks you give it.
- With few checked calls, `n_eff` stays below 30, so nothing changes; this is by design.
- The calibration suites are at ceiling for gpt-oss-20b (Phase 7), so a real per-model difference has not yet been
  demonstrated. That is the research question for Phase 12.

**LLM rewriter.**
- The judge is a model and can be wrong. The exact checks are the floor.
- Rewrites run synchronously inside `agenerate` planning.

**Priors.** The LeBlanc priors are weak (1.5 pseudo-observations), and they cover only three model IDs and one
category.

**Stored evaluation details.** Sandbox test output may contain lines from the model's response. Evaluation
details are stored regardless of `prompt_storage`.

**Not built yet.**
- Automatic (unattended) calibration scheduling. A stale profile waits until someone runs `Calibrator`.
- Phases 11–13: orchestration, research experiments, packaging and CI.
