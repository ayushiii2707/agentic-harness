# Phase 11 and the known-limits pass (2026-11-10)

This follows `review_phases_1_10.md`, which listed six known limits. Each one, and the Phase 11 acceptance
criteria, are checked below.

## Phase 11 acceptance (DESIGN §14)

| Criterion | Result | Evidence |
|---|---|---|
| `generate()` runs the full path | yes | `tests/unit/test_system_loop.py` |
| An evaluation failure never loses a response | yes: a broken store, check or evidence update is logged and the answer returned | `test_an_evaluation_failure_never_loses_the_response` |
| Background evaluation | yes (`background=True`, `flush()`) | `test_background_check_returns_at_once_and_counts_later` |
| Streaming | yes: the trial is recorded and the check runs after the final event | `test_stream_records_and_evaluates_after_the_last_event` |
| Tools pass-through | yes: definitions via `extra=`; tool calls normalised for 3 provider APIs | `tests/contract/test_tools.py` (synthetic bodies) |
| Overhead O5 (< 5 ms p50, profile lookup < 1 ms) | 0.8 ms, 1.6 ms with a check; 0.03 ms (it was 20 ms) | `scripts/bench_overhead.py` |
| End to end on mock and Ollama | yes | `scripts/e2e_ollama.py qwen2.5-coder:7b` |

## Known limits: before → after

| Limit | After | Evidence |
|---|---|---|
| 59/100 held-out math prompts labelled `general` | 0/100 | `scripts/loop_audit.py` |
| Lexicon classifier at 0.688 on the test split | unchanged without the opt-in LLM fallback; with it 0.766, coverage 0.953 (dev 0.938 → 0.927) | `scripts/eval_llm_classifier.py` |
| No per-model difference shown (gpt-oss at ceiling) | for qwen3.8-27b, two strategies are eliminated as harmful; identity is kept. gpt-oss-20b is at ceiling on the same suite | `calibration_cal-97472c9e95.md` |
| A stale profile waits for a manual calibration | opt-in automatic calibration: free for local models, budget-gated for paid ones | `test_orchestration.py`; Ollama run |
| Sandbox output stored whatever `prompt_storage` is | follows `prompt_storage` | `test_evaluation_details_follow_prompt_storage` |
| Rewriter blocks `agenerate` | runs in a worker thread | `test_async_planning_and_checks_run_off_the_event_loop` |

## Defects found on the way, all fixed with regression tests

- **Rate limits (D48).** A tokens-per-minute 429 from Groq was mapped to a terminal quota error, because its
  upgrade link contains "billing". It is now retried.
- **Account IDs (D48).** Provider error text carried an organization ID into trace notes and reports. Account IDs
  are now masked.
- **Classifier labels (D47).** LLM classifier labels lived only in memory, so a restart could re-label a prompt
  and split its evidence. They are now stored.

## Verification

- 256 tests on an isolated `git archive` tree: 211 unit and contract tests plus 45 real-Docker sandbox tests.
- ruff and mypy `--strict` clean.
- A clean wheel install, with the Phase 11 paths exercised from the installed package.
- Policy simulation with the exact comparison: 0/40 false switches; the true effect found in 40/40.

Spend: $0.00. The live calls used Groq's free tier and local Ollama.

## Still open

**Detection limits.** No strategy has yet been shown to *help* a model. With about 55 tasks per arm, the smallest
detectable effect is about 20 pp. Phase 12 is where effects get measured properly, with pre-registration.

**Classifier trade-off.** The LLM fallback gives a category to some prompts whose true label is `general` (dev
accuracy −0.011). It is opt-in for this reason.

**Tool conversations.** Sending tool results back to the model is not abstracted across providers (D45).
