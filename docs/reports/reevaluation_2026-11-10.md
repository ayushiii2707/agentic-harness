# Re-evaluation after the fixes (2026-11-10)

The same method as `evaluation_2026-11-10.md` was used again:
- the full suite on an isolated `git archive` copy;
- ruff and mypy;
- every original probe, re-run unchanged against the fixed code;
- the simulations;
- live checks on free tiers.

No paid calls were made.

## Findings, before → after

| # | Severity | Before (evidence) | After (evidence) | Regression test |
|---|---|---|---|---|
| F1 | Critical | 50 paid calls; ledger $0 | a zero cap on a hosted provider is refused before any call; with a $0.001 cap and $0.005 budget it stops at exactly 5 calls | `test_a_paid_provider_cannot_be_calibrated_with_a_zero_reservation`, `test_a_paid_auxiliary_model_needs_a_positive_reservation` |
| F2 | High | host switches gave a stale profile and identity forever | the same probe learns `role_framing`; profile not stale; no change events. Drift starts profile v2 at once | `test_a_changed_serving_host_is_a_fact_not_a_model_change`, `test_drift_marks_the_profile_stale_and_starts_a_new_version_at_once` |
| F3 | High | defaults: identity forever | defaults explore while undersampled, learn the better strategy, then stop exploring; local models auto-calibrate | `test_the_default_config_learns_and_then_stops_exploring`, `test_auto_calibration_defaults_to_real_local_providers_only` |
| F4 | Medium | double-counted 48/50 and 30/30 | 0/50 and 0/30 | `test_concurrent_reports_for_one_response_count_once` |
| F5 | Medium | feedback credited to profile v2 | credited to the serving profile v1 | `test_feedback_is_credited_to_the_profile_that_served_the_response` |
| F6 | Medium | user spend $0.000633 with no user calls | $0 | covered by probe P6 |
| F7 | Medium | 2 KB 14.6 ms, 10 KB 79 ms, 100 KB ~1 s | 2.5 ms, 16 ms, 56 ms (400 KB: 0.2 s); classifier results byte-identical | `test_analysis_time_stays_bounded_for_long_prompts`; docs restate O5 by size |
| F8 | Medium | environment raised the code's $0 budget to $5 | code wins; `.env` only inside the project | `test_config_precedence.py` |
| F9 | Medium | hallucinated import → excluded | failure (opt-out flag) | `test_an_unavailable_import_is_a_model_failure_unless_allowed` (real Docker) |
| F10 | Medium | `close()` 6.3 s | 0.1 s; calibration reported as cancelled | `test_close_cancels_a_running_calibration_promptly` |
| F11 | Medium | `role="tool"` rejected | two-way tools; live round trips on Groq and Gemini | `tests/contract/test_tools.py` (8 tests) |
| F12 | Low | abandoned stream: no record | recorded as failed, never counted | `test_an_abandoned_stream_is_recorded_but_never_counted` |
| F13 | Low | wheel from the working folder ships a stale module | **needs the owner:** the safety hook blocks the delete command; build releases from a clean checkout | — |
| F14 | Low | same model via two providers passed as an independent judge | refused | `test_the_same_model_through_another_provider_is_not_an_independent_judge` |
| F15 | Low | empty prompts sent | rejected before any call | `test_empty_prompts_are_rejected_before_any_call` |
| F16 | Low | raw sqlite errors escaped | `StoreError` | `test_sqlite_errors_inside_a_transaction_become_store_errors` |
| R2 | risk | unbounded caches | bounded LRU (4,096) | `test_bounded_cache_evicts_the_least_recently_used` |

**Found during the re-evaluation and fixed:** the Gemini live tool round trip failed with HTTP 400, because
thinking models need their `thoughtSignature` sent back with the function call. It is now carried on `ToolCall`,
and `finish_reason` is `tool` when the model calls a tool (`test_gemini_thought_signatures_are_returned_with_their_function_call`).
Synthetic contract tests could not have found this.

## Other checks

- **Tests:** 283 pass, including 46 real-Docker sandbox tests. ruff and mypy `--strict` are clean.
- **Policy simulation** (40 × 400 calls, simulated): 0/40 false switches; the true effect found in 40/40.
- **Loop audit:** decisions matched calibration in 200/200 held-out prompts.
- **Overhead:** 0.5 ms p50 per call and 1.0 ms with a check, short prompt.
- **Unchanged probes:** stateless fallback; no exploration without a check; malformed checks become
  `evaluation_error`.

## Still open (unchanged by these fixes)

- **R1:** elimination is not corrected for repeated tests; the error leans toward identity.
- **R3:** auto-calibration shares the user's rate limits.
- **R4:** sandbox scoring is not safe against adversarial code (documented).
- **R5:** the LLM classifier's confidence is a constant.
- **R6:** prompts can steer the LLM classifier, with bounded impact.
- **R7:** online learning needs many checked calls.
- **U1–U6:** no live OpenRouter host-switch observation; OpenAI and Anthropic not completed; Python 3.10–3.12 not
  run; no strategy yet shown to help a model; suite transfer unproven.
- **Design gaps for Phase 12:** no "model-only" policy mode (condition C); task complexity is not used by the
  policy.
- **Missing for release (Phase 13):** CI, changelog, contribution guide and API reference.
