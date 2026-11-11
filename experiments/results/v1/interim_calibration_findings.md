# Interim, exploratory: calibration data (2026-11-11)

**Status.** These numbers come from the **calibration** split: 35 tasks per suite, each run once under every
strategy. They are the data the selections were made on, so they are **descriptive and exploratory**, not the
pre-registered confirmatory test. The confirmatory results are the held-out analysis in `report.md`, which is
still being collected.

**Test.** Exact McNemar, each strategy against identity on the same tasks, with outcomes credited to the assigned
strategy (intent to treat). Benjamini–Hochberg across all 54 comparisons (3 models × 3 suites × 6 strategies).

## Success, out of 35 tasks, by assigned strategy

| model | suite | identity | concise | role | stepwise | schema | structured | checklist |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | math | 35 | 35 | 35 | 35 | 35 | 35 | 35 |
| gpt-oss-20b | structured output | 35 | 35 | 35 | 34 | 35 | 35 | 35 |
| gpt-oss-20b | instructions | 34 | 29 | 33 | 35 | 33 | 35 | 35 |
| qwen3.8-27b | math | 34 | **9** | 35 | 35 | 32 | 34 | 35 |
| qwen3.8-27b | structured output | 35 | 35 | 35 | 34 | 35 | 34 | 34 |
| qwen3.8-27b | instructions | 2 | 1 | 3 | 2 | 1 | 1 | 4 |
| gemini-3.5-flash-lite | math | 34 | 34 | 34 | 33 | 34 | 34 | 34 |
| gemini-3.5-flash-lite | structured output | 35 | 35 | 35 | 35 | 35 | 35 | 35 |
| gemini-3.5-flash-lite | instructions | 26 | 22 | 26 | 14 | 26 | 28 | 28 |

## What survives correction

**One comparison: qwen3.8-27b on math, `concise_directive` against identity.**
- −71 pp (9/35 against 34/35).
- Discordant pairs: 1 task where only the strategy succeeded, 26 where only identity did.
- p = 4.2×10⁻⁷, BH q = 2.3×10⁻⁵.
- The same instruction has no effect on gpt-oss-20b (35/35) or gemini-3.5-flash-lite (34/35).
- This is a **model-specific harm** of one prompting strategy.

**No strategy improved any model** after correction (smallest q for a positive difference: 1.0). Gemini's −34 pp
with `stepwise_plan` on instructions (14/35 against 26/35) does not survive correction across 54 tests.

## What this means, and what it does not

- **Consistent with the problem statement.** The same prompting intervention can help, do nothing, or harm,
  depending on the model. The harness's selection rule never chose the harmful strategy, because it keeps
  identity unless a strategy is shown to be better.
- **Not shown.** That any strategy helps any of these models on these tasks. Two of the three models are at or near
  ceiling on math and extraction, and qwen3.8-27b is near the floor on the instruction suite. Both conditions
  leave little room to detect an improvement.
- **Not a final result.** The pre-registered held-out comparison (B, `structured_sections`, against the original
  prompt) is still being collected.
