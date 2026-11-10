# Calibration report `cal-eecdf4af35`

Model `openai/gpt-oss-20b`, profile 1. Calls this run: 10; spend this run: $0.000000.

Evidence is stored under the category the harness's classifier assigns (the same key used when
choosing a strategy for a new prompt), so one suite can produce several cells.

## suite `math` → cell `general`: 49 tasks; stopped: nothing to compare
Usable by the policy (enough outcomes): no. Smallest difference from identity this cell could reliably detect (80% power, alpha 0.05): -.

| strategy | n | successes | rate | 95% interval | eliminated |
|---|---|---|---|---|---|
| identity | 0 | 0 | 0.50 | [0.02, 0.98] | no |

## suite `math` → cell `math`: 51 tasks; stopped: max_calls (10) reached
Usable by the policy (enough outcomes): no. Smallest difference from identity this cell could reliably detect (80% power, alpha 0.05): 31 pp.

| strategy | n | successes | rate | 95% interval | eliminated |
|---|---|---|---|---|---|
| identity | 8 | 8 | 0.90 | [0.66, 1.00] | no |
| concise_directive | 8 | 8 | 0.90 | [0.67, 1.00] | no |
| requirements_checklist | 8 | 8 | 0.90 | [0.66, 1.00] | no |
| role_framing | 8 | 8 | 0.90 | [0.66, 1.00] | no |
| stepwise_plan | 8 | 8 | 0.90 | [0.66, 1.00] | no |
