# Calibration report `cal-97472c9e95`

Model `qwen/qwen3.8-27b`, profile 1. Calls this run: 62; spend this run: $0.000000.

Evidence is stored under the category the harness's classifier assigns (the same key used when
choosing a strategy for a new prompt), so one suite can produce several cells.

## suite `instructions` → cell `api_development`: 4 tasks; stopped: suite exhausted
Usable by the policy (enough outcomes): no. Smallest difference from identity this cell could reliably detect (80% power, alpha 0.05): 43 pp.

| strategy | n | successes | rate | 95% interval | eliminated |
|---|---|---|---|---|---|
| identity | 4 | 0 | 0.17 | [0.01, 0.51] | no |
| concise_directive | 4 | 0 | 0.17 | [0.00, 0.51] | no |
| requirements_checklist | 4 | 0 | 0.17 | [0.01, 0.53] | no |
| role_framing | 4 | 1 | 0.33 | [0.05, 0.72] | no |
| security_generic | 4 | 1 | 0.33 | [0.05, 0.71] | no |

## suite `instructions` → cell `content_generation`: 55 tasks; stopped: suite exhausted
Usable by the policy (enough outcomes): yes. Smallest difference from identity this cell could reliably detect (80% power, alpha 0.05): 20 pp.

| strategy | n | successes | rate | 95% interval | eliminated |
|---|---|---|---|---|---|
| identity | 55 | 9 | 0.18 | [0.09, 0.28] | no |
| requirements_checklist | 55 | 8 | 0.16 | [0.08, 0.26] | no |
| role_framing | 25 | 0 | 0.04 | [0.00, 0.13] | yes |
| structured_sections | 25 | 0 | 0.04 | [0.00, 0.13] | yes |

## suite `instructions` → cell `research`: 1 tasks; stopped: suite exhausted
Usable by the policy (enough outcomes): no. Smallest difference from identity this cell could reliably detect (80% power, alpha 0.05): 86 pp.

| strategy | n | successes | rate | 95% interval | eliminated |
|---|---|---|---|---|---|
| identity | 1 | 0 | 0.33 | [0.01, 0.84] | no |
| requirements_checklist | 1 | 1 | 0.67 | [0.15, 0.99] | no |
| role_framing | 1 | 0 | 0.33 | [0.01, 0.85] | no |
| structured_sections | 1 | 0 | 0.33 | [0.01, 0.83] | no |
