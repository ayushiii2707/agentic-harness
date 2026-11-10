# Phase 12 experiment

Research code for the pre-registered comparison of the harness against simpler baselines. **Not part of the
installed package** (the wheel ships only `src/adaptive_harness`).

| File | What it is |
|---|---|
| `preregistration_v1.md` | hypotheses, conditions, splits, outcomes, analysis plan and decision bands, committed before any data, plus dated amendments |
| `config_v1.json` | the machine-readable settings the runner uses |
| `runner.py` | resumable phases: `calibrate`, `select`, `evaluate`, `export` |
| `analysis/stats.py` | exact McNemar, Benjamini–Hochberg, cluster bootstrap, Wilson, minimum detectable effect |
| `analysis/report.py` | builds `results/v1/report.md` from the exported CSVs and the frozen `selections.json` |
| `results/v1/` | `selections.json`, one CSV per model (one row per trial, no response text), `report.md` |

Reproduce:

```bash
python -m experiments.runner --phase calibrate        # all models in the config; rerun to resume
python -m experiments.runner --phase select           # once, after every model has finished calibrating
python -m experiments.runner --phase evaluate         # rerun to resume
python -m experiments.runner --phase export
python -m experiments.analysis.report --results experiments/results/v1
```

Raw trial databases, with full prompts and responses, are written to `.harness/experiments/v1/` (not committed).
Every CSV row carries the `trial_id` of its database row.
