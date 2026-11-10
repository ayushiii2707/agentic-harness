"""Build the Phase 12 analysis report from the exported per-trial CSVs and the frozen selections.

    python -m experiments.analysis.report --results experiments/results/v1

Implements preregistration_v1.md §8 exactly; anything else it prints is labelled exploratory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .stats import benjamini_hochberg, cluster_bootstrap_diff, mcnemar_exact, mde_paired, wilson

CONTRASTS = (("D", "A"), ("D", "B"), ("D", "C"), ("C", "A"), ("B", "A"), ("B+", "A"))
CONDITIONS = ("A", "B", "B+", "C", "D")
IN_DIST = ("math", "structured_output", "instructions")


def strategies_for(sel: dict[str, Any], key: str, task_id: str) -> dict[str, str]:
    m = sel["models"][key]
    return {"A": "identity", "B": sel["B"], "B+": sel["B+"]["strategy"], "C": m["C"]["strategy"],
            "D": m["D"][task_id]["strategy"]}  # fmt: skip


def load_rows(results: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for f in sorted(results.glob("*.csv")):
        with f.open(encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def outcome(r: dict[str, str]) -> float | None:
    if (
        r["status"] != "done"
        or r["eval_status"] not in ("success", "failure", "partial")
        or r["outcome"] == ""
    ):
        return None
    return float(r["outcome"])


def condition_table(
    rows: Sequence[dict[str, str]], sel: dict[str, Any], key: str
) -> dict[tuple[str, str, int], dict[str, float | None]]:
    """(suite, task_id, rep) -> condition -> outcome (None when missing or excluded)."""
    by_ref = {
        (r["task_id"], int(r["rep"]), r["strategy"]): r
        for r in rows
        if r["phase"] == "heldout" and r["model"] == key
    }
    table: dict[tuple[str, str, int], dict[str, float | None]] = {}
    for (task, rep, _), r in by_ref.items():
        strat = strategies_for(sel, key, task)
        table[(r["suite"], task, rep)] = {
            c: (outcome(by_ref[(task, rep, s)]) if (task, rep, s) in by_ref else None)
            for c, s in strat.items()
        }
    return table


def contrast(
    table: dict[tuple[str, str, int], dict[str, float | None]],
    first: str,
    second: str,
    suites: Sequence[str],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[tuple[float, float]]] = defaultdict(list)
    b = c = excluded = 0
    for (suite, task, _), cond in table.items():
        if suite not in suites:
            continue
        x, y = cond[first], cond[second]
        if x is None or y is None:
            excluded += 1
            continue
        clusters[task].append((x, y))
        b += x > y
        c += y > x
    n = sum(len(v) for v in clusters.values())
    point, lo, hi = cluster_bootstrap_diff(clusters, resamples, seed) if n else (math.nan, math.nan, math.nan)
    return {"n": n, "tasks": len(clusters), "b": b, "c": c, "excluded": excluded, "diff": point, "lo": lo, "hi": hi,
            "p": mcnemar_exact(b, c), "discordance": (b + c) / n if n else math.nan}  # fmt: skip


def label(r: dict[str, Any], margin_pp: float, primary: bool) -> str:
    if r["n"] == 0:
        return "no data"
    if r["q"] < 0.05 and r["lo"] > 0:
        return "improves"
    if r["q"] < 0.05 and r["hi"] < 0:
        return "harms"
    if primary and r["lo"] > -margin_pp / 100:
        return "non-inferior"
    return "inconclusive"


def pp(x: float) -> str:
    return "-" if math.isnan(x) else f"{100 * x:+.1f}"


def rate_line(table: dict[tuple[str, str, int], dict[str, float | None]], suites: Sequence[str]) -> list[str]:
    out = []
    for cond in CONDITIONS:
        xs = [v[cond] for (s, _, _), v in table.items() if s in suites and v[cond] is not None]
        lo, hi = wilson(sum(xs), len(xs)) if xs else (math.nan, math.nan)  # type: ignore[arg-type]
        rate = sum(xs) / len(xs) if xs else math.nan  # type: ignore[arg-type]
        out.append(f"| {cond} | {len(xs)} | {'-' if math.isnan(rate) else f'{100 * rate:.1f}%'} | "
                   f"{'-' if math.isnan(lo) else f'[{100 * lo:.1f}, {100 * hi:.1f}]'} |")  # fmt: skip
    return out


def git_commit_of(path: str) -> str:
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h %cs", "--", path], capture_output=True, text=True,
                              check=False).stdout.strip() or "unknown"  # fmt: skip
    except OSError:
        return "unknown"


def build(results: Path, cfg: dict[str, Any]) -> str:
    rows = load_rows(results)
    sel = json.loads((results / "selections.json").read_text(encoding="utf-8"))
    simulated = sum(r["simulated"] == "1" for r in rows)
    res, seed, margin = (
        cfg["bootstrap"]["resamples"],
        cfg["bootstrap"]["seed"],
        cfg["noninferiority_margin_pp"],
    )
    lines = ["# Phase 12 results (preregistration v1)", ""]
    if simulated:
        lines += [f"> **SIMULATED: {simulated} of {len(rows)} trials come from a mock model. Nothing in this report "
                  "is evidence about any real model.**", ""]  # fmt: skip
    lines += [
        f"Pre-registration: `{cfg['preregistration']}`, committed {git_commit_of(cfg['preregistration'])}. "
        f"Trials: {len(rows)} ({sum(r['phase'] == 'calibration' for r in rows)} calibration, "
        f"{sum(r['phase'] == 'heldout' for r in rows)} held-out).",
        "",
        "## Selections (made from calibration data before any held-out call)",
        "",
        f"- **B** (a priori): `{sel['B']}`",
        f"- **B+** (pooled over models): `{sel['B+']['strategy']}`: {sel['B+']['reason']}",
    ]
    for key, m in sel["models"].items():
        d_counts: dict[str, int] = defaultdict(int)
        for v in m["D"].values():
            d_counts[v["strategy"]] += 1
        lines.append(f"- **{key}**: C = `{m['C']['strategy']}` ({m['C']['reason']}); D per task: "
                     + ", ".join(f"`{k}` x {n}" for k, n in sorted(d_counts.items())))  # fmt: skip
    for key in sel["models"]:
        model_rows = [r for r in rows if r["model"] == key]
        table = condition_table(model_rows, sel, key)
        lines += ["", f"## {key}", "", "### Success rate per condition (in-distribution held-out: math, structured "
                  "output, instructions)", "", "| condition | outcomes | success | 95% Wilson |", "|---|---|---|---|"]  # fmt: skip
        lines += rate_line(table, IN_DIST)
        prim = [contrast(table, a, b, IN_DIST, res, seed) for a, b in CONTRASTS]
        for r, q in zip(prim, benjamini_hochberg([r["p"] for r in prim]), strict=True):
            r["q"] = q
        lines += ["", "### Primary family (pre-registered; BH within the family)", "",
                  "| contrast | pairs (tasks) | diff pp | 95% cluster-bootstrap CI | b / c | McNemar p | BH q | result |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for (a, b), r in zip(CONTRASTS, prim, strict=True):
            lines.append(f"| {a} - {b} | {r['n']} ({r['tasks']}) | {pp(r['diff'])} | [{pp(r['lo'])}, {pp(r['hi'])}] | "
                         f"{r['b']} / {r['c']} | {r['p']:.3g} | {r['q']:.3g} | {label(r, margin, (a, b) == ('D', 'A'))} |")  # fmt: skip
        da = prim[0]
        disc = da["discordance"]
        disc_txt = "-" if math.isnan(disc) else f"{100 * disc:.0f}%"
        mde = mde_paired(da["n"], 0.0 if math.isnan(disc) else disc)
        lines.append("")
        lines.append(
            f"Minimum detectable difference for D - A (80% power, alpha 0.05, observed discordance {disc_txt}): "
            f"{pp(mde)} pp. Pairs excluded for a missing or invalid outcome: {da['excluded']}."
        )
        sec: list[tuple[str, tuple[str, str], dict[str, Any]]] = []
        for suite in (*IN_DIST, *cfg["unseen_categories"]):
            for a, b in CONTRASTS:
                sec.append((suite, (a, b), contrast(table, a, b, (suite,), res, seed)))
        for (_, _, r), q in zip(sec, benjamini_hochberg([r["p"] for _, _, r in sec]), strict=True):
            r["q"] = q
        lines += ["", "### Secondary family: per suite, and the held-out category (descriptive; BH within the family)", "",
                  "| suite | contrast | pairs | diff pp | 95% CI | BH q | result |", "|---|---|---|---|---|---|---|"]  # fmt: skip
        for suite, (a, b), r in sec:
            tag = " (held-out category)" if suite in cfg["unseen_categories"] else ""
            lines.append(f"| {suite}{tag} | {a} - {b} | {r['n']} | {pp(r['diff'])} | [{pp(r['lo'])}, {pp(r['hi'])}] | "
                         f"{r['q']:.3g} | {label(r, margin, False)} |")  # fmt: skip
        held = [r for r in model_rows if r["phase"] == "heldout"]
        lines += ["", "### Cost (held-out, mean tokens per call)", "", "| condition | input | output | reasoning | vs A |",
                  "|---|---|---|---|---|"]  # fmt: skip
        base = None
        for cond in CONDITIONS:
            calls = [
                r
                for r in held
                if r["strategy"] == strategies_for(sel, key, r["task_id"])[cond] and r["status"] == "done"
            ]

            def mean(field: str, calls: list[dict[str, str]] = calls) -> float:
                xs = [float(r[field]) for r in calls if r[field] not in ("", None)]
                return sum(xs) / len(xs) if xs else math.nan

            tot = mean("input_tokens") + mean("output_tokens")
            base = tot if cond == "A" else base
            ratio = "1.00" if cond == "A" else ("-" if not base or math.isnan(tot) else f"{tot / base:.2f}")
            lines.append(f"| {cond} | {mean('input_tokens'):.0f} | {mean('output_tokens'):.0f} | "
                         f"{mean('reasoning_tokens'):.0f} | {ratio} |")  # fmt: skip
        fallbacks = sum(r["strategy"] != r["applied_strategy"] for r in model_rows)
        failed = sum(r["status"] == "failed" for r in model_rows)
        eval_err = sum(
            r["status"] == "done" and r["eval_status"] not in ("success", "failure", "partial")
            for r in model_rows
        )
        fps = {r["fingerprint"] for r in model_rows if r["status"] == "done"}
        lines += ["", "### Reliability", "",
                  f"- Trials: {len(model_rows)}; API failures after retries: {failed}; evaluation errors: {eval_err}.",
                  f"- Transformation fallbacks (strategy not applicable, original prompt sent): {fallbacks}.",
                  f"- Distinct model fingerprints during collection: {len(fps)}"
                  + (" (flag: the served model changed during collection)" if len(fps) > 1 else "") + "."]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", default="experiments/config_v1.json")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    results = Path(a.results)
    text = build(results, json.loads(Path(a.config).read_text(encoding="utf-8")))
    out = Path(a.out) if a.out else results / "report.md"
    out.write_text(text, encoding="utf-8")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
