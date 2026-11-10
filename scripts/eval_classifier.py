"""Evaluate the task classifier on data/task_classification/prompts.jsonl.

    python scripts/eval_classifier.py --check-leakage   # lexicon entries that match test prompts but no dev prompt
    python scripts/eval_classifier.py --tune            # grid-search thresholds on the DEV split only
    python scripts/eval_classifier.py --report          # metrics on DEV and TEST with the frozen taxonomy.json

Split rule (fixed before tuning): items -01..-12 of each category are dev, -13..-20 are test.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_harness.tasks import GENERAL, TaskClassifier, Taxonomy  # noqa: E402
from adaptive_harness.tasks.classifier import _phrase_regex  # noqa: E402

DATA = ROOT / "data" / "task_classification" / "prompts.jsonl"
TAXONOMY = ROOT / "src" / "adaptive_harness" / "tasks" / "taxonomy.json"
REPORT = ROOT / "docs" / "reports" / "phase5_classifier.md"


def load() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev = [r for r in rows if int(r["id"].rsplit("-", 1)[1]) <= 12]
    test = [r for r in rows if int(r["id"].rsplit("-", 1)[1]) > 12]
    return dev, test


def check_leakage(raw: dict, dev: list[dict[str, str]], test: list[dict[str, str]]) -> list[str]:
    import re

    flagged = []
    for cid, c in raw["categories"].items():
        entries = [(p, _phrase_regex(p), True) for p in c.get("phrases", {})]
        entries += [(p, re.compile(p, re.I | re.M), False) for p in c.get("patterns", {})]
        for label, rx, lower in entries:

            def hit(rows: list[dict[str, str]], rx=rx, lower=lower) -> bool:
                return any(rx.search(r["prompt"].lower() if lower else r["prompt"]) for r in rows)

            if hit(test) and not hit(dev):
                flagged.append(f"{cid}: {label!r}")
    return flagged


def evaluate(clf: TaskClassifier, rows: list[dict[str, str]]) -> dict:
    tax = clf.taxonomy
    preds = []
    for r in rows:
        c = clf.classify(r["prompt"])
        gold = r["category"]
        parent_ok = c.category == gold or (
            tax.categories.get(gold) and tax.categories[gold].parent == c.category
        )
        preds.append(
            {
                "id": r["id"],
                "gold": gold,
                "pred": c.category,
                "conf": c.confidence,
                "fallback": c.fallback,
                "correct": c.category == gold,
                "parent_ok": bool(parent_ok),
            }
        )
    n = len(preds)
    covered = [p for p in preds if p["pred"] != GENERAL]
    return {
        "n": n,
        "accuracy": sum(p["correct"] for p in preds) / n,
        "parent_accepted_accuracy": sum(p["parent_ok"] for p in preds) / n,
        "coverage": len(covered) / n,
        "selective_accuracy": (sum(p["correct"] for p in covered) / len(covered)) if covered else None,
        "preds": preds,
    }


def with_thresholds(raw: dict, p: float, m: float, k: float) -> TaskClassifier:
    d = copy.deepcopy(raw)
    d["thresholds"] = {"min_probability": p, "min_margin": m, "unexplained_mass": k}
    return TaskClassifier(Taxonomy(d))


def tune(raw: dict, dev: list[dict[str, str]]) -> tuple[float, float, float, float]:
    grid_p = [round(0.20 + 0.05 * i, 2) for i in range(11)]  # 0.20 .. 0.70
    grid_m = [round(0.05 * i, 2) for i in range(7)]  # 0.00 .. 0.30
    grid_k = [0.5, 1.0, 1.5, 2.0, 3.0]
    best: tuple[float, float, float, float] | None = None
    for k, p, m in itertools.product(grid_k, grid_p, grid_m):
        acc = evaluate(with_thresholds(raw, p, m, k), dev)["accuracy"]
        # ties go to the more conservative setting (higher thresholds)
        if best is None or (acc, p, m) > (best[0], best[1], best[2]):
            best = (acc, p, m, k)
    assert best is not None
    return best


def per_category(preds: list[dict]) -> list[tuple[str, int, float, float]]:
    cats = sorted({p["gold"] for p in preds})
    out = []
    for c in cats:
        tp = sum(1 for p in preds if p["pred"] == c and p["gold"] == c)
        fp = sum(1 for p in preds if p["pred"] == c and p["gold"] != c)
        fn = sum(1 for p in preds if p["pred"] != c and p["gold"] == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append((c, tp + fn, prec, rec))
    return out


def calibration(preds: list[dict]) -> list[tuple[str, int, float]]:
    bins: dict[str, list[bool]] = defaultdict(list)
    for p in preds:
        lo = min(int(p["conf"] * 5), 4) / 5
        bins[f"{lo:.1f}-{lo + 0.2:.1f}"].append(p["correct"])
    return [(b, len(v), sum(v) / len(v)) for b, v in sorted(bins.items())]


def report(raw: dict, dev: list[dict[str, str]], test: list[dict[str, str]]) -> str:
    clf = TaskClassifier(Taxonomy(raw))
    d, t = evaluate(clf, dev), evaluate(clf, test)
    th = raw["thresholds"]
    lines = [
        "# Phase 5: task classifier evaluation",
        "",
        f"Taxonomy version {raw['version']}; thresholds frozen from dev tuning: min_probability "
        f"{th['min_probability']}, min_margin {th['min_margin']}, unexplained_mass {th['unexplained_mass']}.",
        "",
        "Data: `data/task_classification/prompts.jsonl` (author-written; see its README for the limitation that "
        "prompts, labels and lexicons share one author, so these numbers are likely optimistic).",
        "",
        "| split | n | accuracy | parent-accepted accuracy | coverage (not `general`) | accuracy when not `general` |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in (("dev (used for tuning)", d), ("**test (held out)**", t)):
        sel = "-" if r["selective_accuracy"] is None else f"{r['selective_accuracy']:.3f}"
        lines.append(
            f"| {name} | {r['n']} | {r['accuracy']:.3f} | {r['parent_accepted_accuracy']:.3f} | "
            f"{r['coverage']:.3f} | {sel} |"
        )
    lines += ["", "## Test: per category", "", "| category | n | precision | recall |", "|---|---|---|---|"]
    lines += [f"| {c} | {n} | {p:.2f} | {r:.2f} |" for c, n, p, r in per_category(t["preds"])]
    wrong = [p for p in t["preds"] if not p["correct"]]
    to_general = sum(1 for p in wrong if p["pred"] == GENERAL)
    lines += [
        "",
        f"Of the {len(wrong)} test errors, {to_general} fell back to `general` (conservative: only identity and "
        f"general-safe strategies are then allowed) and {len(wrong) - to_general} assigned a different specific "
        "category. A `general` prediction with no matching evidence has confidence 1.0 by construction "
        "(all mass is unexplained), which is why that bin looks over-confident.",
    ]
    errors = Counter((p["gold"], p["pred"]) for p in t["preds"] if not p["correct"])
    lines += [
        "",
        "## Test: most frequent confusions (gold → predicted)",
        "",
        "| gold | predicted | count |",
        "|---|---|---|",
    ]
    lines += [f"| {g} | {pr} | {n} |" for (g, pr), n in errors.most_common(12)]
    lines += [
        "",
        "## Test: calibration (confidence bin → accuracy)",
        "",
        "| confidence | n | accuracy |",
        "|---|---|---|",
    ]
    lines += [f"| {b} | {n} | {a:.2f} |" for b, n, a in calibration(t["preds"])]
    lines += ["", "## Test: misclassified items", ""]
    lines += [
        f"- `{p['id']}`: gold `{p['gold']}`, predicted `{p['pred']}` (confidence {p['conf']:.2f})"
        for p in t["preds"]
        if not p["correct"]
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-leakage", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    raw = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    dev, test = load()
    if a.check_leakage:
        flagged = check_leakage(raw, dev, test)
        print("\n".join(flagged) if flagged else "no lexicon entry matches only test prompts")
        return 1 if flagged else 0
    if a.tune:
        acc, p, m, k = tune(raw, dev)
        print(
            json.dumps(
                {"dev_accuracy": round(acc, 4), "min_probability": p, "min_margin": m, "unexplained_mass": k}
            )
        )
        return 0
    if a.report:
        text = report(raw, dev, test)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(text, encoding="utf-8")
        print(text)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
