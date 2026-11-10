"""Measure the LLM fallback classifier on data/task_classification/prompts.jsonl (dev and test splits).

    python scripts/eval_llm_classifier.py groq openai/gpt-oss-20b

Only prompts the deterministic classifier labels ``general`` are sent to the model (free tiers or local models
recommended). Reports accuracy and coverage with and without the fallback, per split.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_harness import BudgetConfig, HarnessConfig, HTTPConfig  # noqa: E402
from adaptive_harness.auxiliary import AuxModel  # noqa: E402
from adaptive_harness.budget import BudgetLedger  # noqa: E402
from adaptive_harness.providers import create_adapter  # noqa: E402
from adaptive_harness.store import Store  # noqa: E402
from adaptive_harness.tasks import GENERAL, TaskClassifier  # noqa: E402
from adaptive_harness.tasks.llm_classifier import LlmFallbackClassifier  # noqa: E402

DATA = ROOT / "data" / "task_classification" / "prompts.jsonl"


def split_of(item_id: str) -> str:
    return "dev" if int(item_id.rsplit("-", 1)[1]) <= 12 else "test"  # the fixed Phase 5 split rule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("provider")
    ap.add_argument("model")
    ap.add_argument(
        "--budget-usd", type=float, default=0.0, help="harness budget for paid providers (default 0)"
    )
    ap.add_argument("--max-cost-per-call-usd", type=float, default=0.001)
    ap.add_argument("--pause-s", type=float, default=0.0)
    args = ap.parse_args()
    cfg = HarnessConfig(budget=BudgetConfig(total_usd=args.budget_usd), http=HTTPConfig(timeout_s=300))
    store = Store(Path(tempfile.mkdtemp()) / "eval.db")
    ledger = BudgetLedger(store, total_usd=cfg.budget.total_usd)
    adapter = create_adapter(args.provider, http=cfg.http)
    if not adapter.capabilities().local and args.budget_usd == 0:
        ledger = BudgetLedger(
            store, total_usd=1.0
        )  # free-tier bookkeeping only: unknown costs commit the cap
        print("note: non-local provider with no budget given; assuming a free tier (bookkeeping budget only)")
    base = TaskClassifier()
    llm = LlmFallbackClassifier(
        AuxModel(
            adapter, args.model, ledger, max_cost_per_call_usd=args.max_cost_per_call_usd, max_tokens=1000
        ),
        base,
    )
    items = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats: dict[str, dict[str, int]] = {}
    for it in items:
        s = stats.setdefault(
            split_of(it["id"]), {"n": 0, "base_ok": 0, "llm_ok": 0, "base_cov": 0, "llm_cov": 0, "asked": 0}
        )
        b = base.classify(it["prompt"]).category
        if b == GENERAL:
            s["asked"] += 1
            if args.pause_s:
                time.sleep(args.pause_s)
        lab = llm.classify(it["prompt"]).category
        s["n"] += 1
        s["base_ok"] += b == it["category"]
        s["llm_ok"] += lab == it["category"]
        s["base_cov"] += b != GENERAL
        s["llm_cov"] += lab != GENERAL
    print(f"model {args.provider}/{args.model}; LLM calls {llm.model.calls}")
    for split in ("dev", "test"):
        s = stats[split]
        print(
            f"{split}: n={s['n']} asked={s['asked']} | accuracy {s['base_ok'] / s['n']:.3f} -> {s['llm_ok'] / s['n']:.3f}"
            f" | coverage {s['base_cov'] / s['n']:.3f} -> {s['llm_cov'] / s['n']:.3f}"
        )


if __name__ == "__main__":
    main()
