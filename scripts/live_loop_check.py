"""Live check that the decision loop runs end to end against a real model (free providers by default).

It verifies the mechanism (strategies applied, real calls, real checks, evidence written). With a handful of calls
it cannot and does not claim that any strategy is better.

    python scripts/live_loop_check.py groq openai/gpt-oss-20b --calls 16
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_harness import AdaptiveHarness, HarnessConfig, PersistenceConfig, PolicyConfig  # noqa: E402

PROBLEMS = [
    ("A shop sells pens at 3 for $2. How many dollars do 24 pens cost?", 16),
    ("What is 17 multiplied by 23?", 391),
    ("A train travels 300 km in 2.5 hours. What is its average speed in km/h?", 120),
    ("Solve for x: 3x + 7 = 22.", 5),
    ("What is the sum of the first 20 positive integers?", 210),
    ("If 5 workers build a wall in 12 days, how many days do 10 workers need at the same rate?", 6),
    ("How many minutes are there in 7.5 hours?", 450),
    ("What is 15% of 840?", 126),
]


def last_number(text: str) -> float | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("provider")
    ap.add_argument("model")
    ap.add_argument("--calls", type=int, default=16)
    a = ap.parse_args()
    db = Path(tempfile.mkdtemp()) / "live.db"
    cfg = HarnessConfig(persistence=PersistenceConfig(path=db), policy=PolicyConfig(exploration_rate=1.0))
    h = AdaptiveHarness(provider=a.provider, model=a.model, config=cfg, policy_seed=1)
    used: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for i in range(a.calls):
        question, answer = PROBLEMS[i % len(PROBLEMS)]
        r = h.generate(
            question + " End your answer with the number only.",
            max_tokens=1024,
            check=lambda t, ans=answer: last_number(t) == ans,
        )
        used[r.trace.strategy_id] += 1
        statuses[r.trace.evaluation_status] += 1
        print(
            f"{i + 1:2d} {r.trace.category:18s} {r.trace.strategy_id:22s} {r.trace.evaluation_status:8s} "
            f"optimized={r.trace.optimized}"
        )
        time.sleep(1.0)  # stay well under free-tier rate limits
    prof = h.profile() or {}
    print("strategies used:", dict(used))
    print("evaluation statuses:", dict(statuses))
    print("evidence rows written:")
    for e in prof.get("evidence", []):
        print(
            f"  {e['category_id']:14s} {e['strategy_id']:22s} n={e['n_trials']} "
            f"successes={e['alpha'] - 1:.0f}"
        )
    assert h.store is not None
    n_trials = h.store.query_one("SELECT COUNT(*) AS n FROM trial")["n"]  # type: ignore[index]
    n_eval = h.store.query_one("SELECT COUNT(*) AS n FROM evaluation")["n"]  # type: ignore[index]
    print(f"trials stored: {n_trials}, evaluations stored: {n_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
