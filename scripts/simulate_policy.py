"""Simulation of the decision loop (results are simulated, never model evidence).

A mock model succeeds with probability p_role when role_framing's system line is sent and p_base otherwise.
Each run makes N checked calls with exploration, then a fresh client (no exploration, no check) on the same store
reports which strategy it would now use.

    python scripts/simulate_policy.py --runs 40 --calls 400
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_harness import AdaptiveHarness, HarnessConfig, PersistenceConfig, PolicyConfig  # noqa: E402
from adaptive_harness.providers import MockProvider  # noqa: E402
from adaptive_harness.types import CompletionRequest  # noqa: E402

PROMPT = "Write a Python function that checks whether a number is prime."
ROLE_LINE = "You are an experienced software engineer."


def run(p_role: float, p_base: float, calls: int, seed: int, exploration: float) -> str:
    rng = random.Random(seed)

    def respond(req: CompletionRequest) -> str:
        has_role = any(m.role == "system" and m.content == ROLE_LINE for m in req.messages)
        return "ok" if rng.random() < (p_role if has_role else p_base) else "no"

    model = MockProvider(respond)
    path = Path(tempfile.mkdtemp()) / "h.db"

    def make(rate: float, pseed: int) -> AdaptiveHarness:
        cfg = HarnessConfig(
            persistence=PersistenceConfig(path=path), policy=PolicyConfig(exploration_rate=rate)
        )
        return AdaptiveHarness(provider="mock", model="sim", adapter=model, config=cfg, policy_seed=pseed)

    h = make(exploration, seed)
    for _ in range(calls):
        h.generate(PROMPT, check=lambda t: t == "ok")
    h.close()
    final = make(0.0, seed + 10_000).generate(PROMPT).trace.strategy_id
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--calls", type=int, default=400)
    ap.add_argument("--exploration", type=float, default=0.5)
    a = ap.parse_args()
    null = [run(0.5, 0.5, a.calls, s, a.exploration) for s in range(a.runs)]
    alt = [run(0.9, 0.3, a.calls, s, a.exploration) for s in range(a.runs)]
    fs = sum(x != "identity" for x in null)
    hit = sum(x == "role_framing" for x in alt)
    print(f"calls per run {a.calls}, exploration {a.exploration}, runs {a.runs} (simulated)")
    print(f"no strategy better:         switched away from identity in {fs}/{a.runs}")
    print(
        f"role_framing truly better:  chose role_framing in {hit}/{a.runs}; other picks: "
        f"{sorted(set(alt) - {'role_framing'})}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
