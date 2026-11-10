"""Loop audit (simulated): does evidence learned in calibration reach new prompts on the normal request path?

A simulated model answers suite questions correctly with p=0.9 when role_framing's line is sent, else p=0.4.
After calibrating each category on its calibration split, the held-out split goes through ``generate()``
(no check, no exploration), and we count how often the request path (a) labelled the prompt with the right
category and (b) used the calibrated strategy.
"""

from __future__ import annotations

import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_harness import AdaptiveHarness, HarnessConfig, PersistenceConfig  # noqa: E402
from adaptive_harness.calibration import Calibrator, suite  # noqa: E402
from adaptive_harness.providers import MockProvider  # noqa: E402
from adaptive_harness.strategies.builtin import RoleFraming  # noqa: E402
from adaptive_harness.types import CompletionRequest  # noqa: E402

CATEGORIES = ("math", "structured_output")  # code needs Docker per call; covered separately


def main() -> int:
    roles = set(RoleFraming.templates.values())
    rng = random.Random(0)

    def respond(req: CompletionRequest) -> str:
        has_role = any(m.role == "system" and m.content in roles for m in req.messages)
        return "RIGHT" if rng.random() < (0.9 if has_role else 0.4) else "WRONG"

    class Right:
        evaluator_id = "sim"

        def evaluate(self, text: str):  # type: ignore[no-untyped-def]
            from adaptive_harness.evaluation import EvalStatus, EvaluationResult

            return EvaluationResult(EvalStatus.SUCCESS if text == "RIGHT" else EvalStatus.FAILURE, "sim")

    # calibration with a simulated evaluator: swap each task's evaluator for the simulated one
    from dataclasses import replace

    import adaptive_harness.calibration.calibrator as cal_mod

    original_suite = cal_mod.suite
    cal_mod.suite = lambda c, split="calibration": [
        replace(t, evaluator=Right()) for t in original_suite(c, split)
    ]

    db = Path(tempfile.mkdtemp()) / "audit.db"
    h = AdaptiveHarness(
        provider="mock", model="sim", adapter=MockProvider(respond),
        config=HarnessConfig(persistence=PersistenceConfig(path=db)), policy_seed=0,
    )  # fmt: skip
    report = Calibrator(h, max_calls=5000, max_cost_per_call_usd=0.0).run(categories=CATEGORIES)
    best: dict[str, str] = {}
    for c in report.cells:
        print(f"suite {c.suite} -> cell {c.category}: {c.stopped}; usable={c.usable}; "
              f"{ {a.strategy_id: (a.n, round(a.mean, 2)) for a in c.arms} }")  # fmt: skip
        alive = [a for a in c.arms if not a.eliminated]
        best[c.category] = max(alive, key=lambda a: a.mean).strategy_id if c.usable and alive else "identity"
    print()
    for category in CATEGORIES:
        labels: Counter[str] = Counter()
        used: Counter[str] = Counter()
        matches = 0
        for t in suite(category, split="heldout"):
            r = h.generate(t.prompt)
            label = r.trace.category or "?"
            labels[label] += 1
            used[r.trace.strategy_id] += 1
            matches += r.trace.strategy_id == best.get(label, "identity")
        n = sum(labels.values())
        print(f"{category}: {n} held-out prompts; labels {dict(labels)}; strategies used {dict(used)}")
        print(f"  decision matched the calibrated best for its own cell: {matches}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
