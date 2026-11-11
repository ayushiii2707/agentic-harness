"""Experiment v2 pipeline (few-shot arm, pilot headroom rule, every arm on every held-out task) on a mock model.
Simulated; never reported as model evidence."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from adaptive_harness.evaluation.builtin import JsonFields, ReferenceAnswer
from adaptive_harness.providers import MockProvider
from adaptive_harness.types import CompletionRequest
from experiments import runner
from experiments.analysis import report


def config() -> dict[str, Any]:
    cfg = json.loads((Path(__file__).resolve().parents[2] / "experiments" / "config_v1.json").read_text())
    cfg.update(
        experiment="v2-test",
        models=[{"provider": "mock", "model": "sim", "max_tokens": 50, "pause_s": 0}],
        calibration={"math": 32, "structured_output": 32},
        heldout={"math": 12, "structured_output": 12},
        unseen_categories=[],
        calibration_arms=["identity", "few_shot", "structured_sections"],
        few_shot={"k": 3, "suites": ["math", "structured_output"]},
        heldout_all_arms=True,
        repetitions=1,
        pilot={"suites": ["math", "structured_output"], "n": 12, "min_successes": 2, "max_successes": 10},
        bootstrap={"resamples": 400, "seed": 12},
    )
    return cfg


def answers() -> dict[str, str]:
    out = {}
    for s in ("math", "structured_output"):
        for split in ("calibration", "heldout", "pilot"):
            for t in runner.suite_tasks(s, split, 40):
                ev = t.evaluator
                if isinstance(ev, ReferenceAnswer):
                    out[t.prompt] = f"{ev.expected}"
                elif isinstance(ev, JsonFields):
                    out[t.prompt] = json.dumps(dict(ev.expected))
    return out


def model(ans: dict[str, str], p_math: float) -> MockProvider:
    rng = random.Random(3)

    def respond(r: CompletionRequest) -> str:
        user = r.messages[-1].content
        shots = sum(m.role == "assistant" for m in r.messages)  # few-shot examples arrive as assistant turns
        is_math = "number only" in user
        p = (0.95 if shots else p_math) if is_math else 1.0  # structured output is at ceiling
        return ans.get(user, "?") if rng.random() < p else "unsure"

    return MockProvider(respond)


def test_v2_pipeline_detects_a_real_few_shot_gain_and_applies_the_pilot_rule(tmp_path: Path) -> None:
    cfg = config()
    mock = model(answers(), p_math=0.4)
    key = "mock:sim"
    h = runner.make_harness(cfg, cfg["models"][0], tmp_path / "sim.db", adapter=mock)
    c = runner.Collector(h, cfg, cfg["models"][0], cfg["experiment"], log=lambda s: None)
    pil = runner.pilot(c)
    assert pil["math"][2] is True  # 40% base rate: headroom
    assert pil["structured_output"][2] is False  # at ceiling: excluded by the rule
    runner.calibrate(c)
    sel = runner.select({key: h}, cfg, cfg["experiment"])
    results = tmp_path / "results"
    results.mkdir()
    (results / "selections.json").write_text(json.dumps(sel))
    runner.evaluate(c, sel, key)
    runner.export(h, key, cfg["experiment"], results / "sim.csv")
    rows = report.load_rows(results)
    held = [r for r in rows if r["phase"] == "heldout"]
    assert {r["strategy"] for r in held} == {"identity", "few_shot", "structured_sections"}
    assert len(held) == 24 * 3  # every arm on every held-out task, 1 repetition
    text = report.build(results, cfg)
    assert "**SIMULATED" in text
    fam = text.split("### Arm family: every strategy")[1].split("###")[0]
    few = next(x for x in fam.splitlines() if x.startswith("| few_shot |"))
    assert "improves" in few
    assert "improves" not in next(x for x in fam.splitlines() if x.startswith("| structured_sections |"))
    # the few-shot examples come from the disjoint examples split, never from calibration or held-out
    fs = h.strategies.get("few_shot")
    shown = {q for pairs in fs.examples.values() for q, _ in pairs}  # type: ignore[attr-defined]
    used = {
        t.prompt
        for s in ("math", "structured_output")
        for sp in ("calibration", "heldout")
        for t in runner.suite_tasks(s, sp, 40)
    }
    assert shown and not (shown & used)
