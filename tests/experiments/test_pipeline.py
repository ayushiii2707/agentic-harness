"""The Phase 12 pipeline end to end on a mock model (simulated; never reported as model evidence)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from adaptive_harness.evaluation.builtin import JsonFields, ReferenceAnswer
from adaptive_harness.providers import MockProvider
from adaptive_harness.types import CompletionRequest
from experiments import runner
from experiments.analysis import report


def config(tmp_path: Path) -> dict[str, Any]:
    cfg = json.loads((Path(__file__).resolve().parents[2] / "experiments" / "config_v1.json").read_text())
    cfg.update(
        models=[{"provider": "mock", "model": "sim", "max_tokens": 50, "pause_s": 0}],
        calibration={"math": 32, "structured_output": 32},
        heldout={"math": 10, "structured_output": 10},
        unseen_categories=[],
        calibration_arms=["identity", "role_framing", "structured_sections"],
        bootstrap={"resamples": 500, "seed": 12},
    )
    return cfg


def oracle(cfg: dict[str, Any]) -> dict[str, str]:
    answers = {}
    for s in ("math", "structured_output"):
        for split, n in (("calibration", 32), ("heldout", 10)):
            for t in runner.suite_tasks(s, split, n):
                ev = t.evaluator
                if isinstance(ev, ReferenceAnswer):
                    answers[t.prompt] = f"The answer is {ev.expected}"
                elif isinstance(ev, JsonFields):
                    answers[t.prompt] = json.dumps(dict(ev.expected))
    return answers


def model(answers: dict[str, str]) -> MockProvider:
    rng = random.Random(5)

    def respond(r: CompletionRequest) -> str:
        user = r.messages[-1].content
        prompt = next((p for p in answers if p in user), None)
        role = any(m.role == "system" for m in r.messages)  # only role_framing adds a system line here
        ok = prompt is not None and rng.random() < (0.95 if role else 0.45)
        return answers[prompt] if ok and prompt else "I am not sure."

    return MockProvider(respond)


def run_all(tmp_path: Path, cfg: dict[str, Any], mock: MockProvider) -> tuple[Path, dict[str, Any]]:
    results = tmp_path / "results"
    key = "mock:sim"
    h = runner.make_harness(cfg, cfg["models"][0], tmp_path / "sim.db", adapter=mock)
    c = runner.Collector(h, cfg, cfg["models"][0], cfg["experiment"], log=lambda s: None)
    runner.calibrate(c)
    sel = runner.select({key: h}, cfg, cfg["experiment"])
    results.mkdir()
    (results / "selections.json").write_text(json.dumps(sel))
    runner.evaluate(c, sel, key)
    runner.export(h, key, cfg["experiment"], results / "sim.csv")
    return results, sel


def test_pipeline_runs_and_the_report_is_marked_simulated(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    mock = model(oracle(cfg))
    results, sel = run_all(tmp_path, cfg, mock)
    m = sel["models"]["mock:sim"]
    assert m["C"]["strategy"] == "role_framing" and sel["B+"]["strategy"] == "role_framing"
    assert {v["strategy"] for v in m["D"].values()} == {"role_framing"}
    text = report.build(results, cfg)
    assert "**SIMULATED" in text
    assert "| D - A |" in text and "improves" in text  # 0.95 vs 0.45 must be detected
    calls = len(mock.requests)

    # resuming every phase makes no new calls and changes nothing
    h = runner.make_harness(cfg, cfg["models"][0], tmp_path / "sim.db", adapter=mock)
    c = runner.Collector(h, cfg, cfg["models"][0], cfg["experiment"], log=lambda s: None)
    runner.calibrate(c)
    runner.evaluate(c, sel, "mock:sim")
    assert len(mock.requests) == calls


def test_identical_conditions_share_trials_and_pairs_are_complete(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    results, sel = run_all(tmp_path, cfg, model(oracle(cfg)))
    rows = report.load_rows(results)
    held = [r for r in rows if r["phase"] == "heldout"]
    # A=identity, B=structured_sections, B+=C=D=role_framing -> 3 distinct strategies x 20 tasks x 2 reps
    assert len(held) == 3 * 20 * 2
    table = report.condition_table(rows, sel, "mock:sim")
    assert all(v["C"] == v["D"] for v in table.values())  # shared trial, identical outcome


def test_leakage_and_frozen_selections_are_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config(tmp_path)
    runner.check_no_leakage(cfg)
    monkeypatch.setattr(runner, "suite_tasks", lambda s, split, n: runner.math_suite(split="calibration")[:n])
    with pytest.raises(SystemExit, match="leakage"):
        runner.check_no_leakage(cfg)


def test_no_real_effect_is_not_reported_as_an_effect(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    answers = oracle(cfg)
    rng = random.Random(9)

    def respond(r: CompletionRequest) -> str:  # success does not depend on the strategy at all
        prompt = next((p for p in answers if p in r.messages[-1].content), None)
        return answers[prompt] if prompt and rng.random() < 0.6 else "no idea"

    results, _ = run_all(tmp_path, cfg, MockProvider(respond))
    text = report.build(results, cfg)
    primary = text.split("### Primary family")[1].split("###")[0]
    assert "improves" not in primary and "harms" not in primary


def test_pooled_selection_compares_arms_on_the_same_tasks() -> None:
    """Regression (found before any held-out call): an arm that applies only to easy tasks, and falls back to
    the original prompt elsewhere, looked better pooled because its fallbacks were credited to identity."""
    rows = []
    for i in range(40):  # easy tasks: both arms succeed
        rows += [{"task_ref": f"cal:easy{i}:identity", "score": 1.0, "tokens": 10},
                 {"task_ref": f"cal:easy{i}:narrow", "score": 1.0, "tokens": 10}]  # fmt: skip
    for i in range(40):  # hard tasks: the narrow arm cannot apply, the original prompt is sent and fails
        rows += [{"task_ref": f"cal:hard{i}:identity", "score": 0.0, "tokens": 10},
                 {"task_ref": f"cal:hard{i}:narrow", "score": 0.0, "tokens": 10}]  # fmt: skip
    for r in rows:
        r["strategy_id"] = r["task_ref"].rsplit(":", 1)[1]
    a = runner._arm_stats(rows, "identity")
    b = runner._arm_stats(rows, "narrow")
    assert (a.n_trials, b.n_trials) == (80, 80) and a.mean == b.mean  # same tasks, same outcomes
