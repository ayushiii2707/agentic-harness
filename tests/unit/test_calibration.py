"""Calibration with a simulated model (results marked simulated). Real-model runs live in scripts/."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from adaptive_harness import (
    AdaptiveHarness,
    BudgetConfig,
    ConfigurationError,
    HarnessConfig,
    PersistenceConfig,
    ProviderUnavailableError,
)
from adaptive_harness.calibration import Calibrator, suite
from adaptive_harness.providers import MockProvider
from adaptive_harness.types import CompletionRequest

ROLE_MATH = "You are a careful mathematician."  # role_framing's line for math


def answer_key() -> dict[str, float]:
    return {t.prompt: float(t.evaluator.expected) for t in suite("math")}  # type: ignore[attr-defined]


def math_model(
    p_role: float, p_other: float, seed: int = 0, fail_after: int | None = None, price: float = 0.0
) -> MockProvider:
    """Answers a suite question correctly with a probability that depends on role_framing's line."""
    key = answer_key()
    rng = random.Random(seed)
    calls = {"n": 0}

    def respond(req: CompletionRequest) -> str:
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            raise ProviderUnavailableError("simulated outage", provider="mock")
        text = req.messages[-1].content
        ans = next((v for p, v in key.items() if p in text), None)
        if ans is None:
            return "I am not sure."  # not a suite question
        has_role = any(m.role == "system" and m.content == ROLE_MATH for m in req.messages)
        right = rng.random() < (p_role if has_role else p_other)
        return f"... so the answer is {ans if right else ans + 1}"

    return MockProvider(respond, price_per_token_usd=price)


def harness(tmp_path: Path, model: MockProvider, budget: float = 0.0) -> AdaptiveHarness:
    cfg = HarnessConfig(
        persistence=PersistenceConfig(path=tmp_path / "h.db"), budget=BudgetConfig(total_usd=budget)
    )
    return AdaptiveHarness(provider="mock", model="sim", adapter=model, config=cfg, policy_seed=0)


def test_calibration_finds_the_better_strategy_and_the_request_path_uses_it(tmp_path: Path) -> None:
    h = harness(tmp_path, math_model(p_role=0.9, p_other=0.4))
    report = Calibrator(h, max_calls=5000, max_cost_per_call_usd=0.0).run(categories=("math",))
    cells = {c.category: c for c in report.cells}
    assert set(cells) == {
        "math"
    }  # every suite prompt is labelled math (the word-problem rule, DECISIONS D41)
    math_arms = {a.strategy_id: a for a in cells["math"].arms}
    assert math_arms["role_framing"].mean > math_arms["identity"].mean + 0.25
    assert not math_arms["role_framing"].eliminated and not math_arms["identity"].eliminated
    assert any(a.eliminated for a in cells["math"].arms)  # clearly worse arms were dropped early
    assert cells["math"].usable
    assert (h.profile() or {})["status"] == "active"

    # The loop, checked on every held-out prompt (never seen in calibration), on the normal request path:
    # prompts labelled math must use what calibration found; in the general cell role_framing does not apply
    # and nothing beats identity, so identity must be kept.
    for task in suite("math", split="heldout"):
        r = h.generate(task.prompt)
        expected = {"math": "role_framing", "general": "identity"}[
            r.trace.category or ""
        ]  # general: none expected
        assert r.trace.strategy_id == expected, (r.trace.category, r.trace.reason)


def test_budget_refusal_stops_the_run_without_overspending(tmp_path: Path) -> None:
    # each simulated call reports a real cost of roughly $0.005-0.01; each reserves $0.01 first
    h = harness(tmp_path, math_model(0.5, 0.5, price=0.0001), budget=0.025)
    report = Calibrator(h, max_calls=1000, max_cost_per_call_usd=0.01).run(categories=("math",))
    assert report.calls >= 1 and any("budget" in c.stopped for c in report.cells)
    assert h.ledger is not None and h.ledger.committed_and_reserved_usd() <= 0.025 + 1e-12
    assert report.spend_usd == pytest.approx(h.ledger.spent_usd())  # committed at the reported cost


def test_max_calls_then_resume_without_repeating_work(tmp_path: Path) -> None:
    model = math_model(0.5, 0.5)
    h = harness(tmp_path, model)
    first = Calibrator(h, max_calls=10, max_cost_per_call_usd=0.0).run(("math",), experiment_id="exp1")
    assert first.calls == 10 and any("max_calls" in c.stopped for c in first.cells)
    second = Calibrator(h, max_calls=1000, max_cost_per_call_usd=0.0).run(("math",), experiment_id="exp1")
    assert h.store is not None
    dupes = h.store.query(
        "SELECT task_ref, strategy_id, COUNT(*) AS n FROM trial WHERE experiment_id='exp1' "
        "GROUP BY task_ref, strategy_id HAVING n > 1"
    )
    assert dupes == [] and second.calls > 0


def test_provider_outage_stops_cleanly_and_resume_finishes(tmp_path: Path) -> None:
    h = harness(tmp_path, math_model(0.5, 0.5, fail_after=7))
    r1 = Calibrator(h, max_calls=1000, max_cost_per_call_usd=0.0).run(("math",), experiment_id="exp2")
    assert any("provider error" in c.stopped for c in r1.cells) and r1.calls == 7
    h.adapter = math_model(0.5, 0.5)  # the provider is back
    r2 = Calibrator(h, max_calls=1000, max_cost_per_call_usd=0.0).run(("math",), experiment_id="exp2")
    assert all(
        c.stopped in ("suite exhausted", "only identity remains", "nothing to compare") for c in r2.cells
    )


def test_calibration_requires_persistence() -> None:
    h = AdaptiveHarness(
        provider="mock", model="sim", config=HarnessConfig(persistence=PersistenceConfig(enabled=False))
    )
    with pytest.raises(ConfigurationError):
        Calibrator(h, max_calls=1, max_cost_per_call_usd=0.0)


def test_report_markdown_lists_every_arm(tmp_path: Path) -> None:
    h = harness(tmp_path, math_model(0.6, 0.6))
    report = Calibrator(h, max_calls=40, max_cost_per_call_usd=0.0).run(("math",))
    md = report.to_markdown()
    for a in [a for c in report.cells for a in c.arms]:
        assert a.strategy_id in md
    assert "could reliably detect" in md


def test_a_paid_provider_cannot_be_calibrated_with_a_zero_reservation(tmp_path: Path) -> None:
    """Regression (evaluation F1): a $0 reservation let a provider that reports no cost spend past the cap."""
    from adaptive_harness.providers.base import ProviderCapabilities

    class PaidNoCost(MockProvider):
        def capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(local=False)

        def complete(self, request):  # type: ignore[no-untyped-def]
            from dataclasses import replace

            return replace(super().complete(request), cost_usd=None)

    model = PaidNoCost(lambda r: "1")
    h = harness(tmp_path, model, budget=0.005)
    with pytest.raises(ConfigurationError):
        Calibrator(h, max_calls=50, max_cost_per_call_usd=0.0)
    report = Calibrator(h, max_calls=50, max_cost_per_call_usd=0.001).run(("math",))
    assert report.calls == len(model.requests) == 5  # the cap stops it: 5 x $0.001 = $0.005
    assert h.ledger is not None and h.ledger.spent_usd() == pytest.approx(0.005)
