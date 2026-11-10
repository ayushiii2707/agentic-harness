"""Phase 11: background evaluation, response safety, streaming checks, async planning, automatic recalibration."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from adaptive_harness import (
    AdaptiveHarness,
    CalibrationConfig,
    HarnessConfig,
    PersistenceConfig,
    PolicyConfig,
)
from adaptive_harness.calibration import Calibrator
from adaptive_harness.providers import MockProvider
from adaptive_harness.providers.base import ProviderCapabilities
from adaptive_harness.strategies import BUILTIN, StrategyRegistry

PROMPT = "What is 17 times 23?"


class PaidMock(MockProvider):
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(local=False)


def make(tmp_path: Path, model: MockProvider | None = None, **cfg: object) -> AdaptiveHarness:
    config = HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db"), **cfg)  # type: ignore[arg-type]
    return AdaptiveHarness(
        provider="mock", model="m", adapter=model or MockProvider(lambda r: "391"), config=config
    )


def evidence_n(h: AdaptiveHarness) -> int:
    assert h.store is not None
    row = h.store.query_one("SELECT COALESCE(SUM(n_trials), 0) AS n FROM arm_stats")
    return int(row["n"]) if row else 0


# ------------------------------------------------------------------ background evaluation
def test_background_check_returns_at_once_and_counts_later(tmp_path: Path) -> None:
    h = make(tmp_path)
    gate = threading.Event()

    def slow_check(text: str) -> bool:
        gate.wait(5)
        return text == "391"

    t = time.perf_counter()
    r = h.generate(PROMPT, check=slow_check, background=True)
    assert time.perf_counter() - t < 1.0 and r.trace.evaluation_status == "pending" and r.text == "391"
    assert evidence_n(h) == 0
    gate.set()
    assert h.flush(5)
    assert evidence_n(h) == 1
    assert h.store is not None
    row = h.store.query_one("SELECT status FROM evaluation WHERE trial_id=?", (r.id,))
    assert row is not None and row["status"] == "success"


def test_an_evaluation_failure_never_loses_the_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = make(tmp_path)
    assert h.evidence is not None

    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("store is on fire")

    monkeypatch.setattr(h.evidence, "record", boom)
    r = h.generate(PROMPT, check=lambda t: True)
    assert r.text == "391"
    monkeypatch.setattr(h, "_record", boom)
    assert h.generate(PROMPT, check=lambda t: True).text == "391"
    assert h.generate(PROMPT, check=lambda t: 1 / 0).text == "391"  # type: ignore[arg-type,return-value]


# ------------------------------------------------------------------ streaming
def test_stream_records_and_evaluates_after_the_last_event(tmp_path: Path) -> None:
    h = make(tmp_path)
    events = list(h.stream(PROMPT, check=lambda t: t == "391"))
    assert "".join(e.delta for e in events) == "391" and events[-1].result is not None
    assert evidence_n(h) == 1
    list(h.stream(PROMPT))  # without a check: recorded, never counted
    assert evidence_n(h) == 1
    assert h.store is not None and len(h.store.query("SELECT id FROM trial")) == 2


# ------------------------------------------------------------------ async
def test_async_planning_and_checks_run_off_the_event_loop(tmp_path: Path) -> None:
    h = make(tmp_path)
    loop_thread: list[int] = []
    check_thread: list[int] = []

    def check(text: str) -> bool:
        check_thread.append(threading.get_ident())
        return True

    async def main() -> str:
        loop_thread.append(threading.get_ident())
        r = await h.agenerate(PROMPT, check=check)
        return r.trace.evaluation_status

    assert asyncio.run(main()) == "success"
    assert check_thread and check_thread[0] != loop_thread[0]


# ------------------------------------------------------------------ privacy of evaluation details
def test_evaluation_details_follow_prompt_storage(tmp_path: Path) -> None:
    from adaptive_harness.evaluation import EvalStatus, EvaluationResult

    class Leaky:
        evaluator_id, version = "leaky", "1"

        def evaluate(self, text: str) -> EvaluationResult:
            return EvaluationResult(
                EvalStatus.FAILURE, "leaky", "1", details=f"failure: AssertionError: got {text!r} | tail"
            )

    h = make(tmp_path, MockProvider(lambda r: "ACME-PRIVATE answer"))
    h.generate(PROMPT, check=Leaky())  # type: ignore[arg-type]
    assert h.store is not None
    stored = json.loads(h.store.query_one("SELECT details_json FROM evaluation")["details_json"])  # type: ignore[index]
    assert stored == {"details": "failure"}


# ------------------------------------------------------------------ automatic recalibration
def test_auto_calibration_runs_for_a_local_model_and_never_twice_at_once(tmp_path: Path) -> None:
    h = make(
        tmp_path,
        calibration=CalibrationConfig(auto=True, max_calls=6, categories=("math",), min_interval_s=3600),
    )
    h.generate(PROMPT)  # first call: cold profile -> a background calibration starts
    assert h.flush(60)
    report = h.orchestrator.last_report
    assert report is not None and report.calls > 0 and h.orchestrator.last_error is None
    _, status = h._current_profile()
    assert status == "active"
    assert not h.orchestrator.maybe_recalibrate("stale")  # within min_interval: no second start


def test_auto_calibration_never_spends_on_a_paid_provider_without_permission(tmp_path: Path) -> None:
    h = make(tmp_path, PaidMock(lambda r: "391"), calibration=CalibrationConfig(auto=True, min_interval_s=0))
    assert h.orchestrator.allowed() == (False, "paid provider: set calibration.max_cost_per_call_usd > 0")
    h.generate(PROMPT)
    assert h.flush(5) and h.orchestrator.last_report is None
    h2 = make(
        tmp_path / "b",
        PaidMock(lambda r: "391"),
        calibration=CalibrationConfig(auto=True, max_cost_per_call_usd=0.01, min_interval_s=0),
    )
    assert h2.orchestrator.allowed() == (
        False,
        "paid provider: harness budget exhausted",
    )  # default budget $0


def test_auto_calibration_off_by_default(tmp_path: Path) -> None:
    h = make(tmp_path)
    h.generate(PROMPT)
    assert h.orchestrator.allowed()[0] is False and h.orchestrator.last_report is None


def test_a_stale_profile_is_recalibrated_into_a_new_version_automatically(tmp_path: Path) -> None:
    h = make(
        tmp_path,
        calibration=CalibrationConfig(auto=True, max_calls=4, categories=("math",), min_interval_s=0),
    )
    h.generate(PROMPT)
    assert h.flush(60)
    first, _ = h._current_profile()
    assert h.store is not None and first is not None
    h.store.execute("UPDATE profile SET status='stale' WHERE id=?", (first,))
    h.generate(PROMPT)
    assert h.flush(60)
    second, status = h._current_profile()
    assert second != first and status == "active"


def test_identity_only_cells_cost_nothing(tmp_path: Path) -> None:
    h = AdaptiveHarness(
        provider="mock",
        model="m",
        adapter=MockProvider(lambda r: "1"),
        strategies=StrategyRegistry((BUILTIN[0],)),
        config=HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db"), policy=PolicyConfig()),
    )
    report = Calibrator(h, max_calls=100, max_cost_per_call_usd=0.0).run(categories=("math",))
    assert report.calls == 0 and all(c.stopped == "nothing to compare" for c in report.cells)
