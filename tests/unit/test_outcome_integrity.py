"""Evaluation findings F4 and F5: one counted outcome per response, credited to the profile that served it."""

from __future__ import annotations

import sqlite3
import threading
from importlib import resources
from pathlib import Path

from adaptive_harness import AdaptiveHarness, HarnessConfig, PersistenceConfig
from adaptive_harness.calibration import Calibrator
from adaptive_harness.providers import MockProvider
from adaptive_harness.store import Store

PROMPT = "What is 17 times 23?"


def make(path: Path) -> AdaptiveHarness:
    return AdaptiveHarness(
        provider="mock", model="m", adapter=MockProvider(lambda r: "391"),
        config=HarnessConfig(persistence=PersistenceConfig(path=path)),
    )  # fmt: skip


def outcomes(h: AdaptiveHarness) -> int:
    assert h.store is not None
    row = h.store.query_one("SELECT COALESCE(SUM(n_trials), 0) AS n FROM arm_stats")
    return int(row["n"]) if row else 0


def _race(path: Path) -> AdaptiveHarness:
    """A background check and two feedback calls for the same response, all at once."""
    h = make(path)
    gate = threading.Event()
    r = h.generate(PROMPT, check=lambda t: (gate.wait(2), True)[1], background=True)
    barrier = threading.Barrier(2)

    def report() -> None:
        barrier.wait()
        h.feedback(r.id, success=True)

    threads = [threading.Thread(target=report) for _ in range(2)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join()
    assert h.flush(5)
    assert h.store is not None
    rows = h.store.query("SELECT counted FROM evaluation WHERE trial_id=?", (r.id,))
    assert sorted(x["counted"] for x in rows) == [0, 0, 1]  # every report is kept; exactly one counts
    return h


def test_concurrent_reports_for_one_response_count_once(tmp_path: Path) -> None:
    for i in range(30):
        assert outcomes(_race(tmp_path / f"{i}.db")) == 1, f"run {i}"


def test_feedback_is_credited_to_the_profile_that_served_the_response(tmp_path: Path) -> None:
    h = make(tmp_path / "h.db")
    r = h.generate(PROMPT)
    served, _ = h._current_profile()
    assert h.store is not None and h.models is not None and served is not None
    ref = h.store.query_one("SELECT model_ref FROM profile WHERE id=?", (served,))
    assert ref is not None
    newer = h.models.next_profile(int(ref["model_ref"]), status="cold")
    Calibrator(h, max_calls=2, max_cost_per_call_usd=0.0).run(categories=("math",))
    before = h.store.query_one(
        "SELECT COALESCE(SUM(n_trials),0) AS n FROM arm_stats WHERE profile_id=?", (newer,)
    )
    h.feedback(r.id, success=True)
    after = h.store.query_one(
        "SELECT COALESCE(SUM(n_trials),0) AS n FROM arm_stats WHERE profile_id=?", (newer,)
    )
    old = h.store.query_one(
        "SELECT COALESCE(SUM(n_trials),0) AS n FROM arm_stats WHERE profile_id=?", (served,)
    )
    assert before is not None and after is not None and old is not None
    assert after["n"] == before["n"] and old["n"] == 1


def test_migration_0003_backfills_a_database_that_already_double_counted(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, name TEXT, applied_at REAL)")
    mig = resources.files("adaptive_harness.store") / "migrations"
    for n, name in ((1, "0001_init.sql"), (2, "0002_classification_cache.sql")):
        con.executescript((mig / name).read_text(encoding="utf-8"))
        con.execute("INSERT INTO schema_version VALUES (?,?,0)", (n, name))
    con.execute(
        "INSERT INTO trial(id, provider, model_requested, strategy_id, strategy_version, prompt_hash, status, "
        "created_at) VALUES ('t1','mock','m','identity','1','h','done',0)"
    )
    for status, score in (("success", 1.0), ("failure", 0.0), ("evaluation_error", None)):
        con.execute(
            "INSERT INTO evaluation(trial_id, evaluator_id, evaluator_version, status, score, created_at) "
            "VALUES ('t1','x','1',?,?,0)",
            (status, score),
        )
    con.commit()
    con.close()
    store = Store(path)  # applies 0003
    rows = store.query("SELECT status, counted FROM evaluation ORDER BY id")
    assert [(r["status"], r["counted"]) for r in rows] == [
        ("success", 1),
        ("failure", 0),
        ("evaluation_error", 0),
    ]
    assert store.query_one("SELECT profile_id FROM trial WHERE id='t1'") == {"profile_id": None}
