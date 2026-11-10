from __future__ import annotations

import threading
from pathlib import Path

import pytest

from adaptive_harness import BudgetExceededError, StoreError
from adaptive_harness.budget import BudgetLedger
from adaptive_harness.store import Store

EXPECTED_TABLES = {
    "model",
    "model_fact",
    "strategy",
    "task_category",
    "profile",
    "prior_evidence",
    "arm_stats",
    "experiment",
    "trial",
    "evaluation",
    "change_event",
    "budget_ledger",
    "schema_version",
}


def test_migrations_create_schema_and_are_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "h.db"
    s = Store(path)
    assert set(s.tables()) >= EXPECTED_TABLES
    v = s.schema_version()
    s.close()
    s2 = Store(path)  # re-open: nothing re-applied
    assert s2.schema_version() == v
    assert len(s2.query("SELECT * FROM schema_version")) == v


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "h.db"
    s = Store(path)
    s.execute("INSERT INTO schema_version(version, name, applied_at) VALUES (999, 'future', 0)")
    s.close()
    with pytest.raises(StoreError):
        Store(path)


def test_check_constraints_reject_bad_status() -> None:
    s = Store(":memory:")
    with pytest.raises(StoreError):
        s.execute(
            "INSERT INTO evaluation(trial_id, evaluator_id, evaluator_version, status, created_at) "
            "VALUES ('t','e','1','great',0)"
        )


def test_budget_reserve_commit_release() -> None:
    ledger = BudgetLedger(Store(":memory:"), total_usd=1.0)
    r1 = ledger.reserve(0.6)
    assert ledger.remaining_usd() == pytest.approx(0.4)
    with pytest.raises(BudgetExceededError) as exc:
        ledger.reserve(0.5)  # open reservation counts against the cap
    assert exc.value.remaining_usd == pytest.approx(0.4)
    ledger.commit(r1, 0.1)  # actual cost lower than the reservation
    assert ledger.spent_usd() == pytest.approx(0.1)
    assert ledger.remaining_usd() == pytest.approx(0.9)
    r2 = ledger.reserve(0.9)
    ledger.release(r2)
    assert ledger.remaining_usd() == pytest.approx(0.9)
    with pytest.raises(ValueError):
        ledger.commit(r2, 0.1)  # already closed


def test_zero_budget_refuses_any_paid_reservation() -> None:
    ledger = BudgetLedger(Store(":memory:"), total_usd=0.0)
    with pytest.raises(BudgetExceededError):
        ledger.reserve(0.0001)
    ledger.reserve(0.0)  # free operations are fine


def test_user_spend_is_recorded_but_never_capped() -> None:
    ledger = BudgetLedger(Store(":memory:"), total_usd=0.0)
    ledger.record_user_spend(5.0)
    assert ledger.spent_usd("user") == 5.0
    assert ledger.spent_usd("harness") == 0.0


def test_budget_persists_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "h.db"
    BudgetLedger(Store(path), total_usd=1.0).commit(
        BudgetLedger(Store(path), total_usd=1.0).reserve(0.7), 0.7
    )
    assert BudgetLedger(Store(path), total_usd=1.0).remaining_usd() == pytest.approx(0.3)


def test_concurrent_reservations_never_exceed_cap(tmp_path: Path) -> None:
    ledger = BudgetLedger(Store(tmp_path / "h.db"), total_usd=1.0)
    ok: list[float] = []

    def worker() -> None:
        try:
            ledger.reserve(0.1)
            ok.append(0.1)
        except BudgetExceededError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(ok) <= 1.0 + 1e-9
    assert len(ok) == 10


def test_sqlite_errors_inside_a_transaction_become_store_errors(tmp_path: Path) -> None:
    """Regression (evaluation F16): raw sqlite3 errors escaped the store-unavailable fallbacks."""
    import pytest as _pytest

    from adaptive_harness.errors import StoreError
    from adaptive_harness.store import Store

    s = Store(tmp_path / "s.db")
    with _pytest.raises(StoreError), s.transaction() as con:
        con.execute("INSERT INTO no_such_table VALUES (1)")
    assert s.query_one("SELECT 1 AS ok") == {"ok": 1}  # the connection is still usable


def test_bounded_cache_evicts_the_least_recently_used() -> None:
    from adaptive_harness._lru import BoundedCache

    c: BoundedCache[str, int] = BoundedCache(2)
    c["a"], c["b"] = 1, 2
    assert c["a"] == 1
    c["c"] = 3
    assert "b" not in c and "a" in c and "c" in c and len(c) == 2
