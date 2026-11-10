"""Budget ledger.

Money the harness spends on its own behalf (calibration, exploration, judges, experiments, live tests) is
capped. Before a paid call the caller *reserves* an upper-bound estimate; afterwards it *commits* the actual
cost (or *releases* the reservation if the call never happened). Open reservations count against the cap, so
concurrent calls cannot overshoot it. The ledger is persistent, so the cap holds across processes and restarts.

The developer's own ``generate`` calls are recorded with scope ``user`` and are never blocked.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Literal

from .errors import BudgetExceededError
from .store import Store

Scope = Literal["harness", "user"]


@dataclass(frozen=True)
class Reservation:
    id: str
    amount_usd: float
    scope: Scope
    experiment_id: str | None


class BudgetLedger:
    def __init__(self, store: Store, *, total_usd: float, daily_usd: float | None = None) -> None:
        self.store = store
        self.total_usd = float(total_usd)
        self.daily_usd = daily_usd

    # ------------------------------------------------------------------ accounting
    _EXPOSURE_SQL = """
        SELECT
          COALESCE((SELECT SUM(e.amount_usd) FROM budget_ledger e
                    WHERE e.kind = 'commit' AND e.scope = 'harness' AND e.at >= ?), 0)
        + COALESCE((SELECT SUM(e.amount_usd) FROM budget_ledger e
                    WHERE e.kind = 'reserve' AND e.scope = 'harness' AND e.at >= ?
                      AND NOT EXISTS (SELECT 1 FROM budget_ledger c
                                      WHERE c.reservation = e.reservation AND c.kind IN ('commit','release'))
                   ), 0) AS s
    """

    def _exposure(self, since: float = 0.0) -> float:
        """Committed harness spend plus still-open reservations, from ``since`` onwards."""
        row = self.store.query_one(self._EXPOSURE_SQL, (since, since))
        return float(row["s"]) if row else 0.0

    def spent_usd(self, scope: Scope = "harness") -> float:
        row = self.store.query_one(
            "SELECT COALESCE(SUM(amount_usd), 0) AS s FROM budget_ledger WHERE kind='commit' AND scope=?",
            (scope,),
        )
        return float(row["s"]) if row else 0.0

    def committed_and_reserved_usd(self) -> float:
        return self._exposure()

    def remaining_usd(self) -> float:
        remaining = self.total_usd - self._exposure()
        if self.daily_usd is not None:
            remaining = min(remaining, self.daily_usd - self._exposure(time.time() - 86400))
        return max(0.0, remaining)

    # ------------------------------------------------------------------ operations
    def reserve(self, amount_usd: float, *, experiment_id: str | None = None, note: str = "") -> Reservation:
        if amount_usd < 0:
            raise ValueError("reservation amount cannot be negative")
        with self.store.transaction() as con:
            remaining = self.remaining_usd()
            if amount_usd > remaining + 1e-12:
                raise BudgetExceededError(
                    f"refused: needs up to ${amount_usd:.6f}, ${remaining:.6f} left of ${self.total_usd:.2f}",
                    requested_usd=amount_usd,
                    remaining_usd=remaining,
                )
            res = Reservation(uuid.uuid4().hex, float(amount_usd), "harness", experiment_id)
            con.execute(
                "INSERT INTO budget_ledger(experiment_id, scope, kind, reservation, amount_usd, note, at) "
                "VALUES (?,?,?,?,?,?,?)",
                (experiment_id, "harness", "reserve", res.id, res.amount_usd, note, time.time()),
            )
        return res

    def commit(self, reservation: Reservation, actual_usd: float, note: str = "") -> None:
        self._close(reservation, "commit", max(0.0, float(actual_usd)), note)

    def release(self, reservation: Reservation, note: str = "") -> None:
        self._close(reservation, "release", 0.0, note)

    def _close(self, reservation: Reservation, kind: str, amount: float, note: str) -> None:
        with self.store.transaction() as con:
            done = con.execute(
                "SELECT 1 FROM budget_ledger WHERE reservation=? AND kind IN ('commit','release')",
                (reservation.id,),
            ).fetchone()
            if done:
                raise ValueError(f"reservation {reservation.id} is already closed")
            con.execute(
                "INSERT INTO budget_ledger(experiment_id, scope, kind, reservation, amount_usd, note, at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    reservation.experiment_id,
                    reservation.scope,
                    kind,
                    reservation.id,
                    amount,
                    note,
                    time.time(),
                ),
            )

    def record_user_spend(self, amount_usd: float, note: str = "") -> None:
        """Record (never block) the cost of a developer's own call."""
        self.store.execute(
            "INSERT INTO budget_ledger(experiment_id, scope, kind, reservation, amount_usd, note, at) "
            "VALUES (?,?,?,?,?,?,?)",
            (None, "user", "commit", uuid.uuid4().hex, max(0.0, float(amount_usd)), note, time.time()),
        )
