"""Outcome drift detection (DESIGN §6.7, trigger 2).

After each counted outcome, the cell's most recent ``window`` outcomes are compared with the ``reference`` outcomes
before them by a two-proportion z statistic. If |z| >= ``z_crit``, the profile is marked ``stale`` and a ``drift``
change event is recorded. A stale profile keeps serving but the policy stops exploiting non-identity strategies
until a recalibration creates the next profile version.

The test is repeated after every outcome, so ``z_crit`` is much higher than a one-off 1.96. Its default was tuned
by simulation (``scripts/simulate_drift.py``) for fewer than 5% false alarms over 500 stationary outcomes.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..store import Store


@dataclass(frozen=True)
class DriftResult:
    drifted: bool
    z: float | None
    recent_mean: float | None = None
    reference_mean: float | None = None


def drift_statistic(outcomes: Sequence[float], window: int, reference: int) -> DriftResult:
    """Compare the last ``window`` outcomes with up to ``reference`` outcomes before them (needs >= window)."""
    if len(outcomes) < 2 * window:
        return DriftResult(False, None)
    recent = list(outcomes[-window:])
    ref = list(outcomes[:-window][-reference:])
    m1, m0 = sum(recent) / len(recent), sum(ref) / len(ref)
    pooled = (sum(recent) + sum(ref)) / (len(recent) + len(ref))
    var = pooled * (1.0 - pooled) * (1.0 / len(recent) + 1.0 / len(ref))
    if var <= 0:
        return DriftResult(False, 0.0, m1, m0)
    return DriftResult(False, (m1 - m0) / math.sqrt(var), m1, m0)


def check_drift(outcomes: Sequence[float], *, window: int, reference: int, z_crit: float) -> DriftResult:
    r = drift_statistic(outcomes, window, reference)
    if r.z is not None and abs(r.z) >= z_crit:
        return DriftResult(True, r.z, r.recent_mean, r.reference_mean)
    return r


def mark_drift(store: Store, profile_id: int, detail: dict[str, object]) -> bool:
    """Mark the profile stale and record a drift event. Returns False if it was already stale."""
    with store.transaction() as con:
        changed = con.execute(
            "UPDATE profile SET status='stale' WHERE id=? AND status != 'stale'", (profile_id,)
        ).rowcount
        if changed:
            con.execute(
                "INSERT INTO change_event(profile_id, kind, detail_json, at) VALUES (?,?,?,?)",
                (profile_id, "drift", json.dumps(detail), time.time()),
            )
    return bool(changed)
