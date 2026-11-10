"""Per-profile evidence: a Beta posterior over the success rate of each (category, strategy@version) cell.

Each evaluated call adds its outcome o in [0, 1] as a (fractional) Bernoulli observation: alpha += o,
beta += 1 - o, starting from Beta(1, 1). Hierarchical backoff, time decay and prior evidence (DESIGN §6.2, §6.6)
build on this table in Phase 10.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..store import Store


@dataclass(frozen=True)
class ArmStats:
    strategy_id: str
    strategy_version: str
    alpha: float = 1.0
    beta: float = 1.0
    n_trials: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class EvidenceStore:
    def __init__(self, store: Store) -> None:
        self.store = store

    def arms(self, profile_id: int, category: str) -> dict[tuple[str, str], ArmStats]:
        rows = self.store.query(
            "SELECT strategy_id, strategy_version, alpha, beta, n_trials FROM arm_stats "
            "WHERE profile_id=? AND category_id=?",
            (profile_id, category),
        )
        return {
            (r["strategy_id"], r["strategy_version"]): ArmStats(
                r["strategy_id"], r["strategy_version"], r["alpha"], r["beta"], r["n_trials"]
            )
            for r in rows
        }

    def record(self, profile_id: int, category: str, strategy_id: str, version: str, outcome: float) -> None:
        if not 0.0 <= outcome <= 1.0:
            raise ValueError("outcome must be in [0, 1]")
        self.store.execute(
            """INSERT INTO arm_stats(profile_id, category_id, strategy_id, strategy_version, alpha, beta,
                                     n_tasks, n_trials, updated_at)
               VALUES (?,?,?,?,?,?,1,1,?)
               ON CONFLICT(profile_id, category_id, strategy_id, strategy_version) DO UPDATE SET
                 alpha = alpha + excluded.alpha - 1.0,
                 beta = beta + excluded.beta - 1.0,
                 n_tasks = n_tasks + 1,
                 n_trials = n_trials + 1,
                 updated_at = excluded.updated_at""",
            (profile_id, category, strategy_id, version, 1.0 + outcome, 1.0 + (1.0 - outcome), time.time()),
        )
