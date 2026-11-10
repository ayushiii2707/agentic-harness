"""Per-profile evidence: a Beta posterior over the success rate of each (category, strategy@version) cell.

Each evaluated call adds its outcome o in [0, 1] as a (fractional) Bernoulli observation: alpha += o,
beta += 1 - o, starting from Beta(1, 1). On top of that (DESIGN §6.2, §6.6):

- **Time decay.** Before each update, and when read, the cell's observations are discounted by
  0.5 ** (age / half_life). ``n_eff`` is the decayed count of the cell's *own* observations; the policy compares
  only cells whose ``n_eff`` reaches ``min_trials``, so old evidence has to be re-confirmed.
- **Hierarchical backoff.** When read, a cell borrows a weak prior: the parent category's mean for the same
  profile and strategy (weight up to 4 pseudo-observations), otherwise the strategy's mean over the profile's
  other categories (weight up to 2). The weight never exceeds the evidence it comes from.
- **Prior evidence.** ``prior_evidence`` rows whose tested model ID matches this model (vendor prefix ignored)
  add pseudo-counts, weight capped at 3 (halved when the host differs or is unknown).

Backoff and priors change ``alpha``/``beta`` (the estimate) but never ``n_eff`` (the gate), so they cannot by
themselves make the policy leave identity. A token summary per cell feeds the cost term of the policy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import Any

from ..store import Store

W_PARENT = 4.0
W_GLOBAL = 2.0
W_PRIOR = 3.0


@dataclass(frozen=True)
class ArmStats:
    strategy_id: str
    strategy_version: str
    alpha: float = 1.0
    beta: float = 1.0
    n_trials: int = 0  # raw count of evaluated outcomes ever recorded
    n_eff: float = 0.0  # decayed count of the cell's own outcomes (the policy's gate)
    mean_tokens: float | None = None  # decayed mean total tokens per call
    sources: tuple[str, ...] = ()  # where any borrowed prior came from

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def normalise_model_id(model_id: str) -> str:
    """``openai/gpt-oss-20b`` and ``gpt-oss-20b`` name the same model; hosts are compared separately."""
    return model_id.rsplit("/", 1)[-1].strip().lower()


class EvidenceStore:
    def __init__(self, store: Store, *, half_life_s: float | None = None) -> None:
        if half_life_s is not None and half_life_s <= 0:
            raise ValueError("half_life_s must be positive")
        self.store = store
        self.half_life_s = half_life_s

    # ------------------------------------------------------------------ decay
    def _factor(self, age_s: float) -> float:
        if self.half_life_s is None or age_s <= 0:
            return 1.0
        return float(0.5 ** (age_s / self.half_life_s))

    @staticmethod
    def _summary(raw: str | None) -> dict[str, float]:
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}

    def _own(self, row: Any, now: float) -> ArmStats:
        f = self._factor(now - row["updated_at"])
        s = self._summary(row["summary_json"])
        tok_w, tok_sum = s.get("tok_w", 0.0) * f, s.get("tok_sum", 0.0) * f
        return ArmStats(
            row["strategy_id"],
            row["strategy_version"],
            alpha=1.0 + (row["alpha"] - 1.0) * f,
            beta=1.0 + (row["beta"] - 1.0) * f,
            n_trials=row["n_trials"],
            n_eff=row["n_tasks"] * f,
            mean_tokens=tok_sum / tok_w if tok_w > 0 else None,
        )

    # ------------------------------------------------------------------ write
    def record(
        self,
        profile_id: int,
        category: str,
        strategy_id: str,
        version: str,
        outcome: float,
        *,
        tokens: int | None = None,
        now: float | None = None,
    ) -> None:
        if not 0.0 <= outcome <= 1.0:
            raise ValueError("outcome must be in [0, 1]")
        now = time.time() if now is None else now
        key = (profile_id, category, strategy_id, version)
        with self.store.transaction() as con:
            row = con.execute(
                "SELECT * FROM arm_stats WHERE profile_id=? AND category_id=? AND strategy_id=? "
                "AND strategy_version=?",
                key,
            ).fetchone()
            if row is None:
                a, b, n, n_trials, s = 1.0, 1.0, 0.0, 0, {}
            else:
                f = self._factor(now - row["updated_at"])
                a = 1.0 + (row["alpha"] - 1.0) * f
                b = 1.0 + (row["beta"] - 1.0) * f
                n, n_trials = row["n_tasks"] * f, row["n_trials"]
                s = {k: v * f for k, v in self._summary(row["summary_json"]).items()}
            if tokens is not None and tokens >= 0:
                s["tok_w"] = s.get("tok_w", 0.0) + 1.0
                s["tok_sum"] = s.get("tok_sum", 0.0) + float(tokens)
            con.execute(
                """INSERT INTO arm_stats(profile_id, category_id, strategy_id, strategy_version, alpha, beta,
                                         n_tasks, n_trials, summary_json, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(profile_id, category_id, strategy_id, strategy_version) DO UPDATE SET
                     alpha=excluded.alpha, beta=excluded.beta, n_tasks=excluded.n_tasks,
                     n_trials=excluded.n_trials, summary_json=excluded.summary_json,
                     updated_at=excluded.updated_at""",
                (*key, a + outcome, b + (1.0 - outcome), n + 1.0, n_trials + 1, json.dumps(s), now),
            )

    # ------------------------------------------------------------------ read
    def arms(
        self,
        profile_id: int,
        category: str,
        *,
        parent: str | None = None,
        model_id: str | None = None,
        host: str | None = None,
        now: float | None = None,
    ) -> dict[tuple[str, str], ArmStats]:
        """The cell's arms with their own (decayed) evidence plus any borrowed prior. Only arms with own data."""
        now = time.time() if now is None else now
        rows = self.store.query("SELECT * FROM arm_stats WHERE profile_id=?", (profile_id,))
        by_cat: dict[str, dict[tuple[str, str], ArmStats]] = {}
        for r in rows:
            by_cat.setdefault(r["category_id"], {})[(r["strategy_id"], r["strategy_version"])] = self._own(
                r, now
            )
        own = by_cat.get(category, {})
        priors = self._prior_rows(model_id, category) if model_id else []
        out = {}
        for key, arm in own.items():
            out[key] = self._with_priors(arm, key, category, parent, by_cat, priors, host)
        return out

    def _with_priors(
        self,
        arm: ArmStats,
        key: tuple[str, str],
        category: str,
        parent: str | None,
        by_cat: dict[str, dict[tuple[str, str], ArmStats]],
        priors: list[Any],
        host: str | None,
    ) -> ArmStats:
        add_a = add_b = 0.0
        sources: list[str] = []
        borrowed = by_cat.get(parent, {}).get(key) if parent else None
        if borrowed is not None and borrowed.n_eff > 0:
            w = min(W_PARENT, borrowed.n_eff)
            sources.append(f"parent:{parent}")
        else:
            others = [c[key] for cat, c in by_cat.items() if cat != category and key in c]
            n = sum(o.n_eff for o in others)
            borrowed = None
            if n > 0:
                succ = sum(o.alpha - 1.0 for o in others)
                fail = sum(o.beta - 1.0 for o in others)
                borrowed = ArmStats(*key, alpha=1.0 + succ, beta=1.0 + fail, n_eff=n)
                w = min(W_GLOBAL, n)
                sources.append("model-global")
        if borrowed is not None:
            add_a += w * borrowed.mean
            add_b += w * (1.0 - borrowed.mean)
        for p in priors:
            if p["strategy"] != key[0]:
                continue
            w = min(W_PRIOR, float(p["weight"]))
            if not host or not p["host"] or p["host"] != host:
                w /= 2.0
            add_a += w * float(p["effect"])
            add_b += w * (1.0 - float(p["effect"]))
            sources.append(f"prior:{p['id']}")
        if not sources:
            return arm
        return replace(arm, alpha=arm.alpha + add_a, beta=arm.beta + add_b, sources=tuple(sources))

    def _prior_rows(self, model_id: str, category: str) -> list[Any]:
        rows = self.store.query(
            "SELECT * FROM prior_evidence WHERE category=? AND effect IS NOT NULL", (category,)
        )
        want = normalise_model_id(model_id)
        return [
            r for r in rows if normalise_model_id(r["model_id_as_tested"]) == want and 0 <= r["effect"] <= 1
        ]

    def outcomes(
        self, profile_id: int, category: str, strategy_id: str, version: str, limit: int
    ) -> list[float]:
        """The cell's most recent counted outcomes, oldest first (for drift detection)."""
        rows = self.store.query(
            """SELECT e.score FROM evaluation e JOIN trial t ON t.id = e.trial_id
               JOIN profile p ON p.model_ref = t.model_ref AND p.id = ?
               WHERE t.category_id=? AND t.strategy_id=? AND t.strategy_version=?
                 AND e.status IN ('success','failure','partial') AND e.score IS NOT NULL
                 AND t.created_at >= p.created_at
                 AND e.id = (SELECT MIN(e2.id) FROM evaluation e2 WHERE e2.trial_id = e.trial_id
                             AND e2.status IN ('success','failure','partial'))
               ORDER BY e.id DESC LIMIT ?""",
            (profile_id, category, strategy_id, version, limit),
        )
        return [float(r["score"]) for r in reversed(rows)]
