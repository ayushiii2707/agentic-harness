"""Cold-start calibration (DESIGN §6.5).

Evidence is keyed by the category the harness's own classifier assigns to a prompt, because that is the key the
request path uses to look evidence up (DECISIONS D29). So each suite's tasks are first grouped into *cells* by
predicted category; within a cell, every surviving strategy is run on the same tasks (a paired design) through
the normal request path (``AdaptiveHarness.run_trial``). After each batch, a strategy is dropped when
P(best > it + min_gain) >= elimination_confidence. Identity is never dropped: it is the reference every switch
is measured against.

Budget: every call first reserves ``max_cost_per_call_usd`` in the persistent ledger and then commits the
provider-reported cost (or the reservation when no cost is reported). A refused reservation stops the run.
Resumable: trials carry the experiment id and task id, and finished (task, strategy) pairs are skipped on resume.
"""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .._version import __version__
from ..errors import BudgetExceededError, ConfigurationError, ProviderError
from ..hashing import canonical_hash
from ..learning.evidence import ArmStats
from ..strategies import Strategy
from ..tasks import TaskSpec
from .suites import SUITE_CATEGORIES, SUITE_VERSION, CalibrationTask, suite

if TYPE_CHECKING:  # pragma: no cover
    from ..api import AdaptiveHarness


@dataclass(frozen=True)
class ArmReport:
    strategy_id: str
    strategy_version: str
    n: int
    successes: float
    mean: float
    ci_low: float
    ci_high: float
    eliminated: bool


@dataclass(frozen=True)
class CellReport:
    suite: str
    category: str  # the predicted category: the key under which this evidence is stored and later used
    tasks: int
    arms: tuple[ArmReport, ...]
    min_detectable_effect: float | None
    stopped: str
    usable: bool  # True when identity and at least one other arm reached the policy's min_trials


@dataclass(frozen=True)
class CalibrationReport:
    experiment_id: str
    model: str
    profile_id: int | None
    calls: int
    spend_usd: float
    cells: tuple[CellReport, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_markdown(self) -> str:
        lines = [
            f"# Calibration report `{self.experiment_id}`",
            "",
            f"Model `{self.model}`, profile {self.profile_id}. Calls this run: {self.calls}; "
            f"spend this run: ${self.spend_usd:.6f}.",
            "",
            "Evidence is stored under the category the harness's classifier assigns (the same key used when",
            "choosing a strategy for a new prompt), so one suite can produce several cells.",
        ]
        for c in self.cells:
            mde = "-" if c.min_detectable_effect is None else f"{100 * c.min_detectable_effect:.0f} pp"
            lines += [
                "",
                f"## suite `{c.suite}` → cell `{c.category}`: {c.tasks} tasks; stopped: {c.stopped}",
                f"Usable by the policy (enough outcomes): {'yes' if c.usable else 'no'}. Smallest difference from "
                f"identity this cell could reliably detect (80% power, alpha 0.05): {mde}.",
                "",
                "| strategy | n | successes | rate | 95% interval | eliminated |",
                "|---|---|---|---|---|---|",
            ]
            lines += [
                f"| {a.strategy_id} | {a.n} | {a.successes:g} | {a.mean:.2f} | [{a.ci_low:.2f}, {a.ci_high:.2f}] | "
                f"{'yes' if a.eliminated else 'no'} |"
                for a in c.arms
            ]
        if self.notes:
            lines += ["", "Notes:"] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


class Calibrator:
    def __init__(
        self,
        harness: AdaptiveHarness,
        *,
        max_calls: int,
        max_cost_per_call_usd: float,
        max_arms: int = 5,
        batch_size: int = 10,
        elimination_confidence: float = 0.95,
        seed: int = 0,
        max_tokens: int | None = None,
        pause_s: float = 0.0,
    ) -> None:
        if harness.store is None or harness.ledger is None or harness.evidence is None:
            raise ConfigurationError("calibration needs persistence enabled (an evidence store)")
        if max_calls < 1 or max_cost_per_call_usd < 0:
            raise ConfigurationError("max_calls must be >= 1 and max_cost_per_call_usd >= 0")
        self.h = harness
        self.max_calls = max_calls
        self.max_cost = max_cost_per_call_usd
        self.max_arms = max_arms
        self.batch_size = batch_size
        self.elimination_confidence = elimination_confidence
        self.seed = seed
        self.max_tokens = max_tokens
        self.pause_s = pause_s
        self.calls = 0
        self.spend = 0.0
        self.notes: list[str] = []

    # ------------------------------------------------------------------ public
    def run(
        self, categories: tuple[str, ...] = SUITE_CATEGORIES, experiment_id: str | None = None
    ) -> CalibrationReport:
        exp = experiment_id or f"cal-{uuid.uuid4().hex[:10]}"
        self._ensure_experiment(exp, categories)
        cells: list[CellReport] = []
        stop_reason: str | None = None
        for name in categories:
            tasks = suite(name)
            random.Random(self.seed).shuffle(tasks)
            groups: dict[str, list[tuple[CalibrationTask, TaskSpec]]] = defaultdict(list)
            for t in tasks:
                spec = self.h.analyzer.analyze(t.prompt)
                groups[spec.category].append((t, spec))
            for category in sorted(groups):
                if stop_reason is not None:
                    cells.append(
                        self._report(exp, name, category, groups[category], [], set(), "not started")
                    )
                    continue
                cell, stop_reason = self._calibrate_cell(exp, name, category, groups[category])
                cells.append(cell)
        profile_id, _ = self.h._current_profile()
        assert self.h.store is not None
        if profile_id is not None and self.calls > 0:
            self.h.store.execute(
                "UPDATE profile SET status='active', last_calibrated_at=? WHERE id=? AND status != 'stale'",
                (time.time(), profile_id),
            )
        self.h.store.execute(
            "UPDATE experiment SET status=?, finished_at=? WHERE id=?",
            ("stopped" if stop_reason else "finished", time.time(), exp),
        )
        return CalibrationReport(
            exp, self.h.model, profile_id, self.calls, round(self.spend, 8), tuple(cells), tuple(self.notes)
        )

    # ------------------------------------------------------------------ internals
    def _ensure_experiment(self, exp: str, categories: tuple[str, ...]) -> None:
        assert self.h.store is not None
        config = {
            "categories": list(categories),
            "max_arms": self.max_arms,
            "batch_size": self.batch_size,
            "elimination_confidence": self.elimination_confidence,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "suite_version": SUITE_VERSION,
            "model": self.h.model,
            "provider": self.h.adapter.name,
        }
        self.h.store.execute(
            "INSERT OR IGNORE INTO experiment(id, kind, config_hash, suite_version, software_version, seed, "
            "budget_json, status, started_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                exp,
                "calibration",
                canonical_hash(config),
                SUITE_VERSION,
                __version__,
                self.seed,
                json.dumps({"max_calls": self.max_calls, "max_cost_per_call_usd": self.max_cost}),
                "running",
                time.time(),
            ),
        )

    def _arms_for(self, items: list[tuple[CalibrationTask, TaskSpec]]) -> list[Strategy]:
        seen: dict[tuple[str, str], Strategy] = {}
        for _, spec in items:
            for s in self.h.strategies.candidates(spec):
                seen.setdefault((s.info.id, s.info.version), s)
        arms = sorted(seen.values(), key=lambda s: (s.info.id != "identity", s.info.id))
        return arms[: self.max_arms]

    def _counts(self, exp: str, category: str) -> dict[tuple[str, str], tuple[int, float]]:
        assert self.h.store is not None
        rows = self.h.store.query(
            "SELECT t.strategy_id, t.strategy_version, COUNT(*) AS n, SUM(e.score) AS s FROM trial t "
            "JOIN evaluation e ON e.trial_id = t.id AND e.status IN ('success','failure','partial') "
            "WHERE t.experiment_id=? AND t.category_id=? GROUP BY t.strategy_id, t.strategy_version",
            (exp, category),
        )
        return {(r["strategy_id"], r["strategy_version"]): (int(r["n"]), float(r["s"] or 0.0)) for r in rows}

    def _done(self, exp: str) -> set[tuple[str, str]]:
        assert self.h.store is not None
        rows = self.h.store.query(
            "SELECT DISTINCT task_ref, strategy_id FROM trial t JOIN evaluation e ON e.trial_id = t.id "
            "WHERE t.experiment_id=? AND t.status='done'",
            (exp,),
        )
        return {(r["task_ref"], r["strategy_id"]) for r in rows}

    def _calibrate_cell(
        self, exp: str, name: str, category: str, items: list[tuple[CalibrationTask, TaskSpec]]
    ) -> tuple[CellReport, str | None]:
        arms = self._arms_for(items)
        if len(arms) < 2:  # only identity applies: the request path will send the original prompt anyway
            return self._report(exp, name, category, items, arms, set(), "nothing to compare"), None
        surviving = list(arms)
        eliminated: set[tuple[str, str]] = set()
        done = self._done(exp)
        for start in range(0, len(items), self.batch_size):
            for task, spec in items[start : start + self.batch_size]:
                for arm in surviving:
                    if (task.id, arm.info.id) in done or not arm.applicable(spec):
                        continue
                    stop = self._one_call(exp, task, arm)
                    if stop:
                        return self._report(exp, name, category, items, arms, eliminated, stop), stop
            surviving = self._eliminate(exp, category, surviving, eliminated)
            if len(surviving) == 1:
                return self._report(
                    exp, name, category, items, arms, eliminated, "only identity remains"
                ), None
        return self._report(exp, name, category, items, arms, eliminated, "suite exhausted"), None

    def _one_call(self, exp: str, task: CalibrationTask, arm: Strategy) -> str | None:
        """Run one call under the budget. Returns a stop reason, or None to continue."""
        assert self.h.ledger is not None
        if self.calls >= self.max_calls:
            return f"max_calls ({self.max_calls}) reached"
        try:
            reservation = self.h.ledger.reserve(
                self.max_cost, experiment_id=exp, note=f"{task.id}:{arm.info.id}"
            )
        except BudgetExceededError as e:
            return f"budget: {e}"
        settings = {"max_tokens": self.max_tokens} if self.max_tokens else {}
        try:
            r = self.h.run_trial(
                task.prompt,
                strategy=arm,
                check=task.evaluator,
                experiment_id=exp,
                task_ref=task.id,
                **settings,
            )
        except ProviderError as e:
            self.h.ledger.release(reservation, note="call failed")
            self.notes.append(f"{task.id}/{arm.info.id}: {type(e).__name__}: {e}"[:300])
            return f"provider error ({type(e).__name__}); resume later with the same experiment id"
        cost = r.result.cost_usd if r.result.cost_usd is not None else self.max_cost
        self.h.ledger.commit(reservation, cost, note="calibration")
        self.spend += cost
        self.calls += 1
        if r.trace.strategy_id != arm.info.id:
            self.notes.append(f"{task.id}: {arm.info.id} could not be applied; identity was sent instead")
        if self.pause_s:
            time.sleep(self.pause_s)
        return None

    def _eliminate(
        self, exp: str, category: str, surviving: list[Strategy], eliminated: set[tuple[str, str]]
    ) -> list[Strategy]:
        counts = self._counts(exp, category)

        def stats(s: Strategy) -> ArmStats:
            n, succ = counts.get((s.info.id, s.info.version), (0, 0.0))
            return ArmStats(s.info.id, s.info.version, 1.0 + succ, 1.0 + n - succ, n)

        best = max(surviving, key=lambda s: stats(s).mean)
        keep = []
        for s in surviving:
            if s.info.id == "identity" or s is best:
                keep.append(s)
            elif self.h.policy.p_better(stats(best), stats(s)) >= self.elimination_confidence:
                eliminated.add((s.info.id, s.info.version))
            else:
                keep.append(s)
        return keep

    def _report(
        self,
        exp: str,
        name: str,
        category: str,
        items: list[tuple[CalibrationTask, TaskSpec]],
        arms: list[Strategy],
        eliminated: set[tuple[str, str]],
        stopped: str,
    ) -> CellReport:
        counts = self._counts(exp, category)
        rng = random.Random(self.seed)
        out = []
        for s in arms:
            n, succ = counts.get((s.info.id, s.info.version), (0, 0.0))
            a, b = 1.0 + succ, 1.0 + n - succ
            draws = sorted(rng.betavariate(a, b) for _ in range(4000))
            out.append(
                ArmReport(s.info.id, s.info.version, n, succ, a / (a + b), draws[100], draws[3899],
                          (s.info.id, s.info.version) in eliminated)
            )  # fmt: skip
        ident = next((x for x in out if x.strategy_id == "identity"), None)
        mde = None
        if ident is not None and ident.n > 0:
            p = min(max(ident.successes / ident.n, 0.05), 0.95)
            mde = (1.96 + 0.84) * math.sqrt(2 * p * (1 - p) / ident.n)
        # The policy can act on this cell when identity has enough outcomes and every challenger that was not
        # eliminated does too (if all were eliminated, keeping identity is itself the evidence-backed choice).
        need = self.h.config.policy.min_trials
        challengers = [x for x in out if x.strategy_id != "identity"]
        usable = (
            ident is not None
            and bool(challengers)
            and ident.n >= need
            and all(x.n >= need for x in challengers if not x.eliminated)
        )
        return CellReport(name, category, len(items), tuple(out), mde, stopped, usable)
