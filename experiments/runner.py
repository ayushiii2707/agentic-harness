"""Phase 12 experiment runner (see experiments/preregistration_v1.md). Not part of the installed package.

Phases, each resumable (a (task, strategy, repetition) that already has a finished trial is skipped):

    calibrate  every calibration task once under every calibration arm (full paired design)
    select     B+, C and D selections from calibration outcomes -> selections.json (before any held-out call)
    evaluate   every held-out task x repetition under each distinct strategy among A, B, B+, C, D
    export     one CSV row per trial (no response text)

    python -m experiments.runner --config experiments/config_v1.json --model groq:openai/gpt-oss-20b --phase calibrate

Every call goes through ``AdaptiveHarness.run_trial`` (the product path with the strategy fixed) and reserves its
per-call cap in the experiment database's budget ledger first. A provider error stops the phase cleanly; run it
again later to resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from adaptive_harness import (  # noqa: E402
    AdaptiveHarness,
    BudgetConfig,
    CalibrationConfig,
    HarnessConfig,
    HTTPConfig,
    PersistenceConfig,
    PolicyConfig,
    RewriterConfig,
)
from adaptive_harness._version import __version__  # noqa: E402
from adaptive_harness.calibration.suites import (  # noqa: E402
    SUITE_VERSION,
    CalibrationTask,
    Split,
    code_suite,
    instructions_suite,
    json_suite,
    math_suite,
)
from adaptive_harness.errors import BudgetExceededError, ProviderError  # noqa: E402
from adaptive_harness.evaluation.builtin import JsonFields, ReferenceAnswer  # noqa: E402
from adaptive_harness.hashing import canonical_hash  # noqa: E402
from adaptive_harness.learning import ArmStats, Policy  # noqa: E402
from adaptive_harness.providers.base import ProviderAdapter  # noqa: E402
from adaptive_harness.strategies import Strategy, StrategyRegistry  # noqa: E402
from adaptive_harness.strategies.few_shot import build_few_shot  # noqa: E402
from adaptive_harness.tasks import TaskAnalyzer  # noqa: E402

CONDITIONS = ("A", "B", "B+", "C", "D")


class StopCollection(Exception):
    """A provider error or the budget stopped this phase; rerun later to resume."""


# ------------------------------------------------------------------ configuration and tasks
def load_config(path: str | Path) -> dict[str, Any]:
    cfg: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["suite_version"] != SUITE_VERSION:
        raise SystemExit(f"config expects suite version {cfg['suite_version']}, package has {SUITE_VERSION}")
    return cfg


def slug(provider: str, model: str) -> str:
    return f"{provider}-{model}".replace("/", "_").replace(":", "_")


def model_entry(cfg: dict[str, Any], spec: str) -> dict[str, Any]:
    provider, _, model = spec.partition(":")
    for m in cfg["models"]:
        if m["provider"] == provider and m["model"] == model:
            return dict(m)
    raise SystemExit(f"model {spec!r} is not in the config")


def suite_tasks(name: str, split: str, n: int) -> list[CalibrationTask]:
    if name == "code_generation":
        if split != "heldout":
            raise ValueError("code_generation is a held-out category only")
        return code_suite()[:n]
    splits: dict[str, Split] = {
        "calibration": "calibration",
        "heldout": "heldout",
        "examples": "examples",
        "pilot": "pilot",
    }
    if split not in splits:
        raise ValueError(f"unknown split {split!r}")
    sp = splits[split]
    if name == "math":
        return math_suite(split=sp)[:n]
    if name == "structured_output":
        return json_suite(split=sp)[:n]
    if name == "instructions":
        return instructions_suite(split=sp)[:n]
    raise ValueError(f"unknown suite {name!r}")


def check_no_leakage(cfg: dict[str, Any]) -> None:
    cal = {t.prompt for s, n in cfg["calibration"].items() for t in suite_tasks(s, "calibration", n)}
    held = {t.prompt for s, n in cfg["heldout"].items() for t in suite_tasks(s, "heldout", n)}
    overlap = cal & held
    if overlap:
        raise SystemExit(f"leakage: {len(overlap)} held-out prompts also appear in calibration")
    unseen = set(cfg["unseen_categories"])
    if unseen & set(cfg["calibration"]):
        raise SystemExit("an unseen category is also calibrated")


# ------------------------------------------------------------------ harness
def make_harness(
    cfg: dict[str, Any],
    entry: dict[str, Any],
    db: Path,
    adapter: ProviderAdapter | None = None,
    aux_adapters: dict[str, ProviderAdapter] | None = None,
) -> AdaptiveHarness:
    rw = cfg.get("rewriter")
    config = HarnessConfig(
        persistence=PersistenceConfig(path=db, prompt_storage="full"),  # our own synthetic tasks: keep all
        budget=BudgetConfig(total_usd=entry.get("budget_usd", cfg["bookkeeping"]["budget_usd"])),
        http=HTTPConfig(timeout_s=300.0, max_attempts=4, max_total_wait_s=180.0),
        policy=PolicyConfig(exploration_rate=0.0),  # frozen defaults otherwise (preregistration §4)
        calibration=CalibrationConfig(auto=False),
        rewriter=RewriterConfig(**rw) if rw else None,
    )
    options: dict[str, Any] = {}
    if entry.get("extra_body") and adapter is None:
        options["extra_body"] = entry["extra_body"]
    registry = StrategyRegistry()
    if cfg.get("few_shot"):
        registry.register(few_shot_strategy(cfg["few_shot"]))
    return AdaptiveHarness(
        provider=entry["provider"],
        model=entry["model"],
        config=config,
        adapter=adapter,
        strategies=registry,
        aux_adapters=aux_adapters,
        **options,
    )


def example_answer(task: CalibrationTask) -> str:
    """The reference answer of an ``examples``-split task, in the form its prompt asks for."""
    ev = task.evaluator
    if isinstance(ev, ReferenceAnswer):
        x = ev.expected
        return f"{x:g}" if isinstance(x, float) else str(x)
    if isinstance(ev, JsonFields):
        return json.dumps(dict(ev.expected))
    raise ValueError(f"no example answer for {type(ev).__name__}")


def few_shot_strategy(fs: dict[str, Any]) -> Strategy:
    """few_shot with ``k`` examples per suite from the disjoint ``examples`` split, keyed by predicted category."""
    analyzer = TaskAnalyzer()
    examples: dict[str, list[tuple[str, str]]] = {}
    for s in fs["suites"]:
        for t in suite_tasks(s, "examples", fs["k"]):
            examples.setdefault(analyzer.analyze(t.prompt).category, []).append((t.prompt, example_answer(t)))
    return build_few_shot(examples)


def ensure_experiment(h: AdaptiveHarness, cfg: dict[str, Any], exp_id: str) -> None:
    assert h.store is not None
    h.store.execute(
        "INSERT OR IGNORE INTO experiment(id, kind, config_hash, suite_version, software_version, seed, budget_json, "
        "status, started_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (exp_id, "research", canonical_hash(cfg), SUITE_VERSION, __version__, cfg["order_seed"],
         json.dumps(cfg["bookkeeping"]), "running", time.time()),
    )  # fmt: skip


def done_refs(h: AdaptiveHarness, exp_id: str) -> set[str]:
    assert h.store is not None
    rows = h.store.query("SELECT task_ref FROM trial WHERE experiment_id=? AND status='done'", (exp_id,))
    return {r["task_ref"] for r in rows}


@dataclass
class Collector:
    h: AdaptiveHarness
    cfg: dict[str, Any]
    entry: dict[str, Any]
    exp_id: str
    log: Callable[[str], None] = print

    def __post_init__(self) -> None:
        ensure_experiment(self.h, self.cfg, self.exp_id)
        self.done = done_refs(self.h, self.exp_id)
        self.calls = 0

    def trial(self, task: CalibrationTask, strategy_id: str, ref: str) -> None:
        if ref in self.done:
            return
        assert self.h.ledger is not None
        local = self.h.adapter.capabilities().local
        cap = (
            0.0
            if local
            else float(
                self.entry.get("max_cost_per_call_usd", self.cfg["bookkeeping"]["max_cost_per_call_usd"])
            )
        )
        try:
            reservation = self.h.ledger.reserve(cap, experiment_id=self.exp_id, note=ref)
        except BudgetExceededError as e:
            raise StopCollection(f"budget: {e}") from e
        try:
            r = self.h.run_trial(
                task.prompt,
                strategy=self.h.strategies.get(strategy_id),
                check=task.evaluator,
                experiment_id=self.exp_id,
                task_ref=ref,
                max_tokens=self.entry["max_tokens"],
            )
        except ProviderError as e:
            self.h.ledger.release(reservation, note="call failed")
            raise StopCollection(f"{type(e).__name__}: {e}"[:300]) from e
        self.h.ledger.commit(
            reservation, r.result.cost_usd if r.result.cost_usd is not None else cap, note=ref
        )
        self.done.add(ref)
        self.calls += 1
        if self.calls % 25 == 0:
            self.log(f"  {self.calls} calls this run")
        if self.entry.get("pause_s"):
            time.sleep(float(self.entry["pause_s"]))


# ------------------------------------------------------------------ phases
def suites_for(cfg: dict[str, Any], entry: dict[str, Any], split: str) -> dict[str, int]:
    """Suites (and task counts) for this model: all of the config's, or the model's own list (v2 pilot rule)."""
    allowed = entry.get("suites")
    sizes: dict[str, int] = cfg[split]
    return {
        s: n for s, n in sizes.items() if allowed is None or s in allowed or s in cfg["unseen_categories"]
    }


def calibrate(c: Collector) -> None:
    items = [
        (t, arm)
        for s, n in suites_for(c.cfg, c.entry, "calibration").items()
        for t in suite_tasks(s, "calibration", n)
        for arm in c.cfg["calibration_arms"]
    ]
    random.Random(c.cfg["order_seed"]).shuffle(items)
    for t, arm in items:
        c.trial(t, arm, f"cal:{t.id}:{arm}")


def _arm_stats(rows: Sequence[dict[str, Any]], strategy: str) -> ArmStats:
    xs = [r for r in rows if r["strategy_id"] == strategy and r["score"] is not None]
    n = len(xs)
    succ = sum(float(r["score"]) for r in xs)
    tok = [r["tokens"] for r in xs if r["tokens"] is not None]
    return ArmStats(strategy, "1", 1.0 + succ, 1.0 + n - succ, n, n_eff=float(n),
                    mean_tokens=sum(tok) / len(tok) if tok else None)  # fmt: skip


def calibration_rows(h: AdaptiveHarness, exp_id: str) -> list[dict[str, Any]]:
    """Calibration outcomes credited to the arm each trial was **assigned** (intent to treat).

    When an arm cannot be applied to a task, the harness sends the original prompt and records the trial under
    identity. Crediting such trials to identity would compare each arm on a different set of tasks (an arm that
    only applies to easy tasks would look better pooled). Grouping by the assigned arm compares every arm on the
    same tasks, as the paired design intends (preregistration amendment 5).
    """
    assert h.store is not None
    rows = h.store.query(
        """SELECT t.task_ref, t.category_id, e.score,
                  COALESCE(t.input_tokens, 0) + COALESCE(t.output_tokens, 0) AS tokens
           FROM trial t JOIN evaluation e ON e.trial_id = t.id AND e.counted = 1
           WHERE t.experiment_id=? AND t.task_ref LIKE 'cal:%' AND t.status='done'""",
        (exp_id,),
    )
    for r in rows:
        r["strategy_id"] = r["task_ref"].rsplit(":", 1)[1]  # cal:<task id>:<assigned arm>
    return rows


def pooled_choice(rows: Sequence[dict[str, Any]], arms: Sequence[str], h: AdaptiveHarness) -> tuple[str, str]:
    """The harness's decision rule on outcomes pooled over categories (C) or over models too (B+)."""
    stats = {(a, "1"): _arm_stats(rows, a) for a in arms}
    cands = [h.strategies.get(a) for a in arms]
    d = Policy(PolicyConfig(exploration_rate=0.0), random.Random(0)).choose(
        cands, stats, can_explore=False, exploit_allowed=True
    )
    return d.strategy.info.id, d.reason


def select(harnesses: dict[str, AdaptiveHarness], cfg: dict[str, Any], exp_id: str) -> dict[str, Any]:
    arms = cfg["calibration_arms"]
    all_rows: list[dict[str, Any]] = []
    out: dict[str, Any] = {"experiment": exp_id, "created_at": time.time(), "models": {}}
    for key, h in harnesses.items():
        rows = calibration_rows(h, exp_id)
        entry = harness_entry(cfg, key)
        need = sum(suites_for(cfg, entry, "calibration").values()) * len(arms)
        if len(rows) < need:
            raise SystemExit(
                f"{key}: calibration incomplete ({len(rows)}/{need} outcomes); run calibrate first"
            )
        all_rows += rows
        c_strategy, c_reason = pooled_choice(rows, arms, h)
        d_map: dict[str, dict[str, str]] = {}
        for s, n in suites_for(cfg, entry, "heldout").items():
            for t in suite_tasks(s, "heldout", n):
                p = h.preview(t.prompt, max_tokens=entry["max_tokens"])
                d_map[t.id] = {
                    "strategy": p.trace.strategy_id,
                    "category": p.trace.category or "",
                    "reason": p.trace.reason,
                }
        out["models"][key] = {"C": {"strategy": c_strategy, "reason": c_reason}, "D": d_map}
    bplus, bplus_reason = pooled_choice(all_rows, arms, next(iter(harnesses.values())))
    out["B"] = cfg["fixed_strategy_b"]
    out["B+"] = {"strategy": bplus, "reason": bplus_reason}
    return out


def harness_entry(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    return model_entry(cfg, key)


def strategies_for(selections: dict[str, Any], key: str, task_id: str) -> dict[str, str]:
    m = selections["models"][key]
    return {
        "A": "identity",
        "B": selections["B"],
        "B+": selections["B+"]["strategy"],
        "C": m["C"]["strategy"],
        "D": m["D"][task_id]["strategy"],
    }


def evaluate(c: Collector, selections: dict[str, Any], key: str) -> None:
    items = []
    for s, n in suites_for(c.cfg, c.entry, "heldout").items():
        for t in suite_tasks(s, "heldout", n):
            chosen = set(strategies_for(selections, key, t.id).values())
            if c.cfg.get("heldout_all_arms"):  # v2: every arm on every held-out task
                chosen |= set(c.cfg["calibration_arms"])
            distinct = sorted(chosen)
            for rep in range(1, c.cfg["repetitions"] + 1):
                items += [(t, strat, rep) for strat in distinct]
    random.Random(c.cfg["order_seed"] + 1).shuffle(items)
    for t, strat, rep in items:
        c.trial(t, strat, f"ho:{t.id}:r{rep}:{strat}")


def pilot(c: Collector) -> dict[str, tuple[int, int, bool]]:
    """Identity on the pilot split; returns {suite: (successes, n, included)} by the pre-registered headroom rule."""
    p = c.cfg["pilot"]
    for s in p["suites"]:
        for t in suite_tasks(s, "pilot", p["n"]):
            c.trial(t, "identity", f"pilot:{t.id}:identity")
    assert c.h.store is not None
    rows = c.h.store.query(
        """SELECT t.task_ref, e.score FROM trial t JOIN evaluation e ON e.trial_id = t.id AND e.counted = 1
           WHERE t.experiment_id=? AND t.task_ref LIKE 'pilot:%'""",
        (c.exp_id,),
    )
    by_task = {r["task_ref"].split(":")[1]: float(r["score"]) for r in rows}
    out = {}
    for s in p["suites"]:
        scores = [by_task[t.id] for t in suite_tasks(s, "pilot", p["n"]) if t.id in by_task]
        k = int(sum(scores))
        out[s] = (k, len(scores), p["min_successes"] <= k <= p["max_successes"])
    return out


# ------------------------------------------------------------------ export
CSV_FIELDS = [
    "model", "phase", "suite", "task_id", "rep", "strategy", "applied_strategy", "category", "status",
    "eval_status", "outcome", "input_tokens", "output_tokens", "reasoning_tokens", "latency_s", "finish_reason",
    "model_reported", "fingerprint", "simulated", "trial_id",
]  # fmt: skip


def export(h: AdaptiveHarness, key: str, exp_id: str, path: Path) -> int:
    assert h.store is not None
    rows = h.store.query(
        """SELECT t.*, e.status AS eval_status, e.score AS outcome
           FROM trial t LEFT JOIN evaluation e ON e.trial_id = t.id AND e.counted = 1
           WHERE t.experiment_id=? ORDER BY t.created_at""",
        (exp_id,),
    )
    suites = {"math": "math", "json": "structured_output", "instr": "instructions", "code": "code_generation"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            parts = r["task_ref"].split(":")
            phase = {"cal": "calibration", "ho": "heldout", "pilot": "pilot"}[
                parts[0]
            ]  # never default to held-out
            task_id = parts[1]
            rep = parts[2][1:] if phase == "heldout" else "1"
            strategy = parts[-1]
            trace = json.loads(r["trace_json"] or "{}")
            w.writerow({
                "model": key, "phase": phase, "suite": suites.get(task_id.split("-")[0], task_id.split("-")[0]),
                "task_id": task_id, "rep": rep, "strategy": strategy,
                "applied_strategy": trace.get("strategy_id", r["strategy_id"]), "category": r["category_id"],
                "status": r["status"], "eval_status": r["eval_status"], "outcome": r["outcome"],
                "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                "reasoning_tokens": r["reasoning_tokens"], "latency_s": r["latency_s"],
                "finish_reason": r["finish_reason"], "model_reported": r["model_reported"],
                "fingerprint": r["fingerprint_json"], "simulated": r["simulated"], "trial_id": r["id"],
            })  # fmt: skip
    return len(rows)


# ------------------------------------------------------------------ command line
def db_path(results: Path, key: str) -> Path:
    provider, _, model = key.partition(":")
    return ROOT / ".harness" / "experiments" / results.name / f"{slug(provider, model)}.db"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "config_v1.json"))
    ap.add_argument("--phase", choices=["pilot", "calibrate", "select", "evaluate", "export"], required=True)
    ap.add_argument(
        "--model", action="append", help="provider:model (repeatable; default: all in the config)"
    )
    ap.add_argument("--results", help="default: experiments/results/<results_dir from the config>")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    check_no_leakage(cfg)
    results = (
        Path(a.results) if a.results else ROOT / "experiments" / "results" / cfg.get("results_dir", "v1")
    )
    exp_id = cfg["experiment"]
    keys = a.model or [f"{m['provider']}:{m['model']}" for m in cfg["models"]]
    harnesses = {k: make_harness(cfg, model_entry(cfg, k), db_path(results, k)) for k in keys}
    sel_path = results / "selections.json"
    try:
        if a.phase == "select":
            if sel_path.exists():
                raise SystemExit(f"{sel_path} exists: selections are fixed once made (preregistration §5)")
            sel = select(harnesses, cfg, exp_id)
            results.mkdir(parents=True, exist_ok=True)
            sel_path.write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")
            print(f"selections written to {sel_path}")
            return 0
        for k, h in harnesses.items():
            if a.phase == "export":
                n = export(h, k, exp_id, results / f"{slug(*k.split(':', 1))}.csv")
                print(f"{k}: {n} trials exported")
                continue
            c = Collector(h, cfg, model_entry(cfg, k), exp_id)
            print(f"{k}: {a.phase} ({len(c.done)} trials already done)")
            if a.phase == "pilot":
                for suite, (succ, n, inc) in pilot(c).items():
                    print(f"{k}: pilot {suite}: identity {succ}/{n} -> {'INCLUDED' if inc else 'excluded'}")
            elif a.phase == "calibrate":
                calibrate(c)
            else:
                evaluate(c, json.loads(sel_path.read_text(encoding="utf-8")), k)
            print(f"{k}: {a.phase} complete ({c.calls} calls this run)")
    except StopCollection as e:
        print(f"stopped: {e}\nrerun the same command later to resume")
        return 2
    finally:
        for h in harnesses.values():
            h.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
