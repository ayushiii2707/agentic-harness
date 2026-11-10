"""The decision loop: evidence -> choice -> transform -> call -> evaluate -> evidence.

The end-to-end tests use MockProvider, whose results are marked simulated. They verify the mechanism (does the
harness find a better strategy, and does it leave the prompt alone when nothing is better), not real model behaviour.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import ClassVar

import pytest

from adaptive_harness import AdaptiveHarness, HarnessConfig, Message, PersistenceConfig, PolicyConfig
from adaptive_harness.evaluation import EvalStatus, feedback_result, run_check
from adaptive_harness.learning import ArmStats, EvidenceStore, Policy
from adaptive_harness.providers import MockProvider
from adaptive_harness.store import Store
from adaptive_harness.strategies import BUILTIN, Strategy, StrategyInfo, StrategyRegistry
from adaptive_harness.types import CompletionRequest

CODE_PROMPT = "Write a Python function that checks whether a number is prime."
ROLE_LINE = "You are an experienced software engineer."  # role_framing's text for code_generation


def cfg(tmp_path: Path, **policy: float) -> HarnessConfig:
    return HarnessConfig(
        persistence=PersistenceConfig(path=tmp_path / "h.db"),
        policy=PolicyConfig(**policy),  # type: ignore[arg-type]
    )


def simulated_model(p_with_role: float, p_without: float, seed: int = 7) -> MockProvider:
    """Answers 'ok' with a probability that depends on whether role_framing's system line was sent."""
    rng = random.Random(seed)

    def respond(req: CompletionRequest) -> str:
        has_role = any(m.role == "system" and m.content == ROLE_LINE for m in req.messages)
        return "ok" if rng.random() < (p_with_role if has_role else p_without) else "wrong"

    return MockProvider(respond)


def ok(text: str) -> bool:
    return text == "ok"


# ------------------------------------------------------------------ units
def test_checks_map_to_typed_outcomes() -> None:
    assert run_check(lambda t: True, "x").status is EvalStatus.SUCCESS
    assert run_check(lambda t: 0.0, "x").outcome == 0.0
    partial = run_check(lambda t: 0.25, "x")
    assert partial.status is EvalStatus.PARTIAL and partial.outcome == 0.25
    assert run_check(lambda t: "yes", "x").status is EvalStatus.ERROR  # type: ignore[arg-type,return-value]
    crashed = run_check(lambda t: 1 / 0, "x")
    assert crashed.status is EvalStatus.ERROR and crashed.outcome is None  # never counted as a model failure
    with pytest.raises(ValueError):
        feedback_result()


def test_evidence_accumulates_fractional_outcomes() -> None:
    ev = EvidenceStore(Store(":memory:"))
    store = ev.store
    store.execute(
        "INSERT INTO model(provider, model_id, deployment, first_seen, last_seen) VALUES ('p','m','hosted',0,0)"
    )
    store.execute("INSERT INTO profile(model_ref, profile_version, status, created_at) VALUES (1,1,'cold',0)")
    ev.record(1, "math", "identity", "1", 1.0)
    ev.record(1, "math", "identity", "1", 0.0)
    ev.record(1, "math", "identity", "1", 0.5)
    arm = ev.arms(1, "math")[("identity", "1")]
    assert (arm.alpha, arm.beta, arm.n_trials) == (2.5, 2.5, 3)
    with pytest.raises(ValueError):
        ev.record(1, "math", "identity", "1", 1.5)


def test_policy_needs_strong_evidence_to_leave_identity() -> None:
    reg = StrategyRegistry()
    cands = [reg.get("identity"), reg.get("role_framing")]
    pol = Policy(PolicyConfig(), random.Random(0))
    # no evidence -> identity
    assert pol.choose(cands, {}, can_explore=False, exploit_allowed=True).strategy.info.id == "identity"
    # clear evidence -> switch
    strong = {
        ("identity", "1"): ArmStats("identity", "1", 1 + 6, 1 + 34, 40, n_eff=40),
        ("role_framing", "1"): ArmStats("role_framing", "1", 1 + 30, 1 + 10, 40, n_eff=40),
    }
    d = pol.choose(cands, strong, can_explore=False, exploit_allowed=True)
    assert d.strategy.info.id == "role_framing" and d.p_better is not None and d.p_better >= 0.9
    # weak evidence (3 vs 2 successes) -> stay on identity
    weak = {
        ("identity", "1"): ArmStats("identity", "1", 1 + 2, 1 + 3, 5, n_eff=5),
        ("role_framing", "1"): ArmStats("role_framing", "1", 1 + 3, 1 + 2, 5, n_eff=5),
    }
    assert pol.choose(cands, weak, can_explore=False, exploit_allowed=True).strategy.info.id == "identity"
    # stale or missing profile -> identity even with strong evidence
    assert pol.choose(cands, strong, can_explore=False, exploit_allowed=False).strategy.info.id == "identity"


def test_policy_explores_only_when_allowed() -> None:
    reg = StrategyRegistry()
    cands = [reg.get("identity"), reg.get("role_framing")]
    pol = Policy(PolicyConfig(exploration_rate=1.0), random.Random(1))
    assert not pol.choose(cands, {}, can_explore=False, exploit_allowed=True).explored
    assert pol.choose(cands, {}, can_explore=True, exploit_allowed=True).explored


# ------------------------------------------------------------------ end to end (simulated)
def test_harness_learns_a_better_strategy_and_then_uses_it(tmp_path: Path) -> None:
    model = simulated_model(p_with_role=0.9, p_without=0.3)
    h = AdaptiveHarness(
        provider="mock", model="sim", adapter=model, config=cfg(tmp_path, exploration_rate=0.5), policy_seed=3
    )
    for _ in range(400):
        h.generate(CODE_PROMPT, check=ok)
    # A fresh client on the same store, no exploration, no check: it must now exploit what was learned.
    h2 = AdaptiveHarness(provider="mock", model="sim", adapter=model, config=cfg(tmp_path), policy_seed=4)
    r = h2.generate(CODE_PROMPT)
    assert r.trace.strategy_id == "role_framing", r.trace.reason
    assert r.trace.optimized and r.trace.p_better is not None and r.trace.p_better >= 0.9
    sent = model.requests[-1].messages
    assert sent[0] == Message("system", ROLE_LINE) and sent[-1] == Message("user", CODE_PROMPT)
    assert h2.store is not None
    assert h2.store.query_one("SELECT COUNT(*) AS n FROM trial WHERE simulated=0")["n"] == 0  # type: ignore[index]


def test_harness_keeps_the_original_prompt_when_nothing_is_better(tmp_path: Path) -> None:
    model = simulated_model(p_with_role=0.5, p_without=0.5)
    h = AdaptiveHarness(
        provider="mock", model="sim", adapter=model, config=cfg(tmp_path, exploration_rate=0.5), policy_seed=3
    )
    for _ in range(400):
        h.generate(CODE_PROMPT, check=ok)
    h2 = AdaptiveHarness(provider="mock", model="sim", adapter=model, config=cfg(tmp_path), policy_seed=4)
    r = h2.generate(CODE_PROMPT)
    assert r.trace.strategy_id == "identity" and not r.trace.optimized
    assert model.requests[-1].messages == (Message("user", CODE_PROMPT),)


def test_without_checks_nothing_is_learned(tmp_path: Path) -> None:
    h = AdaptiveHarness(provider="mock", model="sim", config=cfg(tmp_path, exploration_rate=1.0))
    for _ in range(20):
        r = h.generate(CODE_PROMPT)  # no check: no exploration and no evidence
        assert r.trace.strategy_id == "identity" and not r.trace.explored
    assert h.store is not None
    assert h.store.query_one("SELECT COUNT(*) AS n FROM arm_stats")["n"] == 0  # type: ignore[index]


def test_feedback_counts_once_per_response(tmp_path: Path) -> None:
    h = AdaptiveHarness(provider="mock", model="sim", config=cfg(tmp_path))
    r = h.generate(CODE_PROMPT)
    assert h.feedback(r.id, success=True) == "success"
    assert h.feedback(r.id, success=False) == "failure"  # stored, but not counted a second time
    prof = h.profile()
    assert prof is not None and prof["evidence"][0]["n_trials"] == 1
    assert prof["evidence"][0]["alpha"] == 2.0
    with pytest.raises(KeyError):
        h.feedback("nope", success=True)


def test_crashing_check_is_recorded_but_not_counted(tmp_path: Path) -> None:
    h = AdaptiveHarness(provider="mock", model="sim", config=cfg(tmp_path))
    r = h.generate(CODE_PROMPT, check=lambda t: 1 / 0)
    assert r.trace.evaluation_status == "evaluation_error"
    assert h.store is not None
    assert h.store.query_one("SELECT COUNT(*) AS n FROM arm_stats")["n"] == 0  # type: ignore[index]


def test_failing_or_unfaithful_strategy_falls_back_to_the_original(tmp_path: Path) -> None:
    class Paraphrase(Strategy):
        info: ClassVar[StrategyInfo] = StrategyInfo("paraphrase", "1", "structural", "d", "u", "l")

        def transform(self, messages, spec):  # type: ignore[no-untyped-def]
            return self._result([Message("user", "something else entirely")])

    class Broken(Strategy):
        info: ClassVar[StrategyInfo] = StrategyInfo("broken", "1", "additive", "d", "u", "l")

        def transform(self, messages, spec):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    for bad in (Paraphrase(), Broken()):
        reg = StrategyRegistry((BUILTIN[0], bad))
        model = MockProvider()
        h = AdaptiveHarness(
            provider="mock",
            model="sim",
            adapter=model,
            strategies=reg,
            config=cfg(tmp_path / bad.info.id, exploration_rate=1.0),
            policy_seed=0,
        )
        for _ in range(10):
            r = h.generate(CODE_PROMPT, check=ok)
            assert model.requests[-1].messages[-1].content == CODE_PROMPT
            assert r.trace.strategy_id == "identity"


def test_optimize_false_never_changes_the_prompt(tmp_path: Path) -> None:
    c = HarnessConfig(
        optimize=False,
        persistence=PersistenceConfig(path=tmp_path / "h.db"),
        policy=PolicyConfig(exploration_rate=1.0),
    )
    model = MockProvider()
    h = AdaptiveHarness(provider="mock", model="sim", adapter=model, config=c)
    for _ in range(10):
        r = h.generate(CODE_PROMPT, check=ok)
        assert r.trace.strategy_id == "identity" and model.requests[-1].messages == (
            Message("user", CODE_PROMPT),
        )


def test_the_default_config_learns_and_then_stops_exploring(tmp_path: Path) -> None:
    """Regression (evaluation F3): with defaults the harness used to send identity forever, whatever the checks."""
    import random as _random

    from adaptive_harness import AdaptiveHarness, HarnessConfig, PersistenceConfig
    from adaptive_harness.providers import MockProvider

    role = "You are an experienced software engineer."
    rng = _random.Random(4)
    model = MockProvider(
        lambda r: (
            "ok"
            if rng.random()
            < (0.95 if any(role in m.content for m in r.messages if m.role == "system") else 0.3)
            else "no"
        )
    )
    h = AdaptiveHarness(
        provider="mock", model="m", adapter=model,
        config=HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db")), policy_seed=3,
    )  # fmt: skip
    prompt = "Write a Python function that checks whether a number is prime."
    traces = [h.generate(prompt, check=lambda t: t == "ok").trace for _ in range(2500)]
    assert h.generate(prompt).trace.strategy_id == "role_framing"  # learned with the default settings
    assert 0 < sum(t.explored for t in traces) < 0.1 * len(traces)
    assert not any(h.generate(prompt).trace.explored for _ in range(50))  # never without a check

    spec = h.analyzer.analyze(prompt)
    target = h.config.policy.exploration_target

    def all_sampled() -> bool:
        pid, _ = h._current_profile()
        assert h.evidence is not None and pid is not None
        arms = h.evidence.arms(pid, spec.category)
        keys = [(c.info.id, c.info.version) for c in h.strategies.candidates(spec)]
        return all(k in arms and arms[k].n_eff >= target for k in keys)

    for _ in range(6000):
        if all_sampled():
            break
        h.generate(prompt, check=lambda t: t == "ok")
    assert all_sampled()  # exploration stops once every arm has enough data
    assert not any(h.generate(prompt, check=lambda t: t == "ok").trace.explored for _ in range(200))
