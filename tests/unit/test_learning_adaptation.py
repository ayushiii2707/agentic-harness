"""Phase 10: evidence decay, hierarchical backoff, prior evidence, drift detection and the cost term."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from adaptive_harness import AdaptiveHarness, HarnessConfig, PersistenceConfig, PolicyConfig
from adaptive_harness.calibration import Calibrator
from adaptive_harness.learning import LEBLANC_V4, ArmStats, EvidenceStore, Policy, check_drift, load_priors
from adaptive_harness.providers import MockProvider
from adaptive_harness.store import Store
from adaptive_harness.strategies import StrategyRegistry

DAY = 86400.0


def _store() -> Store:
    s = Store(":memory:")
    s.execute(
        "INSERT INTO model(provider, model_id, deployment, first_seen, last_seen) VALUES ('p','m','hosted',0,0)"
    )
    s.execute("INSERT INTO profile(model_ref, profile_version, status, created_at) VALUES (1,1,'active',0)")
    return s


# ------------------------------------------------------------------ decay
def test_old_evidence_halves_per_half_life_and_must_be_reconfirmed() -> None:
    ev = EvidenceStore(_store(), half_life_s=10 * DAY)
    for i in range(40):
        ev.record(1, "math", "identity", "1", float(i % 2), now=0.0)
    now = ev.arms(1, "math", now=0.0)[("identity", "1")]
    later = ev.arms(1, "math", now=10 * DAY)[("identity", "1")]
    assert now.n_eff == pytest.approx(40) and later.n_eff == pytest.approx(20)
    assert later.alpha - 1 == pytest.approx((now.alpha - 1) / 2) and later.n_trials == 40
    # a new outcome is added to the decayed counts, not the raw ones
    ev.record(1, "math", "identity", "1", 1.0, now=10 * DAY)
    assert ev.arms(1, "math", now=10 * DAY)[("identity", "1")].n_eff == pytest.approx(21)


def test_policy_stops_exploiting_when_evidence_has_decayed() -> None:
    reg = StrategyRegistry()
    cands = [reg.get("identity"), reg.get("role_framing")]
    ev = EvidenceStore(_store(), half_life_s=10 * DAY)
    for i in range(40):
        ev.record(1, "math", "identity", "1", float(i < 8), now=0.0)
        ev.record(1, "math", "role_framing", "1", float(i < 32), now=0.0)
    pol = Policy(PolicyConfig(cost_weight=0.0), random.Random(0))
    fresh = pol.choose(cands, ev.arms(1, "math", now=0.0), can_explore=False, exploit_allowed=True)
    stale = pol.choose(cands, ev.arms(1, "math", now=10 * DAY), can_explore=False, exploit_allowed=True)
    assert fresh.strategy.info.id == "role_framing"
    assert stale.strategy.info.id == "identity" and "decayed" in stale.reason


# ------------------------------------------------------------------ backoff
def test_backoff_borrows_a_capped_prior_but_never_counts_it_as_own_trials() -> None:
    ev = EvidenceStore(_store())
    for _ in range(50):
        ev.record(1, "code_generation", "identity", "1", 1.0)
    ev.record(1, "api_development", "identity", "1", 0.0)
    child = ev.arms(1, "api_development", parent="code_generation")[("identity", "1")]
    assert child.n_eff == pytest.approx(1)  # the gate sees only its own outcome
    assert child.alpha == pytest.approx(1 + 4 * (51 / 52))  # parent mean, weight capped at 4
    assert child.sources == ("parent:code_generation",)
    # no parent: the model-global pool of other categories, weight capped at 2
    glob = ev.arms(1, "api_development")[("identity", "1")]
    assert glob.sources == ("model-global",) and glob.alpha + glob.beta == pytest.approx(3 + 2)
    # nothing to borrow: the plain posterior
    alone = EvidenceStore(_store())
    alone.record(1, "math", "identity", "1", 1.0)
    assert alone.arms(1, "math")[("identity", "1")].sources == ()


# ------------------------------------------------------------------ prior evidence
def test_leblanc_priors_apply_only_to_the_tested_model_and_category() -> None:
    store = _store()
    assert load_priors(store) == len(LEBLANC_V4) == 6
    assert load_priors(store) == 0  # idempotent
    ev = EvidenceStore(store)
    ev.record(1, "security_sensitive_code", "security_generic", "1", 1.0)
    ev.record(1, "math", "security_generic", "1", 1.0)

    def sources(model: str, category: str) -> tuple[str, ...]:
        arm = ev.arms(1, category, model_id=model, host="groq")[("security_generic", "1")]
        return tuple(s for s in arm.sources if s.startswith("prior:"))

    assert sources("openai/gpt-oss-20b", "security_sensitive_code")  # vendor prefix ignored
    assert not sources("gpt-oss-120b", "security_sensitive_code")  # another model
    assert not sources("openai/gpt-oss-20b", "math")  # another category
    arm = ev.arms(1, "security_sensitive_code", model_id="deepseek-v4.1-flash")[("security_generic", "1")]
    # host unknown -> half of the capped weight: 1.5 pseudo-observations at 44.8%; plus the model-global prior,
    # whose weight (1) is capped by the single outcome it comes from
    assert arm.n_eff == pytest.approx(1)
    assert arm.alpha + arm.beta == pytest.approx(3 + 1.5 + 1)


def test_priors_alone_never_switch_the_policy(tmp_path: Path) -> None:
    model = MockProvider(lambda r: "ok")
    h = AdaptiveHarness(
        provider="mock",
        model="deepseek-v4.1-flash",
        adapter=model,
        config=HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db")),
    )
    prompt = "Write a Flask login endpoint that stores passwords securely and prevents SQL injection."
    assert h.analyzer.analyze(prompt).category == "security_sensitive_code"
    for _ in range(3):
        r = h.generate(prompt, check=lambda t: True)
        assert r.trace.strategy_id == "identity"


# ------------------------------------------------------------------ drift
def test_drift_statistic_alarms_on_a_shift_and_not_on_noise() -> None:
    rng = random.Random(1)
    steady = [float(rng.random() < 0.7) for _ in range(500)]
    assert not any(check_drift(steady[:i], window=30, reference=120, z_crit=4.0).drifted for i in range(500))
    shifted = [1.0] * 120 + [0.0] * 30
    assert check_drift(shifted, window=30, reference=120, z_crit=4.0).drifted
    assert not check_drift([1.0] * 40, window=30, reference=120, z_crit=4.0).drifted  # too few outcomes


def test_drift_marks_the_profile_stale_and_recalibration_starts_a_new_version(tmp_path: Path) -> None:
    state = {"good": True}
    model = MockProvider(lambda r: "ok" if state["good"] else "bad")
    h = AdaptiveHarness(
        provider="mock",
        model="m",
        adapter=model,
        config=HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db")),
    )
    prompt = "What is 17 times 23?"
    for _ in range(120):
        h.generate(prompt, check=lambda t: t == "ok")
    first, status = h._current_profile()
    assert status != "stale"
    state["good"] = False
    for _ in range(30):
        h.generate(prompt, check=lambda t: t == "ok")
    pid, status = h._current_profile()
    assert (pid, status) == (first, "stale")
    profile = h.profile()
    assert profile is not None and [e["kind"] for e in profile["change_events"]] == ["drift"]
    r = h.generate(prompt, check=lambda t: True)
    assert r.trace.strategy_id == "identity" and "no current profile evidence" in r.trace.reason

    report = Calibrator(h, max_calls=2, max_cost_per_call_usd=0.0).run(categories=("math",))
    new_pid, new_status = h._current_profile()
    assert new_pid != first and new_status == "active"
    assert any("was stale" in n for n in report.notes)
    assert h.store is not None
    row = h.store.query_one("SELECT profile_version FROM profile WHERE id=?", (new_pid,))
    assert row is not None and row["profile_version"] == 2


# ------------------------------------------------------------------ cost term
def test_cost_penalty_raises_the_bar_for_token_hungry_strategies() -> None:
    reg = StrategyRegistry()
    cands = [reg.get("identity"), reg.get("role_framing")]
    arms = {
        ("identity", "1"): ArmStats("identity", "1", 1 + 60, 1 + 40, 100, n_eff=100, mean_tokens=500),
        ("role_framing", "1"): ArmStats(
            "role_framing", "1", 1 + 80, 1 + 20, 100, n_eff=100, mean_tokens=2000
        ),
    }
    cheap = Policy(PolicyConfig(cost_weight=0.0), random.Random(0))
    priced = Policy(PolicyConfig(cost_weight=0.1), random.Random(0))
    assert priced.cost_penalty(arms[("role_framing", "1")], arms[("identity", "1")]) == pytest.approx(0.3)
    assert (
        cheap.choose(cands, arms, can_explore=False, exploit_allowed=True).strategy.info.id == "role_framing"
    )
    # +20 pp is not worth 4x the tokens at lambda = 0.1 (needs > 2 + 30 pp)
    assert priced.choose(cands, arms, can_explore=False, exploit_allowed=True).strategy.info.id == "identity"
    # a cheaper strategy gets no bonus: the penalty is never negative
    cheaper = ArmStats("role_framing", "1", 1, 1, 0, mean_tokens=100)
    assert priced.cost_penalty(cheaper, arms[("identity", "1")]) == 0.0


def test_token_summaries_are_recorded_through_the_harness(tmp_path: Path) -> None:
    h = AdaptiveHarness(
        provider="mock",
        model="m",
        adapter=MockProvider(lambda r: "ok"),
        config=HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db")),
    )
    prompt = "What is 17 times 23?"
    h.generate(prompt, check=lambda t: True)
    pid, _ = h._current_profile()
    assert h.evidence is not None and pid is not None
    arm = h.evidence.arms(pid, h.analyzer.analyze(prompt).category)[("identity", "1")]
    assert arm.mean_tokens is not None and arm.mean_tokens > 0
