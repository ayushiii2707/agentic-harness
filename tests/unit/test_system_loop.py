"""End-to-end lifecycle of the learning loop (simulated model; mechanism only, never model evidence).

One store, several harness instances (process restarts):
cold start -> exploration on checked calls -> evidence -> exploitation after restart -> unchecked calls change
nothing -> feedback counted once -> broken checks never count -> the model changes -> drift -> the old profile is
stale and a new profile version starts with no inherited evidence -> identity -> recalibration fills it.
"""

from __future__ import annotations

import random
from pathlib import Path

from adaptive_harness import AdaptiveHarness, HarnessConfig, PersistenceConfig, PolicyConfig
from adaptive_harness.calibration import Calibrator
from adaptive_harness.providers import MockProvider
from adaptive_harness.types import CompletionRequest

PROMPT = "Write a Python function that checks whether a number is prime."
ROLE_LINE = "You are an experienced software engineer."


def test_full_learning_lifecycle(tmp_path: Path) -> None:
    db = tmp_path / "h.db"
    rng = random.Random(7)
    world = {"broken": False}

    def respond(r: CompletionRequest) -> str:
        if world["broken"]:
            return "no"
        role = any(ROLE_LINE in m.content for m in r.messages if m.role == "system")
        return "ok" if rng.random() < (0.9 if role else 0.4) else "no"

    model = MockProvider(respond)

    def harness(**policy: float) -> AdaptiveHarness:
        cfg = HarnessConfig(persistence=PersistenceConfig(path=db), policy=PolicyConfig(**policy))
        return AdaptiveHarness(provider="mock", model="sim", adapter=model, config=cfg, policy_seed=1)

    def counts(h: AdaptiveHarness) -> dict[str, int]:
        pid, _ = h._current_profile()
        assert h.evidence is not None and pid is not None
        cat = h.analyzer.analyze(PROMPT).category
        return {k[0]: a.n_trials for k, a in h.evidence.arms(pid, cat).items()}

    # 1. cold start: the original prompt, unchanged
    h = harness()
    r = h.generate(PROMPT)
    assert r.trace.strategy_id == "identity" and model.requests[-1].messages[-1].content == PROMPT
    assert h.profile() is not None

    # 2. exploration happens only on checked calls, and only those create evidence
    h = harness(exploration_rate=0.6)
    for _ in range(20):
        assert not h.generate(PROMPT).trace.explored
    assert counts(h) == {}
    for _ in range(400):
        h.generate(PROMPT, check=lambda t: t == "ok")
    learned = counts(h)
    assert sum(learned.values()) == 400 and learned["identity"] >= 30 and learned["role_framing"] >= 30

    # 3. restart, no exploration: the better strategy is used; the user's text is still sent verbatim
    h = harness()
    r = h.generate(PROMPT)
    assert r.trace.strategy_id == "role_framing" and r.trace.p_better is not None
    assert model.requests[-1].messages[-1].content == PROMPT
    assert counts(h) == learned  # an unchecked call changes nothing

    # 4. feedback counts once; a crashing check never counts as a model failure
    assert h.feedback(r.id, success=True)
    h.feedback(r.id, success=False)
    after = counts(h)
    assert after["role_framing"] == learned["role_framing"] + 1
    h.generate(PROMPT, check=lambda t: 1 / 0)  # type: ignore[arg-type,return-value]
    assert counts(h) == after

    # 5. keep serving with checks, then the model changes: drift -> old profile stale, new version, identity
    first, _ = h._current_profile()
    for _ in range(80):
        h.generate(PROMPT, check=lambda t: t == "ok")
    world["broken"] = True
    for _ in range(40):
        h.generate(PROMPT, check=lambda t: t == "ok")
    assert h.store is not None
    old = h.store.query_one("SELECT status FROM profile WHERE id=?", (first,))
    assert old is not None and old["status"] == "stale"
    assert "drift" in [
        e["kind"] for e in h.store.query("SELECT kind FROM change_event WHERE profile_id=?", (first,))
    ]
    second, status = h._current_profile()
    assert second != first and status == "cold"
    pid = second
    cat = h.analyzer.analyze(PROMPT).category
    assert h.evidence is not None
    assert all(
        a.alpha == 1.0 for a in h.evidence.arms(pid, cat).values()
    )  # only post-drift failures, none inherited
    r = harness().generate(PROMPT)
    assert r.trace.strategy_id == "identity"

    # 6. recalibration fills the new version
    world["broken"] = False
    h = harness()
    Calibrator(h, max_calls=4, max_cost_per_call_usd=0.0).run(categories=("math",))
    pid, status = h._current_profile()
    assert pid == second and status == "active"
    assert h.ledger is not None and h.ledger.spent_usd() == 0.0
