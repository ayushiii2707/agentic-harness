"""The instruction-following suite: every task is satisfiable, and each kind of violation is caught."""

from __future__ import annotations

from adaptive_harness.calibration import suite
from adaptive_harness.calibration.suites import instructions_suite
from adaptive_harness.evaluation import EvalStatus
from adaptive_harness.evaluation.builtin import BulletConstraints


def reference(c: BulletConstraints) -> str:
    """A correct answer built from the constraints alone."""
    placed = {i: w for w, i in c.placed}
    bullets = [
        "- " + " ".join([placed.get(i, "tip"), *["x"] * (c.exact_words - 1)])
        for i in range(1, c.n_bullets + 1)
    ]
    return "\n".join([*bullets, c.last_line])


def test_every_task_has_a_passing_answer_and_splits_are_disjoint() -> None:
    cal, held = instructions_suite(split="calibration"), instructions_suite(split="heldout")
    assert len(cal) == len(held) == 60 and not ({t.prompt for t in cal} & {t.prompt for t in held})
    for t in cal + held:
        c = t.evaluator
        assert isinstance(c, BulletConstraints)
        assert c.evaluate(reference(c)).status is EvalStatus.SUCCESS, t.id
    assert [t.id for t in suite("instructions")] == [t.id for t in cal]


def test_each_violation_fails() -> None:
    c = BulletConstraints(3, 3, (("helmet", 1), ("route", 2)), ("very",), "END")
    ok = "- Wear a helmet\n- Plan your route\n- Ride safely please\nEND"
    assert c.evaluate(ok).status is EvalStatus.SUCCESS
    bad = {
        "two bullets": "- Wear a helmet\n- Plan your route\nEND",
        "wrong marker": "* Wear a helmet\n- Plan your route\n- Ride safely please\nEND",
        "word count": "- Wear a helmet daily\n- Plan your route\n- Ride safely please\nEND",
        "wrong bullet": "- Plan your route\n- Wear a helmet\n- Ride safely please\nEND",
        "forbidden word": "- Wear a helmet\n- Plan your route\n- Ride very safely\nEND",
        "no final line": "- Wear a helmet\n- Plan your route\n- Ride safely please",
        "extra text": "Here you go:\n- Wear a helmet\n- Plan your route\n- Ride safely please\nEND",
        "substring is not the word": "- Wear a helmets\n- Plan your route\n- Ride safely please\nEND",
    }
    for why, text in bad.items():
        assert c.evaluate(text).status is EvalStatus.FAILURE, why


def test_all_four_splits_are_disjoint_and_heldout_is_unchanged() -> None:
    from adaptive_harness.calibration.suites import json_suite, math_suite

    for build in (math_suite, json_suite, instructions_suite):
        splits = {
            s: {t.prompt for t in build(split=s)} for s in ("calibration", "heldout", "examples", "pilot")
        }  # type: ignore[operator]
        names = sorted(splits)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                assert not (splits[a] & splits[b]), (build.__name__, a, b)
    # the held-out set used by experiment v1 must never change (content hash recorded 2026-11-11)
    import hashlib

    text = "".join(
        t.prompt for build in (math_suite, json_suite, instructions_suite) for t in build(split="heldout")
    )  # type: ignore[operator]
    assert hashlib.sha256(text.encode()).hexdigest()[:16] == "8a7227aca87eb490"
