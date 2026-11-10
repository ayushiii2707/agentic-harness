from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from adaptive_harness import Message
from adaptive_harness.strategies import BUILTIN, Strategy, StrategyInfo, StrategyRegistry
from adaptive_harness.tasks import TaskAnalyzer

PROMPTS = Path(__file__).resolve().parents[2] / "data" / "task_classification" / "prompts.jsonl"
ANALYZER = TaskAnalyzer()


def prompts() -> list[str]:
    return [
        json.loads(line)["prompt"]
        for line in PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_builtin_ids_versions_and_hashes_are_unique() -> None:
    keys = [(s.info.id, s.info.version) for s in BUILTIN]
    assert len(keys) == len(set(keys))
    assert len({s.content_hash for s in BUILTIN}) == len(BUILTIN)
    assert {s.info.kind for s in BUILTIN} <= {"identity", "additive", "structural", "rewriting"}
    # rewriting strategies are gated by the preservation check (tests/unit/test_transform.py)
    assert [s.info.id for s in BUILTIN if s.info.kind == "rewriting"] == ["minimal_context"]


def test_identity_is_unchanged() -> None:
    msgs = [Message("system", "be kind"), Message("user", "hi")]
    r = StrategyRegistry().get("identity").transform(msgs, ANALYZER.analyze("hi"))
    assert list(r.messages) == msgs and r.added_text == ()


@pytest.mark.parametrize(
    "strategy", [s for s in BUILTIN if s.info.kind != "rewriting"], ids=lambda s: s.info.id
)
def test_user_text_and_user_system_are_kept_verbatim_on_every_dataset_prompt(strategy: Strategy) -> None:
    user_system = Message("system", "Use British spelling.")
    applied = 0
    for p in prompts():
        spec = ANALYZER.analyze(p)
        if not strategy.applicable(spec):
            continue
        r = strategy.transform([user_system, Message("user", p)], spec)
        applied += 1
        users = [m.content for m in r.messages if m.role == "user"]
        assert len(users) == 1 and p in users[0], f"{strategy.info.id} lost the user text"
        assert user_system in r.messages, f"{strategy.info.id} changed the user's system message"
        assert r.messages[-1].role == "user"
        assert (r.strategy_id, r.strategy_version) == (strategy.info.id, strategy.info.version)
        for added in r.added_text:  # everything the strategy added is visible in the messages
            assert any(added in m.content for m in r.messages)
    assert applied > 0, f"{strategy.info.id} applied to no dataset prompt"


def test_general_category_only_offers_general_safe_strategies() -> None:
    spec = ANALYZER.analyze("hi")
    assert spec.category == "general"
    ids = [s.info.id for s in StrategyRegistry().candidates(spec)]
    assert ids[0] == "identity"
    assert all(StrategyRegistry().get(i).info.general_safe for i in ids)


def test_candidates_depend_on_category_and_requirements() -> None:
    reg = StrategyRegistry()
    sec = [
        s.info.id
        for s in reg.candidates(ANALYZER.analyze("Write a secure login function that hashes passwords."))
    ]
    assert "security_generic" in sec and sec[0] == "identity"
    js = [s.info.id for s in reg.candidates(ANALYZER.analyze("Extract the names and return them as JSON."))]
    assert "output_schema" in js
    plain = [s.info.id for s in reg.candidates(ANALYZER.analyze("Who wrote Pride and Prejudice?"))]
    assert "output_schema" not in plain and "requirements_checklist" not in plain


def test_not_applicable_transform_raises() -> None:
    reg = StrategyRegistry()
    spec = ANALYZER.analyze("Who wrote Pride and Prejudice?")
    with pytest.raises(ValueError):
        reg.get("output_schema").transform([Message("user", "Who wrote Pride and Prejudice?")], spec)


def test_reregistering_changed_content_requires_a_new_version() -> None:
    class Custom(Strategy):
        info: ClassVar[StrategyInfo] = StrategyInfo("custom", "1", "additive", "d", "u", "l")
        templates: ClassVar[dict[str, str]] = {"system": "A"}

        def transform(self, messages, spec):  # type: ignore[no-untyped-def]
            return self._result(messages)

    class Changed(Custom):
        templates: ClassVar[dict[str, str]] = {"system": "B"}

    reg = StrategyRegistry()
    reg.register(Custom())
    reg.register(Custom())  # identical content is fine
    with pytest.raises(ValueError):
        reg.register(Changed())


def test_changed_wording_under_the_same_version_is_disabled_against_stored_evidence(tmp_path: Path) -> None:
    from adaptive_harness.store import Store

    class V1(Strategy):
        info: ClassVar[StrategyInfo] = StrategyInfo("custom", "1", "additive", "d", "u", "l")
        templates: ClassVar[dict[str, str]] = {"system": "Old wording."}

        def transform(self, messages, spec):  # type: ignore[no-untyped-def]
            return self._result(messages)

    class V1Edited(V1):
        templates: ClassVar[dict[str, str]] = {"system": "New wording."}

    store = Store(tmp_path / "h.db")
    assert StrategyRegistry((BUILTIN[0], V1())).sync_with_store(store) == []
    reg = StrategyRegistry((BUILTIN[0], V1Edited()))
    assert reg.sync_with_store(store) == ["custom@1"]
    assert [s.info.id for s in reg.all()] == ["identity"]
