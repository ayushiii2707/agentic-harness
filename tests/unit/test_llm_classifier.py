"""The opt-in LLM fallback classifier: asked only for `general` prompts; any failure keeps `general`."""

from __future__ import annotations

from pathlib import Path

from adaptive_harness import AdaptiveHarness, BudgetConfig, ClassifierConfig, HarnessConfig, PersistenceConfig
from adaptive_harness.providers import MockProvider
from adaptive_harness.providers.base import ProviderCapabilities
from adaptive_harness.tasks.llm_classifier import parse_category

VAGUE = "Help me figure out what to pack for a week in Iceland in March."
CLEAR = "Write a Python function that returns the n-th Fibonacci number using memoization."


class PaidMock(MockProvider):
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(local=False)


def harness(tmp_path: Path, clf: MockProvider, budget: float = 1.0) -> AdaptiveHarness:
    return AdaptiveHarness(
        provider="mock",
        model="t",
        adapter=MockProvider(lambda r: "ok"),
        config=HarnessConfig(
            persistence=PersistenceConfig(path=tmp_path / "h.db"),
            budget=BudgetConfig(total_usd=budget),
            llm_classifier=ClassifierConfig("mock", "clf"),
        ),
        aux_adapters={"classifier": clf},
    )


def test_only_general_prompts_are_sent_and_the_answer_is_cached(tmp_path: Path) -> None:
    clf = PaidMock(lambda r: '{"category": "question_answering"}')
    h = harness(tmp_path, clf)
    assert h.analyzer.analyze(CLEAR).category == "code_generation" and clf.requests == []
    for _ in range(3):
        spec = h.analyzer.analyze(VAGUE)
        assert spec.category == "question_answering"
        assert spec.classification.evidence["question_answering"] == ("llm:clf",)
    assert len(clf.requests) == 1
    system = clf.requests[0].messages[0].content
    assert VAGUE not in system and VAGUE in clf.requests[0].messages[-1].content


def test_bad_answers_and_refusals_keep_general(tmp_path: Path) -> None:
    for reply in ("I think it's travel.", '{"category": "travel_planning"}', '{"category": 3}', "{oops"):
        h = harness(tmp_path / str(abs(hash(reply))), PaidMock([reply]))
        assert h.analyzer.analyze(VAGUE).category == "general", reply
    refused = PaidMock(['{"category": "question_answering"}'])
    h = harness(tmp_path / "zero", refused, budget=0.0)
    assert h.analyzer.analyze(VAGUE).category == "general" and refused.requests == []


def test_parse_category() -> None:
    known = {"math", "general"}
    assert parse_category('```json\n{"category": "math"}\n```', known) == "math"
    assert parse_category('{"category": "algebra"}', known) is None


def test_labels_persist_across_processes_and_failures_are_not_remembered(tmp_path: Path) -> None:
    first = PaidMock(['{"category": "question_answering"}'])
    assert harness(tmp_path, first).analyzer.analyze(VAGUE).category == "question_answering"
    later = PaidMock(
        []
    )  # a new process: any call would fail (empty script), so the label must come from the store
    assert (
        harness(tmp_path, later).analyzer.analyze(VAGUE).category == "question_answering"
        and later.requests == []
    )
    # an outage is not stored: the next process asks again
    down = tmp_path / "down"
    assert harness(down, PaidMock([])).analyzer.analyze(VAGUE).category == "general"
    retry = PaidMock(['{"category": "research"}'])
    assert harness(down, retry).analyzer.analyze(VAGUE).category == "research" and len(retry.requests) == 1
