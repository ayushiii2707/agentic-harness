from __future__ import annotations

from pathlib import Path

from adaptive_harness import AdaptiveHarness, HarnessConfig, PersistenceConfig
from adaptive_harness.evaluation import EvalStatus, run_check
from adaptive_harness.evaluation.builtin import (
    JsonOutput,
    PythonSyntax,
    ReferenceAnswer,
    UnitTests,
    extract_code,
)
from adaptive_harness.providers import MockProvider
from adaptive_harness.sandbox import DockerSandbox


def test_extract_code_prefers_tagged_python_block() -> None:
    text = "Here:\n```json\n{}\n```\nand\n```python\ndef f():\n    return 1\n```"
    assert extract_code(text) == "def f():\n    return 1"
    assert extract_code("no code") is None


def test_python_syntax() -> None:
    assert PythonSyntax().evaluate("```python\nx = 1\n```").status is EvalStatus.SUCCESS
    assert PythonSyntax().evaluate("```python\ndef f(:\n```").status is EvalStatus.FAILURE
    assert PythonSyntax().evaluate("just text").status is EvalStatus.FAILURE


def test_json_output() -> None:
    assert JsonOutput(("name",)).evaluate('{"name": "x"}').status is EvalStatus.SUCCESS
    assert JsonOutput(("name",)).evaluate('```json\n{"name": 1}\n```').status is EvalStatus.SUCCESS
    assert JsonOutput(("name",)).evaluate('{"other": 1}').status is EvalStatus.FAILURE
    assert JsonOutput().evaluate("not json").status is EvalStatus.FAILURE


def test_reference_answer_numeric_and_text() -> None:
    assert ReferenceAnswer(391).evaluate("17 * 23 = 391").status is EvalStatus.SUCCESS
    assert ReferenceAnswer(391).evaluate("about 1,000").status is EvalStatus.FAILURE
    assert ReferenceAnswer(0.5).evaluate("The answer is 0.5.").status is EvalStatus.SUCCESS
    assert ReferenceAnswer("Canberra").evaluate(" canberra. ").status is EvalStatus.SUCCESS


def test_unit_tests_report_infrastructure_problems_as_errors() -> None:
    ev = UnitTests(
        "def check(f):\n    assert f() == 1\n", "f", sandbox=DockerSandbox(docker="/nonexistent/docker")
    )
    r = ev.evaluate("```python\ndef f():\n    return 1\n```")
    assert r.status is EvalStatus.ERROR and r.outcome is None  # never counted against the model
    assert ev.evaluate("no code here").status is EvalStatus.FAILURE


def test_evaluators_plug_into_generate_as_check(tmp_path: Path) -> None:
    h = AdaptiveHarness(
        provider="mock",
        model="m",
        adapter=MockProvider(["The answer is 391."]),
        config=HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db")),
    )
    r = h.generate("What is 17 * 23?", check=ReferenceAnswer(391))
    assert r.trace.evaluation_status == "success"
    row = h.store.query_one("SELECT evaluator_id, status FROM evaluation")  # type: ignore[union-attr]
    assert row == {"evaluator_id": "reference_answer", "status": "success"}


def test_run_check_rejects_wrong_return_type_from_evaluator() -> None:
    class Bad:
        evaluator_id = "bad"

        def evaluate(self, text: str) -> object:
            return True

    r = run_check(Bad(), "x")  # type: ignore[arg-type]
    assert r.status is EvalStatus.ERROR and r.evaluator_id == "bad"
