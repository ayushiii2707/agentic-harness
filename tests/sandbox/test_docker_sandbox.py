"""Real Docker tests. Skipped (not faked) when Docker or the pinned image is unavailable."""

from __future__ import annotations

import pytest

from adaptive_harness.evaluation import EvalStatus
from adaptive_harness.evaluation.builtin import UnitTests
from adaptive_harness.sandbox import DockerSandbox, selfcheck

SANDBOX = DockerSandbox()
AVAILABLE, WHY = SANDBOX.available()
pytestmark = [pytest.mark.sandbox, pytest.mark.skipif(not AVAILABLE, reason=f"sandbox unavailable: {WHY}")]

TEST = "def check(f):\n    assert f(3) == 9\n"


def test_pass_and_fail() -> None:
    assert SANDBOX.run("def sq(x):\n    return x * x\n", TEST, "sq").status == "pass"
    r = SANDBOX.run("def sq(x):\n    return x + x\n", TEST, "sq")
    assert r.status == "fail" and "AssertionError" in r.detail


def test_missing_entry_and_broken_code_are_model_failures() -> None:
    assert SANDBOX.run("def other(x):\n    return x\n", TEST, "sq").status == "fail"
    assert SANDBOX.run("def sq(x)\n    return x\n", TEST, "sq").status == "fail"


def test_missing_third_party_module_is_an_environment_error() -> None:
    assert (
        SANDBOX.run("import not_a_real_module_xyz\ndef sq(x):\n    return x\n", TEST, "sq").status
        == "env_error"
    )


@pytest.mark.parametrize("probe", selfcheck.PROBES, ids=lambda p: p.name)
def test_isolation_and_abuse_probes(probe: selfcheck.Probe) -> None:
    r = SANDBOX.run(probe.code, selfcheck.T_OK, "f", probe.timeout_s)
    assert r.status == probe.expected or (probe.expected == "fail" and r.status == "timeout"), r


def test_unit_tests_evaluator_end_to_end() -> None:
    ev = UnitTests(TEST, "sq", sandbox=SANDBOX)
    assert ev.evaluate("```python\ndef sq(x):\n    return x ** 2\n```").status is EvalStatus.SUCCESS
    assert ev.evaluate("```python\ndef sq(x):\n    return 0\n```").status is EvalStatus.FAILURE


def test_missing_image_is_an_infrastructure_error() -> None:
    sb = DockerSandbox(image="adaptive-harness/does-not-exist:never")
    assert sb.run("def sq(x):\n    return x\n", TEST, "sq").status == "infra_error"
