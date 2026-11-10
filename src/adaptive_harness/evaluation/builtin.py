"""Built-in evaluators (DESIGN §5.6).

Each one turns a model answer into a typed ``EvaluationResult``. They can be passed as ``check=`` to
``generate``. Only use one as a *primary* outcome when it measures what the task needs: syntax or JSON validity
is a weak proxy for correctness (LeBlanc Finding C), whereas unit tests and reference answers measure the result.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from ..sandbox import DockerSandbox
from . import EvalStatus, EvaluationResult

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.S)


def extract_code(text: str, language: str = "python") -> str | None:
    """The first fenced block tagged with ``language`` (or untagged), else None."""
    for m in _FENCE.finditer(text or ""):
        tag, body = m.group(1), m.group(2)
        if tag.lower() in ("", language, "py" if language == "python" else language):
            return body.strip("\n")
    return None


@dataclass(frozen=True)
class PythonSyntax:
    """Success when the answer contains a fenced Python block that compiles. A weak, structural check."""

    evaluator_id: str = "python_syntax"
    version: str = "1"

    def evaluate(self, text: str) -> EvaluationResult:
        code = extract_code(text)
        if code is None:
            return EvaluationResult(
                EvalStatus.FAILURE, self.evaluator_id, self.version, details="no python block"
            )
        try:
            compile(code, "answer.py", "exec")
        except (SyntaxError, ValueError) as e:
            return EvaluationResult(EvalStatus.FAILURE, self.evaluator_id, self.version, details=f"{e}"[:300])
        return EvaluationResult(EvalStatus.SUCCESS, self.evaluator_id, self.version)


@dataclass(frozen=True)
class JsonOutput:
    """Success when the answer is (or contains, in a fenced block) JSON that parses, with the required keys."""

    required_keys: tuple[str, ...] = ()
    evaluator_id: str = "json_output"
    version: str = "1"

    def evaluate(self, text: str) -> EvaluationResult:
        candidate = extract_code(text, "json")
        candidate = candidate if candidate is not None else (text or "").strip()
        try:
            value = json.loads(candidate)
        except ValueError as e:
            return EvaluationResult(
                EvalStatus.FAILURE, self.evaluator_id, self.version, details=f"not JSON: {e}"
            )
        missing = [k for k in self.required_keys if not (isinstance(value, dict) and k in value)]
        if missing:
            return EvaluationResult(
                EvalStatus.FAILURE, self.evaluator_id, self.version, details=f"missing keys {missing}"
            )
        return EvaluationResult(EvalStatus.SUCCESS, self.evaluator_id, self.version)


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", re.I)


@dataclass(frozen=True)
class ReferenceAnswer:
    """Compare with a known answer: the last number in the answer (numeric) or a normalised exact match."""

    expected: str | float
    rel_tol: float = 1e-6
    evaluator_id: str = "reference_answer"
    version: str = "1"

    def evaluate(self, text: str) -> EvaluationResult:
        if isinstance(self.expected, (int, float)):
            nums = _NUMBER.findall((text or "").replace(",", ""))
            if not nums:
                return EvaluationResult(
                    EvalStatus.FAILURE, self.evaluator_id, self.version, details="no number"
                )
            got = float(nums[-1])
            ok = math.isclose(got, float(self.expected), rel_tol=self.rel_tol, abs_tol=self.rel_tol)
            return EvaluationResult(
                EvalStatus.SUCCESS if ok else EvalStatus.FAILURE,
                self.evaluator_id,
                self.version,
                details=f"got {got}, expected {self.expected}",
            )

        def norm(s: str) -> str:
            return " ".join(s.lower().strip().strip(".").split())

        ok = norm(text or "") == norm(str(self.expected))
        return EvaluationResult(
            EvalStatus.SUCCESS if ok else EvalStatus.FAILURE, self.evaluator_id, self.version
        )


@dataclass(frozen=True)
class UnitTests:
    """Run ``test`` (which defines ``check(candidate)``) against ``entry`` from the answer's Python block.

    Runs only inside the Docker sandbox. Infrastructure problems (Docker down, image missing, missing third-party
    module) are ``evaluation_error``, never a model failure.
    """

    test: str
    entry: str
    timeout_s: float = 10.0
    sandbox: DockerSandbox = field(default_factory=DockerSandbox)
    evaluator_id: str = "unit_tests"
    version: str = "1"

    def evaluate(self, text: str) -> EvaluationResult:
        code = extract_code(text)
        if code is None:
            return EvaluationResult(
                EvalStatus.FAILURE, self.evaluator_id, self.version, details="no python block"
            )
        r = self.sandbox.run(code, self.test, self.entry, self.timeout_s)
        if r.status == "pass":
            return EvaluationResult(EvalStatus.SUCCESS, self.evaluator_id, self.version, details=r.detail)
        if r.status in ("fail", "timeout"):
            return EvaluationResult(
                EvalStatus.FAILURE, self.evaluator_id, self.version, details=f"{r.status}: {r.detail}"
            )
        return EvaluationResult(
            EvalStatus.ERROR, self.evaluator_id, self.version, details=f"{r.status}: {r.detail}"
        )
