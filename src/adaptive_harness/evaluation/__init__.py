"""Typed evaluation outcomes (DESIGN §5.6).

Only real outcome signals update evidence: a check the developer supplies (``check=``) or explicit feedback.
``evaluation_error``, ``not_applicable`` and ``no_evaluation`` never count as success or failure.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable


@runtime_checkable
class Evaluator(Protocol):
    """Anything with ``evaluate(text) -> EvaluationResult`` (see ``evaluation.builtin``)."""

    def evaluate(self, text: str) -> EvaluationResult: ...


Check = Union[Callable[[str], "bool | float"], Evaluator]  # noqa: UP007 - runtime alias on Python 3.10


class EvalStatus(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "evaluation_error"
    NOT_APPLICABLE = "not_applicable"
    NO_EVALUATION = "no_evaluation"


@dataclass(frozen=True)
class EvaluationResult:
    status: EvalStatus
    evaluator_id: str
    evaluator_version: str = "1"
    score: float | None = None
    details: str = ""

    @property
    def outcome(self) -> float | None:
        """1.0, 0.0 or a partial score in (0, 1) when the result is usable evidence; otherwise None."""
        if self.status is EvalStatus.SUCCESS:
            return 1.0
        if self.status is EvalStatus.FAILURE:
            return 0.0
        if self.status is EvalStatus.PARTIAL:
            return self.score
        return None


def _from_value(value: object, evaluator_id: str) -> EvaluationResult:
    if isinstance(value, bool):
        return EvaluationResult(EvalStatus.SUCCESS if value else EvalStatus.FAILURE, evaluator_id)
    if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
        v = float(value)
        if v == 1.0:
            return EvaluationResult(EvalStatus.SUCCESS, evaluator_id, score=1.0)
        if v == 0.0:
            return EvaluationResult(EvalStatus.FAILURE, evaluator_id, score=0.0)
        return EvaluationResult(EvalStatus.PARTIAL, evaluator_id, score=v)
    return EvaluationResult(
        EvalStatus.ERROR,
        evaluator_id,
        details=f"expected bool or a number in [0, 1], got {type(value).__name__}",
    )


def run_check(check: Check, text: str) -> EvaluationResult:
    """Run a check (a function or an Evaluator) on the model's answer. A crashing check is an evaluation error."""
    evaluator_id = getattr(check, "evaluator_id", "user_check")
    try:
        if isinstance(check, Evaluator):
            result = check.evaluate(text)
            if not isinstance(result, EvaluationResult):
                raise TypeError("evaluate() must return an EvaluationResult")
            return result
        return _from_value(check(text), evaluator_id)
    except Exception as e:  # the check is the developer's code; its failure is not the model's
        return EvaluationResult(EvalStatus.ERROR, evaluator_id, details=f"{type(e).__name__}: {e}"[:300])


def feedback_result(success: bool | None = None, score: float | None = None) -> EvaluationResult:
    if (success is None) == (score is None):
        raise ValueError("give exactly one of success= or score=")
    return _from_value(success if success is not None else score, "human")
