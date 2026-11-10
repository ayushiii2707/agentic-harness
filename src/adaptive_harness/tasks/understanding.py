"""Task understanding: one call that labels a prompt (DESIGN §5.2).

Combines the classifier, requirement extraction, a coarse complexity estimate and the list of evaluator kinds that
could score the task. The evaluators themselves arrive in Phase 9; here they are only named.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .classifier import Classification, TaskClassifier
from .requirements import RequirementSet, extract_requirements

Complexity = Literal["low", "medium", "high"]

_CODE_CATEGORIES = {
    "code_generation", "code_debugging", "code_refactoring", "api_development", "security_sensitive_code",
}  # fmt: skip
_HAS_TESTS = re.compile(r"\bdef test_\w+|\bassert\b|\bunittest\b|\bpytest\b")


@dataclass(frozen=True)
class TaskSpec:
    classification: Classification
    requirements: RequirementSet
    complexity: Complexity
    suitable_evaluators: tuple[str, ...]

    @property
    def category(self) -> str:
        return self.classification.category

    @property
    def confidence(self) -> float:
        return self.classification.confidence


def estimate_complexity(prompt: str, req: RequirementSet) -> Complexity:
    words = len(prompt.split())
    parts = req.count + len(re.findall(r"^\s*(?:[-*]|\d+[.)])\s+", prompt, re.M))
    if words > 150 or parts >= 6:
        return "high"
    if words > 40 or parts >= 2:
        return "medium"
    return "low"


def suitable_evaluators(prompt: str, category: str, req: RequirementSet) -> tuple[str, ...]:
    out: list[str] = []
    if category in _CODE_CATEGORIES or req.languages:
        out.append("code_syntax")
    if _HAS_TESTS.search(prompt):
        out.append("unit_tests")
    if {"json", "yaml", "xml", "csv", "toml"} & set(req.output_formats):
        out.append("format_validation")
    if req.numbers or req.negatives:
        out.append("constraint_check")
    if category == "math":
        out.append("reference_answer")
    return tuple(out)


class TaskAnalyzer:
    def __init__(self, classifier: TaskClassifier | None = None) -> None:
        self.classifier = classifier or TaskClassifier()

    def analyze(self, prompt: str) -> TaskSpec:
        classification = self.classifier.classify(prompt)
        req = extract_requirements(prompt)
        return TaskSpec(
            classification=classification,
            requirements=req,
            complexity=estimate_complexity(prompt, req),
            suitable_evaluators=suitable_evaluators(prompt, classification.category, req),
        )
