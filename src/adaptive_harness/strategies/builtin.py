"""Version 1 built-in strategies (DESIGN §5.3).

These are hypotheses to be tested per model and per category, not recommendations. None of them paraphrases the
user's text: it always appears verbatim in the transformed messages.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ..tasks import TaskSpec
from ..types import Message
from .base import (
    Strategy,
    StrategyInfo,
    TransformResult,
    last_user_text,
    replace_last_user,
    with_harness_system,
)

CODE = ("code_generation", "code_debugging", "code_refactoring", "api_development", "security_sensitive_code")


class Identity(Strategy):
    info = StrategyInfo(
        id="identity",
        version="1",
        kind="identity",
        description="Send the prompt unchanged.",
        intended_use="The default arm; every other strategy must beat it with evidence.",
        limitations="None; it is the baseline.",
        general_safe=True,
    )

    def transform(self, messages: Sequence[Message], spec: TaskSpec) -> TransformResult:
        return self._result(messages)


class ConciseDirective(Strategy):
    info = StrategyInfo(
        id="concise_directive",
        version="1",
        kind="additive",
        description="Adds one system line asking for a direct answer without preamble.",
        intended_use="Models that pad answers; tasks where length costs tokens without adding value.",
        limitations="Can shorten explanations the user wanted; not applied to long-form writing or research.",
        applies_to=(
            *CODE,
            "code_explanation",
            "question_answering",
            "math",
            "structured_output",
            "data_analysis",
        ),
        general_safe=True,
    )
    templates: ClassVar[dict[str, str]] = {
        "system": "Answer directly. Do not add a preamble, and do not restate the question."
    }

    def transform(self, messages: Sequence[Message], spec: TaskSpec) -> TransformResult:
        text = self.templates["system"]
        return self._result(with_harness_system(messages, text), text)


class RoleFraming(Strategy):
    info = StrategyInfo(
        id="role_framing",
        version="1",
        kind="additive",
        description="Adds a short system line describing an expert role suited to the task category.",
        intended_use="Categories where a domain role may focus the answer.",
        limitations="Effect is model-dependent and often small; not defined for `general`.",
        applies_to=(
            *CODE,
            "code_explanation",
            "architecture",
            "data_analysis",
            "math",
            "research",
            "summarisation",
            "content_generation",
            "structured_output",
            "question_answering",
        ),
    )
    templates: ClassVar[dict[str, str]] = {
        "code_generation": "You are an experienced software engineer.",
        "code_debugging": "You are an experienced software engineer who debugs systematically.",
        "code_refactoring": "You are an experienced software engineer who values readable code.",
        "code_explanation": "You are an experienced software engineer who explains code clearly.",
        "api_development": "You are an experienced backend engineer.",
        "security_sensitive_code": "You are a security-conscious software engineer.",
        "architecture": "You are an experienced software architect.",
        "data_analysis": "You are an experienced data analyst.",
        "math": "You are a careful mathematician.",
        "research": "You are a careful researcher who separates strong evidence from weak evidence.",
        "summarisation": "You are a precise editor.",
        "content_generation": "You are a skilled writer.",
        "structured_output": "You produce output in exactly the requested format.",
        "question_answering": "You are a knowledgeable assistant.",
    }

    def applicable(self, spec: TaskSpec) -> bool:
        return spec.category in self.templates

    def transform(self, messages: Sequence[Message], spec: TaskSpec) -> TransformResult:
        text = self.templates[spec.category]
        return self._result(with_harness_system(messages, text), text)


class StepwisePlan(Strategy):
    info = StrategyInfo(
        id="stepwise_plan",
        version="1",
        kind="additive",
        description="Asks the model to reason step by step and put the final answer last, under a fixed heading.",
        intended_use="Multi-step reasoning: math, debugging, architecture, analysis, multi-step tasks.",
        limitations="Longer outputs (more tokens); can hurt when the output must be only code or only data.",
        applies_to=("math", "multi_step", "code_debugging", "architecture", "data_analysis"),
    )
    templates: ClassVar[dict[str, str]] = {
        "system": "Work through the problem step by step before answering. "
        "Then give the final answer last, under the heading 'Final answer'."
    }

    def transform(self, messages: Sequence[Message], spec: TaskSpec) -> TransformResult:
        text = self.templates["system"]
        return self._result(with_harness_system(messages, text), text)


class OutputSchema(Strategy):
    info = StrategyInfo(
        id="output_schema",
        version="1",
        kind="additive",
        description="When the task names a structured format, asks for that format only, with nothing around it.",
        intended_use="Extraction and structured-output tasks whose result is parsed by a program.",
        limitations="Only applies when a format (JSON, YAML, XML, CSV, TOML) is detected in the prompt.",
    )
    templates: ClassVar[dict[str, str]] = {
        "system": "Return the output strictly as {fmt}, with no text before or after it."
    }
    _formats: ClassVar[dict[str, str]] = {
        "json": "JSON",
        "yaml": "YAML",
        "xml": "XML",
        "csv": "CSV",
        "toml": "TOML",
    }

    def _format(self, spec: TaskSpec) -> str | None:
        found = [self._formats[f] for f in spec.requirements.output_formats if f in self._formats]
        return found[0] if len(found) == 1 else None  # ambiguous when several formats are mentioned

    def applicable(self, spec: TaskSpec) -> bool:
        return self._format(spec) is not None

    def transform(self, messages: Sequence[Message], spec: TaskSpec) -> TransformResult:
        fmt = self._format(spec)
        if fmt is None:
            raise ValueError("output_schema is not applicable: no single structured format in the task")
        text = self.templates["system"].format(fmt=fmt)
        return self._result(with_harness_system(messages, text), text)


class SecurityGeneric(Strategy):
    info = StrategyInfo(
        id="security_generic",
        version="1",
        kind="additive",
        description="Adds one generic security instruction to code-writing tasks.",
        intended_use="Code generation where security matters.",
        limitations="Prior evidence is mixed and model-dependent (see evidence_note).",
        applies_to=("security_sensitive_code", "code_generation", "api_development"),
        evidence_note=(
            "LeBlanc v4 (56 Python CodeSecEval tasks, 3 reps): this exact sentence changed the security-test "
            "failure rate by -11.9 pp on deepseek-v4.1-flash (significant), -3.0 pp on gemini-3.5-flash-lite and "
            "-1.6 pp on gpt-oss-20b (no effect). Prior evidence only; it does not transfer to other models or tasks."
        ),
    )
    templates: ClassVar[dict[str, str]] = {
        "system": "Write secure code and avoid common security vulnerabilities."
    }

    def transform(self, messages: Sequence[Message], spec: TaskSpec) -> TransformResult:
        text = self.templates["system"]
        return self._result(with_harness_system(messages, text), text)


class StructuredSections(Strategy):
    info = StrategyInfo(
        id="structured_sections",
        version="1",
        kind="structural",
        description="Wraps the user's text under a 'Task' heading and lists its extracted requirements, quoted.",
        intended_use="Long prompts with several requirements that a model may overlook.",
        limitations="Adds tokens; the requirement list is only as good as the deterministic extraction.",
    )
    templates: ClassVar[dict[str, str]] = {
        "task_heading": "## Task",
        "requirements_heading": "## Requirements stated in the task (quoted)",
    }

    def transform(self, messages: Sequence[Message], spec: TaskSpec) -> TransformResult:
        original = last_user_text(messages)
        req = spec.requirements
        quoted = [*req.constraints, *req.negatives]
        parts = [self.templates["task_heading"], original]
        added = [self.templates["task_heading"]]
        if quoted:
            block = "\n".join(f'- "{q}"' for q in quoted)
            parts += ["", self.templates["requirements_heading"], block]
            added += [self.templates["requirements_heading"], block]
        return self._result(replace_last_user(messages, "\n".join(parts)), *added)


class RequirementsChecklist(Strategy):
    info = StrategyInfo(
        id="requirements_checklist",
        version="1",
        kind="structural",
        description="Appends a checklist of the task's own requirements, quoted, and asks the model to satisfy each.",
        intended_use="Tasks with explicit constraints or negative instructions.",
        limitations="Only applies when requirements were extracted; adds tokens.",
    )
    templates: ClassVar[dict[str, str]] = {
        "intro": "Before answering, make sure the answer satisfies each of these points from the task:"
    }

    def _items(self, spec: TaskSpec) -> list[str]:
        r = spec.requirements
        return [*r.constraints, *r.negatives]

    def applicable(self, spec: TaskSpec) -> bool:
        return bool(self._items(spec))

    def transform(self, messages: Sequence[Message], spec: TaskSpec) -> TransformResult:
        items = self._items(spec)
        if not items:
            raise ValueError("requirements_checklist is not applicable: no requirements were extracted")
        block = self.templates["intro"] + "\n" + "\n".join(f'- "{i}"' for i in items)
        original = last_user_text(messages)
        return self._result(replace_last_user(messages, original + "\n\n" + block), block)


BUILTIN: tuple[Strategy, ...] = (
    Identity(),
    ConciseDirective(),
    RoleFraming(),
    StepwisePlan(),
    OutputSchema(),
    SecurityGeneric(),
    StructuredSections(),
    RequirementsChecklist(),
)
