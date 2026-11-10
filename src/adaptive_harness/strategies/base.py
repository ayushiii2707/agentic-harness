"""Prompt-strategy interface (DESIGN §5.3).

A strategy is a pure function from (messages, TaskSpec) to new messages. Version 1 strategies are ``identity``,
``additive`` (adds a harness instruction) or ``structural`` (wraps the user's text in a structure). All of them
keep the user's own text verbatim. ``rewriting`` strategies change the user's text and are accepted only through
the preservation checks in ``transform/``.

Evidence is keyed by (id, version). ``content_hash`` covers the metadata and every template string, so any change
to a strategy's wording changes its hash and must come with a new version.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import ClassVar, Literal

from ..hashing import canonical_hash
from ..tasks import TaskSpec
from ..types import Message

Kind = Literal["identity", "additive", "structural", "rewriting"]
ALL = "*"


@dataclass(frozen=True)
class StrategyInfo:
    id: str
    version: str
    kind: Kind
    description: str
    intended_use: str
    limitations: str
    applies_to: tuple[str, ...] = (ALL,)  # task categories, or "*"
    general_safe: bool = False  # allowed when the task category is uncertain (`general`)
    evidence_note: str = ""  # what is known about it; never a model-to-strategy mapping


@dataclass(frozen=True)
class TransformResult:
    messages: tuple[Message, ...]
    strategy_id: str
    strategy_version: str
    added_text: tuple[str, ...] = ()  # every piece of text the strategy added, for traces and audits
    notes: tuple[str, ...] = field(default_factory=tuple)


class Strategy(abc.ABC):
    info: ClassVar[StrategyInfo]
    templates: ClassVar[dict[str, str]] = {}

    @property
    def content_hash(self) -> str:
        return canonical_hash({"info": asdict(self.info), "templates": self.templates})

    def applicable(self, spec: TaskSpec) -> bool:
        cats = self.info.applies_to
        return ALL in cats or spec.category in cats

    @abc.abstractmethod
    def transform(self, messages: Sequence[Message], spec: TaskSpec) -> TransformResult: ...

    # ------------------------------------------------------------------ helpers for subclasses
    def _result(self, messages: Sequence[Message], *added: str) -> TransformResult:
        return TransformResult(tuple(messages), self.info.id, self.info.version, tuple(a for a in added if a))


def with_harness_system(messages: Sequence[Message], text: str) -> list[Message]:
    """Put a harness instruction first, as its own system message.

    The user's own system messages and turns are kept unchanged, after it.
    """
    return [Message("system", text), *messages]


def replace_last_user(messages: Sequence[Message], new_content: str) -> list[Message]:
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].role == "user":
            out[i] = Message("user", new_content)
            return out
    raise ValueError("no user message to transform")


def last_user_text(messages: Sequence[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    raise ValueError("no user message to transform")
