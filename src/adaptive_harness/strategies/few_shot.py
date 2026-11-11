"""Few-shot prompting: worked examples shown as earlier conversation turns (master prompt Part 7).

The examples are inserted as (user, assistant) pairs just before the user's own message, which is kept
verbatim. They never go into a system message. Each category has its own examples, and the strategy applies only
to categories that have some.

The examples are part of the strategy. Build it with :func:`build_few_shot`; its version includes a hash of every
example, so evidence for one example set is never reused for another.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

from ..hashing import canonical_hash
from ..tasks import TaskSpec
from ..types import Message
from .base import Strategy, StrategyInfo, TransformResult


class FewShot(Strategy):
    info: ClassVar[StrategyInfo] = StrategyInfo(
        "few_shot",
        "1",
        "additive",
        "Shows worked examples of the same kind of task, with their answers, before the user's request.",
        "Tasks with a consistent answer format where demonstrations help (calculation, extraction).",
        "Adds input tokens; examples from a different distribution can mislead; quality depends on the examples.",
        applies_to=(),
        general_safe=False,
    )

    def __init__(self, examples: Mapping[str, Sequence[tuple[str, str]]]) -> None:
        self.examples = {cat: tuple(pairs) for cat, pairs in examples.items() if pairs}

    def applicable(self, spec: TaskSpec) -> bool:
        return spec.category in self.examples

    def transform(self, messages: Sequence[Message], spec: TaskSpec) -> TransformResult:
        pairs = self.examples[spec.category]
        last = max(i for i, m in enumerate(messages) if m.role == "user")
        shots = [m for q, a in pairs for m in (Message("user", q), Message("assistant", a))]
        new = [*messages[:last], *shots, *messages[last:]]
        return self._result(new, *(x for pair in pairs for x in pair))


def build_few_shot(examples: Mapping[str, Sequence[tuple[str, str]]]) -> FewShot:
    """A ``few_shot`` strategy whose version identifies its exact examples."""
    cats = {c: [list(p) for p in pairs] for c, pairs in examples.items() if pairs}
    if not cats:
        raise ValueError("few_shot needs at least one category with examples")
    info = StrategyInfo(
        **{
            **FewShot.info.__dict__,
            "version": f"1-{canonical_hash(cats)[:10]}",
            "applies_to": tuple(sorted(cats)),
        }
    )
    cls = type("FewShot", (FewShot,), {"info": info})
    strategy: FewShot = cls(examples)
    return strategy
