"""The few-shot strategy: examples as earlier turns, the user's text verbatim, accepted by the transform guard."""

from __future__ import annotations

from adaptive_harness.strategies.few_shot import build_few_shot
from adaptive_harness.tasks import TaskAnalyzer
from adaptive_harness.transform import apply_strategy
from adaptive_harness.types import Message

EXAMPLES = {"math": [("What is 2 + 3? End your answer with the final number only.", "5"),
                     ("What is 10% of 50? End your answer with the final number only.", "5")]}  # fmt: skip


def test_examples_go_before_the_user_turn_and_pass_the_guard() -> None:
    s = build_few_shot(EXAMPLES)
    prompt = "A tank holds 120 litres and fills at 8 litres per minute. After how many minutes is it full?"
    spec = TaskAnalyzer().analyze(prompt)
    assert spec.category == "math" and s.applicable(spec)
    msgs = [Message("system", "be careful"), Message("user", prompt)]
    out = apply_strategy(s, msgs, spec)
    assert out.ok and out.changed
    roles = [m.role for m in out.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert out.messages[-1].content == prompt and out.messages[0] == msgs[0]
    assert all(m.role != "system" or m.content == "be careful" for m in out.messages)


def test_applies_only_where_there_are_examples_and_version_tracks_them() -> None:
    s = build_few_shot(EXAMPLES)
    assert not s.applicable(TaskAnalyzer().analyze("Summarise this article about rivers in two sentences."))
    other = build_few_shot({"math": [("What is 1 + 1?", "2")]})
    assert s.info.version != other.info.version and s.info.version.startswith("1-")
    assert s.info.applies_to == ("math",)
