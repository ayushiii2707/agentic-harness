"""Apply a strategy and decide whether its result may be sent (DESIGN §5.4).

Rules, for every strategy kind:
- every message other than the last user message must survive unchanged and in order (harness text may only be
  added around it);
- harness-added system messages may not contain the user's text (user content never moves into an instruction
  position);
- identity, additive and structural strategies must keep the last user message verbatim;
- rewriting strategies must pass ``check_rewrite``.

Any exception or failed rule returns ``ok=False`` with the reason; the caller then sends the original messages.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..strategies import Strategy
from ..tasks import TaskSpec
from ..types import Message
from .preservation import check_rewrite


@dataclass(frozen=True)
class Applied:
    messages: tuple[Message, ...]
    ok: bool
    reason: str = ""
    changed: bool = False


def _last_user(messages: Sequence[Message]) -> tuple[int, str] | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            return i, messages[i].content
    return None


def apply_strategy(strategy: Strategy, messages: Sequence[Message], spec: TaskSpec) -> Applied:
    original = tuple(messages)
    if strategy.info.kind == "identity":
        return Applied(original, True)
    try:
        new = tuple(strategy.transform(original, spec).messages)
    except Exception as e:
        return Applied(original, False, f"{strategy.info.id} raised {type(e).__name__}: {e}"[:300])

    old_last, new_last = _last_user(original), _last_user(new)
    if old_last is None or new_last is None:
        return Applied(original, False, "no user message")
    (oi, user_text), (_, new_user_text) = old_last, new_last

    # every other original message must still be there, unchanged and in order
    others = [m for i, m in enumerate(original) if i != oi]
    it = iter(new)
    if not all(any(n == m for n in it) for m in others):
        return Applied(original, False, "an original message was changed, dropped or reordered")
    added = [m for m in new if m not in original and m.content != new_user_text]
    if any(m.role == "system" and user_text.strip() and user_text.strip() in m.content for m in added):
        return Applied(original, False, "user text was placed inside a system instruction")

    if strategy.info.kind == "rewriting":
        result = check_rewrite(user_text, new_user_text)
        if not result.ok:
            return Applied(original, False, "rewrite rejected: " + "; ".join(result.problems)[:300])
    elif user_text not in new_user_text:
        return Applied(original, False, "the user's text was not kept verbatim")
    return Applied(new, True, changed=new != original)
