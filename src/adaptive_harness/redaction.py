"""Redaction of personal or sensitive data from prompts before they are stored (privacy level ``redacted``)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ._keys import redact as redact_keys

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "<email>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b(?:\d[ \-]?){12,19}\b"), "<number>"),  # card-, phone- and account-like digit runs
    (re.compile(r"(?i)\b(password|passwd|pwd|secret|token)\s*[:=]\s*\S+"), r"\1=<redacted>"),
]


def redact_text(text: str, extra_patterns: Iterable[str] = ()) -> str:
    out = redact_keys(text)
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    for p in extra_patterns:
        out = re.sub(p, "<redacted>", out)
    return out
