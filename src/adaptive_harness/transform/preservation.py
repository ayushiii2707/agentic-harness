"""Preservation check for strategies that rewrite the user's text (DESIGN §5.4).

A rewrite is accepted only if it keeps what the user asked for:
- **exactly:** every number, quoted or backticked literal, code identifier, path and URL of the original, and
  every named programming language and output format;
- **in substance:** every content word of the original (only stop words and a short, explicit list of
  pleasantries may go), and every constraint sentence and every negative instruction, measured as the share of its
  content words still present, with at least as many negation words as the original;
- **in size:** the rewrite may not be drastically shorter or longer.

The check is deterministic and conservative: when in doubt it rejects, and the harness then sends the original.
Because it compares words, it cannot recognise a faithful *paraphrase*: rewrites may delete stop words and the
listed pleasantries, nothing else. Paraphrasing rewriters (e.g. an LLM rewrite) need an independent equivalence
judge on top of this check; that is not implemented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..tasks import extract_requirements

_WORD = re.compile(r"[a-z0-9_]+(?:'[a-z]+)?")
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
_NEG = re.compile(
    r"\b(not|no|never|without|avoid|don't|dont|doesn't|mustn't|shouldn't|cannot|can't|nor)\b", re.I
)
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "at",
        "as",
        "is",
        "are",
        "be",
        "it",
        "this",
        "that",
        "these",
        "those",
        "you",
        "your",
        "i",
        "me",
        "my",
        "we",
        "our",
        "please",
        "kindly",
        "can",
        "could",
        "would",
        "will",
        "shall",
        "should",
        "must",
        "need",
        "needs",
        "do",
        "does",
        "it's",
    }
)

# Pleasantries a rewrite may drop. Every other content word of the original is protected by MIN_OVERALL.
FILLER = frozenset(
    {"hi", "hello", "hey", "thanks", "thank", "advance", "wondering", "morning", "afternoon", "evening",
     "good", "there", "everyone", "really", "like", "lot", "much", "so", "very", "many"}
)  # fmt: skip
MIN_COVERAGE = 0.7  # per constraint / negative sentence
MIN_OVERALL = 1.0  # every content word of the original must remain (see module docstring)
LENGTH_RATIO = (0.4, 2.5)


@dataclass(frozen=True)
class PreservationResult:
    ok: bool
    problems: tuple[str, ...] = ()


def _content(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _coverage(sentence: str, rewritten_words: set[str]) -> float:
    words = _content(sentence)
    if not words:
        return 1.0
    return sum(w in rewritten_words for w in words) / len(words)


def check_rewrite(original: str, rewritten: str) -> PreservationResult:
    problems: list[str] = []
    if not rewritten.strip():
        return PreservationResult(False, ("the rewrite is empty",))
    req = extract_requirements(original)
    low = rewritten.lower()
    words = set(_WORD.findall(low))

    for n in sorted(set(_NUM.findall(original))):
        if n not in _NUM.findall(rewritten):
            problems.append(f"number {n!r} was dropped or changed")
    for kind, items in (
        ("literal", req.literals),
        ("identifier", req.identifiers),
        ("path", req.paths),
        ("url", req.urls),
        ("unit amount", req.numbers),
    ):
        for item in items:
            if item not in rewritten:
                problems.append(f"{kind} {item!r} was dropped or changed")
    for name in (*req.languages, *req.output_formats):
        if re.search(r"(?<![a-z0-9_])" + re.escape(name) + r"(?![a-z0-9_])", low) is None:
            problems.append(f"{name!r} is no longer mentioned")

    if len(_NEG.findall(rewritten)) < len(_NEG.findall(original)):
        problems.append("a negation was dropped")
    for s in req.negatives:
        if _coverage(s, words) < MIN_COVERAGE:
            problems.append(f"negative instruction not preserved: {s[:80]!r}")
    for s in req.constraints:
        if _coverage(s, words) < MIN_COVERAGE:
            problems.append(f"constraint not preserved: {s[:80]!r}")

    content = [w for w in _content(original) if w not in FILLER]
    if content:
        overall = sum(w in words for w in content) / len(content)
        if overall < MIN_OVERALL:
            missing = sorted({w for w in content if w not in words})[:8]
            problems.append(f"only {overall:.0%} of the original's content words remain (missing: {missing})")

    if len(original) >= 40:
        ratio = len(rewritten) / len(original)
        if not LENGTH_RATIO[0] <= ratio <= LENGTH_RATIO[1]:
            problems.append(f"length changed too much (ratio {ratio:.2f})")
    return PreservationResult(not problems, tuple(problems))
