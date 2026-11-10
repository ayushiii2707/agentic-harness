"""Deterministic requirement extraction (DESIGN §5.2).

Pulls out the parts of a prompt that a transformation must preserve: constraint sentences, negative instructions,
numbers with units, quoted literals, code identifiers, file paths, URLs, named languages and the requested output
format. It only reads the prompt; nothing here changes it. Phase 8's preservation check consumes the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
_NEGATIVE = re.compile(
    r"\b(do not|don't|dont|never|without|avoid|no longer|must not|mustn't|should not|shouldn't)\b", re.I
)
_DEONTIC = re.compile(
    r"\b(must|should|need to|needs to|required|requires|only|at least|at most|exactly|make sure|ensure|keep|"
    r"maximum|minimum|max|min|under|no more than|strictly)\b",
    re.I,
)
_NUMBER_UNIT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|words?|sentences?|bullets?|bullet points?|items?|lines?|characters?|chars?|"
    r"tokens?|seconds?|minutes?|hours?|days?|ms|kb|mb|gb|km|cm|requests?)\b",
    re.I,
)
_QUOTED = re.compile(r"\"([^\"\n]{1,200})\"|'([^'\n]{2,200})'|`([^`\n]{1,200})`")
_IDENTIFIER = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*\(\)|\b[a-z]+(?:_[a-z0-9]+)+\b|\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"
)
_PATH = re.compile(
    r"(?:^|\s)((?:/|\./|~/)?[\w.\-]+(?:/[\w.\-]+)+|\b[\w\-]+\.(?:py|js|ts|java|go|rs|c|cpp|json|yaml|yml|csv|md|txt|toml|sql|html|css))\b"
)
_URL = re.compile(r"https?://[^\s)>\]]+")
_LANGUAGES = (
    "python", "java", "javascript", "typescript", "go", "golang", "rust", "c++", "c#", "kotlin", "swift", "ruby",
    "php", "sql", "bash", "powershell", "scala", "r", "html", "css",
)  # fmt: skip
_FORMATS = ("json", "yaml", "xml", "csv", "toml", "markdown", "table", "bullet", "numbered list", "html")


@dataclass(frozen=True)
class RequirementSet:
    constraints: tuple[str, ...]  # sentences carrying a requirement word (must, only, at most, ...)
    negatives: tuple[str, ...]  # sentences carrying a negative instruction
    numbers: tuple[str, ...]  # numbers with units ("100 words", "3 bullets")
    literals: tuple[str, ...]  # quoted or backticked text
    identifiers: tuple[str, ...]
    paths: tuple[str, ...]
    urls: tuple[str, ...]
    languages: tuple[str, ...]
    output_formats: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.constraints) + len(self.negatives) + len(self.numbers) + len(self.literals)


def _unique(items: list[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for i in items:
        i = i.strip()
        if i and i not in seen:
            seen[i] = None
    return tuple(seen)


def _word(term: str, low: str) -> bool:
    if term not in low:  # the regex can only match where the term occurs; identical result, much cheaper (F7)
        return False
    left = r"(?<![a-z0-9_])" if term[0].isalnum() else ""
    right = r"(?![a-z0-9_])" if term[-1].isalnum() else ""
    return re.search(left + re.escape(term) + right, low) is not None


def extract_requirements(prompt: str) -> RequirementSet:
    sentences = [s.strip() for s in _SENTENCE.findall(prompt) if s.strip()]
    low = prompt.lower()
    languages = [lang for lang in _LANGUAGES if lang != "r" and _word(lang, low)]
    if re.search(r"\bin R\b|\bR (code|script)\b", prompt):
        languages.append("r")
    return RequirementSet(
        constraints=_unique([s for s in sentences if _DEONTIC.search(s) and not _NEGATIVE.search(s)]),
        negatives=_unique([s for s in sentences if _NEGATIVE.search(s)]),
        numbers=_unique([m.group(0) for m in _NUMBER_UNIT.finditer(prompt)]),
        literals=_unique([next(g for g in m.groups() if g) for m in _QUOTED.finditer(prompt)]),
        identifiers=_unique(_IDENTIFIER.findall(prompt)),
        paths=_unique([m.group(1) for m in _PATH.finditer(prompt)]),
        urls=_unique(_URL.findall(prompt)),
        languages=_unique(languages),
        output_formats=_unique([f for f in _FORMATS if _word(f, low)]),
    )
