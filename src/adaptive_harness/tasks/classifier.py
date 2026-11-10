"""Deterministic task classifier (DESIGN §5.2).

Each category has weighted phrases (matched on word boundaries, case-insensitive) and weighted regex patterns.
A category's score is the sum of the weights that match. Scores become a distribution with an explicit
"unexplained" mass assigned to ``general``:

    p(c) = s_c / (S + k),   p(general) = k / (S + k),   S = sum of all scores

so weak evidence yields low confidence. The prediction falls back to ``general`` when the top probability is
below ``min_probability`` or the margin to the runner-up is below ``min_margin``. A runner-up that is the top
category's parent or child does not count against the margin.

The classifier only labels the prompt; it never changes it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

GENERAL = "general"


@dataclass(frozen=True)
class Category:
    id: str
    parent: str | None
    description: str
    phrases: tuple[tuple[re.Pattern[str], str, float], ...]
    patterns: tuple[tuple[re.Pattern[str], str, float], ...]


@dataclass(frozen=True)
class Classification:
    category: str
    confidence: float
    margin: float
    distribution: dict[str, float]
    fallback: bool  # True when the top category was rejected in favour of ``general``
    evidence: dict[str, tuple[str, ...]] = field(default_factory=dict)
    taxonomy_version: str = ""


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    left = r"(?<![a-z0-9_])" if phrase[:1].isalnum() else ""
    right = r"(?![a-z0-9_])" if phrase[-1:].isalnum() else ""
    return re.compile(left + re.escape(phrase.lower()) + right)


class Taxonomy:
    def __init__(self, data: dict[str, Any]) -> None:
        self.version = str(data["version"])
        t = data["thresholds"]
        self.min_probability = float(t["min_probability"])
        self.min_margin = float(t["min_margin"])
        self.unexplained_mass = float(t["unexplained_mass"])
        self.categories: dict[str, Category] = {}
        for cid, c in data["categories"].items():
            self.categories[cid] = Category(
                id=cid,
                parent=c.get("parent"),
                description=c.get("description", ""),
                phrases=tuple((_phrase_regex(p), p, float(w)) for p, w in c.get("phrases", {}).items()),
                patterns=tuple(
                    (re.compile(p, re.IGNORECASE | re.MULTILINE), p, float(w))
                    for p, w in c.get("patterns", {}).items()
                ),
            )
        for c in self.categories.values():
            if c.parent is not None and c.parent not in self.categories:
                raise ValueError(f"category {c.id} has unknown parent {c.parent}")

    @classmethod
    def load_default(cls) -> Taxonomy:
        text = (resources.files("adaptive_harness.tasks") / "taxonomy.json").read_text(encoding="utf-8")
        return cls(json.loads(text))

    def related(self, a: str, b: str) -> bool:
        ca, cb = self.categories.get(a), self.categories.get(b)
        return bool((ca and ca.parent == b) or (cb and cb.parent == a))


class TaskClassifier:
    def __init__(self, taxonomy: Taxonomy | None = None) -> None:
        self.taxonomy = taxonomy or Taxonomy.load_default()

    def scores(self, prompt: str) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
        low = prompt.lower()
        scores: dict[str, float] = {}
        evidence: dict[str, tuple[str, ...]] = {}
        for c in self.taxonomy.categories.values():
            s = 0.0
            hits: list[str] = []
            for rx, label, w in c.phrases:
                if rx.search(low):
                    s += w
                    hits.append(label)
            for rx, label, w in c.patterns:
                if rx.search(prompt):
                    s += w
                    hits.append(label)
            if s > 0:
                scores[c.id] = s
                evidence[c.id] = tuple(hits)
        return scores, evidence

    def classify(self, prompt: str) -> Classification:
        t = self.taxonomy
        scores, evidence = self.scores(prompt)
        total = sum(scores.values()) + t.unexplained_mass
        dist = {cid: s / total for cid, s in scores.items()}
        dist[GENERAL] = t.unexplained_mass / total
        ranked = sorted(((p, c) for c, p in dist.items() if c != GENERAL), reverse=True)
        if not ranked:
            return Classification(GENERAL, dist[GENERAL], 1.0, dist, False, evidence, t.version)
        top_p, top = ranked[0]
        rival = next((p for p, c in ranked[1:] if not t.related(top, c)), 0.0)
        margin = top_p - rival
        if top_p < t.min_probability or margin < t.min_margin:
            return Classification(GENERAL, top_p, margin, dist, True, evidence, t.version)
        return Classification(top, top_p, margin, dist, False, evidence, t.version)
