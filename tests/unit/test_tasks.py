from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness import AdaptiveHarness, HarnessConfig, PersistenceConfig
from adaptive_harness.providers import MockProvider
from adaptive_harness.tasks import GENERAL, TaskAnalyzer, TaskClassifier, Taxonomy, extract_requirements


def tiny_taxonomy(min_p: float = 0.3, min_m: float = 0.0, k: float = 0.5) -> Taxonomy:
    return Taxonomy(
        {
            "version": "t",
            "thresholds": {"min_probability": min_p, "min_margin": min_m, "unexplained_mass": k},
            "categories": {
                "code_generation": {"parent": None, "phrases": {"implement": 2}},
                "api_development": {"parent": "code_generation", "phrases": {"endpoint": 2}},
                "math": {"parent": None, "phrases": {"solve": 2}, "patterns": {"\\d\\s*\\+\\s*\\d": 1}},
            },
        }
    )


def test_default_taxonomy_loads_with_valid_parents() -> None:
    tax = Taxonomy.load_default()
    assert {"code_generation", "api_development", "security_sensitive_code", "multi_step"} <= set(
        tax.categories
    )
    assert tax.categories["api_development"].parent == "code_generation"


def test_no_evidence_is_general_and_distribution_sums_to_one() -> None:
    c = TaskClassifier(tiny_taxonomy()).classify("hello there")
    assert c.category == GENERAL and not c.fallback
    assert sum(c.distribution.values()) == pytest.approx(1.0)


def test_phrases_match_on_word_boundaries_only() -> None:
    clf = TaskClassifier(tiny_taxonomy())
    assert clf.classify("please implement it").category == "code_generation"
    assert clf.classify("reimplementation").category == GENERAL


def test_low_probability_falls_back_to_general() -> None:
    clf = TaskClassifier(tiny_taxonomy(min_p=0.9))
    c = clf.classify("solve 2 + 2")
    assert c.category == GENERAL and c.fallback and c.confidence < 0.9


def test_parent_child_runner_up_does_not_count_against_margin() -> None:
    clf = TaskClassifier(tiny_taxonomy(min_m=0.2))
    c = clf.classify("implement an endpoint")  # equal scores for a child and its parent
    assert c.category in {"code_generation", "api_development"} and not c.fallback
    unrelated = clf.classify("implement and solve")  # equal scores for unrelated categories
    assert unrelated.category == GENERAL and unrelated.fallback


def test_requirement_extraction() -> None:
    r = extract_requirements(
        "Write a Python function `parse_date()` in utils/dates.py. It must return ISO dates. "
        "Do not use external libraries. Keep the answer under 100 words and return JSON. "
        'See https://example.com/spec and use the key "created_at".'
    )
    assert "Python".lower() in r.languages
    assert any("must return ISO dates" in s for s in r.constraints)
    assert any("Do not use external libraries" in s for s in r.negatives)
    assert "100 words" in r.numbers
    assert "parse_date()" in r.literals and "created_at" in r.literals
    assert "utils/dates.py" in r.paths
    assert "https://example.com/spec" in r.urls
    assert "json" in r.output_formats


def test_r_language_not_matched_by_the_letter_r() -> None:
    assert "r" not in extract_requirements("Write a story about r/askscience readers.").languages
    assert "r" in extract_requirements("Fit a model in R using lm().").languages


def test_analyzer_complexity_and_evaluators() -> None:
    spec = TaskAnalyzer(TaskClassifier(tiny_taxonomy())).analyze("implement add(a, b) in python")
    assert spec.category == "code_generation" and spec.complexity == "low"
    assert "code_syntax" in spec.suitable_evaluators
    long_prompt = "Return JSON. " + " ".join(["word"] * 200)
    spec2 = TaskAnalyzer().analyze(long_prompt)
    assert spec2.complexity == "high" and "format_validation" in spec2.suitable_evaluators


def test_harness_labels_but_never_changes_the_prompt(tmp_path: Path) -> None:
    mock = MockProvider()
    h = AdaptiveHarness(
        provider="mock",
        model="m",
        adapter=mock,
        config=HarnessConfig(persistence=PersistenceConfig(path=tmp_path / "h.db")),
    )
    prompt = "Solve for x: 3x + 7 = 22."
    r = h.generate(prompt)
    assert mock.requests[-1].messages[-1].content == prompt
    assert r.trace.category is not None and r.trace.confidence is not None
    row = h.store.query_one("SELECT category_id FROM trial")  # type: ignore[union-attr]
    assert row is not None and row["category_id"] == r.trace.category


def test_analysis_time_stays_bounded_for_long_prompts() -> None:
    """Regression (evaluation F7): classification time grew linearly with prompt size (about 80 ms at 10 KB)."""
    import time

    from adaptive_harness.tasks import TaskAnalyzer, TaskClassifier
    from adaptive_harness.tasks.classifier import classification_view

    chunk = "Refactor this function so it handles empty input and returns ids sorted by date.\ndef f(x): return x\n"
    long = "Write a Python function that parses ISO dates. " + chunk * 2000 + " Return JSON only."
    view = classification_view(long)
    assert (
        len(view) <= 8001
        and view.startswith("Write a Python function")
        and view.endswith("Return JSON only.")
    )
    assert TaskClassifier().classify(long).category == TaskClassifier().classify(view).category
    a = TaskAnalyzer()
    a.analyze(long)  # warm up
    t = time.perf_counter()
    a.analyze(long)
    assert time.perf_counter() - t < 0.5  # ~200 KB: generous bound for slow CI machines (was ~2 s)
    assert "json" in a.analyze(long).requirements.output_formats  # requirements still read the whole prompt
