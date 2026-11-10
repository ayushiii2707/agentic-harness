"""Evaluation finding F8: explicit configuration wins, and settings files are only read inside the project."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness import AdaptiveHarness, BudgetConfig, HarnessConfig, PersistenceConfig
from adaptive_harness._keys import find_env_file, get_setting


def test_environment_does_not_override_an_explicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADAPTIVE_HARNESS_BUDGET_USD", "5")
    monkeypatch.setenv("ADAPTIVE_HARNESS_PROMPT_STORAGE", "full")
    explicit = HarnessConfig(
        persistence=PersistenceConfig(path=tmp_path / "h.db"), budget=BudgetConfig(total_usd=0.0)
    )
    h = AdaptiveHarness(provider="mock", model="m", config=explicit)
    assert h.config.budget.total_usd == 0.0 and h.config.persistence.prompt_storage == "none"
    # with no config at all, the environment still configures the harness
    monkeypatch.setenv("ADAPTIVE_HARNESS_DB", str(tmp_path / "env.db"))
    assert AdaptiveHarness(provider="mock", model="m").config.budget.total_usd == 5.0


def test_settings_file_is_found_inside_the_project_but_not_above_it(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    project = outer / "project"
    sub = project / "scripts"
    sub.mkdir(parents=True)
    (outer / ".env").write_text("ADAPTIVE_HARNESS_PROMPT_STORAGE=full\n", encoding="utf-8")
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    # an ancestor above the project root is never read
    assert find_env_file(sub) is None
    assert get_setting("ADAPTIVE_HARNESS_PROMPT_STORAGE", env_file=find_env_file(sub)) is None
    # the project's own file is found from a subdirectory
    (project / ".env").write_text("ADAPTIVE_HARNESS_PROMPT_STORAGE=redacted\n", encoding="utf-8")
    found = find_env_file(sub)
    assert found == project / ".env"
    assert get_setting("ADAPTIVE_HARNESS_PROMPT_STORAGE", env_file=found) == "redacted"
