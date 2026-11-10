"""Test isolation: no test may read the developer's real .env, keys or harness home."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

KEY_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "OLLAMA_API_KEY",
    "ADAPTIVE_HARNESS_BUDGET_USD",
    "ADAPTIVE_HARNESS_DB",
    "ADAPTIVE_HARNESS_PROMPT_STORAGE",
]


@pytest.fixture(autouse=True)
def _isolate(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return  # live tests deliberately use the developer's environment
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ADAPTIVE_HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)  # so no parent .env is discovered
    assert not (Path(os.getcwd()) / ".env").exists()
