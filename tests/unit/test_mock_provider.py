from __future__ import annotations

import pytest

from adaptive_harness import Message, ProviderError
from adaptive_harness.providers import MockProvider, available_providers
from adaptive_harness.types import CompletionRequest


def test_scripted_responses_and_errors() -> None:
    m = MockProvider(["a", ProviderError("boom"), "b"])
    req = CompletionRequest.build("m", [Message("user", "x")])
    assert m.complete(req).text == "a"
    with pytest.raises(ProviderError):
        m.complete(req)
    r = m.complete(req)
    assert r.text == "b" and r.simulated is True
    with pytest.raises(ProviderError):
        m.complete(req)  # script exhausted


def test_registry_lists_all_adapters() -> None:
    names = set(available_providers())
    assert {
        "mock",
        "openai",
        "anthropic",
        "gemini",
        "groq",
        "openrouter",
        "deepseek",
        "ollama",
        "vllm",
        "lmstudio",
        "openai_compatible",
        "ollama_cloud",
    } <= names
