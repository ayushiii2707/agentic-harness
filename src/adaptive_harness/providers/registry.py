"""Create adapters by provider name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import ConfigurationError
from .base import ProviderAdapter
from .mock import MockProvider

AdapterFactory = Callable[..., ProviderAdapter]


def _openai_factory(preset: str) -> AdapterFactory:
    def make(**kwargs: Any) -> ProviderAdapter:
        from .openai_compat import OpenAICompatibleAdapter

        return OpenAICompatibleAdapter.from_preset(preset, **kwargs)

    return make


def _anthropic(**kwargs: Any) -> ProviderAdapter:
    from .anthropic import AnthropicAdapter

    return AnthropicAdapter(**kwargs)


def _gemini(**kwargs: Any) -> ProviderAdapter:
    from .gemini import GeminiAdapter

    return GeminiAdapter(**kwargs)


def _mock(**kwargs: Any) -> ProviderAdapter:
    kwargs.pop("api_key", None)
    kwargs.pop("base_url", None)
    kwargs.pop("http", None)
    return MockProvider(**kwargs)


_FACTORIES: dict[str, AdapterFactory] = {
    "mock": _mock,
    "anthropic": _anthropic,
    "gemini": _gemini,
    **{
        name: _openai_factory(name)
        for name in (
            "openai",
            "groq",
            "openrouter",
            "deepseek",
            "ollama",
            "ollama_cloud",
            "vllm",
            "lmstudio",
            "openai_compatible",
        )
    },
}


def register_provider(name: str, factory: AdapterFactory) -> None:
    """Register a custom adapter factory (third-party providers plug in here)."""
    _FACTORIES[name.lower()] = factory


def available_providers() -> list[str]:
    return sorted(_FACTORIES)


def create_adapter(provider: str, **kwargs: Any) -> ProviderAdapter:
    factory = _FACTORIES.get(provider.lower())
    if factory is None:
        raise ConfigurationError(
            f"unknown provider {provider!r}; available: {', '.join(available_providers())}"
        )
    return factory(**kwargs)
