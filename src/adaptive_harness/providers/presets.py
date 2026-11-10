"""Presets for providers that speak the OpenAI chat-completions protocol.

A preset only holds facts about the endpoint (URL, key variable, which settings it accepts). Nothing in a preset
says how to *prompt* a model; that is learned, never configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpenAIPreset:
    name: str
    base_url: str
    key_env: str | None  # None: no key needed
    key_required: bool = True
    local: bool = False
    max_tokens_field: str = "max_tokens"
    supports_seed: bool = False
    supports_stream_usage: bool = True
    reports_cost: bool = False
    default_body: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


PRESETS: dict[str, OpenAIPreset] = {
    p.name: p
    for p in [
        OpenAIPreset(
            "openai",
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
            max_tokens_field="max_completion_tokens",
            supports_seed=True,
        ),
        OpenAIPreset("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", supports_seed=True),
        OpenAIPreset(
            "openrouter",
            "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY",
            supports_seed=True,
            reports_cost=True,
            default_body={"usage": {"include": True}},
        ),
        OpenAIPreset("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
        OpenAIPreset(
            "ollama",
            "http://localhost:11434/v1",
            None,
            key_required=False,
            local=True,
            supports_seed=True,
        ),
        OpenAIPreset(
            "ollama_cloud",
            "https://ollama.com/v1",
            "OLLAMA_API_KEY",
            notes="Ollama's hosted service; OpenAI-compatible endpoint",
        ),
        OpenAIPreset(
            "vllm",
            "http://localhost:8000/v1",
            "VLLM_API_KEY",
            key_required=False,
            local=True,
            supports_seed=True,
        ),
        OpenAIPreset("lmstudio", "http://localhost:1234/v1", None, key_required=False, local=True),
    ]
}


def custom_preset(base_url: str, key_env: str | None = None, *, local: bool = False) -> OpenAIPreset:
    """A preset for any other OpenAI-compatible server."""
    return OpenAIPreset("openai_compatible", base_url, key_env, key_required=key_env is not None, local=local)
