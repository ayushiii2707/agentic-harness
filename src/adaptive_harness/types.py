"""Provider-neutral request and response types.

These types are the contract between the harness core and the provider adapters. No provider-specific field
appears here; adapters translate to and from them.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
_ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"unknown role {self.role!r}; expected one of {_ROLES}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")

    @classmethod
    def coerce(cls, m: Message | Mapping[str, Any]) -> Message:
        if isinstance(m, Message):
            return m
        return cls(role=m["role"], content=m["content"])


class FinishReason(str, enum.Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL = "tool"
    FILTERED = "filtered"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Usage:
    """Token counts as reported by the provider. ``None`` means the provider did not report it."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass(frozen=True)
class GenerationSettings:
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: tuple[Message, ...]
    settings: GenerationSettings = field(default_factory=GenerationSettings)
    extra: Mapping[str, Any] = field(default_factory=dict)  # adapter-specific pass-through options

    @classmethod
    def build(
        cls,
        model: str,
        messages: Sequence[Message | Mapping[str, Any]],
        settings: GenerationSettings | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> CompletionRequest:
        msgs = tuple(Message.coerce(m) for m in messages)
        if not msgs:
            raise ValueError("at least one message is required")
        return cls(
            model=model, messages=msgs, settings=settings or GenerationSettings(), extra=dict(extra or {})
        )


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model, normalised across providers. ``arguments`` is the provider's JSON."""

    name: str
    arguments: str
    id: str | None = None


@dataclass(frozen=True)
class CompletionResult:
    text: str
    finish_reason: FinishReason
    usage: Usage
    model_reported: str | None  # the model id the provider says served the request
    latency_s: float
    provider: str
    cost_usd: float | None = None
    cost_is_estimated: bool = False
    request_id: str | None = None
    fingerprint: Mapping[str, str] = field(default_factory=dict)  # system_fingerprint, serving host, version
    dropped_settings: tuple[str, ...] = ()  # settings the adapter could not send (recorded, never silent)
    retries: int = 0
    simulated: bool = False
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Mapping[str, Any] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class StreamEvent:
    """One streaming event. ``delta`` carries new text; the final event carries ``result``."""

    delta: str = ""
    result: CompletionResult | None = None
