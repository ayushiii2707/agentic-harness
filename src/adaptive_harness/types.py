"""Provider-neutral request and response types.

These types are the contract between the harness core and the provider adapters. No provider-specific field
appears here; adapters translate to and from them.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
_ROLES = ("system", "user", "assistant", "tool")


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model, normalised across providers. ``arguments`` is the provider's JSON.

    ``signature`` is an opaque provider token that must be sent back with the call (Gemini's ``thoughtSignature``
    for thinking models; without it the next turn is rejected). It is never interpreted.
    """

    name: str
    arguments: str
    id: str | None = None
    signature: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Message:
    """One chat message.

    Tool use (finding F11): an ``assistant`` message may carry the model's ``tool_calls``; a ``tool`` message
    returns one call's result, identified by ``tool_call_id`` (OpenAI-compatible, Anthropic) and ``name`` (Gemini).
    Each adapter translates these to its provider's format.
    """

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"unknown role {self.role!r}; expected one of {_ROLES}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages carry tool_calls")
        if self.role == "tool" and not (self.tool_call_id or self.name):
            raise ValueError("a tool message needs tool_call_id (or name, for Gemini)")

    @classmethod
    def coerce(cls, m: Message | Mapping[str, Any]) -> Message:
        """Accept a Message or an OpenAI-style dict (``tool_calls`` as ToolCall objects or OpenAI dicts)."""
        if isinstance(m, Message):
            return m
        calls = []
        for c in m.get("tool_calls") or ():
            if isinstance(c, ToolCall):
                calls.append(c)
            else:
                fn = c.get("function") or {}
                args = fn.get("arguments", "{}")
                calls.append(
                    ToolCall(
                        str(fn.get("name", "")),
                        args if isinstance(args, str) else json.dumps(args),
                        c.get("id"),
                    )
                )
        return cls(
            role=m["role"],
            content=m.get("content") or "",
            tool_calls=tuple(calls),
            tool_call_id=m.get("tool_call_id"),
            name=m.get("name"),
        )


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
