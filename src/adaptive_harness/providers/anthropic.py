"""Adapter for Anthropic's Messages API (raw HTTP, no SDK, per docs/DECISIONS.md D4).

API version ``2026-10-21``: the Messages API no longer accepts ``temperature``, ``top_p`` or ``top_k``. When a
caller sets them they are **not sent** and are listed in ``CompletionResult.dropped_settings``, so experiments
can record the difference instead of hiding it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from typing import Any

import httpx

from .._keys import Secret, get_secret
from ..config import HTTPConfig
from ..errors import ConfigurationError, MalformedResponseError, ProviderError
from ..types import CompletionRequest, CompletionResult, FinishReason, Message, StreamEvent, ToolCall, Usage
from ._http import HTTPTransport, map_error
from .base import MetadataFact, ModelInfo, ProviderAdapter, ProviderCapabilities

API_VERSION = "2026-10-21"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_MAX_TOKENS = 4096  # the Messages API requires max_tokens

_FINISH = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "model_context_window_exceeded": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL,
    "refusal": FinishReason.FILTERED,
}


def _usage(u: Mapping[str, Any] | None) -> Usage:
    if not u:
        return Usage()
    return Usage(
        input_tokens=u.get("input_tokens"),
        output_tokens=u.get("output_tokens"),
        cached_input_tokens=u.get("cache_read_input_tokens"),
    )


def _blocks(m: Message) -> list[dict[str, Any]]:
    """Content blocks for one message. Tool calls become ``tool_use`` blocks; tool results ``tool_result``."""
    if m.role == "tool":
        return [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}]
    blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
    for c in m.tool_calls:
        try:
            args = json.loads(c.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": c.arguments}
        blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": args})
    return blocks


def _messages(messages: tuple[Message, ...]) -> tuple[str | None, list[dict[str, Any]]]:
    """Move system messages to the top-level field and merge consecutive same-role turns.

    Tool results are sent in a ``user`` turn, as the Messages API requires.
    """
    system = [m.content for m in messages if m.role == "system"]
    turns: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        role = "user" if m.role == "tool" else m.role
        if (
            not m.tool_calls
            and m.role != "tool"
            and not any(t for t in turns[-1:] if isinstance(t["content"], list))
        ):
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n\n" + m.content
            else:
                turns.append({"role": role, "content": m.content})
            continue
        blocks = _blocks(m)
        if turns and turns[-1]["role"] == role:
            prev = turns[-1]["content"]
            turns[-1]["content"] = (
                [{"type": "text", "text": prev}] if isinstance(prev, str) else prev
            ) + blocks
        else:
            turns.append({"role": role, "content": blocks})
    return ("\n\n".join(system) if system else None), turns


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http: HTTPConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._key: Secret | None = get_secret("ANTHROPIC_API_KEY", explicit=api_key)
        if self._key is None:
            raise ConfigurationError("anthropic: no API key. Set ANTHROPIC_API_KEY or pass api_key=")
        headers = {
            "x-api-key": self._key.reveal(),
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        self.transport = HTTPTransport(self.name, base_url or DEFAULT_BASE_URL, headers, http, client)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(temperature=False, top_p=False, seed=False, tools=True)

    def _body(self, request: CompletionRequest, stream: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
        s = request.settings
        dropped = [
            name
            for name, v in (("temperature", s.temperature), ("top_p", s.top_p), ("seed", s.seed))
            if v is not None
        ]
        system, turns = _messages(request.messages)
        if not turns or turns[0]["role"] != "user":
            raise ConfigurationError("anthropic: the first non-system message must be from the user")
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": s.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": turns,
        }
        if system:
            body["system"] = system
        if s.stop:
            body["stop_sequences"] = list(s.stop)
        if stream:
            body["stream"] = True
        for key, value in request.extra.items():
            if key in ("temperature", "top_p", "top_k"):
                dropped.append(key)
                continue
            body[key] = value
        return body, tuple(dropped)

    def complete(self, request: CompletionRequest) -> CompletionResult:
        body, dropped = self._body(request, stream=False)
        resp = self.transport.request_json("POST", "messages", body)
        d = resp.data
        if d.get("type") == "error":
            raise map_error(self.name, 502, str(d), resp.headers)
        content = d.get("content")
        if not isinstance(content, list):
            raise MalformedResponseError("anthropic: response has no content list", provider=self.name)
        text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        return CompletionResult(
            text=text,
            finish_reason=_FINISH.get(str(d.get("stop_reason")), FinishReason.UNKNOWN),
            usage=_usage(d.get("usage")),
            model_reported=d.get("model"),
            latency_s=resp.latency_s,
            provider=self.name,
            request_id=resp.headers.get("request-id") or d.get("id"),
            fingerprint={"model_reported": str(d["model"])} if d.get("model") else {},
            dropped_settings=dropped,
            retries=resp.attempts - 1,
            tool_calls=tuple(
                ToolCall(str(b["name"]), json.dumps(b.get("input") or {}), b.get("id"))
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name")
            ),
            raw=d,
        )

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        body, dropped = self._body(request, stream=True)
        t0 = time.perf_counter()
        parts: list[str] = []
        stop_reason = None
        model = None
        msg_id = None
        input_tokens = output_tokens = cached = None
        for event in self.transport.stream_sse("messages", body):
            kind = event.get("type")
            if kind == "error":
                err = event.get("error") or {}
                raise ProviderError(
                    f"anthropic stream error: {err.get('type')}: {err.get('message')}", provider=self.name
                )
            if kind == "message_start":
                msg = event.get("message") or {}
                model, msg_id = msg.get("model"), msg.get("id")
                u = msg.get("usage") or {}
                input_tokens, cached = u.get("input_tokens"), u.get("cache_read_input_tokens")
            elif kind == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    parts.append(delta["text"])
                    yield StreamEvent(delta=delta["text"])
            elif kind == "message_delta":
                stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
                output_tokens = (event.get("usage") or {}).get("output_tokens", output_tokens)
        yield StreamEvent(
            result=CompletionResult(
                text="".join(parts),
                finish_reason=_FINISH.get(str(stop_reason), FinishReason.UNKNOWN),
                usage=Usage(
                    input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached
                ),
                model_reported=model,
                latency_s=time.perf_counter() - t0,
                provider=self.name,
                request_id=msg_id,
                fingerprint={"model_reported": str(model)} if model else {},
                dropped_settings=dropped,
            )
        )

    def model_info(self, model: str) -> ModelInfo:
        try:
            d = self.transport.request_json("GET", f"models/{model}").data
        except ProviderError:
            return ModelInfo(model_id=model)
        facts = [MetadataFact("listed", True, "verified", "anthropic /v1/models")]
        for src, key in (
            ("max_input_tokens", "context_window"),
            ("max_tokens", "max_output_tokens"),
            ("display_name", "display_name"),
            ("created_at", "created_at"),
        ):
            if d.get(src) is not None:
                facts.append(MetadataFact(key, d[src], "verified", f"/v1/models.{src}"))
        if isinstance(d.get("capabilities"), dict):
            facts.append(
                MetadataFact("capabilities", d["capabilities"], "verified", "/v1/models.capabilities")
            )
        facts.append(MetadataFact("deployment", "hosted", "inferred", "adapter"))
        return ModelInfo(model_id=model, facts=tuple(facts))

    def close(self) -> None:
        self.transport.close()
