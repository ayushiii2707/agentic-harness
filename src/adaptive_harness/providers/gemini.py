"""Adapter for Google's Generative Language API (Gemini), raw HTTP."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from typing import Any

import httpx

from .._keys import Secret, get_secret
from ..config import HTTPConfig
from ..errors import ConfigurationError, MalformedResponseError, ProviderError
from ..types import CompletionRequest, CompletionResult, FinishReason, StreamEvent, ToolCall, Usage
from ._http import HTTPTransport, map_error
from .base import MetadataFact, ModelInfo, ProviderAdapter, ProviderCapabilities

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_FINISH = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.FILTERED,
    "RECITATION": FinishReason.FILTERED,
    "BLOCKLIST": FinishReason.FILTERED,
    "PROHIBITED_CONTENT": FinishReason.FILTERED,
    "SPII": FinishReason.FILTERED,
    "IMAGE_SAFETY": FinishReason.FILTERED,
    "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
}


def _model_path(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


def _usage(u: Mapping[str, Any] | None) -> Usage:
    if not u:
        return Usage()
    return Usage(
        input_tokens=u.get("promptTokenCount"),
        output_tokens=u.get("candidatesTokenCount"),
        reasoning_tokens=u.get("thoughtsTokenCount"),
        cached_input_tokens=u.get("cachedContentTokenCount"),
    )


def _text(d: Mapping[str, Any]) -> tuple[str, FinishReason]:
    candidates = d.get("candidates") or []
    if not candidates:
        if (d.get("promptFeedback") or {}).get("blockReason"):
            return "", FinishReason.FILTERED
        return "", FinishReason.UNKNOWN
    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    reason = (
        _FINISH.get(str(cand.get("finishReason")), FinishReason.UNKNOWN)
        if cand.get("finishReason")
        else (FinishReason.UNKNOWN)
    )
    return text, reason


def _tool_calls(d: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    cands = d.get("candidates") or []
    parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
    return tuple(
        ToolCall(
            str(p["functionCall"]["name"]),
            json.dumps(p["functionCall"].get("args") or {}),
            p["functionCall"].get("id"),
        )
        for p in parts
        if isinstance(p, dict) and isinstance(p.get("functionCall"), dict) and p["functionCall"].get("name")
    )


class GeminiAdapter(ProviderAdapter):
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http: HTTPConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._key: Secret | None = get_secret("GEMINI_API_KEY", explicit=api_key) or get_secret(
            "GOOGLE_API_KEY"
        )
        if self._key is None:
            raise ConfigurationError("gemini: no API key. Set GEMINI_API_KEY or pass api_key=")
        headers = {"x-goog-api-key": self._key.reveal(), "Content-Type": "application/json"}
        self.transport = HTTPTransport(self.name, base_url or DEFAULT_BASE_URL, headers, http, client)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(seed=True, json_mode=True, tools=True)

    def _body(self, request: CompletionRequest) -> dict[str, Any]:
        s = request.settings
        system = [m.content for m in request.messages if m.role == "system"]
        contents: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            parts: list[dict[str, Any]] = [{"text": m.content}] if m.content and m.role != "tool" else []
            for c in m.tool_calls:
                try:
                    args = json.loads(c.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": c.arguments}
                parts.append({"functionCall": {"name": c.name, "args": args}})
            if m.role == "tool":  # a tool result goes back as a functionResponse part (finding F11)
                try:
                    response = json.loads(m.content)
                except json.JSONDecodeError:
                    response = {"result": m.content}
                if not isinstance(response, dict):
                    response = {"result": response}
                parts.append({"functionResponse": {"name": m.name or m.tool_call_id, "response": response}})
            if not parts:
                parts = [{"text": ""}]
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        gen: dict[str, Any] = {}
        if s.max_tokens is not None:
            gen["maxOutputTokens"] = s.max_tokens
        if s.temperature is not None:
            gen["temperature"] = s.temperature
        if s.top_p is not None:
            gen["topP"] = s.top_p
        if s.stop:
            gen["stopSequences"] = list(s.stop)
        if s.seed is not None:
            gen["seed"] = s.seed
        if gen:
            body["generationConfig"] = gen
        body.update(dict(request.extra))
        return body

    def complete(self, request: CompletionRequest) -> CompletionResult:
        resp = self.transport.request_json(
            "POST", f"{_model_path(request.model)}:generateContent", self._body(request)
        )
        d = resp.data
        if d.get("error"):
            raise map_error(self.name, int((d["error"] or {}).get("code") or 502), str(d), resp.headers)
        if "candidates" not in d and "promptFeedback" not in d:
            raise MalformedResponseError("gemini: response has no candidates", provider=self.name)
        text, reason = _text(d)
        return CompletionResult(
            text=text,
            finish_reason=reason,
            usage=_usage(d.get("usageMetadata")),
            model_reported=d.get("modelVersion"),
            latency_s=resp.latency_s,
            provider=self.name,
            request_id=d.get("responseId"),
            fingerprint={"model_version": str(d["modelVersion"])} if d.get("modelVersion") else {},
            retries=resp.attempts - 1,
            tool_calls=_tool_calls(d),
            raw=d,
        )

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        t0 = time.perf_counter()
        parts: list[str] = []
        reason = FinishReason.UNKNOWN
        usage: Mapping[str, Any] | None = None
        version = None
        response_id = None
        for chunk in self.transport.stream_sse(
            f"{_model_path(request.model)}:streamGenerateContent", self._body(request), params={"alt": "sse"}
        ):
            if chunk.get("error"):
                raise ProviderError(f"gemini stream error: {chunk['error']}", provider=self.name)
            text, r = _text(chunk)
            if text:
                parts.append(text)
                yield StreamEvent(delta=text)
            if r is not FinishReason.UNKNOWN:
                reason = r
            usage = chunk.get("usageMetadata") or usage
            version = chunk.get("modelVersion") or version
            response_id = chunk.get("responseId") or response_id
        yield StreamEvent(
            result=CompletionResult(
                text="".join(parts),
                finish_reason=reason,
                usage=_usage(usage),
                model_reported=version,
                latency_s=time.perf_counter() - t0,
                provider=self.name,
                request_id=response_id,
                fingerprint={"model_version": str(version)} if version else {},
            )
        )

    def model_info(self, model: str) -> ModelInfo:
        try:
            d = self.transport.request_json("GET", _model_path(model)).data
        except ProviderError:
            return ModelInfo(model_id=model)
        facts = [MetadataFact("listed", True, "verified", "gemini models.get")]
        for src, key in (
            ("inputTokenLimit", "context_window"),
            ("outputTokenLimit", "max_output_tokens"),
            ("version", "version"),
            ("displayName", "display_name"),
            ("supportedGenerationMethods", "generation_methods"),
            ("thinking", "thinking"),
        ):
            if d.get(src) is not None:
                facts.append(MetadataFact(key, d[src], "verified", f"models.get.{src}"))
        facts.append(MetadataFact("deployment", "hosted", "inferred", "adapter"))
        return ModelInfo(model_id=model, facts=tuple(facts))

    def close(self) -> None:
        self.transport.close()
