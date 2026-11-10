"""Adapter for the OpenAI chat-completions protocol (OpenAI, Groq, OpenRouter, DeepSeek, Ollama, vLLM, LM Studio)."""

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
from .presets import PRESETS, OpenAIPreset, custom_preset

_FINISH = {
    "stop": FinishReason.STOP,
    "eos": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL,
    "function_call": FinishReason.TOOL,
    "content_filter": FinishReason.FILTERED,
    "error": FinishReason.ERROR,
}


def _finish(value: Any) -> FinishReason:
    return _FINISH.get(str(value).lower(), FinishReason.UNKNOWN) if value else FinishReason.UNKNOWN


def _usage(u: Mapping[str, Any] | None) -> Usage:
    if not u:
        return Usage()
    details = u.get("completion_tokens_details") or {}
    prompt_details = u.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=u.get("prompt_tokens"),
        output_tokens=u.get("completion_tokens"),
        reasoning_tokens=details.get("reasoning_tokens"),
        cached_input_tokens=prompt_details.get("cached_tokens"),
    )


def _tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    out = []
    for c in message.get("tool_calls") or ():
        fn = c.get("function") if isinstance(c, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            args = fn.get("arguments")
            out.append(
                ToolCall(
                    str(fn["name"]), args if isinstance(args, str) else json.dumps(args or {}), c.get("id")
                )
            )
    return tuple(out)


class OpenAICompatibleAdapter(ProviderAdapter):
    def __init__(
        self,
        preset: OpenAIPreset,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http: HTTPConfig | None = None,
        client: httpx.Client | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.preset = preset
        self.name = preset.name
        self._key: Secret | None = (
            get_secret(preset.key_env, explicit=api_key)
            if preset.key_env
            else (Secret(api_key) if api_key else None)
        )
        if preset.key_required and self._key is None:
            raise ConfigurationError(
                f"{preset.name}: no API key. Set {preset.key_env} in the environment or .env, or pass api_key="
            )
        headers = {"Content-Type": "application/json"}
        if self._key is not None:
            headers["Authorization"] = f"Bearer {self._key.reveal()}"
        if preset.name == "openrouter":
            headers["X-Title"] = "adaptive-prompt-harness"
        self.transport = HTTPTransport(preset.name, base_url or preset.base_url, headers, http, client)
        self.extra_body = dict(extra_body or {})

    @classmethod
    def from_preset(cls, name: str, **kwargs: Any) -> OpenAICompatibleAdapter:
        if name == "openai_compatible":
            base_url = kwargs.get("base_url")
            if not base_url:
                raise ConfigurationError("openai_compatible needs base_url=")
            return cls(custom_preset(base_url, kwargs.pop("key_env", None)), **kwargs)
        if name not in PRESETS:
            raise ConfigurationError(f"unknown OpenAI-compatible preset {name!r}")
        return cls(PRESETS[name], **kwargs)

    def capabilities(self) -> ProviderCapabilities:
        p = self.preset
        return ProviderCapabilities(
            seed=p.supports_seed, json_mode=True, tools=True, reports_cost=p.reports_cost, local=p.local
        )

    # ------------------------------------------------------------------ request building
    def _body(self, request: CompletionRequest, stream: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
        s = request.settings
        dropped: list[str] = []
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if s.max_tokens is not None:
            body[self.preset.max_tokens_field] = s.max_tokens
        if s.temperature is not None:
            body["temperature"] = s.temperature
        if s.top_p is not None:
            body["top_p"] = s.top_p
        if s.stop:
            body["stop"] = list(s.stop)
        if s.seed is not None:
            if self.preset.supports_seed:
                body["seed"] = s.seed
            else:
                dropped.append("seed")
        if stream:
            body["stream"] = True
            if self.preset.supports_stream_usage:
                body["stream_options"] = {"include_usage": True}
        body.update(self.preset.default_body)
        body.update(self.extra_body)
        body.update(dict(request.extra))
        return body, tuple(dropped)

    # ------------------------------------------------------------------ completion
    def complete(self, request: CompletionRequest) -> CompletionResult:
        body, dropped = self._body(request, stream=False)
        resp = self.transport.request_json("POST", "chat/completions", body)
        d = resp.data
        if d.get("error") and not d.get("choices"):  # some gateways return errors with HTTP 200
            err = d["error"]
            code = err.get("code") if isinstance(err, dict) else None
            status = int(code) if isinstance(code, int) or (isinstance(code, str) and code.isdigit()) else 502
            raise map_error(self.name, status, str(d), resp.headers)
        try:
            choice = d["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as e:
            raise MalformedResponseError(f"{self.name}: response has no choices", provider=self.name) from e
        usage_raw = d.get("usage") or {}
        cost = usage_raw.get("cost") if isinstance(usage_raw, dict) else None
        return CompletionResult(
            text=message.get("content") or "",
            finish_reason=_finish(choice.get("finish_reason") or choice.get("native_finish_reason")),
            usage=_usage(usage_raw),
            model_reported=d.get("model"),
            latency_s=resp.latency_s,
            provider=self.name,
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            cost_is_estimated=False,
            request_id=d.get("id") or resp.headers.get("x-request-id"),
            fingerprint=self._fingerprint(d, resp.headers),
            dropped_settings=dropped,
            retries=resp.attempts - 1,
            tool_calls=_tool_calls(message),
            raw=d,
        )

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        body, dropped = self._body(request, stream=True)
        t0 = time.perf_counter()
        parts: list[str] = []
        finish: Any = None
        usage_raw: dict[str, Any] = {}
        model_reported = None
        last: dict[str, Any] = {}
        for chunk in self.transport.stream_sse("chat/completions", body):
            if chunk.get("error"):
                raise ProviderError(f"{self.name}: stream error: {chunk['error']}", provider=self.name)
            last = chunk
            model_reported = chunk.get("model") or model_reported
            if chunk.get("usage"):
                usage_raw = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    parts.append(delta)
                    yield StreamEvent(delta=delta)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        cost = usage_raw.get("cost")
        yield StreamEvent(
            result=CompletionResult(
                text="".join(parts),
                finish_reason=_finish(finish),
                usage=_usage(usage_raw),
                model_reported=model_reported,
                latency_s=time.perf_counter() - t0,
                provider=self.name,
                cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
                request_id=last.get("id"),
                fingerprint=self._fingerprint(last, {}),
                dropped_settings=dropped,
            )
        )

    def _fingerprint(self, d: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, str]:
        fp: dict[str, str] = {}
        if d.get("system_fingerprint"):
            fp["system_fingerprint"] = str(d["system_fingerprint"])
        if d.get("provider"):  # OpenRouter: the host that actually served the call
            fp["serving_host"] = str(d["provider"])
        if d.get("model"):
            fp["model_reported"] = str(d["model"])
        return fp

    # ------------------------------------------------------------------ metadata
    def model_info(self, model: str) -> ModelInfo:
        facts: list[MetadataFact] = []
        try:
            data = self.transport.request_json("GET", "models").data.get("data") or []
        except ProviderError:
            data = []
        entry = next((m for m in data if isinstance(m, dict) and m.get("id") == model), None)
        if entry is not None:
            facts.append(MetadataFact("listed", True, "verified", f"{self.name} /models"))
            for key in ("context_length", "context_window", "max_model_len"):
                if entry.get(key):
                    facts.append(
                        MetadataFact("context_window", int(entry[key]), "verified", f"/models.{key}")
                    )
                    break
            if entry.get("owned_by"):
                facts.append(MetadataFact("owned_by", entry["owned_by"], "provider_reported", "/models"))
        if self.preset.name == "ollama":
            facts.extend(self._ollama_show(model))
        facts.append(
            MetadataFact("deployment", "local" if self.preset.local else "hosted", "inferred", "preset")
        )
        return ModelInfo(model_id=model, facts=tuple(facts))

    def _ollama_show(self, model: str) -> list[MetadataFact]:
        root = self.transport.base_url.removesuffix("/v1")
        try:
            d = self.transport.request_json("POST", f"{root}/api/show", {"model": model}).data
        except ProviderError:
            return []
        facts: list[MetadataFact] = []
        details = d.get("details") or {}
        for key in ("family", "parameter_size", "quantization_level", "format"):
            if details.get(key):
                facts.append(MetadataFact(key, details[key], "verified", "ollama /api/show"))
        for key, value in (d.get("model_info") or {}).items():
            if key.endswith(".context_length"):
                facts.append(MetadataFact("context_window", int(value), "verified", "ollama /api/show"))
        return facts

    def close(self) -> None:
        self.transport.close()
