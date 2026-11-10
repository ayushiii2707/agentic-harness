"""Gemini adapter. Synthetic fixtures in the documented format; not recordings."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from adaptive_harness import AuthenticationError, FinishReason, GenerationSettings, Message, RateLimitError
from adaptive_harness.providers import create_adapter
from adaptive_harness.types import CompletionRequest

BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-test"
KEY = "AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE00"


def body(**over: object) -> dict[str, object]:
    d: dict[str, object] = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "thinking...", "thought": True}, {"text": "Answer"}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 1, "thoughtsTokenCount": 20},
        "modelVersion": "gemini-test-001",
        "responseId": "resp-1",
    }
    d.update(over)
    return d


def request(**settings: object) -> CompletionRequest:
    return CompletionRequest.build(
        "gemini-test",
        [Message("system", "sys"), Message("user", "q"), Message("assistant", "a"), Message("user", "q2")],
        GenerationSettings(**settings),  # type: ignore[arg-type]
    )


@respx.mock
def test_request_shape_and_parsing() -> None:
    route = respx.post(f"{BASE}:generateContent").mock(return_value=httpx.Response(200, json=body()))
    r = create_adapter("gemini", api_key=KEY).complete(request(max_tokens=64, temperature=0.1, seed=3))
    req = route.calls[0].request
    sent = json.loads(req.content)
    assert req.headers["x-goog-api-key"] == KEY and "key=" not in str(req.url)  # key never in the URL
    assert sent["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert [c["role"] for c in sent["contents"]] == ["user", "model", "user"]
    assert sent["generationConfig"] == {"maxOutputTokens": 64, "temperature": 0.1, "seed": 3}
    assert r.text == "Answer"  # thought parts excluded
    assert r.usage.reasoning_tokens == 20 and r.model_reported == "gemini-test-001"
    assert r.finish_reason is FinishReason.STOP


@respx.mock
def test_blocked_prompt_is_filtered_not_malformed() -> None:
    respx.post(f"{BASE}:generateContent").mock(
        return_value=httpx.Response(
            200, json={"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 3}}
        )
    )
    r = create_adapter("gemini", api_key=KEY).complete(request())
    assert r.text == "" and r.finish_reason is FinishReason.FILTERED


@respx.mock
def test_invalid_key_and_quota_errors() -> None:
    respx.post(f"{BASE}:generateContent").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )
    )
    with pytest.raises(AuthenticationError):
        create_adapter("gemini", api_key=KEY).complete(request())


@respx.mock
def test_rate_limit_uses_retry_info() -> None:
    respx.post(f"{BASE}:generateContent").mock(
        return_value=httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Rate limited",
                    "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "500s"}],
                }
            },
        )
    )
    a = create_adapter("gemini", api_key=KEY)
    with pytest.raises(RateLimitError) as exc:
        a.complete(request())
    assert exc.value.retry_after == 500.0  # longer than the wait budget, so it is raised, not slept on


@respx.mock
def test_streaming() -> None:
    chunks = [
        body(candidates=[{"content": {"parts": [{"text": "Hel"}]}}], usageMetadata=None),
        body(candidates=[{"content": {"parts": [{"text": "lo"}]}, "finishReason": "MAX_TOKENS"}]),
    ]
    sse = "".join(f"data: {json.dumps(c)}\r\n\r\n" for c in chunks)
    route = respx.post(f"{BASE}:streamGenerateContent").mock(return_value=httpx.Response(200, text=sse))
    out = list(create_adapter("gemini", api_key=KEY).stream(request()))
    assert route.calls[0].request.url.params["alt"] == "sse"
    assert "".join(e.delta for e in out) == "Hello"
    final = out[-1].result
    assert final is not None and final.finish_reason is FinishReason.LENGTH and final.usage.input_tokens == 7


@respx.mock
def test_model_info() -> None:
    respx.get(BASE).mock(
        return_value=httpx.Response(
            200,
            json={
                "name": "models/gemini-test",
                "inputTokenLimit": 1048576,
                "outputTokenLimit": 65536,
                "version": "001",
            },
        )
    )
    info = create_adapter("gemini", api_key=KEY).model_info("gemini-test")
    assert info.get("context_window") == 1048576 and info.get("version") == "001"
