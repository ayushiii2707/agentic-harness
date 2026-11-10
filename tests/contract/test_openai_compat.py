"""OpenAI-compatible adapter: request building, parsing, streaming, presets.

Fixtures are synthetic bodies in the documented response format; they are not recordings of live calls.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from adaptive_harness import (
    AuthenticationError,
    ConfigurationError,
    FinishReason,
    GenerationSettings,
    Message,
    QuotaExceededError,
)
from adaptive_harness.providers import create_adapter
from adaptive_harness.providers.openai_compat import OpenAICompatibleAdapter
from adaptive_harness.types import CompletionRequest

FAKE_KEY = "gsk_FAKEFAKEFAKEFAKEFAKEFAKEFAKE"


def req(**settings: object) -> CompletionRequest:
    return CompletionRequest.build(
        "test-model",
        [Message("system", "sys"), Message("user", "hi")],
        GenerationSettings(**settings),  # type: ignore[arg-type]
    )


def ok_body(**over: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": "chatcmpl-1",
        "model": "test-model-0613",
        "system_fingerprint": "fp_abc",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "completion_tokens_details": {"reasoning_tokens": 1},
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }
    body.update(over)
    return body


@respx.mock
def test_groq_request_and_response() -> None:
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_body())
    )
    a = create_adapter("groq", api_key=FAKE_KEY)
    r = a.complete(req(max_tokens=50, temperature=0.2, seed=7, stop=("END",)))
    sent = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert sent["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert (
        sent["max_tokens"] == 50
        and sent["temperature"] == 0.2
        and sent["seed"] == 7
        and sent["stop"] == ["END"]
    )
    assert r.text == "hello" and r.finish_reason is FinishReason.STOP
    assert (
        r.usage.input_tokens,
        r.usage.output_tokens,
        r.usage.reasoning_tokens,
        r.usage.cached_input_tokens,
    ) == (12, 3, 1, 4)
    assert r.model_reported == "test-model-0613" and r.fingerprint["system_fingerprint"] == "fp_abc"
    assert r.cost_usd is None and r.simulated is False


@respx.mock
def test_openai_uses_max_completion_tokens() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_body())
    )
    create_adapter("openai", api_key="sk-proj-FAKEFAKEFAKEFAKEFAKE").complete(req(max_tokens=10))
    sent = json.loads(route.calls[0].request.content)
    assert sent["max_completion_tokens"] == 10 and "max_tokens" not in sent


@respx.mock
def test_openrouter_reports_cost_and_serving_host() -> None:
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json=ok_body(
                provider="DeepInfra", usage={"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.000123}
            ),
        )
    )
    a = create_adapter(
        "openrouter",
        api_key="sk-or-v1-FAKEFAKEFAKEFAKEFAKE",
        extra_body={"provider": {"order": ["deepinfra"], "allow_fallbacks": False}},
    )
    r = a.complete(req())
    sent = json.loads(route.calls[0].request.content)
    assert sent["usage"] == {"include": True} and sent["provider"]["order"] == ["deepinfra"]
    assert r.cost_usd == pytest.approx(0.000123) and r.cost_is_estimated is False
    assert r.fingerprint["serving_host"] == "DeepInfra"


@respx.mock
def test_error_inside_http_200_body_is_raised() -> None:
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"error": {"code": 402, "message": "Insufficient credits"}})
    )
    with pytest.raises(QuotaExceededError):
        create_adapter("openrouter", api_key="sk-or-v1-FAKEFAKEFAKEFAKEFAKE").complete(req())


@respx.mock
def test_ollama_needs_no_key_and_drops_nothing() -> None:
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json=ok_body(choices=[{"message": {"content": None}, "finish_reason": "length"}])
        )
    )
    a = create_adapter("ollama")
    r = a.complete(req(seed=1))
    assert "authorization" not in route.calls[0].request.headers
    assert r.text == "" and r.finish_reason is FinishReason.LENGTH and r.dropped_settings == ()
    assert a.capabilities().local is True


@respx.mock
def test_unsupported_setting_is_recorded_not_silently_sent() -> None:
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_body())
    )
    r = create_adapter("deepseek", api_key="sk-FAKEFAKEFAKEFAKEFAKE").complete(req(seed=3))
    assert "seed" not in json.loads(route.calls[0].request.content)
    assert r.dropped_settings == ("seed",)


def test_missing_key_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError) as exc:
        create_adapter("groq")
    assert "GROQ_API_KEY" in str(exc.value)


def test_custom_endpoint_requires_base_url() -> None:
    with pytest.raises(ConfigurationError):
        OpenAICompatibleAdapter.from_preset("openai_compatible")
    a = OpenAICompatibleAdapter.from_preset("openai_compatible", base_url="http://localhost:9999/v1")
    assert a.transport.base_url == "http://localhost:9999/v1"


@respx.mock
def test_auth_error_does_not_leak_key() -> None:
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": f"Invalid API Key {FAKE_KEY}"}})
    )
    with pytest.raises(AuthenticationError) as exc:
        create_adapter("groq", api_key=FAKE_KEY).complete(req())
    assert FAKE_KEY not in str(exc.value)


@respx.mock
def test_streaming_assembles_text_and_usage() -> None:
    chunks = [
        {"id": "c1", "model": "m", "choices": [{"delta": {"role": "assistant"}}]},
        {"id": "c1", "model": "m", "choices": [{"delta": {"content": "hel"}}]},
        {"id": "c1", "model": "m", "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"id": "c1", "model": "m", "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    )
    events = list(create_adapter("groq", api_key=FAKE_KEY).stream(req()))
    sent = json.loads(route.calls[0].request.content)
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert "".join(e.delta for e in events) == "hello"
    final = events[-1].result
    assert final is not None and final.text == "hello" and final.usage.output_tokens == 2
    assert final.finish_reason is FinishReason.STOP


@respx.mock
def test_model_info_from_models_listing() -> None:
    respx.get("https://api.groq.com/openai/v1/models").mock(
        return_value=httpx.Response(
            200, json={"data": [{"id": "test-model", "owned_by": "Meta", "context_window": 131072}]}
        )
    )
    info = create_adapter("groq", api_key=FAKE_KEY).model_info("test-model")
    assert info.get("context_window") == 131072 and info.get("owned_by") == "Meta"
    prov = {f.key: f.provenance for f in info.facts}
    assert prov["context_window"] == "verified" and prov["deployment"] == "inferred"


@respx.mock
def test_model_info_survives_endpoint_failure() -> None:
    respx.get("https://api.groq.com/openai/v1/models").mock(return_value=httpx.Response(404, json={}))
    info = create_adapter("groq", api_key=FAKE_KEY).model_info("x")
    assert info.get("context_window") is None  # incomplete metadata is fine; nothing is invented
