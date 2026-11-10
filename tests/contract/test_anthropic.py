"""Anthropic Messages adapter. Synthetic fixtures in the documented format; not recordings."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from adaptive_harness import (
    ConfigurationError,
    FinishReason,
    GenerationSettings,
    Message,
    ProviderUnavailableError,
)
from adaptive_harness.providers import create_adapter
from adaptive_harness.types import CompletionRequest

URL = "https://api.anthropic.com/v1/messages"
KEY = "sk-ant-api03-FAKEFAKEFAKEFAKEFAKEFAKE"


def body(**over: object) -> dict[str, object]:
    d: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": "Hi"},
            {"type": "text", "text": "!"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 9, "output_tokens": 2, "cache_read_input_tokens": 0},
    }
    d.update(over)
    return d


@respx.mock
def test_request_shape_and_parsing() -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=body(), headers={"request-id": "req_1"})
    )
    a = create_adapter("anthropic", api_key=KEY)
    r = a.complete(
        CompletionRequest.build(
            "claude-haiku-5-5",
            [Message("system", "be terse"), Message("user", "a"), Message("user", "b")],
            GenerationSettings(temperature=0.3, top_p=0.9, stop=("X",)),
        )
    )
    req = route.calls[0].request
    sent = json.loads(req.content)
    assert req.headers["x-api-key"] == KEY and req.headers["anthropic-version"] == "2026-10-21"
    assert sent["system"] == "be terse"
    assert sent["messages"] == [{"role": "user", "content": "a\n\nb"}]  # consecutive turns merged
    assert sent["max_tokens"] == 4096 and sent["stop_sequences"] == ["X"]
    assert "temperature" not in sent and "top_p" not in sent  # rejected by API version 2026-10-21
    assert r.dropped_settings == ("temperature", "top_p")
    assert r.text == "Hi!" and r.finish_reason is FinishReason.STOP
    assert r.usage.input_tokens == 9 and r.request_id == "req_1" and r.model_reported == "claude-haiku-5-5"


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        ("max_tokens", FinishReason.LENGTH),
        ("tool_use", FinishReason.TOOL),
        ("refusal", FinishReason.FILTERED),
        ("pause_turn", FinishReason.UNKNOWN),
    ],
)
@respx.mock
def test_stop_reasons(stop: str, expected: FinishReason) -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json=body(stop_reason=stop)))
    r = create_adapter("anthropic", api_key=KEY).complete(
        CompletionRequest.build("m", [Message("user", "x")])
    )
    assert r.finish_reason is expected


@respx.mock
def test_overloaded_is_retryable_error() -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        )
    )
    a = create_adapter("anthropic", api_key=KEY)
    a.transport._sleep = lambda s: None  # type: ignore[attr-defined]
    with pytest.raises(ProviderUnavailableError):
        a.complete(CompletionRequest.build("m", [Message("user", "x")]))


def test_first_turn_must_be_user() -> None:
    a = create_adapter("anthropic", api_key=KEY)
    with pytest.raises(ConfigurationError):
        a.complete(CompletionRequest.build("m", [Message("assistant", "x")]))


@respx.mock
def test_streaming_events() -> None:
    events = [
        (
            "message_start",
            {"type": "message_start", "message": {"id": "msg_2", "model": "m", "usage": {"input_tokens": 5}}},
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
        ),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    sse = "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events)
    respx.post(URL).mock(return_value=httpx.Response(200, text=sse))
    out = list(
        create_adapter("anthropic", api_key=KEY).stream(CompletionRequest.build("m", [Message("user", "x")]))
    )
    assert "".join(e.delta for e in out) == "Hello"
    final = out[-1].result
    assert final is not None and final.usage.input_tokens == 5 and final.usage.output_tokens == 2
    assert final.finish_reason is FinishReason.STOP


@respx.mock
def test_model_info() -> None:
    respx.get("https://api.anthropic.com/v1/models/claude-haiku-5-5").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "claude-haiku-5-5",
                "display_name": "Claude Haiku 5.5",
                "max_input_tokens": 1000000,
                "max_tokens": 128000,
            },
        )
    )
    info = create_adapter("anthropic", api_key=KEY).model_info("claude-haiku-5-5")
    assert info.get("context_window") == 1000000 and info.get("max_output_tokens") == 128000
