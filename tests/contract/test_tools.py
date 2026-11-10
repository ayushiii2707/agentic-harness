"""Tool pass-through: provider-native tool definitions go out via ``extra``; tool calls come back normalised.

Fixtures are synthetic bodies in each provider's documented format, not recordings of live calls.
"""

from __future__ import annotations

import json

import httpx
import respx

from adaptive_harness import (
    AdaptiveHarness,
    FinishReason,
    HarnessConfig,
    Message,
    PersistenceConfig,
    ToolCall,
)
from adaptive_harness.providers import create_adapter
from adaptive_harness.types import CompletionRequest

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


@respx.mock
def test_openai_compatible_tools_round_trip_through_the_harness(tmp_path: object) -> None:
    body = {
        "id": "c1",
        "model": "m",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=body)
    )
    h = AdaptiveHarness(
        provider="groq",
        model="m",
        api_key="gsk_FAKEFAKEFAKEFAKEFAKEFAKEFAKE",
        config=HarnessConfig(persistence=PersistenceConfig(enabled=False)),
    )
    r = h.generate("What is the weather in Oslo?", extra={"tools": TOOLS, "tool_choice": "auto"})
    sent = json.loads(route.calls[0].request.content)
    assert sent["tools"] == TOOLS and sent["tool_choice"] == "auto"
    assert r.tool_calls == (ToolCall("get_weather", '{"city": "Oslo"}', "call_1"),)
    assert r.text == ""


@respx.mock
def test_anthropic_tool_use_blocks_are_normalised() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "model": "claude-x",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        )
    )
    a = create_adapter("anthropic", api_key="sk-ant-api03-FAKEFAKEFAKEFAKEFAKEFAKE")
    r = a.complete(CompletionRequest.build("claude-x", [Message("user", "q")], extra={"tools": []}))
    assert r.tool_calls == (ToolCall("get_weather", '{"city": "Oslo"}', "toolu_1"),)


@respx.mock
def test_gemini_function_calls_are_normalised() -> None:
    respx.post("https://generativelanguage.googleapis.com/v1beta/models/g:generateContent").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Oslo"}}}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3},
            },
        )
    )
    a = create_adapter("gemini", api_key="AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE00")
    r = a.complete(CompletionRequest.build("g", [Message("user", "q")]))
    assert r.tool_calls == (ToolCall("get_weather", '{"city": "Oslo"}', None),)


# ------------------------------------------------------------------ sending tool results back (finding F11)
CONVERSATION = [
    {"role": "user", "content": "What is the weather in Oslo?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": '{"temp_c": 4}'},
]
OK_OPENAI = {
    "id": "c2",
    "model": "m",
    "choices": [{"message": {"role": "assistant", "content": "It is 4 C."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 9, "completion_tokens": 4},
}


@respx.mock
def test_openai_compatible_tool_results_are_sent_in_openai_format() -> None:
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_OPENAI)
    )
    h = AdaptiveHarness(
        provider="groq",
        model="m",
        api_key="gsk_FAKEFAKEFAKEFAKEFAKEFAKEFAKE",
        config=HarnessConfig(persistence=PersistenceConfig(enabled=False)),
    )
    assert h.chat(CONVERSATION, extra={"tools": TOOLS}).text == "It is 4 C."
    sent = json.loads(route.calls[0].request.content)["messages"]
    assert sent[-2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
            }
        ],
    }
    assert sent[-1] == {
        "role": "tool",
        "content": '{"temp_c": 4}',
        "tool_call_id": "call_1",
        "name": "get_weather",
    }


@respx.mock
def test_anthropic_tool_results_become_tool_use_and_tool_result_blocks() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "m",
                "type": "message",
                "model": "c",
                "content": [{"type": "text", "text": "4 C"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    a = create_adapter("anthropic", api_key="sk-ant-api03-FAKEFAKEFAKEFAKEFAKEFAKE")
    a.complete(CompletionRequest.build("c", CONVERSATION))
    turns = json.loads(route.calls[0].request.content)["messages"]
    assert turns[0] == {"role": "user", "content": "What is the weather in Oslo?"}
    assert turns[1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Oslo"}}],
    }
    assert turns[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": '{"temp_c": 4}'}],
    }


@respx.mock
def test_gemini_tool_results_become_function_response_parts() -> None:
    route = respx.post("https://generativelanguage.googleapis.com/v1beta/models/g:generateContent").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "4 C"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )
    )
    a = create_adapter("gemini", api_key="AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE00")
    a.complete(CompletionRequest.build("g", CONVERSATION))
    contents = json.loads(route.calls[0].request.content)["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Oslo"}, "id": "call_1"}}],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "get_weather", "response": {"temp_c": 4}, "id": "call_1"}}],
    }


def test_tool_messages_are_validated() -> None:
    import pytest

    with pytest.raises(ValueError):
        Message("tool", "{}")  # no tool_call_id or name
    with pytest.raises(ValueError):
        Message("user", "x", tool_calls=(ToolCall("f", "{}", "1"),))
    assert Message.coerce(CONVERSATION[1]).tool_calls == (
        ToolCall("get_weather", '{"city": "Oslo"}', "call_1"),
    )


@respx.mock
def test_gemini_thought_signatures_are_returned_with_their_function_call() -> None:
    """Regression (found live, 2026-11-10): Gemini thinking models reject a function call sent back without the
    thoughtSignature they attached to it (HTTP 400)."""
    route = respx.post("https://generativelanguage.googleapis.com/v1beta/models/g:generateContent").mock(
        side_effect=[
            httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [
                {"functionCall": {"name": "get_weather", "args": {"city": "Oslo"}, "id": "c9"}, "thoughtSignature": "SIG"}
            ]}, "finishReason": "STOP"}]}),
            httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": "4 C"}]},
                                                      "finishReason": "STOP"}]}),
        ]
    )  # fmt: skip
    a = create_adapter("gemini", api_key="AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE00")
    first = a.complete(CompletionRequest.build("g", [Message("user", "Weather in Oslo?")]))
    assert first.finish_reason is FinishReason.TOOL and first.tool_calls[0].signature == "SIG"
    call = first.tool_calls[0]
    a.complete(
        CompletionRequest.build(
            "g",
            [
                Message("user", "Weather in Oslo?"),
                Message("assistant", "", tool_calls=(call,)),
                Message("tool", '{"temp_c": 4}', tool_call_id=call.id, name=call.name),
            ],
        )
    )
    sent = json.loads(route.calls[1].request.content)["contents"][1]["parts"][0]
    assert sent == {
        "functionCall": {"name": "get_weather", "args": {"city": "Oslo"}, "id": "c9"},
        "thoughtSignature": "SIG",
    }
