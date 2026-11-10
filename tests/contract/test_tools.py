"""Tool pass-through: provider-native tool definitions go out via ``extra``; tool calls come back normalised.

Fixtures are synthetic bodies in each provider's documented format, not recordings of live calls.
"""

from __future__ import annotations

import json

import httpx
import respx

from adaptive_harness import AdaptiveHarness, HarnessConfig, Message, PersistenceConfig, ToolCall
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
