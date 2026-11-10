"""Retry policy and error normalisation of the shared HTTP layer (no network: respx mocks every call)."""

from __future__ import annotations

import httpx
import pytest
import respx

from adaptive_harness import (
    AuthenticationError,
    ContextLengthError,
    HTTPConfig,
    InvalidRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)
from adaptive_harness.providers._http import HTTPTransport, map_error, parse_retry_after

URL = "https://api.example.test/v1/thing"


def transport(attempts: int = 3, total_wait: float = 90.0) -> tuple[HTTPTransport, list[float]]:
    slept: list[float] = []
    t = HTTPTransport(
        "example",
        "https://api.example.test/v1",
        {"Authorization": "Bearer x"},
        HTTPConfig(max_attempts=attempts, max_total_wait_s=total_wait),
        sleep=slept.append,
    )
    return t, slept


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, '{"error":{"message":"bad key"}}', AuthenticationError),
        (403, '{"error":{"message":"forbidden"}}', AuthenticationError),
        (
            400,
            '{"error":{"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT"}}',
            AuthenticationError,
        ),
        (402, '{"error":{"message":"no credit"}}', QuotaExceededError),
        (
            429,
            '{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}',
            QuotaExceededError,
        ),
        # Bodies below are the wording real providers returned on 2026-11-10 (key-free).
        (
            400,
            '{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}}',
            QuotaExceededError,
        ),
        (
            429,
            '{"error":{"message":"You have no credits remaining. Add credits to continue."}}',
            QuotaExceededError,
        ),
        (402, '{"error":"Add usage credits or subscribe"}', QuotaExceededError),
        (429, '{"error":{"message":"slow down"}}', RateLimitError),
        (400, '{"error":{"message":"This model maximum context length is 8192 tokens"}}', ContextLengthError),
        (400, '{"error":{"message":"unknown field"}}', InvalidRequestError),
        (404, '{"error":{"message":"model not found"}}', InvalidRequestError),
        (408, "timeout", ProviderTimeoutError),
        (500, "oops", ProviderUnavailableError),
        (
            529,
            '{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
            ProviderUnavailableError,
        ),
    ],
)
def test_error_mapping(status: int, body: str, expected: type) -> None:
    err = map_error("p", status, body, {})
    assert type(err) is expected
    assert err.status_code == status


def test_error_messages_are_redacted() -> None:
    err = map_error(
        "p",
        401,
        '{"error":{"message":"Incorrect API key provided: sk-proj-abcdefghijklmnopqrstuvwxyz123"}}',
        {},
    )
    assert "abcdefghijklmnopqrstuvwxyz123" not in str(err)


def test_retry_after_sources() -> None:
    assert parse_retry_after({"retry-after": "7"}) == 7.0
    assert parse_retry_after({}, '{"details":[{"retryDelay":"31s"}]}') == 31.0
    assert parse_retry_after({}, "Please try again in 250ms.") == 0.25
    assert parse_retry_after({}) is None


@respx.mock
def test_retries_rate_limit_then_succeeds() -> None:
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow"}}),
            httpx.Response(503, text="busy"),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    t, slept = transport()
    resp = t.request_json("POST", "thing", {"a": 1})
    assert resp.data == {"ok": True} and resp.attempts == 3 and route.call_count == 3
    assert slept[0] == 2.0  # Retry-After honoured


@respx.mock
def test_does_not_retry_non_retryable() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(401, json={"error": {"message": "no"}}))
    t, slept = transport()
    with pytest.raises(AuthenticationError):
        t.request_json("POST", "thing", {})
    assert route.call_count == 1 and slept == []


@respx.mock
def test_attempts_are_bounded() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(500, text="down"))
    t, _ = transport(attempts=2)
    with pytest.raises(ProviderUnavailableError):
        t.request_json("POST", "thing", {})
    assert route.call_count == 2


@respx.mock
def test_retry_after_longer_than_wait_budget_is_not_waited_for() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(429, headers={"retry-after": "3600"}, json={}))
    t, slept = transport(total_wait=60)
    with pytest.raises(RateLimitError):
        t.request_json("POST", "thing", {})
    assert route.call_count == 1 and slept == []


@respx.mock
def test_connection_errors_are_retried_and_normalised() -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
    t, slept = transport(attempts=2)
    with pytest.raises(ProviderUnavailableError):
        t.request_json("POST", "thing", {})
    assert len(slept) == 1


@respx.mock
def test_timeouts_are_normalised() -> None:
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    t, _ = transport(attempts=1)
    with pytest.raises(ProviderTimeoutError):
        t.request_json("POST", "thing", {})


def test_groq_token_rate_limit_is_retryable_even_with_a_billing_link() -> None:
    """Regression: Groq's TPM 429 links to /settings/billing and was mapped to a terminal QuotaExceededError."""
    from adaptive_harness.errors import RateLimitError
    from adaptive_harness.providers._http import map_error

    body = (
        '{"error":{"message":"Rate limit reached for model `m` in organization `org_01abcdefghijkl` service tier '
        "`on_demand` on tokens per minute (TPM): Limit 1000, Used 966, Requested 56. Please try again in 1.32s. "
        'Need more tokens? Upgrade at https://console.groq.com/settings/billing","type":"tokens",'
        '"code":"rate_limit_exceeded"}}'
    )
    err = map_error("groq", 429, body, {})
    assert isinstance(err, RateLimitError)
    assert err.retry_after == pytest.approx(1.32, abs=0.01)
    assert "org_01abcdefghijkl" not in str(err) and "org_<redacted>" in str(err)
    quota = map_error(
        "openai",
        429,
        '{"error":{"message":"You exceeded your current quota","code":"insufficient_quota"}}',
        {},
    )
    assert not isinstance(quota, RateLimitError)
