"""Shared HTTP plumbing for adapters: one httpx client, bounded retries, normalised errors, SSE parsing.

Retry policy:
- only rate limits (429 that is not a hard quota), server errors / overload (5xx, 529), timeouts and connection
  failures are retried;
- at most ``HTTPConfig.max_attempts`` attempts in total, and at most ``max_total_wait_s`` seconds spent waiting;
- a provider's ``Retry-After`` is honoured when it fits in the remaining wait budget, otherwise the error is raised.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .._keys import redact
from .._logging import get_logger
from ..config import HTTPConfig
from ..errors import (
    AuthenticationError,
    ContextLengthError,
    InvalidRequestError,
    MalformedResponseError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)

log = get_logger(__name__)

_CONTEXT_HINTS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "context window",
    "input is too long",
    "reduce the length",
)
_QUOTA_HINTS = ("insufficient_quota", "exceeded your current quota", "billing", "credit", "payment required")


@dataclass
class HTTPResponse:
    data: dict[str, Any]
    headers: Mapping[str, str]
    attempts: int
    latency_s: float


def error_message(body: str) -> str:
    """Pull a human-readable message out of an error body, redacted and bounded."""
    msg = body
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if isinstance(parsed, dict):
        err = parsed.get("error", parsed)
        if isinstance(err, dict):
            msg = str(err.get("message") or err.get("type") or err)
        elif isinstance(err, str):
            msg = err
    return redact(" ".join(msg.split()))[:400]


def parse_retry_after(headers: Mapping[str, str], body: str = "") -> float | None:
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)  # Gemini RetryInfo
    if m:
        return float(m.group(1))
    m = re.search(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)\b", body)  # Groq / OpenAI message text
    if m:
        return float(m.group(1)) / (1000.0 if m.group(2) == "ms" else 1.0)
    return None


def map_error(provider: str, status: int, body: str, headers: Mapping[str, str]) -> ProviderError:
    """Translate an HTTP error into the harness's error hierarchy."""
    msg = error_message(body)
    low = body.lower()
    request_id = headers.get("x-request-id") or headers.get("request-id")
    kw: dict[str, Any] = {"provider": provider, "status_code": status, "request_id": request_id}
    text = f"{provider} HTTP {status}: {msg}"
    if status in (401, 403) or "api_key_invalid" in low or "api key not valid" in low:
        return AuthenticationError(text, **kw)
    if status == 402 or (status == 429 and any(h in low for h in _QUOTA_HINTS)):
        return QuotaExceededError(text, **kw)
    if status == 429:
        return RateLimitError(text, retry_after=parse_retry_after(headers, body), **kw)
    if status == 408:
        return ProviderTimeoutError(text, **kw)
    if status in (400, 413, 422) and any(h in low for h in _CONTEXT_HINTS):
        return ContextLengthError(text, **kw)
    if status >= 500 or status == 529:
        return ProviderUnavailableError(text, **kw)
    return InvalidRequestError(text, **kw)


class HTTPTransport:
    """A small JSON-over-HTTP client with the retry policy above."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        headers: Mapping[str, str],
        config: HTTPConfig | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.config = config or HTTPConfig()
        self._client = client or httpx.Client(timeout=self.config.timeout_s, headers=dict(headers))
        if client is not None:
            self._client.headers.update(dict(headers))
        self._sleep = sleep

    def url(self, path: str) -> str:
        return path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ requests
    def request_json(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> HTTPResponse:
        def send() -> httpx.Response:
            return self._client.request(method, self.url(path), json=body, params=params)

        response, attempts, latency = self._with_retries(send)
        try:
            data = response.json()
        except ValueError as e:
            raise MalformedResponseError(
                f"{self.provider}: response is not JSON ({response.text[:120]!r})", provider=self.provider
            ) from e
        if not isinstance(data, dict):
            raise MalformedResponseError(f"{self.provider}: expected a JSON object", provider=self.provider)
        return HTTPResponse(data=data, headers=dict(response.headers), attempts=attempts, latency_s=latency)

    def stream_sse(
        self, path: str, body: Mapping[str, Any], params: Mapping[str, str] | None = None
    ) -> Iterator[dict[str, Any]]:
        """POST and yield each SSE ``data:`` payload parsed as JSON (``[DONE]`` ends the stream).

        Connection set-up is retried like a normal request; once bytes have streamed, errors are raised.
        """
        attempt = 0
        waited = 0.0
        while True:
            attempt += 1
            try:
                with self._client.stream("POST", self.url(path), json=body, params=params) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise map_error(self.provider, response.status_code, response.text, response.headers)
                    yield from _iter_sse(response, self.provider)
                    return
            except ProviderError as e:
                delay = self._retry_delay(e, attempt, waited)
                if delay is None:
                    raise
            except httpx.TimeoutException as e:
                err: ProviderError = ProviderTimeoutError(
                    f"{self.provider}: timed out", provider=self.provider
                )
                delay = self._retry_delay(err, attempt, waited)
                if delay is None:
                    raise err from e
            except httpx.TransportError as e:
                err = ProviderUnavailableError(
                    f"{self.provider}: connection failed: {redact(str(e))}", provider=self.provider
                )
                delay = self._retry_delay(err, attempt, waited)
                if delay is None:
                    raise err from e
            waited += delay
            self._sleep(delay)

    # ------------------------------------------------------------------ retry core
    def _retry_delay(self, error: ProviderError, attempt: int, waited: float) -> float | None:
        if not error.retryable or attempt >= self.config.max_attempts:
            return None
        backoff = min(self.config.max_backoff_s, (2 ** (attempt - 1)) + random.random())
        retry_after = getattr(error, "retry_after", None)
        delay = float(retry_after) if retry_after is not None else backoff
        if waited + delay > self.config.max_total_wait_s:
            return None
        log.info("%s: retrying after %.1fs (attempt %d): %s", self.provider, delay, attempt, error)
        return delay

    def _with_retries(self, send: Callable[[], httpx.Response]) -> tuple[httpx.Response, int, float]:
        attempt = 0
        waited = 0.0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            error: ProviderError
            try:
                response = send()
                latency = time.perf_counter() - t0
                if response.status_code < 400:
                    return response, attempt, latency
                error = map_error(self.provider, response.status_code, response.text, response.headers)
            except httpx.TimeoutException:
                error = ProviderTimeoutError(f"{self.provider}: request timed out", provider=self.provider)
            except httpx.TransportError as e:
                error = ProviderUnavailableError(
                    f"{self.provider}: connection failed: {redact(str(e))}", provider=self.provider
                )
            delay = self._retry_delay(error, attempt, waited)
            if delay is None:
                raise error
            waited += delay
            self._sleep(delay)


def _iter_sse(response: httpx.Response, provider: str) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload.strip() == "[DONE]":
                    return
                try:
                    obj = json.loads(payload)
                except ValueError as e:
                    raise MalformedResponseError(f"{provider}: bad SSE payload", provider=provider) from e
                if isinstance(obj, dict):
                    yield obj
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        payload = "\n".join(data_lines)
        if payload.strip() != "[DONE]":
            try:
                obj = json.loads(payload)
            except ValueError as e:
                raise MalformedResponseError(f"{provider}: bad SSE payload", provider=provider) from e
            if isinstance(obj, dict):
                yield obj
