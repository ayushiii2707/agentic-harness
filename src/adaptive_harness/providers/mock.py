"""A scripted, deterministic provider for tests and simulations.

Every result it returns is marked ``simulated=True``. The store keeps that flag, so simulated outcomes can never
be reported as real model behaviour.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence

from ..errors import ProviderError
from ..types import CompletionRequest, CompletionResult, FinishReason, Usage
from .base import ProviderAdapter, ProviderCapabilities

Responder = Callable[[CompletionRequest], str]


def echo_responder(request: CompletionRequest) -> str:
    return f"[mock] {request.messages[-1].content}"


class MockProvider(ProviderAdapter):
    name = "mock"

    def __init__(
        self,
        responder: Responder | Sequence[str | ProviderError] | None = None,
        *,
        seed: int = 0,
        price_per_token_usd: float = 0.0,
        model_reported: str | None = None,
    ) -> None:
        self._script: list[str | ProviderError] | None = None
        self._responder: Responder = echo_responder
        if callable(responder):
            self._responder = responder
        elif responder is not None:
            self._script = list(responder)
        self.rng = random.Random(seed)
        self.price = price_per_token_usd
        self.model_reported = model_reported
        self.requests: list[CompletionRequest] = []

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(local=True, seed=True)

    def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        t0 = time.perf_counter()
        if self._script is not None:
            if not self._script:
                raise ProviderError("mock script exhausted", provider=self.name)
            item = self._script.pop(0)
            if isinstance(item, ProviderError):
                raise item
            text = item
        else:
            text = self._responder(request)
        prompt_chars = sum(len(m.content) for m in request.messages)
        usage = Usage(input_tokens=max(1, prompt_chars // 4), output_tokens=max(1, len(text) // 4))
        cost = (usage.total_tokens or 0) * self.price
        return CompletionResult(
            text=text,
            finish_reason=FinishReason.STOP,
            usage=usage,
            model_reported=self.model_reported or request.model,
            latency_s=time.perf_counter() - t0,
            provider=self.name,
            cost_usd=cost,
            cost_is_estimated=False,
            simulated=True,
        )
