"""Provider adapter interface.

An adapter translates provider-neutral :class:`~adaptive_harness.types.CompletionRequest` objects into one
provider's wire format and back. Everything provider-specific stays inside the adapter.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from ..types import CompletionRequest, CompletionResult, StreamEvent

Provenance = Literal["verified", "provider_reported", "inferred", "observed"]


@dataclass(frozen=True)
class ProviderCapabilities:
    system_role: bool = True
    streaming: bool = True
    temperature: bool = True
    top_p: bool = True
    seed: bool = False
    json_mode: bool = False
    tools: bool = False
    reports_cost: bool = False
    local: bool = False


@dataclass(frozen=True)
class MetadataFact:
    key: str
    value: Any
    provenance: Provenance
    source: str = ""


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    facts: tuple[MetadataFact, ...] = field(default_factory=tuple)

    def get(self, key: str, default: Any = None) -> Any:
        for f in self.facts:
            if f.key == key:
                return f.value
        return default


class ProviderAdapter(abc.ABC):
    """Base class for all adapters."""

    name: str = "base"

    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @abc.abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResult: ...

    async def acomplete(self, request: CompletionRequest) -> CompletionResult:
        """Async completion. Adapters with a native async client override this."""
        return await asyncio.to_thread(self.complete, request)

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        """Streaming completion. The default yields one event; adapters with SSE support override this."""
        result = self.complete(request)
        if result.text:
            yield StreamEvent(delta=result.text)
        yield StreamEvent(result=result)

    def model_info(self, model: str) -> ModelInfo:
        """Metadata the provider exposes about ``model``. May be empty: providers differ widely."""
        return ModelInfo(model_id=model)

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""
