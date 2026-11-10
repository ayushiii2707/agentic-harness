"""Public facade.

``AdaptiveHarness.generate`` and ``chat`` always go through the harness path. In this release the optimisation
step is the identity strategy (the prompt is sent unchanged); strategy selection arrives in later phases. The
trace says so explicitly, so nobody mistakes pass-through for optimisation.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ._logging import get_logger
from ._version import __version__
from .budget import BudgetLedger
from .config import HarnessConfig
from .errors import StoreError
from .hashing import text_hash
from .models import ModelKey, ModelRegistry, deployment_for
from .providers.base import ModelInfo, ProviderAdapter
from .providers.registry import create_adapter
from .redaction import redact_text
from .store import Store
from .types import CompletionRequest, CompletionResult, GenerationSettings, Message, StreamEvent

log = get_logger(__name__)

IDENTITY = ("identity", "1")


@dataclass(frozen=True)
class Trace:
    strategy_id: str
    strategy_version: str
    reason: str
    optimized: bool
    category: str | None = None
    confidence: float | None = None
    evaluation_status: str = "no_evaluation"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarnessResponse:
    id: str
    text: str
    result: CompletionResult = field(repr=False)
    trace: Trace

    def __str__(self) -> str:
        return self.text

    @property
    def usage(self) -> Any:
        return self.result.usage

    @property
    def cost_usd(self) -> float | None:
        return self.result.cost_usd


class AdaptiveHarness:
    """Drop-in client: ``AdaptiveHarness(provider=..., model=...).generate(prompt)``."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        config: HarnessConfig | None = None,
        adapter: ProviderAdapter | None = None,
        **provider_options: Any,
    ) -> None:
        self.config = HarnessConfig.from_env(config)
        self.provider = provider
        self.model = model
        if adapter is not None:
            self.adapter = adapter
        else:
            opts = dict(provider_options)
            if provider.lower() != "mock":
                opts.update(api_key=api_key, base_url=base_url, http=self.config.http)
            self.adapter = create_adapter(provider, **opts)
        transport = getattr(self.adapter, "transport", None)
        self.model_key = ModelKey(
            provider=self.adapter.name,
            model_id=model,
            deployment=deployment_for(
                getattr(transport, "base_url", None), self.adapter.capabilities().local
            ),
        )
        self.store: Store | None = None
        self.ledger: BudgetLedger | None = None
        self.models: ModelRegistry | None = None
        if self.config.persistence.enabled:
            try:
                self.store = Store(self.config.persistence.resolved_path())
                self.ledger = BudgetLedger(
                    self.store, total_usd=self.config.budget.total_usd, daily_usd=self.config.budget.daily_usd
                )
                self.models = ModelRegistry(self.store)
            except StoreError as e:  # never block the user's call because the store is unavailable
                log.warning("evidence store unavailable, running stateless: %s", e)
                self.store = None

    # ------------------------------------------------------------------ public API
    def generate(self, prompt: str, *, system: str | None = None, **settings: Any) -> HarnessResponse:
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        return self.chat(messages, **settings)

    def chat(self, messages: Sequence[Message | Mapping[str, Any]], **settings: Any) -> HarnessResponse:
        request, trace = self._prepare(messages, settings)
        trial_id = uuid.uuid4().hex
        try:
            result = self.adapter.complete(request)
        except Exception as e:
            self._record(trial_id, request, None, trace, error=e)
            raise
        self._record(trial_id, request, result, trace)
        return HarnessResponse(id=trial_id, text=result.text, result=result, trace=trace)

    async def agenerate(self, prompt: str, *, system: str | None = None, **settings: Any) -> HarnessResponse:
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        return await self.achat(messages, **settings)

    async def achat(
        self, messages: Sequence[Message | Mapping[str, Any]], **settings: Any
    ) -> HarnessResponse:
        request, trace = self._prepare(messages, settings)
        trial_id = uuid.uuid4().hex
        try:
            result = await self.adapter.acomplete(request)
        except Exception as e:
            self._record(trial_id, request, None, trace, error=e)
            raise
        self._record(trial_id, request, result, trace)
        return HarnessResponse(id=trial_id, text=result.text, result=result, trace=trace)

    def stream(self, prompt: str, *, system: str | None = None, **settings: Any) -> Iterator[StreamEvent]:
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        request, trace = self._prepare(messages, settings)
        trial_id = uuid.uuid4().hex
        for event in self.adapter.stream(request):
            if event.result is not None:
                self._record(trial_id, request, event.result, trace)
            yield event

    def profile(self) -> dict[str, Any] | None:
        """What the harness currently holds about the connected model (None before the first call)."""
        if self.models is None:
            return None
        ref = self.models.latest_model_ref(self.model_key)
        return self.models.profile(ref) if ref is not None else None

    def model_info(self) -> ModelInfo:
        """Ask the provider for model metadata (an explicit request) and record the facts with provenance.

        Facts are stored once the model has answered at least one call, because the model row is keyed by the
        version the provider reports in its responses.
        """
        info = self.adapter.model_info(self.model)
        if self.models is not None:
            ref = self.models.latest_model_ref(self.model_key)
            if ref is not None:
                try:
                    self.models.record_facts(ref, info)
                except StoreError as e:
                    log.warning("could not record model facts: %s", e)
        return info

    def close(self) -> None:
        self.adapter.close()
        if self.store is not None:
            self.store.close()

    def __enter__(self) -> AdaptiveHarness:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ internals
    def _prepare(
        self, messages: Sequence[Message | Mapping[str, Any]], settings: Mapping[str, Any]
    ) -> tuple[CompletionRequest, Trace]:
        known = {"max_tokens", "temperature", "top_p", "stop", "seed"}
        unknown = set(settings) - known
        if unknown:
            raise TypeError(f"unknown generation settings: {sorted(unknown)}")
        gen = GenerationSettings(
            max_tokens=settings.get("max_tokens"),
            temperature=settings.get("temperature"),
            top_p=settings.get("top_p"),
            stop=tuple(settings.get("stop") or ()),
            seed=settings.get("seed"),
        )
        request = CompletionRequest.build(self.model, messages, gen)
        reason = (
            "optimisation disabled (pass-through)"
            if not self.config.optimize
            else "strategy selection not implemented yet: identity pass-through"
        )
        trace = Trace(strategy_id=IDENTITY[0], strategy_version=IDENTITY[1], reason=reason, optimized=False)
        return request, trace

    def _record(
        self,
        trial_id: str,
        request: CompletionRequest,
        result: CompletionResult | None,
        trace: Trace,
        error: Exception | None = None,
    ) -> None:
        if self.store is None:
            return
        user_text = "\n\n".join(m.content for m in request.messages if m.role == "user")
        storage = self.config.persistence.prompt_storage
        stored_prompt = (
            None if storage == "none" else (redact_text(user_text) if storage == "redacted" else user_text)
        )
        stored_response = None
        if result is not None and storage != "none":
            stored_response = redact_text(result.text) if storage == "redacted" else result.text
        u = result.usage if result else None
        try:
            model_ref = None
            if result is not None and self.models is not None:
                obs = self.models.observe(self.model_key, result)
                model_ref = obs.model_ref
                if obs.changed:
                    log.warning(
                        "model %s changed (now %s); its earlier profile is marked stale",
                        self.model,
                        obs.resolved_version,
                    )
            self.store.execute(
                """INSERT INTO trial(id, model_ref, provider, model_requested, model_reported, strategy_id, strategy_version,
                     settings_json, prompt_hash, original_prompt, transformed_prompt, response_text, finish_reason,
                     input_tokens, output_tokens, reasoning_tokens, cost_usd, cost_is_estimated, latency_s, retries,
                     fingerprint_json, dropped_settings, error, status, simulated, trace_json, created_at, finished_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trial_id,
                    model_ref,
                    self.adapter.name,
                    request.model,
                    result.model_reported if result else None,
                    trace.strategy_id,
                    trace.strategy_version,
                    json.dumps(asdict(request.settings)),
                    text_hash(user_text),
                    stored_prompt,
                    stored_prompt,  # identity: transformed == original
                    stored_response,
                    result.finish_reason.value if result else None,
                    u.input_tokens if u else None,
                    u.output_tokens if u else None,
                    u.reasoning_tokens if u else None,
                    result.cost_usd if result else None,
                    int(result.cost_is_estimated) if result else 0,
                    result.latency_s if result else None,
                    result.retries if result else 0,
                    json.dumps(dict(result.fingerprint)) if result else "{}",
                    json.dumps(list(result.dropped_settings)) if result else "[]",
                    f"{type(error).__name__}: {error}"[:500] if error else None,
                    "failed" if error else "done",
                    int(result.simulated) if result else 0,
                    json.dumps(asdict(trace)),
                    time.time(),
                    time.time(),
                ),
            )
            if result is not None and result.cost_usd and self.ledger is not None:
                self.ledger.record_user_spend(
                    result.cost_usd, note=f"generate {self.adapter.name}:{request.model}"
                )
        except StoreError as e:
            log.warning("could not record trial: %s", e)


def software_version() -> str:
    return __version__
