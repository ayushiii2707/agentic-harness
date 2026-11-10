"""Budgeted auxiliary model calls (rewriter, judge, LLM classifier).

Every call to a non-local provider first reserves ``max_cost_per_call_usd`` in the persistent budget ledger and
then commits the reported cost (or the whole reservation when no cost is reported). Local providers cost nothing.
Any refusal or failure raises; callers fail closed.
"""

from __future__ import annotations

from .budget import BudgetLedger
from .errors import BudgetExceededError
from .providers.base import ProviderAdapter
from .types import CompletionRequest, GenerationSettings, Message


class AuxModel:
    """One budgeted auxiliary model (rewriter or judge). Raises on refusal or failure; callers fail closed."""

    def __init__(
        self,
        adapter: ProviderAdapter,
        model: str,
        ledger: BudgetLedger | None,
        *,
        max_cost_per_call_usd: float,
        max_tokens: int,
    ) -> None:
        self.adapter, self.model, self.ledger = adapter, model, ledger
        self.max_cost = max_cost_per_call_usd
        self.max_tokens = max_tokens
        self.local = adapter.capabilities().local
        self.calls = 0

    @property
    def key(self) -> tuple[str, str]:
        return (self.adapter.name, self.model)

    def complete(self, system: str, user: str, purpose: str) -> str:
        request = CompletionRequest.build(
            self.model,
            [Message("system", system), Message("user", user)],
            GenerationSettings(max_tokens=self.max_tokens, temperature=0.0),
        )
        if self.local:
            self.calls += 1
            return self.adapter.complete(request).text
        if self.ledger is None:
            raise BudgetExceededError(
                "no budget ledger (persistence disabled): paid auxiliary calls refused",
                requested_usd=self.max_cost,
                remaining_usd=0.0,
            )
        reservation = self.ledger.reserve(self.max_cost, note=f"{purpose}:{self.model}")
        try:
            result = self.adapter.complete(request)
        except BaseException:
            self.ledger.release(reservation, note=f"{purpose} failed")
            raise
        cost = result.cost_usd if result.cost_usd is not None else self.max_cost
        self.ledger.commit(reservation, cost, note=purpose)
        self.calls += 1
        return result.text
