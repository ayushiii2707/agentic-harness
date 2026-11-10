"""Public facade and request path.

Every ``generate`` / ``chat`` call runs one decision loop:

    analyse the task -> candidate strategies -> choose from stored evidence (identity unless the evidence says
    otherwise) -> transform (verbatim guard; any problem falls back to the original prompt) -> call the model ->
    record the trial -> evaluate if the caller supplied a check -> update the evidence the next choice will use.

Evidence only comes from real outcome signals: ``check=`` (a function the developer supplies) or ``feedback()``.
Without either, calls are recorded but never counted as successes or failures.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ._logging import get_logger
from ._version import __version__
from .budget import BudgetLedger
from .config import HarnessConfig
from .errors import StoreError
from .evaluation import Check, EvalStatus, EvaluationResult, feedback_result, run_check
from .hashing import text_hash
from .learning import Decision, EvidenceStore, Policy
from .models import ModelKey, ModelRegistry, deployment_for
from .providers.base import ModelInfo, ProviderAdapter
from .providers.registry import create_adapter
from .redaction import redact_text
from .store import Store
from .strategies import Strategy, StrategyRegistry
from .tasks import TaskAnalyzer, TaskSpec
from .types import CompletionRequest, CompletionResult, GenerationSettings, Message, StreamEvent

log = get_logger(__name__)


@dataclass(frozen=True)
class Trace:
    strategy_id: str
    strategy_version: str
    reason: str
    optimized: bool  # True when the prompt sent differs from the original
    category: str | None = None
    confidence: float | None = None
    candidates: tuple[str, ...] = ()
    explored: bool = False
    p_better: float | None = None
    evaluation_status: str = EvalStatus.NO_EVALUATION.value
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


@dataclass(frozen=True)
class _Plan:
    original: CompletionRequest
    sent: CompletionRequest
    strategy: Strategy
    spec: TaskSpec | None
    trace: Trace
    experiment_id: str | None = None
    task_ref: str | None = None


def _preserves(original: Sequence[Message], new: Sequence[Message]) -> bool:
    """Every original message must still be present, in order, and the last user text verbatim."""
    users = [m.content for m in new if m.role == "user"]
    last_user = next((m.content for m in reversed(original) if m.role == "user"), None)
    if last_user is None or not users or last_user not in users[-1]:
        return False
    it = iter(new)
    for m in original[:-1] if original and original[-1].role == "user" else original:
        if not any(n == m for n in it):
            return False
    return True


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
        strategies: StrategyRegistry | None = None,
        policy_seed: int | None = None,
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
        self.analyzer = TaskAnalyzer()
        self.strategies = strategies or StrategyRegistry()
        self.policy = Policy(self.config.policy, random.Random(policy_seed))
        self.store: Store | None = None
        self.ledger: BudgetLedger | None = None
        self.models: ModelRegistry | None = None
        self.evidence: EvidenceStore | None = None
        if self.config.persistence.enabled:
            try:
                self.store = Store(self.config.persistence.resolved_path())
                self.ledger = BudgetLedger(
                    self.store, total_usd=self.config.budget.total_usd, daily_usd=self.config.budget.daily_usd
                )
                self.models = ModelRegistry(self.store)
                self.evidence = EvidenceStore(self.store)
                for key in self.strategies.sync_with_store(self.store):
                    log.warning(
                        "strategy %s changed its wording without a new version; it is disabled because its stored "
                        "evidence describes the old wording (bump the strategy's version to use it)",
                        key,
                    )
            except StoreError as e:  # never block the user's call because the store is unavailable
                log.warning("evidence store unavailable, running stateless: %s", e)
                self.store = self.models = self.evidence = self.ledger = None

    # ------------------------------------------------------------------ public API
    def generate(
        self, prompt: str, *, system: str | None = None, check: Check | None = None, **settings: Any
    ) -> HarnessResponse:
        return self.chat(self._messages(prompt, system), check=check, **settings)

    def chat(
        self, messages: Sequence[Message | Mapping[str, Any]], *, check: Check | None = None, **settings: Any
    ) -> HarnessResponse:
        plan = self._plan(messages, settings, can_explore=check is not None)
        trial_id = uuid.uuid4().hex
        try:
            result = self.adapter.complete(plan.sent)
        except Exception as e:
            self._record(trial_id, plan, None, error=e)
            raise
        return self._finish(trial_id, plan, result, check)

    async def agenerate(
        self, prompt: str, *, system: str | None = None, check: Check | None = None, **settings: Any
    ) -> HarnessResponse:
        return await self.achat(self._messages(prompt, system), check=check, **settings)

    async def achat(
        self, messages: Sequence[Message | Mapping[str, Any]], *, check: Check | None = None, **settings: Any
    ) -> HarnessResponse:
        plan = self._plan(messages, settings, can_explore=check is not None)
        trial_id = uuid.uuid4().hex
        try:
            result = await self.adapter.acomplete(plan.sent)
        except Exception as e:
            self._record(trial_id, plan, None, error=e)
            raise
        return self._finish(trial_id, plan, result, check)

    def run_trial(
        self,
        prompt: str,
        *,
        strategy: Strategy,
        check: Check,
        experiment_id: str,
        task_ref: str,
        **settings: Any,
    ) -> HarnessResponse:
        """One experiment call with a fixed strategy (used by calibration).

        It goes through the same task analysis, transformation guard, trial record, evaluation and evidence
        update as ``generate``; only the choice of strategy is fixed by the experiment. In particular the
        evidence is keyed by the category the classifier assigns, exactly as it will be looked up later
        (no training/serving skew; see docs/DECISIONS.md D29).
        """
        plan = self._plan(self._messages(prompt, None), settings, can_explore=False, forced=strategy)
        plan = replace(plan, experiment_id=experiment_id, task_ref=task_ref)
        trial_id = uuid.uuid4().hex
        try:
            result = self.adapter.complete(plan.sent)
        except Exception as e:
            self._record(trial_id, plan, None, error=e)
            raise
        return self._finish(trial_id, plan, result, check)

    def stream(self, prompt: str, *, system: str | None = None, **settings: Any) -> Iterator[StreamEvent]:
        """Streaming uses the same strategy choice; there is no check, so it never explores."""
        plan = self._plan(self._messages(prompt, system), settings, can_explore=False)
        trial_id = uuid.uuid4().hex
        for event in self.adapter.stream(plan.sent):
            if event.result is not None:
                self._record(trial_id, plan, event.result)
            yield event

    def feedback(self, response_id: str, *, success: bool | None = None, score: float | None = None) -> str:
        """Report the real outcome of an earlier response. Returns the stored evaluation status."""
        evaluation = feedback_result(success=success, score=score)
        if self.store is None:
            return EvalStatus.NO_EVALUATION.value
        row = self.store.query_one(
            "SELECT model_ref, category_id, strategy_id, strategy_version FROM trial WHERE id=?",
            (response_id,),
        )
        if row is None:
            raise KeyError(f"unknown response id {response_id!r}")
        profile_id = None
        if row["model_ref"] is not None:
            p = self.store.query_one(
                "SELECT id FROM profile WHERE model_ref=? ORDER BY profile_version DESC LIMIT 1",
                (row["model_ref"],),
            )
            profile_id = p["id"] if p else None
        self._store_evaluation(
            response_id,
            evaluation,
            profile_id,
            row["category_id"],
            row["strategy_id"],
            row["strategy_version"],
        )
        return evaluation.status.value

    def profile(self) -> dict[str, Any] | None:
        """What the harness currently holds about the connected model (None before the first call)."""
        if self.models is None:
            return None
        ref = self.models.latest_model_ref(self.model_key)
        if ref is None:
            return None
        prof = self.models.profile(ref)
        if prof is not None and self.store is not None:
            prof["evidence"] = self.store.query(
                "SELECT category_id, strategy_id, strategy_version, alpha, beta, n_trials FROM arm_stats "
                "WHERE profile_id=? ORDER BY category_id, strategy_id",
                (prof["id"],),
            )
        return prof

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

    # ------------------------------------------------------------------ planning
    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[Message]:
        return ([Message("system", system)] if system else []) + [Message("user", prompt)]

    def _current_profile(self) -> tuple[int | None, str | None]:
        if self.models is None:
            return None, None
        ref = self.models.latest_model_ref(self.model_key)
        if ref is None:
            return None, None
        p = self.models.profile(ref)
        return (p["id"], p["status"]) if p else (None, None)

    def _plan(
        self,
        messages: Sequence[Message | Mapping[str, Any]],
        settings: Mapping[str, Any],
        *,
        can_explore: bool,
        forced: Strategy | None = None,
    ) -> _Plan:
        """Build the request to send. ``forced`` is used by calibration only: the strategy is fixed by the
        experiment instead of chosen by the policy. Everything else is identical to a normal call."""
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
        original = CompletionRequest.build(self.model, messages, gen)
        identity = self.strategies.get("identity")

        def passthrough(reason: str, spec: TaskSpec | None = None, notes: tuple[str, ...] = ()) -> _Plan:
            trace = Trace(
                identity.info.id,
                identity.info.version,
                reason,
                optimized=False,
                category=spec.category if spec else None,
                confidence=round(spec.confidence, 4) if spec else None,
                notes=notes,
            )
            return _Plan(original, original, identity, spec, trace)

        user_text = "\n\n".join(m.content for m in original.messages if m.role == "user")
        try:
            spec = self.analyzer.analyze(user_text)
        except Exception as e:  # task understanding must never block the call
            log.warning("task analysis failed: %s", e)
            return passthrough("identity: task analysis failed", notes=("task analysis failed",))
        if not self.config.optimize and forced is None:
            return passthrough("optimisation disabled (pass-through)", spec)

        try:
            candidates = self.strategies.candidates(spec)
            if forced is not None:
                decision = Decision(forced, "calibration: strategy fixed by the experiment")
            else:
                profile_id, status = self._current_profile()
                arms = self.evidence.arms(profile_id, spec.category) if (self.evidence and profile_id) else {}
                decision = self.policy.choose(
                    candidates,
                    arms,
                    can_explore=can_explore and self.evidence is not None,
                    exploit_allowed=profile_id is not None and status != "stale",
                )
        except Exception as e:
            log.warning("strategy choice failed: %s", e)
            return passthrough("identity: strategy choice failed", spec, ("strategy choice failed",))

        strategy = decision.strategy
        notes: tuple[str, ...] = ()
        sent = original
        if strategy.info.id != "identity":
            try:
                out = strategy.transform(original.messages, spec)
                if not _preserves(original.messages, out.messages):
                    raise ValueError("transformation did not keep the original messages verbatim")
                sent = replace(original, messages=out.messages)
            except Exception as e:
                log.warning("strategy %s failed, sending the original prompt: %s", strategy.info.id, e)
                return passthrough(
                    f"identity: {strategy.info.id} could not be applied",
                    spec,
                    (f"{strategy.info.id} failed",),
                )
        trace = Trace(
            strategy.info.id,
            strategy.info.version,
            decision.reason,
            optimized=sent is not original,
            category=spec.category,
            confidence=round(spec.confidence, 4),
            candidates=tuple(s.info.id for s in candidates),
            explored=decision.explored,
            p_better=None if decision.p_better is None else round(decision.p_better, 4),
            notes=notes,
        )
        return _Plan(original, sent, strategy, spec, trace)

    # ------------------------------------------------------------------ outcome
    def _finish(
        self, trial_id: str, plan: _Plan, result: CompletionResult, check: Check | None
    ) -> HarnessResponse:
        profile_id = self._record(trial_id, plan, result)
        trace = plan.trace
        if check is not None:
            evaluation = run_check(check, result.text)
            self._store_evaluation(
                trial_id,
                evaluation,
                profile_id,
                trace.category,
                trace.strategy_id,
                trace.strategy_version,
            )
            trace = replace(trace, evaluation_status=evaluation.status.value)
        return HarnessResponse(id=trial_id, text=result.text, result=result, trace=trace)

    def _store_evaluation(
        self,
        trial_id: str,
        evaluation: EvaluationResult,
        profile_id: int | None,
        category: str | None,
        strategy_id: str,
        strategy_version: str,
    ) -> None:
        if self.store is None:
            return
        try:
            already_counted = self.store.query_one(
                "SELECT 1 FROM evaluation WHERE trial_id=? AND status IN ('success','failure','partial')",
                (trial_id,),
            )
            self.store.execute(
                "INSERT INTO evaluation(trial_id, evaluator_id, evaluator_version, status, score, details_json, "
                "created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    trial_id,
                    evaluation.evaluator_id,
                    evaluation.evaluator_version,
                    evaluation.status.value,
                    evaluation.outcome,
                    json.dumps({"details": evaluation.details}),
                    time.time(),
                ),
            )
            outcome = evaluation.outcome
            # One outcome per trial updates the evidence; later evaluations of the same trial are kept as records.
            if outcome is not None and not already_counted and self.evidence and profile_id and category:
                self.evidence.record(profile_id, category, strategy_id, strategy_version, outcome)
        except StoreError as e:
            log.warning("could not record evaluation: %s", e)

    def _record(
        self, trial_id: str, plan: _Plan, result: CompletionResult | None, error: Exception | None = None
    ) -> int | None:
        """Store the trial; return the profile id of the model that answered (None when unknown)."""
        if self.store is None:
            return None
        storage = self.config.persistence.prompt_storage

        def keep(text: str) -> str | None:
            return None if storage == "none" else (redact_text(text) if storage == "redacted" else text)

        original_text = "\n\n".join(m.content for m in plan.original.messages if m.role == "user")
        sent_text = "\n\n".join(f"[{m.role}] {m.content}" for m in plan.sent.messages)
        u = result.usage if result else None
        trace = plan.trace
        profile_id = None
        try:
            model_ref = None
            if result is not None and self.models is not None:
                obs = self.models.observe(self.model_key, result)
                model_ref, profile_id = obs.model_ref, obs.profile_id
                if obs.changed:
                    log.warning(
                        "model %s changed (now %s); its earlier profile is marked stale",
                        self.model,
                        obs.resolved_version,
                    )
            self.store.execute(
                """INSERT INTO trial(id, experiment_id, task_ref, model_ref, category_id, provider, model_requested,
                     model_reported, strategy_id, strategy_version, settings_json, prompt_hash, original_prompt,
                     transformed_prompt, response_text, finish_reason, input_tokens, output_tokens, reasoning_tokens,
                     cost_usd, cost_is_estimated, latency_s, retries, fingerprint_json, dropped_settings, error,
                     status, simulated, trace_json, created_at, finished_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trial_id,
                    plan.experiment_id,
                    plan.task_ref,
                    model_ref,
                    trace.category,
                    self.adapter.name,
                    plan.sent.model,
                    result.model_reported if result else None,
                    trace.strategy_id,
                    trace.strategy_version,
                    json.dumps(asdict(plan.sent.settings)),
                    text_hash(original_text),
                    keep(original_text),
                    keep(sent_text),
                    keep(result.text) if result is not None else None,
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
                    result.cost_usd, note=f"generate {self.adapter.name}:{plan.sent.model}"
                )
        except StoreError as e:
            log.warning("could not record trial: %s", e)
        return profile_id


def software_version() -> str:
    return __version__
