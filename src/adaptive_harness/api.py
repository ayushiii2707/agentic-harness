"""Public facade and request path.

Every ``generate`` / ``chat`` call runs one decision loop:

    analyse the task -> candidate strategies -> choose from stored evidence (identity unless the evidence says
    otherwise) -> transform (verbatim guard; any problem falls back to the original prompt) -> call the model ->
    record the trial -> evaluate if the caller supplied a check -> update the evidence the next choice will use.

Evidence only comes from real outcome signals: ``check=`` (a function the developer supplies) or ``feedback()``.
Without either, calls are recorded but never counted as successes or failures.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ._logging import get_logger
from ._version import __version__
from .auxiliary import AuxModel
from .budget import BudgetLedger
from .config import HarnessConfig, RewriterConfig
from .errors import ProviderError, StoreError
from .evaluation import Check, EvalStatus, EvaluationResult, feedback_result, run_check
from .hashing import text_hash
from .learning import Decision, EvidenceStore, Policy
from .learning.drift import check_drift, mark_drift
from .learning.priors import load_priors
from .models import ModelKey, ModelRegistry, deployment_for
from .orchestrator import Orchestrator
from .providers.base import ModelInfo, ProviderAdapter
from .providers.registry import create_adapter
from .redaction import redact_text, scrub_quoted
from .store import Store
from .strategies import Strategy, StrategyRegistry
from .tasks import TaskAnalyzer, TaskSpec
from .tasks.llm_classifier import LlmFallbackClassifier
from .transform import apply_strategy
from .transform.llm_rewrite import build_llm_rewrite
from .types import CompletionRequest, CompletionResult, GenerationSettings, Message, StreamEvent, ToolCall

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

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Tool calls the model asked for (pass ``tools`` to the provider via ``extra=``)."""
        return self.result.tool_calls


@dataclass(frozen=True)
class _Plan:
    original: CompletionRequest
    sent: CompletionRequest
    strategy: Strategy
    spec: TaskSpec | None
    trace: Trace
    experiment_id: str | None = None
    task_ref: str | None = None


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
        aux_adapters: Mapping[str, ProviderAdapter] | None = None,
        **provider_options: Any,
    ) -> None:
        # Explicit configuration wins: environment settings apply only when no config is passed (finding F8).
        self.config = config if config is not None else HarnessConfig.from_env()
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
        self.orchestrator = Orchestrator(self)
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
                pc = self.config.policy
                days = pc.alias_half_life_days if self.model_key.alias else pc.half_life_days
                self.evidence = EvidenceStore(
                    self.store, half_life_s=None if days is None else days * 86400.0
                )
                if pc.prior_evidence:
                    load_priors(self.store)
                for key in self.strategies.sync_with_store(self.store):
                    log.warning(
                        "strategy %s changed its wording without a new version; it is disabled because its stored "
                        "evidence describes the old wording (bump the strategy's version to use it)",
                        key,
                    )
            except StoreError as e:  # never block the user's call because the store is unavailable
                log.warning("evidence store unavailable, running stateless: %s", e)
                self.store = self.models = self.evidence = self.ledger = None
        if self.config.rewriter is not None:
            self._add_llm_rewrite(self.config.rewriter, aux_adapters or {})
        if self.config.llm_classifier is not None:
            lc = self.config.llm_classifier
            adapter = (aux_adapters or {}).get("classifier") or create_adapter(
                lc.provider, **({} if lc.provider.lower() == "mock" else {"http": self.config.http})
            )
            self.analyzer = TaskAnalyzer(
                LlmFallbackClassifier(
                    AuxModel(
                        adapter,
                        lc.model,
                        self.ledger,
                        max_cost_per_call_usd=lc.max_cost_per_call_usd,
                        max_tokens=lc.max_tokens,
                    ),
                    store=self.store,
                )
            )

    def _add_llm_rewrite(self, rc: RewriterConfig, adapters: Mapping[str, ProviderAdapter]) -> None:
        """Register ``llm_rewrite``. ``adapters`` may supply the "rewriter"/"judge" adapters (tests, custom)."""

        def aux(role: str, provider: str, model: str) -> AuxModel:
            adapter = adapters.get(role) or create_adapter(
                provider, **({} if provider.lower() == "mock" else {"http": self.config.http})
            )
            return AuxModel(
                adapter,
                model,
                self.ledger,
                max_cost_per_call_usd=rc.max_cost_per_call_usd,
                max_tokens=rc.max_tokens,
            )

        strategy = build_llm_rewrite(
            aux("rewriter", rc.rewriter_provider, rc.rewriter_model),
            aux("judge", rc.judge_provider, rc.judge_model),
        )
        self.strategies.register(strategy)
        if self.store is not None:
            self.strategies.sync_with_store(self.store)

    # ------------------------------------------------------------------ public API
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        check: Check | None = None,
        background: bool = False,
        extra: Mapping[str, Any] | None = None,
        **settings: Any,
    ) -> HarnessResponse:
        """Send ``prompt``. ``check`` scores the answer (the only way the harness learns); with
        ``background=True`` the check runs on a worker thread and the response returns at once. ``extra`` is passed
        to the provider unchanged (e.g. ``tools``, ``tool_choice``, ``response_format``)."""
        return self.chat(
            self._messages(prompt, system), check=check, background=background, extra=extra, **settings
        )

    def chat(
        self,
        messages: Sequence[Message | Mapping[str, Any]],
        *,
        check: Check | None = None,
        background: bool = False,
        extra: Mapping[str, Any] | None = None,
        **settings: Any,
    ) -> HarnessResponse:
        plan = self._plan(messages, settings, can_explore=check is not None, extra=extra)
        trial_id = uuid.uuid4().hex
        try:
            result = self.adapter.complete(plan.sent)
        except Exception as e:
            self._record(trial_id, plan, None, error=e)
            raise
        return self._finish(trial_id, plan, result, check, background=background)

    async def agenerate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        check: Check | None = None,
        extra: Mapping[str, Any] | None = None,
        **settings: Any,
    ) -> HarnessResponse:
        return await self.achat(self._messages(prompt, system), check=check, extra=extra, **settings)

    async def achat(
        self,
        messages: Sequence[Message | Mapping[str, Any]],
        *,
        check: Check | None = None,
        extra: Mapping[str, Any] | None = None,
        **settings: Any,
    ) -> HarnessResponse:
        """Async chat. Planning (which may call a rewriter and judge) and the check run in a worker thread, so they
        never block the event loop."""
        plan = await asyncio.to_thread(
            self._plan, messages, settings, can_explore=check is not None, extra=extra
        )
        trial_id = uuid.uuid4().hex
        try:
            result = await self.adapter.acomplete(plan.sent)
        except Exception as e:
            await asyncio.to_thread(self._record, trial_id, plan, None, e)
            raise
        return await asyncio.to_thread(self._finish, trial_id, plan, result, check)

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

    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        check: Check | None = None,
        background: bool = False,
        extra: Mapping[str, Any] | None = None,
        **settings: Any,
    ) -> Iterator[StreamEvent]:
        """Stream the answer. The trial is recorded, and ``check`` (if given) run on the full text, after the final
        event. Exploration happens only when a check is given, as for ``generate``."""
        plan = self._plan(
            self._messages(prompt, system), settings, can_explore=check is not None, extra=extra
        )
        trial_id = uuid.uuid4().hex
        finished = False
        try:
            for event in self.adapter.stream(plan.sent):
                if event.result is not None:
                    self._finish(trial_id, plan, event.result, check, background=background)
                    finished = True
                yield event
        except ProviderError as e:
            finished = True
            self._record(trial_id, plan, None, error=e)
            raise
        finally:
            if not finished:  # the consumer stopped early, or the stream ended without a result (finding F12)
                self._record(trial_id, plan, None, error=RuntimeError("stream closed before the final event"))

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for background evaluations (and an automatic recalibration, if one is running). True if all done."""
        return self.orchestrator.flush(timeout)

    def feedback(self, response_id: str, *, success: bool | None = None, score: float | None = None) -> str:
        """Report the real outcome of an earlier response. Returns the stored evaluation status."""
        evaluation = feedback_result(success=success, score=score)
        if self.store is None:
            return EvalStatus.NO_EVALUATION.value
        row = self.store.query_one(
            "SELECT model_ref, profile_id, category_id, strategy_id, strategy_version FROM trial WHERE id=?",
            (response_id,),
        )
        if row is None:
            raise KeyError(f"unknown response id {response_id!r}")
        # Credit the profile that served the response (finding F5). Trials recorded before migration 0003 have no
        # profile id; they fall back to the model's latest profile, as before.
        profile_id = row["profile_id"]
        if profile_id is None and row["model_ref"] is not None:
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

    def close(self, timeout: float | None = 30.0) -> None:
        """Cancel background calibration, wait up to ``timeout`` s for pending background checks, release resources."""
        self.orchestrator.close(timeout)
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
        extra: Mapping[str, Any] | None = None,
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
        original = CompletionRequest.build(self.model, messages, gen, extra)
        if not any(m.content.strip() or m.tool_calls for m in original.messages):
            raise ValueError("the prompt is empty")  # never pay for an empty request (finding F15)
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
                arms = (
                    self.evidence.arms(
                        profile_id,
                        spec.category,
                        parent=self._parent(spec.category),
                        model_id=self.model,
                        host=self.adapter.name,
                    )
                    if (self.evidence and profile_id)
                    else {}
                )
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
            applied = apply_strategy(strategy, original.messages, spec)
            if not applied.ok:
                log.warning(
                    "strategy %s not applied, sending the original prompt: %s",
                    strategy.info.id,
                    applied.reason if self.config.log_content else scrub_quoted(applied.reason),
                )
                return passthrough(
                    f"identity: {strategy.info.id} could not be applied",
                    spec,
                    (f"{strategy.info.id} rejected: {applied.reason}"[:300],),
                )
            sent = replace(original, messages=applied.messages)
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
        self,
        trial_id: str,
        plan: _Plan,
        result: CompletionResult,
        check: Check | None,
        *,
        background: bool = False,
    ) -> HarnessResponse:
        """Record the trial and evaluate it. Nothing here can lose the response: any failure is logged."""
        trace = plan.trace
        try:
            profile_id = self._record(trial_id, plan, result)
        except Exception as e:
            log.warning("could not record the trial: %s: %s", type(e).__name__, e)
            profile_id = None
        if self.config.calibration.auto is not False and plan.experiment_id is None:
            try:
                self.orchestrator.maybe_recalibrate(self._current_profile()[1])
            except Exception as e:
                log.warning("automatic calibration check failed: %s", e)
        if check is not None:
            if background:
                self.orchestrator.submit(lambda: self._evaluate(trial_id, plan, result, check, profile_id))
                trace = replace(trace, evaluation_status="pending")
            else:
                trace = replace(
                    trace, evaluation_status=self._evaluate(trial_id, plan, result, check, profile_id)
                )
        return HarnessResponse(id=trial_id, text=result.text, result=result, trace=trace)

    def _evaluate(
        self, trial_id: str, plan: _Plan, result: CompletionResult, check: Check, profile_id: int | None
    ) -> str:
        evaluation = run_check(check, result.text)  # never raises: a crashing check is an evaluation_error
        try:
            self._store_evaluation(
                trial_id,
                evaluation,
                profile_id,
                plan.trace.category,
                plan.trace.strategy_id,
                plan.trace.strategy_version,
            )
        except Exception as e:
            log.warning("could not store the evaluation: %s: %s", type(e).__name__, e)
        return evaluation.status.value

    def _keep_details(self, details: str) -> str:
        """Evaluator details can contain parts of the response (e.g. sandbox test output). With
        ``prompt_storage="none"`` only the diagnosis before the first colon is kept, with quoted spans removed."""
        storage = self.config.persistence.prompt_storage
        if storage == "full":
            return details
        if storage == "redacted":
            return redact_text(details)
        return scrub_quoted(details.split(":", 1)[0].split("|", 1)[0])[:80].strip()

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
            outcome = evaluation.outcome
            row_values = (
                trial_id,
                evaluation.evaluator_id,
                evaluation.evaluator_version,
                evaluation.status.value,
                outcome,
                json.dumps({"details": self._keep_details(evaluation.details)}),
                time.time(),
            )
            insert = (
                "INSERT {} INTO evaluation(trial_id, evaluator_id, evaluator_version, status, score, details_json, "
                "created_at, counted) VALUES (?,?,?,?,?,?,?,?)"
            )
            # At most one evaluation per trial counts as evidence. The unique index on (trial_id) WHERE counted=1
            # decides atomically which one, even across threads and processes (finding F4); a later evaluation is
            # kept as a record only.
            counted = False
            if outcome is not None:
                counted = self.store.execute(insert.format("OR IGNORE"), (*row_values, 1)) == 1
            if not counted:
                self.store.execute(insert.format(""), (*row_values, 0))
            if counted and outcome is not None and self.evidence and profile_id and category:
                row = self.store.query_one(
                    "SELECT input_tokens, output_tokens FROM trial WHERE id=?", (trial_id,)
                )
                tokens = None
                if row is not None and (row["input_tokens"] is not None or row["output_tokens"] is not None):
                    tokens = (row["input_tokens"] or 0) + (row["output_tokens"] or 0)
                self.evidence.record(
                    profile_id, category, strategy_id, strategy_version, outcome, tokens=tokens
                )
                self._check_drift(profile_id, category, strategy_id, strategy_version)
        except StoreError as e:
            log.warning("could not record evaluation: %s", e)

    def _parent(self, category: str) -> str | None:
        c = self.analyzer.classifier.taxonomy.categories.get(category)
        return c.parent if c is not None else None

    def _check_drift(self, profile_id: int, category: str, strategy_id: str, version: str) -> None:
        assert self.evidence is not None and self.store is not None
        pc = self.config.policy
        outcomes = self.evidence.outcomes(
            profile_id, category, strategy_id, version, pc.drift_window + pc.drift_reference
        )
        r = check_drift(outcomes, window=pc.drift_window, reference=pc.drift_reference, z_crit=pc.drift_z)
        if r.drifted:
            detail = {
                "category": category,
                "strategy": f"{strategy_id}@{version}",
                "z": round(r.z or 0.0, 3),
                "recent_mean": r.recent_mean,
                "reference_mean": r.reference_mean,
            }
            if mark_drift(self.store, profile_id, detail):
                # The old evidence describes behaviour that has changed: keep it for audit, and start a fresh
                # profile version at once so learning resumes without a manual recalibration (finding F2).
                row = self.store.query_one("SELECT model_ref FROM profile WHERE id=?", (profile_id,))
                if row is not None and self.models is not None:
                    self.models.next_profile(int(row["model_ref"]), status="cold")
                log.warning(
                    "outcome drift detected for %s (%s); profile marked stale, a new profile version started",
                    self.model,
                    detail,
                )

    def _record(
        self, trial_id: str, plan: _Plan, result: CompletionResult | None, error: Exception | None = None
    ) -> int | None:
        """Store the trial; return the profile id of the model that answered (None when unknown)."""
        if self.store is None:
            return None
        storage = self.config.persistence.prompt_storage

        def keep(text: str) -> str | None:
            return None if storage == "none" else (redact_text(text) if storage == "redacted" else text)

        def keep_note(
            note: str,
        ) -> str:  # notes may quote prompt fragments (e.g. a rejected rewrite's reasons)
            return (
                scrub_quoted(note)
                if storage == "none"
                else (redact_text(note) if storage == "redacted" else note)
            )

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
                """INSERT INTO trial(id, experiment_id, task_ref, model_ref, profile_id, category_id, provider,
                     model_requested,
                     model_reported, strategy_id, strategy_version, settings_json, prompt_hash, original_prompt,
                     transformed_prompt, response_text, finish_reason, input_tokens, output_tokens, reasoning_tokens,
                     cost_usd, cost_is_estimated, latency_s, retries, fingerprint_json, dropped_settings, error,
                     status, simulated, trace_json, created_at, finished_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trial_id,
                    plan.experiment_id,
                    plan.task_ref,
                    model_ref,
                    profile_id,
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
                    json.dumps(asdict(replace(trace, notes=tuple(keep_note(n) for n in trace.notes)))),
                    time.time(),
                    time.time(),
                ),
            )
            # Calibration calls are harness spend, committed by the calibrator; only the developer's own calls are
            # recorded as user spend (finding F6).
            if (
                result is not None
                and result.cost_usd
                and self.ledger is not None
                and plan.experiment_id is None
            ):
                self.ledger.record_user_spend(
                    result.cost_usd, note=f"generate {self.adapter.name}:{plan.sent.model}"
                )
        except StoreError as e:
            log.warning("could not record trial: %s", e)
        return profile_id


def software_version() -> str:
    return __version__
