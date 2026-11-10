"""Configuration.

Defaults are safe: no paid calibration (budget 0), prompts stored only as hashes, optimisation on. Settings can
come from code, from environment variables, or from an ``adaptive_harness.toml`` file.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

from ._keys import get_setting
from .errors import ConfigurationError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

PromptStorage = Literal["none", "redacted", "full"]
_PROMPT_STORAGE = ("none", "redacted", "full")


def default_home() -> Path:
    """Per-user data directory: $ADAPTIVE_HARNESS_HOME, else the platform convention."""
    env = os.environ.get("ADAPTIVE_HARNESS_HOME")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "adaptive-harness"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "adaptive-harness"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "adaptive-harness"


@dataclass(frozen=True)
class BudgetConfig:
    """Hard caps on money the harness itself may spend (calibration, exploration, judges, experiments).

    The developer's own ``generate`` calls are never blocked by this budget; they are recorded in the ledger.
    """

    total_usd: float = 0.0
    daily_usd: float | None = None

    def __post_init__(self) -> None:
        if self.total_usd < 0 or (self.daily_usd is not None and self.daily_usd < 0):
            raise ConfigurationError("budgets cannot be negative")


@dataclass(frozen=True)
class PersistenceConfig:
    enabled: bool = True
    path: Path | None = None  # database file; default: <home>/harness.db
    prompt_storage: PromptStorage = "none"

    def __post_init__(self) -> None:
        if self.prompt_storage not in _PROMPT_STORAGE:
            raise ConfigurationError(f"prompt_storage must be one of {_PROMPT_STORAGE}")

    def resolved_path(self) -> Path:
        return self.path if self.path is not None else default_home() / "harness.db"


@dataclass(frozen=True)
class HTTPConfig:
    timeout_s: float = 120.0
    max_attempts: int = 3  # total attempts, including the first
    max_backoff_s: float = 30.0
    max_total_wait_s: float = 90.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.timeout_s <= 0:
            raise ConfigurationError("max_attempts must be >= 1 and timeout_s > 0")


@dataclass(frozen=True)
class PolicyConfig:
    """Strategy choice (DESIGN §6.4). Defaults: never switch without strong evidence; never explore."""

    exploration_rate: float = 0.0  # share of *checked* calls that try a uniformly random candidate strategy
    min_gain: float = 0.02  # a strategy must beat identity by at least this success-rate margin...
    confidence: float = 0.90  # ...with this posterior probability (Bonferroni-corrected)
    min_trials: int = 30  # evaluated outcomes needed for both identity and the challenger before comparing
    cost_weight: float = 0.1  # lambda of DESIGN §6.3: success-rate points given up per 100% extra tokens
    half_life_days: float | None = 90.0  # evidence decay for pinned model versions (None = no decay)
    alias_half_life_days: float = 21.0  # ...and for moving aliases such as *-latest
    drift_window: int = 30  # recent outcomes compared with the ones before them
    drift_reference: int = 120
    drift_z: float = 4.0  # tuned in scripts/simulate_drift.py: ~1% false alarms over 500 outcomes
    prior_evidence: bool = True  # load the shipped LeBlanc v4 prior records (weak; matching model IDs only)

    def __post_init__(self) -> None:
        if self.cost_weight < 0 or self.alias_half_life_days <= 0:
            raise ConfigurationError("cost_weight must be >= 0 and alias_half_life_days > 0")
        if self.half_life_days is not None and self.half_life_days <= 0:
            raise ConfigurationError("half_life_days must be > 0 or None")
        if self.drift_window < 5 or self.drift_reference < self.drift_window or self.drift_z <= 0:
            raise ConfigurationError("drift_window >= 5, drift_reference >= drift_window and drift_z > 0")
        if not 0.0 <= self.exploration_rate <= 1.0 or not 0.5 <= self.confidence < 1.0:
            raise ConfigurationError("exploration_rate must be in [0, 1] and confidence in [0.5, 1)")
        if self.min_gain < 0 or self.min_trials < 1:
            raise ConfigurationError("min_gain must be >= 0 and min_trials >= 1")


@dataclass(frozen=True)
class CalibrationConfig:
    """Automatic calibration (DESIGN §6.5). Off by default.

    When on, a cold or stale profile is calibrated on a background thread. A local provider needs nothing else. A
    paid provider also needs ``max_cost_per_call_usd > 0`` and harness budget; every call is reserved first.
    """

    auto: bool = False
    max_calls: int = 60  # per automatic run
    max_cost_per_call_usd: float = 0.0
    max_tokens: int | None = 600
    categories: tuple[str, ...] = ("math", "structured_output")  # code_generation needs Docker
    min_interval_s: float = 3600.0  # at most one automatic start per interval

    def __post_init__(self) -> None:
        if self.max_calls < 1 or self.max_cost_per_call_usd < 0 or self.min_interval_s < 0:
            raise ConfigurationError("max_calls >= 1, max_cost_per_call_usd >= 0 and min_interval_s >= 0")
        object.__setattr__(self, "categories", tuple(self.categories))


@dataclass(frozen=True)
class ClassifierConfig:
    """Opt-in LLM fallback for prompts the deterministic classifier labels ``general``."""

    provider: str
    model: str
    max_cost_per_call_usd: float = 0.001
    max_tokens: int = 1000  # reasoning models spend part of this before answering

    def __post_init__(self) -> None:
        if self.max_cost_per_call_usd < 0 or self.max_tokens < 1:
            raise ConfigurationError("max_cost_per_call_usd must be >= 0 and max_tokens >= 1")


@dataclass(frozen=True)
class RewriterConfig:
    """Opt-in ``llm_rewrite`` strategy (DESIGN §5.4): a rewriter model plus an independent judge model.

    Each auxiliary call reserves ``max_cost_per_call_usd`` from the harness budget; local providers are free.
    """

    rewriter_provider: str
    rewriter_model: str
    judge_provider: str
    judge_model: str
    max_cost_per_call_usd: float = 0.001
    max_tokens: int = 2000  # reasoning models spend part of this before answering

    def __post_init__(self) -> None:
        if (self.rewriter_provider, self.rewriter_model) == (self.judge_provider, self.judge_model):
            raise ConfigurationError("the judge must be a different model from the rewriter")
        if self.max_cost_per_call_usd < 0 or self.max_tokens < 1:
            raise ConfigurationError("max_cost_per_call_usd must be >= 0 and max_tokens >= 1")


@dataclass(frozen=True)
class HarnessConfig:
    optimize: bool = True  # False = plain pass-through client (baseline condition A)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    http: HTTPConfig = field(default_factory=HTTPConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    rewriter: RewriterConfig | None = None  # None = no LLM rewriting
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    llm_classifier: ClassifierConfig | None = None  # None = deterministic classifier only
    log_content: bool = False  # log prompt and response text at DEBUG level

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_env(cls, base: HarnessConfig | None = None) -> HarnessConfig:
        """Apply ``ADAPTIVE_HARNESS_*`` environment (or ``.env``) settings on top of ``base``."""
        cfg = base or cls()
        budget = get_setting("ADAPTIVE_HARNESS_BUDGET_USD")
        if budget is not None:
            try:
                cfg = replace(cfg, budget=replace(cfg.budget, total_usd=float(budget)))
            except ValueError as e:
                raise ConfigurationError("ADAPTIVE_HARNESS_BUDGET_USD must be a number") from e
        db = get_setting("ADAPTIVE_HARNESS_DB")
        if db:
            cfg = replace(cfg, persistence=replace(cfg.persistence, path=Path(db).expanduser()))
        storage = get_setting("ADAPTIVE_HARNESS_PROMPT_STORAGE")
        if storage:
            cfg = replace(cfg, persistence=replace(cfg.persistence, prompt_storage=storage))  # type: ignore[arg-type]
        return cfg

    @classmethod
    def from_toml(cls, path: str | Path) -> HarnessConfig:
        try:
            data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigurationError(f"cannot read config file {path}: {e}") from e
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HarnessConfig:
        sections = {
            "budget": BudgetConfig,
            "persistence": PersistenceConfig,
            "http": HTTPConfig,
            "policy": PolicyConfig,
            "rewriter": RewriterConfig,
            "calibration": CalibrationConfig,
            "llm_classifier": ClassifierConfig,
        }
        kwargs: dict[str, Any] = {}
        known = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key not in known:
                raise ConfigurationError(f"unknown config key {key!r}")
            if key in sections:
                if not isinstance(value, dict):
                    raise ConfigurationError(f"[{key}] must be a table")
                sub_known = {f.name for f in fields(sections[key])}
                unknown = set(value) - sub_known
                if unknown:
                    raise ConfigurationError(f"unknown keys in [{key}]: {sorted(unknown)}")
                if key == "persistence" and "path" in value:
                    value = dict(value, path=Path(value["path"]).expanduser())
                kwargs[key] = sections[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)
