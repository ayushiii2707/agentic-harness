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
class HarnessConfig:
    optimize: bool = True  # False = plain pass-through client (baseline condition A)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    http: HTTPConfig = field(default_factory=HTTPConfig)
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
        sections = {"budget": BudgetConfig, "persistence": PersistenceConfig, "http": HTTPConfig}
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
