"""Strategy registry: built-ins plus any registered by the user. Holds no model-to-strategy mapping."""

from __future__ import annotations

import json

from ..store import Store
from ..tasks import GENERAL, TaskSpec
from .base import Strategy
from .builtin import BUILTIN


class StrategyRegistry:
    def __init__(self, strategies: tuple[Strategy, ...] = BUILTIN) -> None:
        self._by_key: dict[tuple[str, str], Strategy] = {}
        for s in strategies:
            self.register(s)

    def register(self, strategy: Strategy) -> None:
        key = (strategy.info.id, strategy.info.version)
        existing = self._by_key.get(key)
        if existing is not None and existing.content_hash != strategy.content_hash:
            raise ValueError(f"strategy {key} is already registered with different content; bump its version")
        self._by_key[key] = strategy

    def get(self, strategy_id: str, version: str | None = None) -> Strategy:
        if version is not None:
            return self._by_key[(strategy_id, version)]
        candidates = [s for (sid, _), s in self._by_key.items() if sid == strategy_id]
        if not candidates:
            raise KeyError(strategy_id)
        return max(candidates, key=lambda s: int(s.info.version) if s.info.version.isdigit() else 0)

    def all(self) -> list[Strategy]:
        return list(self._by_key.values())

    def sync_with_store(self, store: Store) -> list[str]:
        """Record each strategy's content hash; disable any whose wording no longer matches the stored hash.

        Evidence is keyed by (id, version). If the text of a strategy changed without a version bump, its old
        evidence describes a different prompt, so the strategy must not be chosen on that evidence. Returns the
        disabled ``id@version`` keys (identity is never disabled).
        """
        disabled = []
        for (sid, ver), s in list(self._by_key.items()):
            meta = json.dumps({"description": s.info.description, "applies_to": list(s.info.applies_to)})
            store.execute(
                "INSERT OR IGNORE INTO strategy(id, version, content_hash, kind, metadata_json) VALUES (?,?,?,?,?)",
                (sid, ver, s.content_hash, s.info.kind, meta),
            )
            row = store.query_one("SELECT content_hash FROM strategy WHERE id=? AND version=?", (sid, ver))
            if row is not None and row["content_hash"] != s.content_hash and sid != "identity":
                del self._by_key[(sid, ver)]
                disabled.append(f"{sid}@{ver}")
        return disabled

    def candidates(self, spec: TaskSpec) -> list[Strategy]:
        """Strategies that may be tried for this task. Identity always comes first.

        When the task category is ``general`` (uncertain), only strategies marked ``general_safe`` are offered.
        """
        out = [s for s in self._by_key.values() if s.applicable(spec)]
        if spec.category == GENERAL:
            out = [s for s in out if s.info.general_safe]
        out.sort(key=lambda s: (s.info.id != "identity", s.info.id, s.info.version))
        return out
