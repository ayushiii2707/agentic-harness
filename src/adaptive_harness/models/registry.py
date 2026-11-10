"""Persistent model registry: model rows, metadata facts with provenance, profiles and staleness.

Rules (DESIGN §5.1, §6.7):
- a new resolved version of the same (provider, model id, deployment) is a new model row with its own profile;
  the previous version's profiles become ``stale`` and a ``fingerprint`` change event is recorded;
- a changed fingerprint on the same version (e.g. a different serving host) marks that row's profiles stale;
- a stale profile is kept, never deleted; recalibration creates the next profile version (``next_profile``);
- no network call is made here; metadata facts come from responses or from an explicit ``model_info()`` call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ..providers.base import MetadataFact, ModelInfo
from ..store import Store
from ..types import CompletionResult
from .identity import ModelKey, fingerprint_of, resolved_version


@dataclass(frozen=True)
class Observation:
    model_ref: int
    profile_id: int
    resolved_version: str | None
    changed: bool  # True when this call revealed a new version or fingerprint


class ModelRegistry:
    def __init__(self, store: Store) -> None:
        self.store = store

    # ------------------------------------------------------------------ observation
    def observe(self, key: ModelKey, result: CompletionResult) -> Observation:
        version = resolved_version(result)
        fp = fingerprint_of(result)
        now = time.time()
        changed = False
        with self.store.transaction() as con:
            row = con.execute(
                "SELECT id, fingerprint FROM model WHERE provider=? AND model_id=? AND deployment=? "
                "AND resolved_version IS ?",
                (key.provider, key.model_id, key.deployment, version),
            ).fetchone()
            if row is None:
                previous = [
                    r[0]
                    for r in con.execute(
                        "SELECT id FROM model WHERE provider=? AND model_id=? AND deployment=?",
                        (key.provider, key.model_id, key.deployment),
                    ).fetchall()
                ]
                cur = con.execute(
                    "INSERT INTO model(provider, model_id, resolved_version, deployment, fingerprint, alias, "
                    "first_seen, last_seen) VALUES (?,?,?,?,?,?,?,?)",
                    (key.provider, key.model_id, version, key.deployment, fp, int(key.alias), now, now),
                )
                model_ref = int(cur.lastrowid or 0)
                for old in previous:
                    changed |= self._mark_stale(
                        con, old, {"reason": "new resolved version", "to": version}, now
                    )
            else:
                model_ref = int(row[0])
                if fp and row[1] and fp != row[1]:
                    changed |= self._mark_stale(
                        con, model_ref, {"reason": "fingerprint changed", "from": row[1], "to": fp}, now
                    )
                con.execute(
                    "UPDATE model SET last_seen=?, fingerprint=COALESCE(?, fingerprint) WHERE id=?",
                    (now, fp, model_ref),
                )
            profile_id = self._ensure_profile(con, model_ref, now)
            for k, v in result.fingerprint.items():
                self._add_fact_if_new(
                    con, model_ref, MetadataFact(k, v, "provider_reported", "response"), now
                )
        return Observation(model_ref, profile_id, version, changed)

    def record_facts(self, model_ref: int, info: ModelInfo) -> int:
        """Store metadata facts; returns how many were new (unchanged values are not duplicated)."""
        now = time.time()
        with self.store.transaction() as con:
            return sum(self._add_fact_if_new(con, model_ref, f, now) for f in info.facts)

    # ------------------------------------------------------------------ queries
    def latest_model_ref(self, key: ModelKey) -> int | None:
        row = self.store.query_one(
            "SELECT id FROM model WHERE provider=? AND model_id=? AND deployment=? ORDER BY last_seen DESC LIMIT 1",
            (key.provider, key.model_id, key.deployment),
        )
        return int(row["id"]) if row else None

    def facts(self, model_ref: int) -> list[dict[str, Any]]:
        """Latest value of each fact key, with its provenance."""
        rows = self.store.query(
            "SELECT key, value_json, provenance, observed_at FROM model_fact f WHERE model_ref=? AND id = "
            "(SELECT MAX(id) FROM model_fact g WHERE g.model_ref=f.model_ref AND g.key=f.key) ORDER BY key",
            (model_ref,),
        )
        return [dict(r, value=json.loads(r.pop("value_json"))) for r in rows]

    def profile(self, model_ref: int) -> dict[str, Any] | None:
        row = self.store.query_one(
            "SELECT p.*, m.provider, m.model_id, m.resolved_version, m.deployment, m.alias, m.fingerprint "
            "FROM profile p JOIN model m ON m.id = p.model_ref WHERE p.model_ref=? "
            "ORDER BY p.profile_version DESC LIMIT 1",
            (model_ref,),
        )
        if row is None:
            return None
        events = self.store.query(
            "SELECT kind, detail_json, at FROM change_event WHERE profile_id=? ORDER BY at", (row["id"],)
        )
        return dict(row, facts=self.facts(model_ref), change_events=events)

    def next_profile(self, model_ref: int, status: str = "calibrating") -> int:
        """Start the next profile version for a model (after drift, or to recalibrate a stale one). Returns its id.

        The new profile starts with no evidence; the old one is kept for audit.
        """
        with self.store.transaction() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(profile_version), 0) FROM profile WHERE model_ref=?", (model_ref,)
            ).fetchone()
            cur = con.execute(
                "INSERT INTO profile(model_ref, profile_version, status, created_at) VALUES (?,?,?,?)",
                (model_ref, int(row[0]) + 1, status, time.time()),
            )
            return int(cur.lastrowid or 0)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _ensure_profile(con: Any, model_ref: int, now: float) -> int:
        row = con.execute(
            "SELECT id FROM profile WHERE model_ref=? ORDER BY profile_version DESC LIMIT 1", (model_ref,)
        ).fetchone()
        if row is not None:
            return int(row[0])
        cur = con.execute(
            "INSERT INTO profile(model_ref, profile_version, status, created_at) VALUES (?,?,?,?)",
            (model_ref, 1, "cold", now),
        )
        return int(cur.lastrowid or 0)

    @staticmethod
    def _mark_stale(con: Any, model_ref: int, detail: dict[str, Any], now: float) -> bool:
        ids = [
            r[0]
            for r in con.execute(
                "SELECT id FROM profile WHERE model_ref=? AND status != 'stale'", (model_ref,)
            ).fetchall()
        ]
        for pid in ids:
            con.execute("UPDATE profile SET status='stale' WHERE id=?", (pid,))
            con.execute(
                "INSERT INTO change_event(profile_id, kind, detail_json, at) VALUES (?,?,?,?)",
                (pid, "fingerprint", json.dumps(detail), now),
            )
        return bool(ids)

    @staticmethod
    def _add_fact_if_new(con: Any, model_ref: int, fact: MetadataFact, now: float) -> int:
        value_json = json.dumps(fact.value, sort_keys=True, default=str)
        last = con.execute(
            "SELECT value_json, provenance FROM model_fact WHERE model_ref=? AND key=? ORDER BY id DESC LIMIT 1",
            (model_ref, fact.key),
        ).fetchone()
        if last is not None and last[0] == value_json and last[1] == fact.provenance:
            return 0
        con.execute(
            "INSERT INTO model_fact(model_ref, key, value_json, provenance, observed_at) VALUES (?,?,?,?,?)",
            (model_ref, fact.key, value_json, fact.provenance, now),
        )
        return 1
