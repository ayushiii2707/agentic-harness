"""SQLite connection, migrations and small query helpers.

- One database file per harness home (or any path the caller gives); ``:memory:`` is supported for tests.
- Migrations are numbered SQL files shipped inside the package and applied in order, each in a transaction.
- A database written by a newer version of the package (unknown higher migration) is refused, not guessed at.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

from ..errors import StoreError

_MIGRATION_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def _migrations() -> list[tuple[int, str, str]]:
    pkg = resources.files("adaptive_harness.store") / "migrations"
    found = []
    for entry in pkg.iterdir():
        m = _MIGRATION_RE.match(entry.name)
        if m:
            found.append((int(m.group(1)), entry.name, entry.read_text(encoding="utf-8")))
    found.sort()
    numbers = [n for n, _, _ in found]
    if numbers != list(range(1, len(numbers) + 1)):
        raise StoreError(f"migration files are not numbered contiguously from 0001: {numbers}")
    return found


class Store:
    """Thread-safe wrapper around one SQLite connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        try:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._con = sqlite3.connect(self.path, timeout=30, check_same_thread=False, isolation_level=None)
        except (OSError, sqlite3.Error) as e:
            raise StoreError(f"cannot open store at {self.path}: {e}") from e
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA foreign_keys=ON")
        self._con.execute("PRAGMA busy_timeout=30000")
        if self.path != ":memory:":
            self._con.execute("PRAGMA journal_mode=WAL")
        self.migrate()

    # ------------------------------------------------------------------ migrations
    def schema_version(self) -> int:
        with self._lock:
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, name TEXT, applied_at REAL)"
            )
            row = self._con.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            return int(row["v"] or 0)

    def migrate(self) -> int:
        migrations = _migrations()
        with self._lock:
            current = self.schema_version()
            latest = migrations[-1][0] if migrations else 0
            if current > latest:
                raise StoreError(
                    f"store {self.path} has schema version {current}, newer than this package supports ({latest})"
                )
            for number, name, sql in migrations:
                if number <= current:
                    continue
                try:
                    self._con.execute("BEGIN")
                    for statement in _split_sql(sql):
                        self._con.execute(statement)
                    self._con.execute(
                        "INSERT INTO schema_version(version, name, applied_at) VALUES (?,?,?)",
                        (number, name, time.time()),
                    )
                    self._con.execute("COMMIT")
                except sqlite3.Error as e:
                    self._con.execute("ROLLBACK")
                    raise StoreError(f"migration {name} failed: {e}") from e
            return self.schema_version()

    # ------------------------------------------------------------------ queries
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        # sqlite3 errors become StoreError, so callers' "store unavailable" fallbacks catch them (finding F16).
        with self._lock:
            try:
                self._con.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as e:
                raise StoreError(str(e)) from e
            try:
                yield self._con
            except sqlite3.Error as e:
                self._con.execute("ROLLBACK")
                raise StoreError(str(e)) from e
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
            else:
                try:
                    self._con.execute("COMMIT")
                except sqlite3.Error as e:
                    self._con.execute("ROLLBACK")
                    raise StoreError(str(e)) from e

    def execute(self, sql: str, args: Sequence[Any] = ()) -> int:
        """Run one write statement; return the number of affected rows."""
        with self._lock:
            try:
                return self._con.execute(sql, tuple(args)).rowcount
            except sqlite3.Error as e:
                raise StoreError(str(e)) from e

    def query(self, sql: str, args: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            try:
                return [dict(r) for r in self._con.execute(sql, tuple(args)).fetchall()]
            except sqlite3.Error as e:
                raise StoreError(str(e)) from e

    def query_one(self, sql: str, args: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, args)
        return rows[0] if rows else None

    def tables(self) -> list[str]:
        return [
            r["name"] for r in self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]

    def close(self) -> None:
        with self._lock:
            self._con.close()


def _split_sql(sql: str) -> list[str]:
    """Split a migration file into statements (no semicolons inside string literals in our migrations)."""
    no_comments = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
    return [s.strip() for s in no_comments.split(";") if s.strip()]
