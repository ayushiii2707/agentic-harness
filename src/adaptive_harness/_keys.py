"""API key handling.

Rules enforced here:
- a key is held in :class:`Secret`, whose ``repr`` and ``str`` are masked;
- the raw value is only available through :meth:`Secret.reveal`, used when building an HTTP header;
- every revealed secret is registered so :func:`redact` can scrub it from logs and error messages;
- key-shaped strings are also scrubbed by pattern, in case a provider echoes a key back.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

_REGISTRY: set[str] = set()
_LOCK = threading.Lock()

# Common provider key shapes. Deliberately broad: a false positive only masks text in a log line.
_KEY_PATTERNS = [
    re.compile(r"sk-(?:proj-|ant-[a-z0-9]+-|or-v1-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"AQ\.[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}"),
    re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9_\-]{16,}\b"),
]
MASK = "***"


class Secret:
    """A credential that never prints its value."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("a secret must be a non-empty string")
        self._value = value.strip()
        with _LOCK:
            _REGISTRY.add(self._value)

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(("Secret", self._value))

    def __reduce__(self) -> tuple[object, ...]:
        raise TypeError("Secret objects cannot be pickled")


def redact(text: str) -> str:
    """Remove every registered secret and every key-shaped string from ``text``."""
    if not text:
        return text
    with _LOCK:
        known = sorted(_REGISTRY, key=len, reverse=True)
    for value in known:
        if value in text:
            text = text.replace(value, MASK)
    for pat in _KEY_PATTERNS:
        text = pat.sub(lambda m: (m.group(1) if m.groups() else "") + MASK, text)
    return text


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        if value:
            out[key.strip()] = value
    return out


_PROJECT_MARKERS = (".git", "pyproject.toml")


def find_env_file(start: Path | None = None) -> Path | None:
    """Look for a ``.env`` in ``start`` (default: the working directory) and its parents, up to the project root.

    The search stops at the first directory that contains ``.git`` or ``pyproject.toml`` (the project root), and
    never goes above the user's home directory. A ``.env`` in an unrelated ancestor directory can therefore not
    supply keys or settings (evaluation finding F8).
    """
    here = (start or Path.cwd()).resolve()
    try:
        home: Path | None = Path.home().resolve()
    except RuntimeError:  # pragma: no cover - no home directory
        home = None
    for d in (here, *here.parents):
        candidate = d / ".env"
        if candidate.is_file():
            return candidate
        if any((d / m).exists() for m in _PROJECT_MARKERS) or d == home:
            return None
    return None


def get_secret(env_var: str, *, explicit: str | None = None, env_file: Path | None = None) -> Secret | None:
    """Resolve a credential: explicit argument, then the environment, then a ``.env`` file."""
    if explicit:
        return Secret(explicit)
    value = os.environ.get(env_var)
    if value and value.strip():
        return Secret(value)
    path = env_file if env_file is not None else find_env_file()
    if path is not None:
        value = _parse_env_file(path).get(env_var)
        if value:
            return Secret(value)
    return None


def get_setting(name: str, *, env_file: Path | None = None) -> str | None:
    """Read a non-secret setting from the environment or ``.env`` (environment wins)."""
    value = os.environ.get(name)
    if value is not None and value.strip():
        return value.strip()
    path = env_file if env_file is not None else find_env_file()
    return _parse_env_file(path).get(name) if path is not None else None
