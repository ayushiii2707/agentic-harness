"""Stable hashing.

Hashes of configurations, suites and strategies are computed over canonical JSON (sorted keys, no whitespace,
UTF-8, LF), so platform line endings or key order can never change a hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def canonical_hash(obj: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


def text_hash(text: str, length: int = 16) -> str:
    """Hash of a text with line endings normalised to LF."""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:length]
