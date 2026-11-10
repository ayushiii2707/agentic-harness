"""A small bounded mapping for in-process caches (evaluation risk R2: caches grew without limit)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class BoundedCache(Generic[K, V]):
    """Least-recently-used cache with a fixed maximum size."""

    def __init__(self, max_items: int = 4096) -> None:
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self.max_items = max_items
        self._data: OrderedDict[K, V] = OrderedDict()

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getitem__(self, key: K) -> V:
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key: K, value: V) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
