from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping
from threading import RLock
from typing import Iterator, TypeVar, Generic

K = TypeVar("K")
V = TypeVar("V")


class BoundedCache(MutableMapping[K, V], Generic[K, V]):
    def __init__(self, maxsize: int = 4096):
        self.maxsize = max(1, int(maxsize))
        self._data: OrderedDict[K, V] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __getitem__(self, key: K) -> V:
        with self._lock:
            value = self._data[key]
            self._hits += 1
            self._data.move_to_end(key)
            return value

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
                self._evictions += 1

    def __delitem__(self, key: K) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            return iter(tuple(self._data))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, key: K, default=None):
        with self._lock:
            if key in self._data:
                self._hits += 1
                self._data.move_to_end(key)
                return self._data[key]
            self._misses += 1
            return default

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._data),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }
