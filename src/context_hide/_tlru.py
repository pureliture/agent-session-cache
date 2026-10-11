"""Minimal TLRUCache — drop-in for the single cachetools API this project uses.

Why this exists: the host process (Hermes) does not have ``cachetools``
installed, and adding a runtime dependency to a shared interpreter is not an
option for a plugin. ``context_hide.store`` needs exactly one thing from
cachetools — a size- and time-bounded LRU mapping with a TTU (time-to-use)
expiry — so this module implements that one contract with the standard library
only.

Contract implemented (matches cachetools.TLRUCache as used here):

- ``maxsize`` — a byte budget, not an entry count. Insertion is refused when an
  item's ``getsizeof`` does not fit in the remaining budget.
- ``ttu(key, value, now)`` — returns the item's expiry timestamp.
- ``timer`` — the clock used for expiry (the store passes ``time.monotonic``).
- ``getsizeof`` — per-item size, used for ``currsize`` accounting.
- ``expire(now)`` — drops every item whose TTU expiry has passed, returns them.
- LRU eviction: the least recently used entry is dropped first when space is
  needed, but admission is checked BEFORE insertion so a too-large item never
  evicts a live neighbour.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable


class TLRUCache:
    """Thread-safe, size-bounded LRU with per-item TTL."""

    def __init__(
        self,
        maxsize: int,
        ttu: Callable[[Any, Any, float], float],
        timer: Callable[[], float],
        getsizeof: Callable[[Any], int],
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._ttu = ttu
        self._timer = timer
        self._getsizeof = getsizeof
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self._currsize = 0
        self._lock = threading.RLock()

    # -- introspection ----------------------------------------------------
    @property
    def currsize(self) -> int:
        with self._lock:
            return self._currsize

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._data

    def __iter__(self):
        with self._lock:
            return iter(list(self._data))

    # -- expiry -----------------------------------------------------------
    def expire(self, time: float | None = None) -> list[Any]:
        """Drop every item expired at *time*; return the dropped values."""
        moment = self._timer() if time is None else time
        with self._lock:
            dead = []
            for key, value in list(self._data.items()):
                if self._ttu(key, value, moment) <= moment:
                    dead.append((key, value))
            for key, value in dead:
                self._data.pop(key, None)
                self._currsize -= self._getsizeof(value)
            return [value for _key, value in dead]

    # -- access -----------------------------------------------------------
    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            value = self._data[key]
            self._data.move_to_end(key)
            return value

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            value = self._data[key]
            self._data.move_to_end(key)
            return value

    def __setitem__(self, key: Any, value: Any) -> None:
        size = self._getsizeof(value)
        with self._lock:
            if key in self._data:
                previous = self._data.pop(key)
                self._currsize -= self._getsizeof(previous)
            # Admission check: a too-large item must not evict a live neighbour.
            if size > self.maxsize - self._currsize:
                return
            self._data[key] = value
            self._currsize += size
            # Evict least-recently-used until the advertised budget holds.
            while self._currsize > self.maxsize and self._data:
                _, evicted = self._data.popitem(last=False)
                self._currsize -= self._getsizeof(evicted)

    def items(self):
        with self._lock:
            return list(self._data.items())

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def values(self):
        with self._lock:
            return list(self._data.values())

    def pop(self, key: Any, *default: Any) -> Any:
        with self._lock:
            if key not in self._data:
                if default:
                    return default[0]
                raise KeyError(key)
            value = self._data.pop(key)
            self._currsize -= self._getsizeof(value)
            return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._currsize = 0
