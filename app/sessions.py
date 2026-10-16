"""In-memory session stores that forget entries nobody has touched for a while.

The demo keeps chat sessions, pending runs and sequence layouts in memory. A
plain dict grows forever on a public deployment, so these stores drop entries
after a period of inactivity. Reading or writing an entry counts as activity.
Expired entries are swept lazily on access; no background task is needed.
"""

import time
from collections.abc import Iterator, MutableMapping
from typing import Any

DEFAULT_TTL_SECONDS = 60 * 60


class ExpiringStore(MutableMapping):
    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS, sweep_every: float = 60.0, clock=time.monotonic):
        self._ttl = ttl_seconds
        self._sweep_every = sweep_every
        self._clock = clock
        self._data: dict[str, tuple[float, Any]] = {}
        self._last_sweep = clock()

    def _expired(self, touched: float, now: float) -> bool:
        return now - touched > self._ttl

    def _sweep(self, now: float) -> None:
        if now - self._last_sweep < self._sweep_every:
            return
        self._last_sweep = now
        for key in [k for k, (touched, _) in self._data.items() if self._expired(touched, now)]:
            del self._data[key]

    def __getitem__(self, key: str) -> Any:
        now = self._clock()
        self._sweep(now)
        touched, value = self._data[key]
        if self._expired(touched, now):
            del self._data[key]
            raise KeyError(key)
        self._data[key] = (now, value)  # reading counts as activity
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        now = self._clock()
        self._sweep(now)
        self._data[key] = (now, value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        now = self._clock()
        return iter([k for k, (touched, _) in self._data.items() if not self._expired(touched, now)])

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def copy(self) -> dict:
        """Plain-dict snapshot (lets unittest.mock.patch.dict save and restore it)."""
        return {key: self[key] for key in list(self)}
