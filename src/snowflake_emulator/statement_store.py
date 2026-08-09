"""In-memory cache of executed statement results, keyed by statement handle."""

from __future__ import annotations

import threading
from collections import OrderedDict

from snowflake_emulator.executor import ExecutionResult
from snowflake_emulator.settings import settings


class StatementStore:
    """A bounded FIFO cache so `GET /api/v2/statements/{handle}` can replay results."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._results: "OrderedDict[str, ExecutionResult]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, result: ExecutionResult) -> None:
        with self._lock:
            self._results[result.handle] = result
            self._results.move_to_end(result.handle)
            while len(self._results) > self._max_size:
                self._results.popitem(last=False)

    def get(self, handle: str) -> ExecutionResult | None:
        with self._lock:
            return self._results.get(handle)


_store: StatementStore | None = None
_store_lock = threading.Lock()


def get_statement_store() -> StatementStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = StatementStore(settings.max_cached_statements)
    return _store
