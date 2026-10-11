"""In-memory session state, mimicking the Snowflake notion of a connected session.

This is a local emulator, not a security boundary: authentication is intentionally
permissive (any bearer token is accepted) so tools built against the real Snowflake
SQL API v2 can be pointed at this service with minimal changes.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

from snowflake_emulator.settings import settings


@dataclass
class SessionContext:
    """Mutable session-scoped state: current database/schema/warehouse/role."""

    token: str
    database: str = field(default_factory=lambda: settings.default_database)
    schema: str = field(default_factory=lambda: settings.default_schema)
    warehouse: str = field(default_factory=lambda: settings.default_warehouse)
    role: str = field(default_factory=lambda: settings.default_role)
    last_query_id: str | None = None
    # Snowflake session variables set with ``SET name = value`` and read with ``$name``.
    variables: dict[str, object] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class SessionManager:
    """Tracks active sessions keyed by opaque bearer token.

    Sessions are bounded: once `max_sessions` bearer-token sessions are live, creating
    another evicts the least recently used one, which releases the DuckDB connection it
    was using (and with it its temporary tables), the same way a real Snowflake session
    drops its temporary tables when it expires. Requests without a token share one
    pinned anonymous session, so a tokenless client still gets a working session without
    the emulator retaining one session (and one connection) per request.
    """

    def __init__(self, max_sessions: int | None = None) -> None:
        self._sessions: OrderedDict[str, SessionContext] = OrderedDict()
        self._max_sessions = settings.max_sessions if max_sessions is None else max_sessions
        self._anonymous: SessionContext | None = None
        self._lock = threading.Lock()

    def create(self) -> SessionContext:
        return self.create_with_token(secrets.token_urlsafe(32))

    def get_or_create(self, token: str | None) -> SessionContext:
        """Return the session for `token`, creating a fresh default one if unknown/absent."""
        if not token:
            # One shared session for every caller without a token: bounded, and its
            # temporary tables survive from one request to the next.
            return self._anonymous_session()
        with self._lock:
            existing = self._sessions.get(token)
            if existing is not None:
                self._sessions.move_to_end(token)
                return existing
        return self.create_with_token(token)

    def create_with_token(self, token: str) -> SessionContext:
        session = SessionContext(token=token)
        with self._lock:
            self._sessions[token] = session
            self._sessions.move_to_end(token)
            evicted = self._evicted_locked()
        self._release(evicted)
        return session

    def close(self, token: str) -> bool:
        with self._lock:
            closed = self._sessions.pop(token, None) is not None
        if closed:
            self._release([token])
        return closed

    def _anonymous_session(self) -> SessionContext:
        """Return the shared session used by requests that carry no token."""
        with self._lock:
            if self._anonymous is None:
                # Unguessable, so a client-supplied token cannot collide with it.
                self._anonymous = SessionContext(token=f"anonymous-{secrets.token_urlsafe(32)}")
            return self._anonymous

    def _evicted_locked(self) -> list[str]:
        """Drop least-recently-used sessions over the cap (caller holds the lock)."""
        evicted: list[str] = []
        if self._max_sessions <= 0:
            return evicted
        while len(self._sessions) > self._max_sessions:
            token, _ = self._sessions.popitem(last=False)
            evicted.append(token)
        return evicted

    @staticmethod
    def _release(tokens: Iterable[str]) -> None:
        """Release each session's private database (isolated mode) and connection."""
        tokens = list(tokens)
        if not tokens:
            return
        from snowflake_emulator.database import release_isolated_manager, release_session

        for token in tokens:
            release_isolated_manager(token)
            release_session(token)


_manager: SessionManager | None = None
_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = SessionManager()
    return _manager
