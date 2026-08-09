"""In-memory session state, mimicking the Snowflake notion of a connected session.

This is a local emulator, not a security boundary: authentication is intentionally
permissive (any bearer token is accepted) so tools built against the real Snowflake
SQL API v2 can be pointed at this service with minimal changes.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

from snowflake_emulator.settings import settings


@dataclass
class SessionContext:
    """Mutable session-scoped state: current database/schema/warehouse/role."""

    token: str
    database: str = field(default_factory=lambda: settings.default_database)
    schema: str = field(default_factory=lambda: settings.default_schema)
    warehouse: str = field(default_factory=lambda: settings.default_warehouse)
    role: str = field(default_factory=lambda: settings.default_role)
    created_at: float = field(default_factory=time.time)


class SessionManager:
    """Tracks active sessions keyed by opaque bearer token."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionContext] = {}
        self._lock = threading.Lock()

    def create(self) -> SessionContext:
        token = secrets.token_urlsafe(32)
        session = SessionContext(token=token)
        with self._lock:
            self._sessions[token] = session
        return session

    def get_or_create(self, token: str | None) -> SessionContext:
        """Return the session for `token`, creating a fresh default one if unknown/absent."""
        if token:
            with self._lock:
                existing = self._sessions.get(token)
            if existing is not None:
                return existing
        return self.create_with_token(token) if token else self.create()

    def create_with_token(self, token: str) -> SessionContext:
        session = SessionContext(token=token)
        with self._lock:
            self._sessions[token] = session
        return session

    def close(self, token: str) -> bool:
        with self._lock:
            return self._sessions.pop(token, None) is not None


_manager: SessionManager | None = None
_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = SessionManager()
    return _manager
