"""DuckDB connection management.

Snowflake organizes data as ``database.schema.table``. DuckDB natively supports the
same three-level namespacing once additional catalogs are ``ATTACH``-ed, so we map:

    Snowflake database  -> DuckDB attached catalog
    Snowflake schema    -> DuckDB schema (within that catalog)

The default catalog (``memory``) is left untouched; every Snowflake "database" the
client references is transparently attached as its own DuckDB catalog on first use.
"""

from __future__ import annotations

import re
import threading

import duckdb

from snowflake_emulator.settings import settings

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _quote_ident(identifier: str) -> str:
    """Safely quote an identifier for interpolation into DDL statements."""
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Invalid identifier: {identifier!r}")
    return f'"{identifier}"'


class DuckDBManager:
    """Owns the single DuckDB connection used to back the emulator."""

    def __init__(self, database_path: str) -> None:
        self._lock = threading.Lock()
        self._connection = duckdb.connect(database_path)
        self._attached_catalogs: set[str] = set()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._connection

    def ensure_namespace(self, database: str, schema: str) -> None:
        """Ensure the given Snowflake database/schema pair exists as a DuckDB catalog/schema."""
        db_ident = _quote_ident(database)
        schema_ident = _quote_ident(schema)
        with self._lock:
            if database.lower() not in self._attached_catalogs:
                self._connection.execute(f"ATTACH ':memory:' AS {db_ident}")
                self._attached_catalogs.add(database.lower())
            self._connection.execute(f"CREATE SCHEMA IF NOT EXISTS {db_ident}.{schema_ident}")

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """Return an isolated cursor so concurrent requests don't clobber each other's state."""
        return self._connection.cursor()


_manager: DuckDBManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> DuckDBManager:
    """Return the process-wide DuckDB manager, creating it lazily."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DuckDBManager(settings.database_path)
    return _manager
