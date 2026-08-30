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
from snowflake_emulator.translator import INFORMATION_SCHEMA_SCHEMA

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
                self._ensure_information_schema(db_ident, database)
            self._connection.execute(f"CREATE SCHEMA IF NOT EXISTS {db_ident}.{schema_ident}")

    def _ensure_information_schema(self, db_ident: str, database: str) -> None:
        """Create a Snowflake-compatible ``INFORMATION_SCHEMA`` in an attached catalog.

        DuckDB only exposes an ``information_schema`` schema in its ``system`` catalog,
        so attached catalogs (one per Snowflake database) get their own schema with
        ``tables``/``schemata`` views backed by DuckDB's catalog introspection
        functions. Column names are uppercase to match Snowflake's
        ``INFORMATION_SCHEMA``; DuckDB's case-insensitive unquoted identifiers make
        ``TABLE_SCHEMA``/``SCHEMA_NAME``/``CREATED``/``LAST_ALTERED`` resolve to them.
        ``CREATED``/``LAST_ALTERED`` are synthesized since DuckDB tracks neither
        (schemachange only uses them for an existence check + log line). ``tables``
        unions base tables (``duckdb_tables()``) with views (``duckdb_views()``) so
        ``TABLE_TYPE`` is ``BASE TABLE`` or ``VIEW`` as on Snowflake.
        """
        schema = INFORMATION_SCHEMA_SCHEMA
        self._connection.execute(f"CREATE SCHEMA IF NOT EXISTS {db_ident}.{schema}")
        self._connection.execute(
            f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.schemata AS
            SELECT
                database_name AS CATALOG_NAME,
                schema_name AS SCHEMA_NAME,
                NULL::VARCHAR AS SCHEMA_OWNER,
                NULL::TIMESTAMP AS CREATED,
                NULL::TIMESTAMP AS LAST_ALTERED
            FROM duckdb_schemas()
            WHERE database_name = '{database}'
            """
        )
        self._connection.execute(
            f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.tables AS
            SELECT
                database_name AS TABLE_CATALOG,
                schema_name AS TABLE_SCHEMA,
                table_name AS TABLE_NAME,
                'BASE TABLE'::VARCHAR AS TABLE_TYPE,
                NULL::TIMESTAMP AS CREATED,
                NULL::TIMESTAMP AS LAST_ALTERED,
                comment AS COMMENT,
                NULL::BIGINT AS ROW_COUNT,
                estimated_size AS BYTES
            FROM duckdb_tables()
            WHERE database_name = '{database}'
            UNION ALL
            SELECT
                database_name AS TABLE_CATALOG,
                schema_name AS TABLE_SCHEMA,
                view_name AS TABLE_NAME,
                'VIEW'::VARCHAR AS TABLE_TYPE,
                NULL::TIMESTAMP AS CREATED,
                NULL::TIMESTAMP AS LAST_ALTERED,
                comment AS COMMENT,
                NULL::BIGINT AS ROW_COUNT,
                NULL::BIGINT AS BYTES
            FROM duckdb_views()
            WHERE database_name = '{database}'
            """
        )

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
