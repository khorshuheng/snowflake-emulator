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
from typing import Any

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
        # Snowflake-only column metadata DuckDB discards: the declared SQL type
        # (notably VARCHAR(n) lengths) and column comments.
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {db_ident}.{schema}._sf_columns_ext (
                table_catalog VARCHAR,
                table_schema VARCHAR,
                table_name VARCHAR,
                column_name VARCHAR,
                declared_type VARCHAR,
                comment VARCHAR,
                PRIMARY KEY (table_catalog, table_schema, table_name, column_name)
            )
            """
        )
        self._connection.execute(
            f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.columns AS
            SELECT
                c.database_name AS TABLE_CATALOG,
                c.schema_name AS TABLE_SCHEMA,
                c.table_name AS TABLE_NAME,
                c.column_name AS COLUMN_NAME,
                c.column_index + 1 AS ORDINAL_POSITION,
                CASE
                    WHEN e.declared_type LIKE 'VARCHAR(%' OR e.declared_type LIKE 'CHAR(%'
                      OR e.declared_type LIKE 'STRING(%' OR e.declared_type LIKE 'TEXT(%'
                      OR e.declared_type LIKE 'BINARY(%' THEN e.declared_type
                    WHEN c.data_type LIKE 'VARCHAR%' THEN 'VARCHAR'
                    WHEN c.data_type LIKE 'DECIMAL%' THEN 'NUMBER' || substr(c.data_type, 8)
                    WHEN c.data_type IN ('BIGINT','INTEGER','SMALLINT','TINYINT','HUGEINT','UINTEGER','UBIGINT','USMALLINT','UTINYINT','UHUGEINT') THEN 'NUMBER(38,0)'
                    WHEN c.data_type IN ('DOUBLE','FLOAT','REAL') THEN 'FLOAT'
                    WHEN c.data_type LIKE 'TIMESTAMP%' THEN 'TIMESTAMP_NTZ'
                    ELSE c.data_type
                END AS DATA_TYPE,
                CASE WHEN c.is_nullable THEN 'YES' ELSE 'NO' END AS IS_NULLABLE,
                c.column_default AS COLUMN_DEFAULT,
                e.comment AS COMMENT
            FROM duckdb_columns() c
            LEFT JOIN {db_ident}.{schema}._sf_columns_ext e
              ON lower(e.table_catalog) = lower(c.database_name)
             AND lower(e.table_schema) = lower(c.schema_name)
             AND lower(e.table_name) = lower(c.table_name)
             AND lower(e.column_name) = lower(c.column_name)
            WHERE c.database_name = '{database}'
            """
        )

    def record_columns(
        self,
        database: str,
        schema: str,
        table: str,
        columns: list[tuple[str, str | None, str | None]],
    ) -> None:
        """Record the declared Snowflake types for a table's columns.

        DuckDB stores ``VARCHAR`` without its length, so the type as written is kept in
        the per-catalog ``_sf_columns_ext`` side table and surfaced by ``DESCRIBE`` /
        ``INFORMATION_SCHEMA.COLUMNS``.
        """
        db_ident = _quote_ident(database)
        # The side table lives in the emulator's information schema, not the table's schema.
        ext_ident = _quote_ident(INFORMATION_SCHEMA_SCHEMA)
        with self._lock:
            self._connection.execute(
                f"DELETE FROM {db_ident}.{ext_ident}._sf_columns_ext "
                "WHERE lower(table_catalog) = lower(?) AND lower(table_schema) = lower(?) "
                "AND lower(table_name) = lower(?)",
                [database, schema, table],
            )
            for name, declared_type, comment in columns:
                self._connection.execute(
                    f"INSERT INTO {db_ident}.{ext_ident}._sf_columns_ext "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [database, schema, table, name, declared_type, comment],
                )

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """Return an isolated cursor so concurrent requests don't clobber each other's state."""
        return self._connection.cursor()

    def has_database(self, database: str) -> bool:
        """Return True if a Snowflake database is already attached as a catalog."""
        return database.lower() in self._attached_catalogs

    def replace_database(self, database: str) -> None:
        """Recreate a Snowflake database (``CREATE OR REPLACE DATABASE``)."""
        db_ident = _quote_ident(database)
        with self._lock:
            if self.has_database(database):
                self._connection.execute(f"DETACH {db_ident}")
                self._attached_catalogs.discard(database.lower())
        # Re-attach (with INFORMATION_SCHEMA + default PUBLIC schema) via the
        # normal namespace path.
        self.ensure_namespace(database, "PUBLIC")


_manager: DuckDBManager | None = None
_manager_lock = threading.Lock()
# Private managers for isolated mode, keyed by session token.
_isolated_managers: dict[str, DuckDBManager] = {}


def release_isolated_manager(token: str) -> None:
    """Close and forget a session's isolated database when its session ends."""
    with _manager_lock:
        manager = _isolated_managers.pop(token, None)
    if manager is not None:
        try:
            manager.connection.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


def get_manager(session: Any | None = None) -> DuckDBManager:
    """Return the DuckDB manager serving a request.

    In the default ``shared`` mode one process-wide database backs every session. In
    ``isolated`` mode each session gets its own private in-memory database, which keeps
    parallel test runs from seeing each other's tables.
    """
    if session is not None and settings.isolation == "isolated":
        key = getattr(session, "token", None) or "<anonymous>"
        with _manager_lock:
            manager = _isolated_managers.get(key)
            if manager is None:
                manager = DuckDBManager(":memory:")
                _isolated_managers[key] = manager
            return manager
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DuckDBManager(settings.database_path)
    return _manager
