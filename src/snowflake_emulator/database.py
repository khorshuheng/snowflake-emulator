"""DuckDB connection management.

Snowflake organizes data as ``database.schema.table``. DuckDB natively supports the
same three-level namespacing once additional catalogs are ``ATTACH``-ed, so we map:

    Snowflake database  -> DuckDB attached catalog
    Snowflake schema    -> DuckDB schema (within that catalog)

The default catalog (``memory``) is left untouched; every Snowflake "database" the
client references is transparently attached as its own DuckDB catalog on first use.

Storage is ephemeral (``:memory:``) unless ``persistence.database_path`` names a file.
In that case a dedicated directory ``<database_path>.d`` is created next to it and every
Snowflake database is stored there as ``<DATABASE>.duckdb``; ``database_path`` itself is
only the anchor for that directory name. DuckDB maps one catalog to one file, and a
dedicated directory keeps unrelated files in the parent from being attached. Existing
database files are re-attached at startup, so data survives a restart.
"""

from __future__ import annotations

import re
import threading
import warnings
from pathlib import Path
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
        # Reentrant so `replace_database` can hold the lock across DETACH -> unlink ->
        # re-ATTACH and still call `ensure_namespace`. A plain Lock would deadlock.
        self._lock = threading.RLock()
        self._persistent = database_path != ":memory:"
        self._storage_dir = (
            Path(f"{database_path}{_STORAGE_SUFFIX}") if self._persistent else None
        )
        # The root connection is always in-memory: every Snowflake database is attached
        # as its own catalog, so no user data ends up in a catalog the client never
        # addresses.
        self._connection = duckdb.connect(":memory:")
        self._attached_catalogs: set[str] = set()
        if self._persistent:
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            # Re-attach databases left behind by a previous run so they are usable (and
            # visible to SHOW DATABASES) without first being referenced. A single
            # unreadable file must not make the whole emulator unusable, so skip it.
            for database in self._persisted_databases():
                try:
                    self.ensure_namespace(database, "PUBLIC")
                except Exception as exc:  # noqa: BLE001 - best-effort discovery
                    warnings.warn(
                        f"Ignoring unreadable database file for {database!r}: {exc}",
                        stacklevel=2,
                    )

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._connection

    def _catalog_path(self, database: str) -> str:
        """Return the DuckDB file backing a Snowflake database.

        ``:memory:`` keeps every catalog ephemeral. Otherwise each database lives in the
        dedicated ``<database_path>.d`` directory as ``<DATABASE>.duckdb`` (e.g.
        ``/var/lib/sfemu/emulator.duckdb`` ->
        ``/var/lib/sfemu/emulator.duckdb.d/MY_DB.duckdb``). DuckDB maps one catalog to one
        file, and naming each file after its database keeps a database's identity stable
        across restarts. Database names are case-insensitive, as DuckDB catalogs are, so
        the file name is upper-cased.
        """
        if not self._persistent:
            return ":memory:"
        return str(self._storage_dir / f"{database.upper()}{_DATABASE_FILE_SUFFIX}")

    def _persisted_databases(self) -> list[str]:
        """Return the Snowflake databases whose backing files already exist on disk.

        Only the dedicated storage directory is scanned, so files elsewhere on the
        filesystem are never picked up; files that do not match the naming scheme are
        ignored.
        """
        found: list[str] = []
        if self._storage_dir.is_dir():
            for child in sorted(self._storage_dir.iterdir()):
                name = child.name
                if not child.is_file() or not name.endswith(_DATABASE_FILE_SUFFIX):
                    continue
                database = name[: -len(_DATABASE_FILE_SUFFIX)]
                if database and _IDENTIFIER_RE.match(database):
                    found.append(database)
        return found

    def ensure_namespace(self, database: str, schema: str) -> None:
        """Ensure the given Snowflake database/schema pair exists as a DuckDB catalog/schema."""
        db_ident = _quote_ident(database)
        schema_ident = _quote_ident(schema)
        with self._lock:
            if database.lower() not in self._attached_catalogs:
                path = self._catalog_path(database).replace("'", "''")
                self._connection.execute(f"ATTACH '{path}' AS {db_ident}")
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
        """Return True if a Snowflake database is already attached as a catalog.

        A persistent database whose file exists but has not been attached yet (e.g. a
        fresh manager mid-lazy-attach) still counts as existing, so ``CREATE DATABASE``
        reports a collision instead of silently re-attaching it.
        """
        if database.lower() in self._attached_catalogs:
            return True
        return self._persistent and Path(self._catalog_path(database)).exists()

    def replace_database(self, database: str) -> None:
        """Recreate a Snowflake database (``CREATE OR REPLACE DATABASE``)."""
        db_ident = _quote_ident(database)
        # Hold the lock across DETACH, file removal and re-ATTACH so a concurrent
        # reference cannot observe (or re-attach) the database mid-replacement.
        with self._lock:
            if database.lower() in self._attached_catalogs:
                self._connection.execute(f"DETACH {db_ident}")
                self._attached_catalogs.discard(database.lower())
            if self._persistent:
                # Remove the backing file (and any write-ahead log) so re-attaching
                # does not resurrect the replaced database's contents.
                path = Path(self._catalog_path(database))
                for candidate in (path, path.with_name(f"{path.name}.wal")):
                    if candidate.exists():
                        candidate.unlink()
            # Re-attach (with INFORMATION_SCHEMA + default PUBLIC schema) via the
            # normal namespace path.
            self.ensure_namespace(database, "PUBLIC")


_STORAGE_SUFFIX = ".d"  # dedicated directory: <database_path>.d
_DATABASE_FILE_SUFFIX = ".duckdb"  # one file per database inside that directory

_manager: DuckDBManager | None = None
_manager_lock = threading.Lock()
# Private managers for isolated mode, keyed by session token.
_isolated_managers: dict[str, DuckDBManager] = {}
_isolated_persistence_warned = False


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
        global _isolated_persistence_warned
        if settings.database_path != ":memory:" and not _isolated_persistence_warned:
            _isolated_persistence_warned = True
            warnings.warn(
                "persistence.database_path is set but isolation='isolated': each "
                "session gets a private in-memory database, so nothing is persisted.",
                stacklevel=2,
            )
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
