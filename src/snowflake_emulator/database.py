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

import json
import re
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import duckdb

from snowflake_emulator.core.constants import INFORMATION_SCHEMA_SCHEMA
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
        # One connection per session, created lazily. DuckDB scopes TEMP tables to the
        # connection rather than to the database instance, so a session has to keep
        # using the same connection for its temporary tables to survive from one
        # statement to the next, as they do on Snowflake. Regular tables live in the
        # attached catalogs and stay visible from every connection either way.
        self._session_cursors: dict[str, duckdb.DuckDBPyConnection] = {}
        # Serializes the statements of one session: DuckDB connections are not meant to
        # be used concurrently, and a Snowflake session executes one statement at a
        # time anyway. Sessions do not block each other.
        self._session_locks: dict[str, threading.RLock] = {}
        # {session key: {temporary table name: database it was created in}}, so
        # ``CREATE OR REPLACE DATABASE`` can drop the temporary tables that belonged to
        # the replaced database, as Snowflake does. An entry that has gone stale (the
        # user dropped or renamed the table) is harmless: the drop is ``IF EXISTS``.
        self._session_temp_tables: dict[str, dict[str, str]] = {}
        # Native (Python) functions registered for type-overloaded user functions, keyed
        # by the dispatcher's name. They are per-process, cannot be re-registered under
        # the same name, and cannot be unregistered, so the mapping is mutable and the
        # registered callback reads it at call time.
        self.dispatch_groups: dict[str, Any] = {}
        # {UPPER(function name): {arity: callable name}} per (database, schema), loaded
        # from the side table on first use and dropped whenever a function changes.
        self._function_arity_cache: dict[tuple[str, str], dict[str, dict[int, str]]] = {}
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
                # VARIANT columns require storage version v1.5.0 or newer, while DuckDB
                # defaults new files to an older version; passing it here also upgrades
                # a file created before this option was used.
                self._connection.execute(
                    f"ATTACH '{path}' AS {db_ident} (STORAGE_VERSION '{_STORAGE_VERSION}')"
                )
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

        The calling session's temporary tables are included too. DuckDB keeps them in a
        connection-local ``temp`` catalog, so ``duckdb_tables()``/``duckdb_columns()``
        report them to that connection only -- which is exactly the visibility a
        Snowflake temporary table has. They are reported under the session's current
        database and schema (``current_database()``/``current_schema()`` are evaluated
        per connection inside the view), because that is where the user created them.
        """
        schema = INFORMATION_SCHEMA_SCHEMA
        # A scoped object of the catalog this view lives in, or the calling connection's
        # temporary tables when it is currently working in that catalog. A temporary
        # table shadows a permanent one of the same name, so the catalog-side row is
        # dropped while a shadowing temporary table exists.
        def in_scope(alias: str) -> str:
            return (
                f"({alias}.database_name = '{database}'"
                f" AND lower({alias}.schema_name) != lower('{INFORMATION_SCHEMA_SCHEMA}')"
                f" AND NOT EXISTS (SELECT 1 FROM duckdb_tables() sh"
                f" WHERE lower(sh.database_name) = 'temp'"
                f" AND lower(sh.table_name) = lower({alias}.table_name)))"
                f" OR (lower({alias}.database_name) = 'temp'"
                f" AND lower(current_database()) = lower('{database}'))"
            )

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
              AND lower(schema_name) NOT IN ('main', lower('{INFORMATION_SCHEMA_SCHEMA}'))
            """
        )
        self._connection.execute(
            f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.tables AS
            SELECT
                CASE WHEN lower(t.database_name) = 'temp'
                     THEN current_database() ELSE t.database_name END AS TABLE_CATALOG,
                CASE WHEN lower(t.database_name) = 'temp'
                     THEN current_schema() ELSE t.schema_name END AS TABLE_SCHEMA,
                t.table_name AS TABLE_NAME,
                'BASE TABLE'::VARCHAR AS TABLE_TYPE,
                NULL::TIMESTAMP AS CREATED,
                NULL::TIMESTAMP AS LAST_ALTERED,
                t.comment AS COMMENT,
                NULL::BIGINT AS ROW_COUNT,
                t.estimated_size AS BYTES
            FROM duckdb_tables() t
            WHERE {in_scope('t')}
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
              AND lower(schema_name) != lower('{INFORMATION_SCHEMA_SCHEMA}')
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
        # User-defined functions (``CREATE FUNCTION``) are emulated as DuckDB macros,
        # which store no Snowflake-side metadata. The declared signature and body are
        # kept here so ``SHOW FUNCTIONS`` / ``DESCRIBE FUNCTION`` can report them, and so
        # the macro can be re-resolved after a restart with a persistent database.
        #
        # ``signature`` (the declared argument types) is part of the key because one
        # name/arity may carry several overloads; they are dispatched at run time.
        self._create_functions_ext_table(db_ident, schema)
        self._migrate_functions_ext(db_ident, schema)
        self._create_constraints_ext_table(db_ident, schema)
        self._connection.execute(
            f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.columns AS
            SELECT
                CASE WHEN lower(c.database_name) = 'temp'
                     THEN current_database() ELSE c.database_name END AS TABLE_CATALOG,
                CASE WHEN lower(c.database_name) = 'temp'
                     THEN current_schema() ELSE c.schema_name END AS TABLE_SCHEMA,
                c.table_name AS TABLE_NAME,
                c.column_name AS COLUMN_NAME,
                c.column_index + 1 AS ORDINAL_POSITION,
                CASE
                    WHEN regexp_matches(upper(e.declared_type), '^TIMESTAMP_?NTZ\\(([7-9])\\)$')
                      THEN 'TIMESTAMP_NTZ('
                           || regexp_extract(upper(e.declared_type), '\\(([7-9])\\)', 1) || ')'
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
             -- Temporary tables have no row of their own in the side table (their types
             -- are per session, so a shared row could not tell two sessions apart):
             -- joining one would apply some permanent table's declared type to it.
             AND lower(c.database_name) <> 'temp'
            WHERE {in_scope('c')}
            """
        )
        self._create_additional_information_schema_views(db_ident, schema, database)

    def _create_additional_information_schema_views(
        self, db_ident: str, schema: str, database: str
    ) -> None:
        """Create the common INFORMATION_SCHEMA views beyond tables/columns."""
        internal = f"lower('{INFORMATION_SCHEMA_SCHEMA}')"
        views = f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.views AS
            SELECT
                v.database_name AS TABLE_CATALOG,
                v.schema_name AS TABLE_SCHEMA,
                v.view_name AS TABLE_NAME,
                v.sql AS VIEW_DEFINITION,
                'NONE'::VARCHAR AS CHECK_OPTION,
                'NO'::VARCHAR AS IS_UPDATABLE,
                'NO'::VARCHAR AS IS_INSERTABLE_INTO,
                'NO'::VARCHAR AS IS_TRIGGER_UPDATABLE,
                'NO'::VARCHAR AS IS_TRIGGER_DELETABLE,
                'NO'::VARCHAR AS IS_TRIGGER_INSERTABLE_INTO,
                NULL::TIMESTAMP AS CREATED,
                NULL::TIMESTAMP AS LAST_ALTERED,
                v.comment AS COMMENT,
                NULL::VARCHAR AS OWNER,
                v.column_count AS COLUMN_COUNT
            FROM duckdb_views() v
            WHERE v.database_name = '{database}'
              AND lower(v.schema_name) != {internal}
              AND v.internal = false
        """
        sequences = f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.sequences AS
            SELECT
                s.database_name AS SEQUENCE_CATALOG,
                s.schema_name AS SEQUENCE_SCHEMA,
                s.sequence_name AS SEQUENCE_NAME,
                s.start_value AS START_VALUE,
                s.min_value AS MINIMUM_VALUE,
                s.max_value AS MAXIMUM_VALUE,
                s.increment_by AS INCREMENT,
                CASE WHEN s.cycle THEN 'YES' ELSE 'NO' END AS CYCLE_OPTION,
                'NO'::VARCHAR AS ORDERED,
                s.last_value AS LAST_VALUE,
                NULL::TIMESTAMP AS CREATED,
                NULL::TIMESTAMP AS LAST_ALTERED,
                s.comment AS COMMENT,
                NULL::VARCHAR AS OWNER
            FROM duckdb_sequences() s
            WHERE s.database_name = '{database}'
              AND lower(s.schema_name) != {internal}
        """
        databases = """
            CREATE OR REPLACE VIEW {db}.{schema}.databases AS
            SELECT
                d.database_name AS DATABASE_NAME,
                NULL::VARCHAR AS DATABASE_OWNER,
                CASE WHEN lower(current_database()) = lower(d.database_name)
                     THEN 'YES' ELSE 'NO' END AS IS_CURRENT,
                'NO'::VARCHAR AS IS_DEFAULT,
                'LOCAL'::VARCHAR AS ORIGIN,
                d.type AS TYPE,
                NULL::BIGINT AS RETENTION_TIME,
                d.comment AS COMMENT,
                NULL::VARCHAR AS OWNER
            FROM duckdb_databases() d
            WHERE lower(d.database_name) NOT IN ('system', 'temp', 'memory')
        """.format(db=db_ident, schema=schema)
        table_constraints = f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.table_constraints AS
            SELECT
                c.table_catalog AS CONSTRAINT_CATALOG,
                c.table_schema AS CONSTRAINT_SCHEMA,
                c.constraint_name AS CONSTRAINT_NAME,
                c.table_catalog AS TABLE_CATALOG,
                c.table_schema AS TABLE_SCHEMA,
                c.table_name AS TABLE_NAME,
                c.constraint_type AS CONSTRAINT_TYPE,
                'NO'::VARCHAR AS IS_DEFERRABLE,
                'NO'::VARCHAR AS INITIALLY_DEFERRED,
                'NO'::VARCHAR AS ENFORCED,
                c.comment AS COMMENT
            FROM (
                SELECT DISTINCT
                    table_catalog, table_schema, table_name,
                    constraint_name, constraint_type, comment
                FROM {db_ident}.{schema}._sf_constraints_ext
            ) c
        """
        key_column_usage = f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.key_column_usage AS
            SELECT
                table_catalog AS CONSTRAINT_CATALOG,
                table_schema AS CONSTRAINT_SCHEMA,
                constraint_name AS CONSTRAINT_NAME,
                table_catalog AS TABLE_CATALOG,
                table_schema AS TABLE_SCHEMA,
                table_name AS TABLE_NAME,
                column_name AS COLUMN_NAME,
                key_sequence AS ORDINAL_POSITION,
                CASE WHEN constraint_type = 'FOREIGN KEY'
                     THEN NULL ELSE key_sequence END AS POSITION_IN_UNIQUE_CONSTRAINT
            FROM {db_ident}.{schema}._sf_constraints_ext
        """
        constraint_column_usage = f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.constraint_column_usage AS
            SELECT
                table_catalog AS CONSTRAINT_CATALOG,
                table_schema AS CONSTRAINT_SCHEMA,
                constraint_name AS CONSTRAINT_NAME,
                table_catalog AS TABLE_CATALOG,
                table_schema AS TABLE_SCHEMA,
                table_name AS TABLE_NAME,
                column_name AS COLUMN_NAME
            FROM {db_ident}.{schema}._sf_constraints_ext
        """
        referential_constraints = f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.referential_constraints AS
            SELECT DISTINCT
                table_catalog AS CONSTRAINT_CATALOG,
                table_schema AS CONSTRAINT_SCHEMA,
                constraint_name AS CONSTRAINT_NAME,
                NULL::VARCHAR AS UNIQUE_CONSTRAINT_CATALOG,
                NULL::VARCHAR AS UNIQUE_CONSTRAINT_SCHEMA,
                NULL::VARCHAR AS UNIQUE_CONSTRAINT_NAME,
                'NONE'::VARCHAR AS MATCH_OPTION,
                'NO ACTION'::VARCHAR AS UPDATE_RULE,
                'NO ACTION'::VARCHAR AS DELETE_RULE
            FROM {db_ident}.{schema}._sf_constraints_ext
            WHERE constraint_type = 'FOREIGN KEY'
        """
        functions = f"""
            CREATE OR REPLACE VIEW {db_ident}.{schema}.functions AS
            SELECT
                function_catalog AS FUNCTION_CATALOG,
                function_schema AS FUNCTION_SCHEMA,
                function_name AS FUNCTION_NAME,
                NULL::VARCHAR AS FUNCTION_OWNER,
                '(' || replace(signature, '|', ', ') || ')' AS ARGUMENT_SIGNATURE,
                return_type AS DATA_TYPE,
                language AS FUNCTION_LANGUAGE,
                body AS FUNCTION_DEFINITION,
                'SCALAR'::VARCHAR AS FUNCTION_TYPE,
                'N'::VARCHAR AS IS_SECURE,
                'N'::VARCHAR AS IS_BUILTIN,
                created_on AS CREATED,
                created_on AS LAST_ALTERED,
                NULL::VARCHAR AS COMMENT
            FROM {db_ident}.{schema}._sf_functions_ext
        """
        for sql in (
            views,
            sequences,
            databases,
            table_constraints,
            key_column_usage,
            constraint_column_usage,
            referential_constraints,
            functions,
        ):
            self._connection.execute(sql)

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

    def record_constraints(
        self,
        database: str,
        schema: str,
        table: str,
        rows: list[tuple[str, str, str, int, str | None, str | None]],
        *,
        replace: bool = True,
    ) -> None:
        """Record table constraints without letting DuckDB enforce them.

        Snowflake accepts primary/unique/foreign-key declarations but does not enforce
        them; the emulator therefore strips them from the DuckDB DDL and keeps the
        metadata here for ``SHOW ... KEYS``.
        """
        db_ident = _quote_ident(database)
        ext_ident = _quote_ident(INFORMATION_SCHEMA_SCHEMA)
        with self._lock:
            if replace:
                self._connection.execute(
                    f"DELETE FROM {db_ident}.{ext_ident}._sf_constraints_ext "
                    "WHERE lower(table_catalog) = lower(?) AND lower(table_schema) = lower(?) "
                    "AND lower(table_name) = lower(?)",
                    [database, schema, table],
                )
            for constraint_name, constraint_type, column_name, key_sequence, ref_table, ref_column in rows:
                self._connection.execute(
                    f"INSERT OR REPLACE INTO {db_ident}.{ext_ident}._sf_constraints_ext VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        database,
                        schema,
                        table,
                        constraint_name,
                        constraint_type,
                        column_name,
                        key_sequence,
                        ref_table,
                        ref_column,
                        None,
                    ],
                )

    def _create_functions_ext_table(self, db_ident: str, schema: str) -> None:
        """Create the user-defined function metadata side table if it is missing."""
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {db_ident}.{schema}._sf_functions_ext (
                function_catalog VARCHAR,
                function_schema VARCHAR,
                function_name VARCHAR,
                num_arguments INTEGER,
                signature VARCHAR,
                callable_name VARCHAR,
                sub_macro_name VARCHAR,
                arguments VARCHAR,
                return_type VARCHAR,
                return_type_duckdb VARCHAR,
                language VARCHAR,
                body VARCHAR,
                created_on TIMESTAMP,
                PRIMARY KEY (
                    function_catalog, function_schema, function_name,
                    num_arguments, signature
                )
            )
            """
        )

    def _create_constraints_ext_table(self, db_ident: str, schema: str) -> None:
        """Create the constraint metadata side table if it is missing."""
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {db_ident}.{schema}._sf_constraints_ext (
                table_catalog VARCHAR,
                table_schema VARCHAR,
                table_name VARCHAR,
                constraint_name VARCHAR,
                constraint_type VARCHAR,
                column_name VARCHAR,
                key_sequence INTEGER,
                referenced_table VARCHAR,
                referenced_column VARCHAR,
                comment VARCHAR,
                PRIMARY KEY (
                    table_catalog, table_schema, table_name, constraint_name,
                    column_name, key_sequence
                )
            )
            """
        )

    def _migrate_functions_ext(self, db_ident: str, schema: str) -> None:
        """Upgrade a ``_sf_functions_ext`` table created before type overloading.

        The earlier shape keyed on (name, arity) only and called the column
        ``macro_name``, so it cannot hold two overloads of one name/arity. Rebuild it,
        deriving the signature from the stored JSON argument list. ``return_type_duckdb``
        is left NULL: it is only needed to promote a function to a run-time dispatcher,
        which reports the gap rather than guessing a type.
        """
        table = f"{db_ident}.{schema}._sf_functions_ext"
        try:
            # The only expected failure is the missing column, which is the signal that
            # the table predates type overloading (CREATE TABLE IF NOT EXISTS above is a
            # no-op for a table that already exists in the old shape).
            self._connection.execute(f"SELECT signature FROM {table} LIMIT 0")
            return
        except duckdb.Error:
            pass

        legacy_name = "_sf_functions_ext_v1"
        legacy = f"{db_ident}.{schema}.{legacy_name}"
        rows = self._connection.execute(
            "SELECT function_catalog, function_schema, function_name, num_arguments, "
            f"macro_name, arguments, return_type, language, body, created_on FROM {table}"
        ).fetchall()
        self._connection.execute(f"ALTER TABLE {table} RENAME TO {legacy_name}")
        self._create_functions_ext_table(db_ident, schema)
        for row in rows:
            # ``arguments`` holds [[name, type], ...]; the signature is its type list.
            try:
                pairs = json.loads(row[5]) if row[5] else []
            except ValueError:
                pairs = []
            signature = "|".join(str(pair[1]) for pair in pairs)
            self._connection.execute(
                f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, ?, ?, ?)",
                [*row[:4], signature, row[4], row[5], row[6], row[7], row[8], row[9]],
            )
        self._connection.execute(f"DROP TABLE {legacy}")

    def save_function(
        self,
        database: str,
        schema: str,
        *,
        name: str,
        num_arguments: int,
        signature: str,
        callable_name: str,
        sub_macro_name: str | None,
        arguments: str,
        return_type: str,
        return_type_duckdb: str | None,
        language: str,
        body: str,
    ) -> None:
        """Record a user-defined function's Snowflake-side metadata.

        Keyed by (name, arity, signature), so several overloads of one name/arity can
        coexist for run-time dispatch.
        """
        db_ident = _quote_ident(database)
        ext_ident = _quote_ident(INFORMATION_SCHEMA_SCHEMA)
        with self._lock:
            self._connection.execute(
                f"DELETE FROM {db_ident}.{ext_ident}._sf_functions_ext "
                "WHERE lower(function_catalog) = lower(?) AND lower(function_schema) = lower(?) "
                "AND lower(function_name) = lower(?) AND num_arguments = ? AND signature = ?",
                [database, schema, name, num_arguments, signature],
            )
            self._connection.execute(
                f"INSERT INTO {db_ident}.{ext_ident}._sf_functions_ext VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())",
                [
                    database,
                    schema,
                    name,
                    num_arguments,
                    signature,
                    callable_name,
                    sub_macro_name,
                    arguments,
                    return_type,
                    return_type_duckdb,
                    language,
                    body,
                ],
            )
            self._function_arity_cache.pop(self._function_cache_key(database, schema), None)

    def set_function_dispatch(
        self,
        database: str,
        schema: str,
        name: str,
        num_arguments: int,
        callable_name: str,
        sub_macros: dict[str, str],
    ) -> None:
        """Point every overload of one (name, arity) at a shared dispatcher.

        `sub_macros` maps each overload's signature to the macro holding its body.
        """
        db_ident = _quote_ident(database)
        ext_ident = _quote_ident(INFORMATION_SCHEMA_SCHEMA)
        with self._lock:
            for signature, sub_macro_name in sub_macros.items():
                self._connection.execute(
                    f"UPDATE {db_ident}.{ext_ident}._sf_functions_ext "
                    "SET callable_name = ?, sub_macro_name = ? "
                    "WHERE lower(function_catalog) = lower(?) "
                    "AND lower(function_schema) = lower(?) "
                    "AND lower(function_name) = lower(?) AND num_arguments = ? "
                    "AND signature = ?",
                    [
                        callable_name,
                        sub_macro_name,
                        database,
                        schema,
                        name,
                        num_arguments,
                        signature,
                    ],
                )
            self._function_arity_cache.pop(self._function_cache_key(database, schema), None)

    def delete_functions(
        self,
        database: str,
        schema: str,
        name: str,
        num_arguments: int | None = None,
        signature: str | None = None,
    ) -> list[tuple[str, int, str | None]]:
        """Delete a function's metadata, returning removed (name, arity, sub-macro) rows.

        With neither ``num_arguments`` nor ``signature`` every overload of the name is
        removed, matching Snowflake's ``DROP FUNCTION f`` without a signature. The
        signature narrows the drop to one overload, which matters as soon as one name and
        arity carries several of them.
        """
        db_ident = _quote_ident(database)
        ext_ident = _quote_ident(INFORMATION_SCHEMA_SCHEMA)
        where = (
            "WHERE lower(function_catalog) = lower(?) AND lower(function_schema) = lower(?) "
            "AND lower(function_name) = lower(?)"
        )
        params: list[Any] = [database, schema, name]
        if num_arguments is not None:
            where += " AND num_arguments = ?"
            params.append(num_arguments)
        if signature is not None:
            where += " AND signature = ?"
            params.append(signature)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT function_name, num_arguments, sub_macro_name, signature FROM "
                f"{db_ident}.{ext_ident}._sf_functions_ext {where}",
                params,
            ).fetchall()
            self._connection.execute(
                f"DELETE FROM {db_ident}.{ext_ident}._sf_functions_ext {where}", params
            )
            self._function_arity_cache.pop(self._function_cache_key(database, schema), None)
        return [
            (str(row[0]), int(row[1]), row[2] if row[2] is None else str(row[2]))
            for row in rows
        ]

    def query_functions(
        self,
        database: str,
        schema: str | None = None,
        name: str | None = None,
        like: str | None = None,
        signature: str | None = None,
    ) -> list[tuple[Any, ...]]:
        """Return recorded functions as (name, schema, catalog, arity, signature,
        callable_name, sub_macro_name, arguments, return_type, return_type_duckdb,
        language, body, created_on)."""
        db_ident = _quote_ident(database)
        ext_ident = _quote_ident(INFORMATION_SCHEMA_SCHEMA)
        where = "WHERE lower(function_catalog) = lower(?)"
        params: list[Any] = [database]
        if schema is not None:
            where += " AND lower(function_schema) = lower(?)"
            params.append(schema)
        if name is not None:
            where += " AND lower(function_name) = lower(?)"
            params.append(name)
        if like is not None:
            where += " AND lower(function_name) LIKE lower(?)"
            params.append(like)
        if signature is not None:
            where += " AND signature = ?"
            params.append(signature)
        with self._lock:
            return self._connection.execute(
                "SELECT function_name, function_schema, function_catalog, num_arguments, "
                "signature, callable_name, sub_macro_name, arguments, return_type, "
                "return_type_duckdb, language, body, created_on "
                f"FROM {db_ident}.{ext_ident}._sf_functions_ext {where} "
                "ORDER BY lower(function_name), num_arguments, signature",
                params,
            ).fetchall()

    def function_arities(self, database: str, schema: str) -> dict[str, dict[int, str]]:
        """Return ``{UPPERCASE(function name): {arity: callable name}}`` for a schema.

        Used to resolve user-defined function calls to the macro (or dispatch function)
        backing them while translating a query. Every overload of one name/arity shares a
        callable name, so the last row for an arity is the same as the first.
        """
        key = self._function_cache_key(database, schema)
        with self._lock:
            cached = self._function_arity_cache.get(key)
            if cached is None:
                cached = {}
                for row in self.query_functions(database, schema):
                    cached.setdefault(str(row[0]).upper(), {})[int(row[3])] = str(row[5])
                self._function_arity_cache[key] = cached
            return cached

    @staticmethod
    def _function_cache_key(database: str, schema: str) -> tuple[str, str]:
        return (database.lower(), schema.lower())

    def cursor(self, session_token: str | None = None) -> duckdb.DuckDBPyConnection:
        """Return the DuckDB connection to run a statement on.

        With a `session_token` the session's own long-lived connection is returned, so
        tables created with ``CREATE TEMPORARY TABLE`` (which DuckDB scopes to a
        connection) survive from one statement to the next and are dropped when the
        session ends. Without one, an isolated cursor is returned so concurrent callers
        don't clobber each other's state.
        """
        if session_token is None:
            return self._connection.cursor()
        key = _session_key(session_token)
        with self._lock:
            cursor = self._session_cursors.get(key)
            if cursor is None:
                cursor = self._connection.cursor()
                self._session_cursors[key] = cursor
            return cursor

    def session_lock(self, session_token: str | None) -> threading.RLock:
        """Return the lock serializing one session's statements (see `cursor`)."""
        key = _session_key(session_token)
        with self._lock:
            lock = self._session_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._session_locks[key] = lock
            return lock

    @contextmanager
    def session_scope(self, session_token: str | None) -> Iterator[None]:
        """Hold a session's statement lock for the duration of one statement."""
        with self.session_lock(session_token):
            yield

    def release_session(self, session_token: str | None) -> None:
        """Drop a closed session's connection, discarding its temporary tables.

        The session's statement lock is held across the pop-and-close so a logout cannot
        pull the connection out from under a statement that is still running on it.
        """
        key = _session_key(session_token)
        with self.session_lock(key):
            with self._lock:
                cursor = self._session_cursors.pop(key, None)
                self._session_locks.pop(key, None)
                self._session_temp_tables.pop(key, None)
            _close_quietly(cursor)

    def record_temp_table(
        self, session_token: str | None, table: str, database: str
    ) -> None:
        """Remember that a session's temporary table belongs to `database`."""
        key = _session_key(session_token)
        with self._lock:
            self._session_temp_tables.setdefault(key, {})[table.lower()] = database.lower()

    def _drop_replaced_temp_tables(self, database: str) -> None:
        """Drop the temporary tables a replaced database owned, in every session.

        Best effort: a temporary table that has already been dropped, or whose session
        is mid-statement, must not make ``CREATE OR REPLACE DATABASE`` fail.
        """
        target = database.lower()
        with self._lock:
            victims = [
                (key, table)
                for key, tables in self._session_temp_tables.items()
                for table, owner in tables.items()
                if owner == target
            ]
            for key, table in victims:
                self._session_temp_tables.get(key, {}).pop(table, None)
        for key, table in victims:
            escaped = table.replace('"', '""')
            with self.session_lock(key):
                with self._lock:
                    cursor = self._session_cursors.get(key)
                if cursor is None:
                    continue
                try:
                    cursor.execute(f'DROP TABLE IF EXISTS temp."{escaped}"')
                except Exception:  # noqa: BLE001 - best-effort cleanup, see docstring
                    continue

    def has_database(self, database: str) -> bool:
        """Return True if a Snowflake database is already attached as a catalog.

        A persistent database whose file exists but has not been attached yet (e.g. a
        fresh manager mid-lazy-attach) still counts as existing, so ``CREATE DATABASE``
        reports a collision instead of silently re-attaching it.
        """
        if database.lower() in self._attached_catalogs:
            return True
        return self._persistent and Path(self._catalog_path(database)).exists()

    def drop_database(self, database: str) -> None:
        """Drop a Snowflake database, detaching its catalog and deleting its file."""
        db_ident = _quote_ident(database)
        with self._lock:
            if database.lower() in self._attached_catalogs:
                self._connection.execute(f"DETACH {db_ident}")
                self._attached_catalogs.discard(database.lower())
            self._function_arity_cache = {
                key: value
                for key, value in self._function_arity_cache.items()
                if key[0] != database.lower()
            }
            if self._persistent:
                path = Path(self._catalog_path(database))
                for candidate in (path, path.with_name(f"{path.name}.wal")):
                    if candidate.exists():
                        candidate.unlink()

    def replace_database(self, database: str) -> None:
        """Recreate a Snowflake database (``CREATE OR REPLACE DATABASE``)."""
        db_ident = _quote_ident(database)
        # Temporary tables live in DuckDB's connection-local ``temp`` catalog rather than
        # in the catalog being replaced, so they have to be dropped explicitly for the
        # replacement to cascade the way it does on Snowflake.
        self._drop_replaced_temp_tables(database)
        # Hold the lock across DETACH, file removal and re-ATTACH so a concurrent
        # reference cannot observe (or re-attach) the database mid-replacement.
        with self._lock:
            if database.lower() in self._attached_catalogs:
                self._connection.execute(f"DETACH {db_ident}")
                self._attached_catalogs.discard(database.lower())
            self._function_arity_cache = {
                key: value
                for key, value in self._function_arity_cache.items()
                if key[0] != database.lower()
            }
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


def _session_key(session_token: str | None) -> str:
    """Key session-scoped state by token, treating an absent token as one anonymous session."""
    return session_token or "<anonymous>"


def _close_quietly(cursor: duckdb.DuckDBPyConnection | None) -> None:
    """Close a connection, ignoring an already-closed one."""
    if cursor is None:
        return
    try:
        cursor.close()
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass


_STORAGE_SUFFIX = ".d"  # dedicated directory: <database_path>.d
_DATABASE_FILE_SUFFIX = ".duckdb"  # one file per database inside that directory
# Minimum DuckDB storage version that supports VARIANT columns (its default is older).
_STORAGE_VERSION = "v1.5.0"

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


def release_session(token: str | None) -> None:
    """Drop a closed session's connection from the manager serving it.

    In ``isolated`` mode the session's whole private database is released by
    :func:`release_isolated_manager`, which drops the connection with it. In ``shared``
    mode there is one common manager whose per-session connection -- and with it the
    session's temporary tables -- has to be closed here.
    """
    if settings.isolation == "isolated":
        return
    global _manager
    if _manager is not None:
        _manager.release_session(token)


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
