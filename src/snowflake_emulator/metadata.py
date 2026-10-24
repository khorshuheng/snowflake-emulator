"""Snowflake-only object metadata that DuckDB does not store: stages and file formats.

DuckDB has no stage and no file-format object, so their metadata lives in the per-database
``SF_INFORMATION_SCHEMA`` side tables next to the declared column types and the recorded
constraints: ``_sf_stages_ext`` and ``_sf_file_formats_ext``. The tables are created by
``database._ensure_information_schema`` and surfaced by the ``STAGES``/``FILE_FORMATS``
views, so ``SHOW`` and ``INFORMATION_SCHEMA`` read the same rows.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb

from snowflake_emulator.core.constants import INFORMATION_SCHEMA_SCHEMA
from snowflake_emulator.sessions import SessionContext

STAGES_TABLE = "_sf_stages_ext"
FILE_FORMATS_TABLE = "_sf_file_formats_ext"

# The side-table DDL is shared with ``database._ensure_information_schema`` so the tables
# can be recreated by a write when an older database (or a hand-dropped table) lacks them.
STAGES_TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        created_on TIMESTAMP,
        stage_catalog VARCHAR,
        stage_schema VARCHAR,
        stage_name VARCHAR,
        stage_url VARCHAR,
        stage_type VARCHAR,
        stage_owner VARCHAR,
        comment VARCHAR,
        file_format VARCHAR,
        PRIMARY KEY (stage_catalog, stage_schema, stage_name)
    )
"""
FILE_FORMATS_TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        created_on TIMESTAMP,
        file_format_catalog VARCHAR,
        file_format_schema VARCHAR,
        file_format_name VARCHAR,
        file_format_type VARCHAR,
        options VARCHAR,
        file_format_owner VARCHAR,
        comment VARCHAR,
        PRIMARY KEY (file_format_catalog, file_format_schema, file_format_name)
    )
"""


def table_ddl(database: str, table: str) -> str:
    """The ``CREATE TABLE IF NOT EXISTS`` statement for one side table."""
    ddl = STAGES_TABLE_DDL if table == STAGES_TABLE else FILE_FORMATS_TABLE_DDL
    return ddl.format(table=_side_table(database, table))


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _side_table(database: str, table: str) -> str:
    return f"{_quote(database)}.{_quote(INFORMATION_SCHEMA_SCHEMA)}.{_quote(table)}"


def _cursor(manager: Any) -> Any:
    """The connection the metadata writes go through (the manager's root connection)."""
    return manager.cursor(None)


def _has_table(manager: Any, database: str, table: str) -> bool:
    """True when the side table exists; a missing table is created, other errors surface.

    Only DuckDB's catalog error means "this database was created before the table
    existed"; anything else (a closed connection, a permission failure) must not turn a
    metadata write into a silent no-op.
    """
    try:
        _cursor(manager).execute(f"SELECT 1 FROM {_side_table(database, table)} LIMIT 1")
        return True
    except duckdb.CatalogException:
        try:
            _cursor(manager).execute(table_ddl(database, table))
        except duckdb.CatalogException:
            return False  # the database itself is not attached
        return True


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------


def stage_url(database: str, schema: str, name: str) -> str:
    """The synthetic URL the emulator reports for an internal named stage."""
    return f"s3://sfemu/{database}.{schema}.{name}"


def record_stage(
    manager: Any,
    session: SessionContext,
    name: str,
    *,
    database: str | None = None,
    schema: str | None = None,
    file_format: dict[str, Any] | None = None,
    comment: str | None = None,
) -> None:
    """Persist ``CREATE STAGE`` metadata (type, URL, owner, comment, FILE_FORMAT options)."""
    db = database or session.database
    sc = schema or session.schema
    if not _has_table(manager, db, STAGES_TABLE):
        return
    cursor = _cursor(manager)
    cursor.execute(
        f"DELETE FROM {_side_table(db, STAGES_TABLE)} "
        "WHERE stage_catalog = ? AND stage_schema = ? AND stage_name = ?",
        [db, sc, name],
    )
    cursor.execute(
        f"INSERT INTO {_side_table(db, STAGES_TABLE)} VALUES "
        "(current_localtimestamp(), ?, ?, ?, ?, 'Internal Named', ?, ?, ?)",
        [
            db,
            sc,
            name,
            stage_url(db, sc, name),
            session.role,
            comment,
            json.dumps(file_format or {}),
        ],
    )


def get_stage(
    manager: Any, database: str, schema: str, name: str
) -> dict[str, Any] | None:
    """One recorded stage by exact identity, or None when it does not exist."""
    if not _has_table(manager, database, STAGES_TABLE):
        return None
    cursor = _cursor(manager)
    cursor.execute(
        "SELECT stage_name, stage_catalog, stage_schema, stage_url, stage_type, "
        "stage_owner, comment, file_format FROM "
        f"{_side_table(database, STAGES_TABLE)} "
        "WHERE stage_catalog = ? AND stage_schema = ? AND stage_name = ?",
        [database, schema, name],
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "name": row[0],
        "database": row[1],
        "schema": row[2],
        "url": row[3],
        "type": row[4],
        "owner": row[5],
        "comment": row[6],
        "file_format": json.loads(row[7]) if row[7] else {},
    }


def update_stage(
    manager: Any,
    database: str,
    schema: str,
    name: str,
    *,
    file_format: dict[str, Any] | None = None,
    comment: str | None = None,
) -> None:
    """Update an existing stage's FILE_FORMAT/COMMENT in place (CREATED and owner stay)."""
    if not _has_table(manager, database, STAGES_TABLE):
        return
    _cursor(manager).execute(
        f"UPDATE {_side_table(database, STAGES_TABLE)} "
        "SET file_format = ?, comment = ? "
        "WHERE stage_catalog = ? AND stage_schema = ? AND stage_name = ?",
        [json.dumps(file_format or {}), comment, database, schema, name],
    )


def drop_stage(manager: Any, database: str, schema: str, name: str) -> None:
    """Forget a dropped stage's metadata."""
    if not _has_table(manager, database, STAGES_TABLE):
        return
    _cursor(manager).execute(
        f"DELETE FROM {_side_table(database, STAGES_TABLE)} "
        "WHERE stage_catalog = ? AND stage_schema = ? AND stage_name = ?",
        [database, schema, name],
    )


def stages(
    manager: Any, database: str | None = None, schema: str | None = None
) -> list[dict[str, Any]]:
    """The recorded named stages, optionally scoped to one database/schema."""
    databases = [database] if database else _databases(manager)
    rows: list[dict[str, Any]] = []
    for db in databases:
        if not _has_table(manager, db, STAGES_TABLE):
            continue
        clauses = ["1 = 1"]
        params: list[Any] = []
        if schema:
            clauses.append("lower(stage_schema) = lower(?)")
            params.append(schema)
        cursor = _cursor(manager)
        cursor.execute(
            "SELECT stage_name, stage_catalog, stage_schema, stage_url, stage_type, "
            f"stage_owner, comment, file_format FROM {_side_table(db, STAGES_TABLE)} "
            f"WHERE {' AND '.join(clauses)} ORDER BY stage_schema, stage_name",
            params,
        )
        for row in cursor.fetchall():
            rows.append(
                {
                    "name": row[0],
                    "database": row[1],
                    "schema": row[2],
                    "url": row[3],
                    "type": row[4],
                    "owner": row[5],
                    "comment": row[6],
                    "file_format": json.loads(row[7]) if row[7] else {},
                }
            )
    return rows


def stage_file_format(
    manager: Any, database: str, schema: str, name: str
) -> dict[str, Any]:
    """The FILE_FORMAT options a named stage was created with (empty when none)."""
    if not _has_table(manager, database, STAGES_TABLE):
        return {}
    cursor = _cursor(manager)
    cursor.execute(
        f"SELECT file_format FROM {_side_table(database, STAGES_TABLE)} "
        "WHERE stage_catalog = ? AND stage_schema = ? AND stage_name = ?",
        [database, schema, name],
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        return {}
    try:
        options = json.loads(row[0])
    except ValueError:
        return {}
    return {str(key).upper(): value for key, value in options.items()}


# --------------------------------------------------------------------------------------
# File formats
# --------------------------------------------------------------------------------------


def record_file_format(
    manager: Any,
    session: SessionContext,
    name: str,
    options: dict[str, Any],
    *,
    replace: bool = False,
    if_not_exists: bool = False,
    database: str | None = None,
    schema: str | None = None,
) -> bool:
    """Persist a named file format; False when it already exists and IF NOT EXISTS was used."""
    db = database or session.database
    sc = schema or session.schema
    if not _has_table(manager, db, FILE_FORMATS_TABLE):
        return False
    cursor = _cursor(manager)
    existing = file_format(manager, db, sc, name)
    if existing is not None and not replace:
        if if_not_exists:
            return False
        raise FileFormatExists(name)
    cursor.execute(
        f"DELETE FROM {_side_table(db, FILE_FORMATS_TABLE)} "
        "WHERE lower(file_format_schema) = lower(?) AND lower(file_format_name) = lower(?)",
        [sc, name],
    )
    upper = {str(key).upper(): value for key, value in options.items()}
    cursor.execute(
        f"INSERT INTO {_side_table(db, FILE_FORMATS_TABLE)} VALUES "
        "(current_localtimestamp(), ?, ?, ?, ?, ?, ?, ?)",
        [
            db,
            sc,
            name,
            upper.get("TYPE", "CSV"),
            json.dumps(upper),
            session.role,
            upper.get("COMMENT"),
        ],
    )
    return True


class FileFormatExists(Exception):
    """Raised when a named file format already exists and REPLACE was not given."""


def drop_file_format(
    manager: Any,
    database: str,
    schema: str,
    name: str,
    *,
    if_exists: bool = False,
) -> bool:
    """Delete a named file format; False when it did not exist and IF EXISTS was used."""
    existing = file_format(manager, database, schema, name)
    if existing is None:
        if if_exists:
            return False
        raise FileFormatNotFound(name)
    if _has_table(manager, database, FILE_FORMATS_TABLE):
        _cursor(manager).execute(
            f"DELETE FROM {_side_table(database, FILE_FORMATS_TABLE)} "
            "WHERE lower(file_format_schema) = lower(?) AND lower(file_format_name) = lower(?)",
            [schema, name],
        )
    return True


class FileFormatNotFound(Exception):
    """Raised when a named file format does not exist and IF EXISTS was not given."""


def file_format(
    manager: Any, database: str, schema: str, name: str
) -> dict[str, Any] | None:
    """One named file format's options, or None when it does not exist."""
    if not _has_table(manager, database, FILE_FORMATS_TABLE):
        return None
    cursor = _cursor(manager)
    cursor.execute(
        f"SELECT options FROM {_side_table(database, FILE_FORMATS_TABLE)} "
        "WHERE lower(file_format_schema) = lower(?) AND lower(file_format_name) = lower(?)",
        [schema, name],
    )
    row = cursor.fetchone()
    if row is None or not row[0]:
        return None
    try:
        options = json.loads(row[0])
    except ValueError:
        return None
    return {str(key).upper(): value for key, value in options.items()}


def file_formats(
    manager: Any, database: str, schema: str | None = None
) -> list[dict[str, Any]]:
    """The named file formats of a database (optionally one schema)."""
    if not _has_table(manager, database, FILE_FORMATS_TABLE):
        return []
    clauses = ["1 = 1"]
    params: list[Any] = []
    if schema:
        clauses.append("lower(file_format_schema) = lower(?)")
        params.append(schema)
    cursor = _cursor(manager)
    cursor.execute(
        "SELECT file_format_name, file_format_schema, file_format_type, options, "
        f"file_format_owner, comment FROM {_side_table(database, FILE_FORMATS_TABLE)} "
        f"WHERE {' AND '.join(clauses)} ORDER BY file_format_schema, file_format_name",
        params,
    )
    rows: list[dict[str, Any]] = []
    for row in cursor.fetchall():
        options = json.loads(row[3]) if row[3] else {}
        rows.append(
            {
                "name": row[0],
                "schema": row[1],
                "type": row[2],
                "options": options,
                "owner": row[4],
                "comment": row[5],
            }
        )
    return rows


def _databases(manager: Any) -> list[str]:
    return [
        str(row[0])
        for row in _cursor(manager)
        .execute(
            "SELECT database_name FROM duckdb_databases() "
            "WHERE lower(database_name) NOT IN ('system', 'temp', 'memory') ORDER BY 1"
        )
        .fetchall()
    ]
