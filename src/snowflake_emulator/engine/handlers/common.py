"""Helpers shared by the SHOW and DESCRIBE handlers."""

from __future__ import annotations

import re
from typing import Any

from snowflake_emulator.core.results import ColumnMeta
from snowflake_emulator.type_mapping import (
    duckdb_type_to_snowflake,
    duckdb_type_to_snowflake_sql_type,
)

_DECLARED_TYPE_OVERRIDE = re.compile(r"^(VARCHAR|CHAR|STRING|TEXT|BINARY|VARBINARY)\(\d+\)$")


def result_columns(cursor: Any) -> tuple[list[ColumnMeta], list[list[Any]]]:
    """Build ``(row_type, rows)`` from a cursor that just executed a query."""
    description = cursor.description
    row_type = [
        ColumnMeta(
            name=col[0],
            type=duckdb_type_to_snowflake(str(col[1])),
            duckdb_type=str(col[1]),
        )
        for col in description
    ]
    rows = [list(row) for row in cursor.fetchall()]
    return row_type, rows


def convert_describe_types(rows: list[list[Any]]) -> list[list[Any]]:
    """Convert DuckDB type names in a ``DESCRIBE`` result to Snowflake type names.

    DuckDB's ``DESCRIBE`` returns a ``column_type`` column (index 1) whose values are
    DuckDB-specific type names. Snowflake's ``DESCRIBE TABLE`` reports Snowflake SQL
    types in that column, so we rewrite each value in place.
    """
    converted: list[list[Any]] = []
    for row in rows:
        if len(row) >= 2 and isinstance(row[1], str):
            row = list(row)
            row[1] = duckdb_type_to_snowflake_sql_type(row[1])
        converted.append(row)
    return converted


def prefer_declared_type(declared: Any, duckdb_type: Any) -> str | None:
    """Return the declared Snowflake type only where DuckDB loses information.

    DuckDB keeps numeric precision/scale, so ``INT`` must still report as
    ``NUMBER(38,0)``; only character/binary lengths are dropped, so ``VARCHAR(255)``
    is the one case worth restoring from the side table.
    """
    if not isinstance(declared, str):
        return None
    return declared if _DECLARED_TYPE_OVERRIDE.match(declared.upper()) else None


def temp_inclusive_scope(
    alias: str, database: str, schema: str | None, name_column: str = "table_name"
) -> str:
    """Return a WHERE fragment matching an object in scope, or a session's temp table.

    DuckDB reports a temporary table in the connection-local ``temp`` catalog, so an
    object is in scope when it lives in the catalog/schema asked for *or* when it is a
    temporary table of the calling connection and the caller asked about that
    connection's current database (and schema, when one was named). The ``temp`` catalog
    is per connection, so a session never sees another session's temporary tables.

    A temporary table shadows a permanent one of the same name (DuckDB resolves names to
    the temp catalog first), so the catalog-side row is dropped while a shadowing
    temporary table exists -- otherwise the same column would be reported twice.
    `name_column` is the name column of `alias` (``view_name`` for ``duckdb_views()``).
    """
    schema_match = f" AND lower({alias}.schema_name) = lower('{schema}')" if schema else ""
    temp_schema_match = f" AND lower(current_schema()) = lower('{schema}')" if schema else ""
    shadowed = (
        " AND NOT EXISTS (SELECT 1 FROM duckdb_tables() sh"
        " WHERE lower(sh.database_name) = 'temp'"
        f" AND lower(sh.table_name) = lower({alias}.{name_column}))"
    )
    return (
        f"((lower({alias}.database_name) = lower('{database}'){schema_match}{shadowed})"
        f" OR (lower({alias}.database_name) = 'temp'"
        f" AND lower(current_database()) = lower('{database}'){temp_schema_match}))"
    )
