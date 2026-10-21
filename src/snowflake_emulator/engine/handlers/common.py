"""Helpers shared by the SHOW and DESCRIBE handlers."""

from __future__ import annotations

import re
from typing import Any

from sqlglot import exp

from snowflake_emulator.core.constants import INFORMATION_SCHEMA_SCHEMA
from snowflake_emulator.core.results import ColumnMeta
from snowflake_emulator.type_mapping import (
    duckdb_type_to_snowflake,
    duckdb_type_to_snowflake_sql_type,
)

_DECLARED_TYPE_OVERRIDE = re.compile(r"^(VARCHAR|CHAR|STRING|TEXT|BINARY|VARBINARY)\(\d+\)$")

# A ``TIMESTAMP_NTZ(7..9)`` column is stored at microsecond precision (DuckDB's nanosecond
# type only spans ~1677-2262, Snowflake's range is year 1-9999), so the declared precision has
# to come back from the side table. Precision 0-6 needs no restoring: the physical type already
# reports it. The side table keeps sqlglot's spelling (``TIMESTAMPNTZ(9)``).
_TIMESTAMP_NTZ_PRECISION = re.compile(r"^TIMESTAMP_?NTZ\((\d+)\)$")


def show_name_patterns(
    statement: exp.Expression,
) -> tuple[str | None, str | None]:
    """Return the ``(STARTS WITH, LIKE)`` name filters of a SHOW statement."""
    if not isinstance(statement, exp.Show):
        return None, None
    prefix = statement.args.get("starts_with")
    prefix = prefix.this if isinstance(prefix, exp.Literal) else None
    like = statement.args.get("like")
    like = like.this if isinstance(like, exp.Literal) else None
    return prefix, like


def name_matches(value: str, prefix: str | None, like: str | None) -> bool:
    """Case-insensitive match for an in-memory SHOW result row."""
    if prefix and not value.upper().startswith(prefix.upper()):
        return False
    if not like:
        return True
    # Translate SQL LIKE (``%``/``_``, backslash escape) to a regex.
    parts: list[str] = []
    i = 0
    while i < len(like):
        char = like[i]
        if char == "\\" and i + 1 < len(like):
            parts.append(re.escape(like[i + 1]))
            i += 2
            continue
        if char == "%":
            parts.append(".*")
        elif char == "_":
            parts.append(".")
        else:
            parts.append(re.escape(char))
        i += 1
    return re.fullmatch("".join(parts), value, re.IGNORECASE) is not None


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
    upper = declared.upper()
    match = _TIMESTAMP_NTZ_PRECISION.match(upper)
    if match and int(match.group(1)) > 6:
        return f"TIMESTAMP_NTZ({match.group(1)})"
    return declared if _DECLARED_TYPE_OVERRIDE.match(upper) else None


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
    internal = (
        f" AND lower({alias}.schema_name) != lower('{INFORMATION_SCHEMA_SCHEMA}')"
    )
    shadowed = (
        " AND NOT EXISTS (SELECT 1 FROM duckdb_tables() sh"
        " WHERE lower(sh.database_name) = 'temp'"
        f" AND lower(sh.table_name) = lower({alias}.{name_column}))"
    )
    return (
        f"((lower({alias}.database_name) = lower('{database}'){schema_match}{internal}{shadowed})"
        f" OR (lower({alias}.database_name) = 'temp'"
        f" AND lower(current_database()) = lower('{database}'){temp_schema_match}))"
    )
