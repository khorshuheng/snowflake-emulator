"""Catalog lookups the translator needs for a bare column.

A bare column carries no type in the AST, but some Snowflake rewrites depend on one: an array
function has to know whether the column really holds a DuckDB LIST (and of which element
type) or a VARIANT/JSON array, and ``ARRAY_SORT``/``ARRAY_MAX``/``ARRAY_MIN`` may only reject a
VARIANT one. ``column_type_resolver`` answers that question from DuckDB's catalog on demand,
so a statement with no bare-column array operand never pays for the query.
"""

from __future__ import annotations

from typing import Any, Callable

# ``(table_name_or_None, column_name) -> DuckDB type name`` (e.g. ``"VARCHAR[]"``), or None
# when the column cannot be found. Matched case-insensitively, because the translator works on
# upper-cased unquoted identifiers while DuckDB keeps the name the table was created with.
ColumnTypeResolver = Callable[[str | None, str], str | None]


def column_type_resolver(cursor: Any, session: Any) -> ColumnTypeResolver:
    """Build a lazily-populated ``(table, column)`` -> DuckDB type lookup.

    `session` supplies the current database and schema, which decide a tie when two schemas
    hold a table of the same name (a temporary table shadows a permanent one, as it does for
    a query).
    """
    columns: list[tuple[str, str, str, str, str]] | None = None

    def load() -> list[tuple[str, str, str, str, str]]:
        nonlocal columns
        if columns is None:
            columns = [
                tuple(str(value) for value in row)
                for row in cursor.execute(
                    "SELECT database_name, schema_name, table_name, column_name, data_type "
                    "FROM duckdb_columns()"
                ).fetchall()
            ]
        return columns

    def rank(row: tuple[str, str, str, str, str]) -> tuple[int, int, int]:
        """Sort key: a temporary table first, then the session's schema, then its database."""
        database, schema, table, _, _ = row
        return (
            0 if database.lower() == "temp" else 1,
            0 if schema.lower() == str(session.schema).lower() else 1,
            0 if database.lower() == str(session.database).lower() else 1,
        )

    def resolve(table: str | None, column: str) -> str | None:
        wanted_column = column.upper()
        wanted_table = table.upper() if table else None
        candidates = [
            row
            for row in load()
            if row[3].upper() == wanted_column
            and (wanted_table is None or row[2].upper() == wanted_table)
        ]
        if not candidates:
            return None
        if wanted_table is None and len({row[4].upper() for row in candidates}) > 1:
            # The same column name in two differently-typed tables. An unqualified column
            # cannot be attributed to one of them without reading the statement's FROM clause,
            # and guessing a type would silently change which rewrite applies, so leave it
            # unresolved and let the caller keep its conservative behaviour.
            return None
        return sorted(candidates, key=rank)[0][4]

    return resolve
