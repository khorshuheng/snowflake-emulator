"""Catalog lookups the translator needs for a bare column.

A bare column carries no type in the AST, but some Snowflake rewrites depend on one: an array
function has to know whether the column really holds a DuckDB LIST (and of which element
type) or a VARIANT/JSON array, and ``ARRAY_SORT``/``ARRAY_MAX``/``ARRAY_MIN`` may only reject a
VARIANT one. ``column_type_resolver`` answers that question from DuckDB's catalog on demand,
so a statement with no bare-column array operand never pays for the query.
"""

from __future__ import annotations

from typing import Any, Callable

# ``(catalog_or_None, schema_or_None, table_or_None, column) -> DuckDB type name`` (e.g.
# ``"VARCHAR[]"``), or None when the column cannot be attributed. Matched case-insensitively,
# because the translator works on upper-cased unquoted identifiers while DuckDB keeps the name
# the table was created with.
ColumnTypeResolver = Callable[[str | None, str | None, str | None, str], str | None]


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
        database, schema, _, _, _ = row
        return (
            0 if database.lower() == "temp" else 1,
            0 if schema.lower() == str(session.schema).lower() else 1,
            0 if database.lower() == str(session.database).lower() else 1,
        )

    def resolve(
        catalog: str | None, schema: str | None, table: str | None, column: str
    ) -> str | None:
        wanted_column = column.upper()
        candidates = [row for row in load() if row[3].upper() == wanted_column]
        for index, wanted in ((0, catalog), (1, schema), (2, table)):
            if wanted:
                candidates = [
                    row for row in candidates if row[index].upper() == wanted.upper()
                ]
        if not candidates:
            return None
        if len({row[4].upper() for row in candidates}) > 1:
            # The name resolves to more than one type -- the same column in two tables of
            # different type, or the same table name in two schemas when the reference did not
            # name the schema. Guessing would change which rewrite applies (a VARIANT column
            # read as a native list reaches DuckDB's VARIANT ``list_sort``, which fails with an
            # out-of-memory error), so leave it unresolved and let the caller stay conservative.
            return None
        return sorted(candidates, key=rank)[0][4]

    return resolve
