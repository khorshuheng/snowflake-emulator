"""The AST rewrite pipeline shared by statements and UDF bodies."""

from __future__ import annotations

from sqlglot import exp

from snowflake_emulator.translator.arrays import (
    ColumnTypeResolver,
    _rewrite_json_array_functions,
)
from snowflake_emulator.translator.semi_structured import (
    _rewrite_information_schema,
    _rewrite_snowflake_specifics,
    _uppercase_unquoted_identifiers,
)


def _apply_transforms(
    statement: exp.Expression, column_types: ColumnTypeResolver | None = None
) -> exp.Expression:
    """Apply the Snowflake-to-DuckDB AST rewrites shared by statements and UDF bodies.

    A ``CREATE FUNCTION`` body is Snowflake SQL in its own right, so it has to go
    through exactly the same rewrites as a top-level statement -- that is what turns
    e.g. ``EQUAL_NULL(a, b)`` into DuckDB's ``a IS NOT DISTINCT FROM b``. `column_types`
    resolves a bare column's type for the array rewrites; a UDF body has no catalog in
    scope, so it is optional.
    """
    statement = statement.transform(_uppercase_unquoted_identifiers)
    statement = statement.transform(_rewrite_information_schema)
    statement = statement.transform(_rewrite_snowflake_specifics)
    # After the rewrites above: an array call is rebuilt from its operands' SQL, so
    # those operands must already be in their final DuckDB form.
    return _rewrite_json_array_functions(statement, column_types)
