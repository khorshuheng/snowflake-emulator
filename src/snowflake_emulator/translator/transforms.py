"""The AST rewrite pipeline shared by statements and UDF bodies."""

from __future__ import annotations

from sqlglot import exp

from snowflake_emulator.translator import column_types
from snowflake_emulator.translator.aggregates import _rewrite_aggregate_functions
from snowflake_emulator.translator.arrays import (
    ColumnTypeResolver,
    _rewrite_json_array_functions,
)
from snowflake_emulator.translator.regexp import _rewrite_regexp_functions
from snowflake_emulator.translator.semi_structured import (
    _rewrite_information_schema,
    _rewrite_snowflake_specifics,
    _uppercase_unquoted_identifiers,
    rewrite_semi_structured_writes,
)


# Snowflake's TIMESTAMP_NTZ accepts 0-9 fractional digits; DuckDB stores anything above 6 as
# ``TIMESTAMP_NS`` (int64 nanoseconds), whose range is only ~1677-2262. Snowflake's own range is
# year 1-9999, so a nanosecond column cannot hold values Snowflake accepts (a ``9999-12-31``
# expiry fails with "Could not convert string ... to INT64"). Storage is therefore capped at
# microsecond precision, which spans the whole Snowflake range; digits 7-9 are masked on write.
# TIMESTAMP_TZ/TIMESTAMP_LTZ are not affected (sqlglot maps them to TIMESTAMPTZ, µs).
_TIMESTAMP_MICROSECOND_PRECISION = 6


def _rewrite_timestamp_precision(node: exp.Expression) -> exp.Expression:
    """Cap a ``TIMESTAMP_NTZ(n)`` type at microsecond precision so the year range is Snowflake's."""
    if not isinstance(node, exp.DataType) or node.this != exp.DataType.Type.TIMESTAMPNTZ:
        return node
    parameters = node.expressions or []
    if len(parameters) != 1:
        return node
    precision = parameters[0]
    if not isinstance(precision, exp.DataTypeParam) or not isinstance(
        precision.this, exp.Literal
    ):
        return node
    if precision.this.is_string:
        return node
    try:
        digits = int(precision.this.name)
    except ValueError:
        return node
    if digits <= _TIMESTAMP_MICROSECOND_PRECISION:
        return node
    precision.set(
        "this", exp.Literal.number(_TIMESTAMP_MICROSECOND_PRECISION)
    )
    return node


def _apply_transforms(
    statement: exp.Expression, resolver: ColumnTypeResolver | None = None
) -> exp.Expression:
    """Apply the Snowflake-to-DuckDB AST rewrites shared by statements and UDF bodies.

    A ``CREATE FUNCTION`` body is Snowflake SQL in its own right, so it has to go
    through exactly the same rewrites as a top-level statement -- that is what turns
    e.g. ``EQUAL_NULL(a, b)`` into DuckDB's ``a IS NOT DISTINCT FROM b``. `column_types`
    resolves a bare column's type for the array rewrites; a UDF body has no catalog in
    scope, so it is optional.
    """
    # A bare column's type decides several rewrites (an array operand's list form, a VARIANT
    # value's text coercion), so the resolver and the statement's own names are in scope for
    # the whole pipeline, not just the array pass.
    resolver_token = column_types.set_resolver(resolver)
    scoped_token = column_types.set_scoped_names(
        column_types.statement_scoped_names(statement)
    )
    alias_token = column_types.set_table_aliases(
        column_types.statement_table_aliases(statement)
    )
    try:
        statement = statement.transform(_rewrite_timestamp_precision)
        statement = statement.transform(_uppercase_unquoted_identifiers)
        statement = statement.transform(_rewrite_information_schema)
        statement = statement.transform(_rewrite_snowflake_specifics)
        statement = rewrite_semi_structured_writes(statement)
        statement = statement.transform(_rewrite_aggregate_functions)
        statement = _rewrite_regexp_functions(statement)
        # After the rewrites above: an array call is rebuilt from its operands' SQL, so
        # those operands must already be in their final DuckDB form.
        return _rewrite_json_array_functions(statement, resolver)
    finally:
        column_types.reset_table_aliases(alias_token)
        column_types.reset_scoped_names(scoped_token)
        column_types.reset_resolver(resolver_token)
