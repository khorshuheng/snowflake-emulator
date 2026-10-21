"""The AST rewrite pipeline shared by statements and UDF bodies."""

from __future__ import annotations

from sqlglot import exp

from snowflake_emulator.translator import column_types
from snowflake_emulator.translator.aggregates import _rewrite_aggregate_functions
from snowflake_emulator.translator.arrays import (
    ColumnTypeResolver,
    _is_json_value,
    _is_variant_value,
    _rewrite_json_array_functions,
)
from snowflake_emulator.translator.errors import TranslationError
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


def _derived_column_types(statement: exp.Expression) -> dict[tuple[str, str], str]:
    """The output column types of the statement's derived tables and CTEs.

    A derived table has no catalog entry, so a column read from it would otherwise be
    untyped: a staged ``$1:name AS name`` projection then compared as a plain name, and
    DuckDB parsed the VARCHAR side as JSON (``Malformed JSON``). The projection itself says
    what the column holds, so record the semi-structured ones (and any catalog type a
    projected bare column carries).
    """
    types: dict[tuple[str, str], str] = {}

    def record(alias: str, select: exp.Expression) -> None:
        if not alias or not isinstance(select, exp.Select):
            return
        for item in select.expressions:
            value = item.this if isinstance(item, exp.Alias) else item
            name = item.alias_or_name
            if not name or name == "*":
                continue
            if _is_json_value(value):
                declared = "JSON"
            elif _is_variant_value(value):
                declared = "VARIANT"
            else:
                declared = None
            declared = declared or column_types.declared_column_type(value)
            if declared:
                types[(alias.upper(), name.upper())] = declared

    for subquery in statement.find_all(exp.Subquery):
        record(subquery.alias, subquery.this)
    for cte in statement.find_all(exp.CTE):
        record(cte.alias, cte.this)
    return types


def _select_sources(select: exp.Select) -> set[str]:
    """The table names/aliases a SELECT's own FROM and JOIN entries introduce."""
    names: set[str] = set()

    def add(node: exp.Expression | None) -> None:
        if isinstance(node, exp.Table):
            names.add((node.alias or node.name).upper())
        elif isinstance(node, exp.Subquery) and node.alias:
            names.add(node.alias.upper())
        elif isinstance(node, exp.Lateral):
            if node.alias:
                names.add(node.alias.upper())
            else:
                add(node.this)

    from_ = select.args.get("from_")
    if isinstance(from_, exp.From):
        add(from_.this)
    for join in select.args.get("joins") or []:
        if isinstance(join, exp.Join):
            add(join.this)
    return names


def _outer_sources(select: exp.Select) -> set[str]:
    """Names introduced by SELECTs enclosing `select` (the scopes it can correlate to)."""
    names: set[str] = set()
    node = select.parent
    while node is not None:
        if isinstance(node, exp.Select):
            names.update(_select_sources(node))
        node = node.parent
    return names


def _reject_correlated_limit(statement: exp.Expression) -> None:
    """Reject LIMIT/FETCH in a correlated subquery, as Snowflake does.

    Snowflake allows LIMIT/FETCH only in an *uncorrelated scalar* subquery. DuckDB accepts
    the correlated form (often returning rows), so leaving it through would silently report
    results Snowflake rejects. A subquery is treated as correlated when it references a table
    name or alias introduced by an enclosing SELECT.
    """
    for select in statement.find_all(exp.Select):
        if select.args.get("limit") is None:
            continue
        local = _select_sources(select)
        outer = _outer_sources(select)
        if not outer:
            continue
        for column in select.find_all(exp.Column):
            qualifier = column.table.upper()
            if qualifier and qualifier in outer and qualifier not in local:
                raise TranslationError(
                    "LIMIT / FETCH is not supported in a correlated subquery; Snowflake "
                    "allows it only in an uncorrelated scalar subquery."
                )


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
    derived_token = column_types.set_derived_column_types(
        _derived_column_types(statement)
    )
    try:
        _reject_correlated_limit(statement)
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
        column_types.reset_derived_column_types(derived_token)
        column_types.reset_table_aliases(alias_token)
        column_types.reset_scoped_names(scoped_token)
        column_types.reset_resolver(resolver_token)
