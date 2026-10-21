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


def _derived_type_resolver(statement: exp.Expression):
    """Build a scope-aware lookup for derived-table/CTE projection types.

    A derived relation has no catalog entry, but its projection says what a column holds. The
    lookup walks the column's enclosing SELECTs from the innermost outwards and inspects that
    scope's FROM/JOIN entries, the way SQL name resolution does. Keying by alias alone is not
    enough: the same alias may be reused by unrelated subqueries, and a JSON projection in one
    must not decide the type of a text column in another.

    Projections are resolved lazily and recursively, so a derived table that selects from
    another derived table keeps the type (and a UNION uses the type of its arms).
    """
    ctes = {
        cte.alias.upper(): cte.this
        for cte in statement.find_all(exp.CTE)
        if cte.alias
    }

    def enclosing_scopes(node: exp.Expression):
        node = node.parent
        while node is not None:
            if isinstance(node, (exp.Select, exp.Update)):
                yield node
            node = node.parent

    def relation_sources(
        scope: exp.Select | exp.Update,
    ) -> list[tuple[str, exp.Expression]]:
        """The derived relations in `scope`'s own FROM/JOIN, as (alias, projection)."""
        sources: list[tuple[str, exp.Expression]] = []

        def add(node: exp.Expression | None) -> None:
            if isinstance(node, exp.Subquery):
                sources.append(((node.alias or "").upper(), node.this))
            elif isinstance(node, exp.Table):
                cte = ctes.get(node.name.upper())
                if cte is not None:
                    sources.append(((node.alias or node.name).upper(), cte))
            elif isinstance(node, exp.Lateral):
                if isinstance(node.this, exp.Subquery):
                    sources.append(((node.alias or "").upper(), node.this.this))
                elif isinstance(node.this, exp.Table):
                    cte = ctes.get(node.this.name.upper())
                    if cte is not None:
                        sources.append(((node.alias or node.this.name).upper(), cte))

        from_ = scope.args.get("from_")
        if isinstance(from_, exp.From):
            add(from_.this)
        for join in scope.args.get("joins") or []:
            if isinstance(join, exp.Join):
                add(join.this)
        return sources

    def select_arms(relation: exp.Expression) -> list[exp.Select]:
        """The SELECTs whose columns a derived relation exposes (both sides of a UNION)."""
        if isinstance(relation, exp.Select):
            return [relation]
        if isinstance(relation, exp.Union):
            arms: list[exp.Select] = []
            pending = [relation]
            while pending:
                node = pending.pop()
                if isinstance(node, exp.Union):
                    pending.extend((node.this, node.expression))
                elif isinstance(node, exp.Select):
                    arms.append(node)
            return arms
        return []

    def merge_types(types: list[str | None]) -> str | None:
        """One type for a UNION column; any semi-structured arm wins."""
        known = [declared for declared in types if declared]
        semi = [
            declared
            for declared in known
            if declared.strip().upper().startswith(("JSON", "VARIANT"))
        ]
        if semi:
            return semi[0]
        if known and all(item.upper() == known[0].upper() for item in known):
            return known[0]
        return None

    def expression_type(value: exp.Expression, seen: frozenset[tuple[int, str]]) -> str | None:
        if _is_json_value(value):
            return "JSON"
        if _is_variant_value(value):
            return "VARIANT"
        if isinstance(value, exp.Column):
            handled, declared = resolve_column(value, seen)
            if handled:
                return declared
            # Not a derived relation: the catalog can type a real table's column.
            return column_types.catalog_column_type(value)
        return None

    def projection_lookup(
        relation: exp.Expression, name: str, seen: frozenset[tuple[int, str]]
    ) -> tuple[bool, str | None]:
        """Whether `relation` exposes `name`, and its type when known."""
        found = False
        types: list[str | None] = []
        for select in select_arms(relation):
            key = (id(select), name.upper())
            if key in seen:
                continue
            for item in select.expressions:
                item_name = item.alias_or_name
                if not item_name or item_name.upper() != name.upper():
                    continue
                found = True
                value = item.this if isinstance(item, exp.Alias) else item
                types.append(expression_type(value, seen | {key}))
                break
        return found, merge_types(types) if found else None

    def resolve_relation(node: exp.Column) -> exp.Expression | None:
        """The derived relation `node` reads from in its own lexical scope, if any."""
        table = node.table or None
        if table:
            wanted = table.upper()
            for scope in enclosing_scopes(node):
                for alias, relation in relation_sources(scope):
                    if alias == wanted:
                        return relation
            return None
        scopes = list(enclosing_scopes(node))
        if not scopes:
            return None
        # An unqualified name is only attributed to the raw staged relation, whose positional
        # ``$1`` column has no table qualifier. Other unqualified names stay conservative:
        # resolving them against a derived projection changed array-function behavior that is
        # deliberately left to the catalog/unknown path.
        if not (node.name.startswith("$") and node.name[1:].isdigit()):
            return None
        scope = scopes[0]
        if scope.args.get("joins"):
            return None
        sources = relation_sources(scope)
        if len(sources) == 1:
            return sources[0][1]
        return None

    def resolve_column(
        node: exp.Column, seen: frozenset[tuple[int, str]] = frozenset()
    ) -> tuple[bool, str | None]:
        """Resolve `node` against derived relations in its own lexical scopes."""
        relation = resolve_relation(node)
        if relation is None:
            return False, None
        return projection_lookup(relation, node.name, seen)

    return resolve_column, resolve_relation


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


def _relation_projection_names(relation: exp.Expression) -> set[str] | None:
    """The column names a derived relation exposes, or None when it cannot be enumerated."""
    if isinstance(relation, exp.Select):
        arms = [relation]
    elif isinstance(relation, exp.Union):
        arms = []
        pending: list[exp.Expression] = [relation]
        while pending:
            node = pending.pop()
            if isinstance(node, exp.Union):
                pending.extend((node.this, node.expression))
            elif isinstance(node, exp.Select):
                arms.append(node)
    else:
        return None
    if not arms:
        return None
    names = {item.alias_or_name.upper() for item in arms[0].expressions if item.alias_or_name}
    if not names or any(
        not item.alias_or_name or item.alias_or_name == "*" for item in arms[0].expressions
    ):
        return None
    return names


def _source_column_names(
    select: exp.Select, ctes: dict[str, exp.Expression]
) -> tuple[set[str], bool]:
    """Column names supplied by a SELECT's own sources, and whether all are known."""
    names: set[str] = set()
    known = True

    def add(node: exp.Expression | None) -> None:
        nonlocal known
        if isinstance(node, exp.Subquery):
            projected = _relation_projection_names(node.this)
            if projected is None:
                known = False
            else:
                names.update(projected)
        elif isinstance(node, exp.Table):
            cte = ctes.get(node.name.upper())
            if cte is not None:
                projected = _relation_projection_names(cte)
                if projected is None:
                    known = False
                else:
                    names.update(projected)
            else:
                columns = column_types.declared_table_columns(node)
                if columns is None:
                    known = False
                else:
                    names.update(column.upper() for column in columns)
        elif isinstance(node, exp.Lateral):
            add(node.this)
        else:
            known = False

    from_ = select.args.get("from_")
    if isinstance(from_, exp.From):
        add(from_.this)
    for join in select.args.get("joins") or []:
        if isinstance(join, exp.Join):
            add(join.this)
    return names, known


def _direct_columns(select: exp.Select):
    """Columns whose nearest enclosing SELECT is `select` (not a nested subquery)."""
    for column in select.find_all(exp.Column):
        node = column.parent
        while node is not None and not isinstance(node, exp.Select):
            node = node.parent
        if node is select:
            yield column


def _reject_correlated_limit(statement: exp.Expression) -> None:
    """Reject LIMIT/FETCH in a correlated subquery, as Snowflake does.

    Snowflake allows LIMIT/FETCH only in an *uncorrelated scalar* subquery. DuckDB accepts
    the correlated form (often returning rows), so leaving it through would silently report
    results Snowflake rejects. A subquery is correlated when it references a table alias or
    column name introduced by an enclosing SELECT; an unqualified name is checked against the
    source column sets, and unknown metadata keeps the check conservative.
    """
    ctes = {
        cte.alias.upper(): cte.this
        for cte in statement.find_all(exp.CTE)
        if cte.alias
    }
    for select in statement.find_all(exp.Select):
        if select.args.get("limit") is None:
            continue
        local_aliases = _select_sources(select)
        local_names, local_known = _source_column_names(select, ctes)
        outer_aliases: set[str] = set()
        outer_names: set[str] = set()
        outer_known = True
        node = select.parent
        while node is not None:
            if isinstance(node, exp.Select):
                outer_aliases.update(_select_sources(node))
                names, known = _source_column_names(node, ctes)
                outer_names.update(names)
                outer_known = outer_known and known
            node = node.parent
        if not outer_aliases and not outer_names:
            continue
        for column in _direct_columns(select):
            qualifier = column.table.upper()
            if qualifier:
                correlated = qualifier in outer_aliases and qualifier not in local_aliases
            else:
                correlated = (
                    local_known
                    and outer_known
                    and column.name.upper() not in local_names
                    and column.name.upper() in outer_names
                )
            if correlated:
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
    derived_resolver, relation_resolver = _derived_type_resolver(statement)
    derived_token = column_types.set_derived_resolver(derived_resolver)
    relation_token = column_types.set_derived_relation_resolver(relation_resolver)
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
        column_types.reset_derived_relation_resolver(relation_token)
        column_types.reset_derived_resolver(derived_token)
        column_types.reset_table_aliases(alias_token)
        column_types.reset_scoped_names(scoped_token)
        column_types.reset_resolver(resolver_token)
