"""The bare-column type lookup shared by the rewrite passes.

A bare column carries no type in the AST, and several rewrites need one: the array functions
must tell a real DuckDB LIST from a VARIANT/JSON array, and Snowflake's text coercion must
recognise a VARIANT column so it can render JSON text instead of DuckDB's VARIANT repr. The
resolver is supplied by the caller that has a catalog (see ``snowflake_emulator.catalog``) and
published for the duration of a translation pass, because the passes are otherwise pure
transforms.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Callable

from sqlglot import exp

# ``(catalog_or_None, schema_or_None, table_or_None, column) -> DuckDB type name``.
ColumnTypeResolver = Callable[[str | None, str | None, str | None, str], str | None]

_resolver: ContextVar[ColumnTypeResolver | None] = ContextVar(
    "column_type_resolver", default=None
)

# Column and table names the *statement itself* defines (select aliases, CTE names, derived
# table aliases). A bare column with one of these names belongs to the query, not to a catalog
# table that happens to share it, so it must not inherit that table's type.
_scoped: ContextVar[frozenset[str]] = ContextVar("scoped_names", default=frozenset())

# Table alias -> real table name, so a column qualified by an alias (``s.$1:id`` is a stage,
# but ``t.name`` may be ``stage_t AS t``) still resolves. A derived-table alias has no real
# table name and is absent, which leaves it unresolved -- the safe direction.
_aliases: ContextVar[dict[str, str]] = ContextVar("table_aliases", default={})


def current_resolver() -> ColumnTypeResolver | None:
    """The resolver in scope, for a pass that re-enters itself on an operand."""
    return _resolver.get()


def set_resolver(resolver: ColumnTypeResolver | None) -> Token:
    """Publish `resolver` for the current translation pass."""
    return _resolver.set(resolver)


def reset_resolver(token: Token) -> None:
    """Restore the resolver that was in scope before `set_resolver`."""
    _resolver.reset(token)


def set_scoped_names(names: frozenset[str]) -> Token:
    """Publish the names the statement defines itself."""
    return _scoped.set(names)


def reset_scoped_names(token: Token) -> None:
    """Restore the scoped-name set that was in scope before `set_scoped_names`."""
    _scoped.reset(token)


def statement_scoped_names(statement: exp.Expression) -> frozenset[str]:
    """The upper-cased names `statement` defines as *values*: column aliases and CTE names.

    A table alias is not one of these: it names a real table and is translated by
    ``statement_table_aliases`` rather than being excluded from resolution.
    """
    names = {
        name.upper()
        for name in (
            *[alias.alias for alias in statement.find_all(exp.Alias) if alias.alias],
            *[cte.alias for cte in statement.find_all(exp.CTE) if cte.alias],
            # A derived table's alias has no catalog meaning: if it shares a name with a real
            # table, resolving to that table's column type is wrong (and reaches DuckDB with a
            # type that does not match the value, e.g. LIST_SORT over a VARIANT -> OOM).
            *[
                subquery.alias
                for subquery in statement.find_all(exp.Subquery)
                if subquery.alias
            ],
        )
        if name
    }
    return frozenset(names)


def statement_table_aliases(statement: exp.Expression) -> dict[str, str]:
    """The upper-cased ``alias -> table`` map for the tables the statement names."""
    return {
        table.alias.upper(): table.name.upper()
        for table in statement.find_all(exp.Table)
        if table.alias and table.name
    }


def set_table_aliases(aliases: dict[str, str]) -> Token:
    """Publish the statement's table aliases for the current translation pass."""
    return _aliases.set(aliases)


def reset_table_aliases(token: Token) -> None:
    """Restore the alias map that was in scope before `set_table_aliases`."""
    _aliases.reset(token)


def declared_column_type(node: exp.Expression | None) -> str | None:
    """The DuckDB type of a bare column, when the caller supplied a catalog lookup.

    A name the statement defines itself is not looked up: an unrelated table with the same
    column name would otherwise decide how the column is rewritten.
    """
    resolver = _resolver.get()
    if resolver is None or not isinstance(node, exp.Column):
        return None
    scoped = _scoped.get()
    if node.name.upper() in scoped:
        return None
    table = node.table or None
    if table:
        if table.upper() in scoped:
            # A derived-table alias: it names a query result, not a catalog table.
            return None
        # A table alias names a real table; a stage reference has no entry and stays unresolved.
        table = _aliases.get().get(table.upper(), table)
    return resolver(node.catalog or None, node.db or None, table, node.name)


def is_variant_typed(node: exp.Expression | None) -> bool:
    """True when `node` evaluates to a VARIANT value: a VARIANT column, cast or array."""
    if node is None:
        return False
    if isinstance(node, exp.ToVariant):
        return True
    if (
        isinstance(node, exp.Cast)
        and isinstance(node.to, exp.DataType)
        and node.to.this == exp.DataType.Type.VARIANT
    ):
        return True
    declared = declared_column_type(node)
    return declared is not None and declared.strip().upper().startswith("VARIANT")
