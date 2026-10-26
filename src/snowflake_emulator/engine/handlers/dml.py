"""Snowflake's result columns for DML statements.

DuckDB reports every DML statement as a single ``Count`` column holding the number of rows
it touched. Snowflake instead reports one column per action: ``number of rows inserted``,
``number of rows updated`` (plus ``number of multi-joined rows updated``), and ``number of
rows deleted``. A ``MERGE`` reports only the columns for the actions its ``WHEN`` clauses
name, so its per-action split has to be counted before DuckDB collapses it into one total.
"""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp

from snowflake_emulator.core.results import ColumnMeta

INSERT_COLUMN = "number of rows inserted"
UPDATE_COLUMN = "number of rows updated"
MULTI_JOINED_COLUMN = "number of multi-joined rows updated"
DELETE_COLUMN = "number of rows deleted"

_ACTION_COLUMNS = {
    "insert": INSERT_COLUMN,
    "update": UPDATE_COLUMN,
    "delete": DELETE_COLUMN,
}
_ACTION_ORDER = ("insert", "update", "delete")


def result_column(name: str) -> ColumnMeta:
    """Metadata for one of Snowflake's DML count columns."""
    return ColumnMeta(name=name, type="fixed", duckdb_type="BIGINT")


def parse_executed_statement(duckdb_sql: str) -> exp.Expression | None:
    """Parse the translated DuckDB statement the executor is about to run.

    ``None`` when the translation is not a single statement (``INSERT OVERWRITE`` expands
    into a ``DELETE`` and an ``INSERT``), in which case no pre-execution counting is done.
    """
    try:
        parsed = sqlglot.parse(duckdb_sql, read="duckdb")
    except sqlglot.errors.SqlglotError:
        return None
    if len(parsed) != 1:
        return None
    return parsed[0]


def _action_of(then: exp.Expression | None) -> str | None:
    """Classify the action of one ``WHEN ... THEN ...`` clause."""
    if isinstance(then, exp.Insert):
        return "insert"
    if isinstance(then, exp.Update):
        return "update"
    if isinstance(then, exp.Delete):
        return "delete"
    if isinstance(then, exp.Var) and then.name.upper() == "DELETE":
        return "delete"
    return None


def merge_actions(merge: exp.Merge) -> list[str]:
    """Return the action kinds a ``MERGE`` names, in Snowflake's column order."""
    whens = merge.args.get("whens")
    found: set[str] = set()
    for when in whens.expressions if isinstance(whens, exp.Whens) else []:
        action = _action_of(when.args.get("then"))
        if action is not None:
            found.add(action)
    return [action for action in _ACTION_ORDER if action in found]


def result_columns(
    statement: exp.Expression, executed: exp.Expression | None = None
) -> list[str] | None:
    """Return Snowflake's result column names for a DML statement, or ``None``."""
    if isinstance(statement, exp.Insert):
        return [INSERT_COLUMN]
    if isinstance(statement, exp.Update):
        return [UPDATE_COLUMN, MULTI_JOINED_COLUMN]
    if isinstance(statement, exp.Delete):
        return [DELETE_COLUMN]
    if isinstance(statement, exp.Merge):
        actions = merge_actions(statement)
        if not actions and isinstance(executed, exp.Merge):
            actions = merge_actions(executed)
        return [_ACTION_COLUMNS[action] for action in actions]
    return None


def result_values(statement: exp.Expression, preview: list[int] | None, count: int) -> list[int]:
    """Return the values for a DML statement's result columns.

    `preview` holds counts taken before a statement ran (``UPDATE ... FROM`` and a
    ``MERGE`` with more than one action); otherwise DuckDB's own affected-row count is the
    action count for a single-action statement.
    """
    columns = result_columns(statement) or []
    if preview is not None and len(preview) == len(columns):
        return preview
    if isinstance(statement, exp.Update):
        return [count, 0]
    # A single-action INSERT/DELETE/MERGE, or a shape whose preview could not be taken.
    return [count, *([0] * (len(columns) - 1))]


def preview_values(
    executed: exp.Expression | None,
    cursor: Any,
    bind_params: list[Any] | dict[str, Any] | None,
) -> list[int] | None:
    """Count the actions a DML statement is about to take, before it takes them.

    Only statements whose single DuckDB count cannot be split need this: an ``UPDATE``
    with a ``FROM`` clause (which also reports multi-joined rows) and a ``MERGE`` naming
    more than one action. Returns ``None`` for everything else.
    """
    if isinstance(executed, exp.Update) and _update_from(executed) is not None:
        return _update_preview(executed, cursor, bind_params)
    if isinstance(executed, exp.Merge) and len(merge_actions(executed)) > 1:
        return _merge_preview(executed, cursor, bind_params)
    return None


# ---------------------------------------------------------------------------
# Bind parameters
# ---------------------------------------------------------------------------
#
# The preview queries each use a *subset* of the statement's placeholders, while DuckDB
# requires exactly one value per placeholder. A named binding already carries its names;
# positional bindings are renamed here so a preview can bind only the values it uses.


def _retag_positions(
    statement: exp.Expression, bind_params: list[Any] | dict[str, Any] | None
) -> tuple[exp.Expression, dict[str, Any] | None]:
    """Copy `statement`, naming positional placeholders so their subsets can be bound."""
    if bind_params is None:
        return statement, None
    if isinstance(bind_params, dict):
        return statement, dict(bind_params)
    values: dict[str, Any] = {}
    index = 0

    def rename(node: exp.Expression) -> exp.Expression:
        nonlocal index
        if isinstance(node, exp.Placeholder):
            name = f"_sfb{index}"
            if index < len(bind_params):
                values[name] = bind_params[index]
            index += 1
            return exp.Placeholder(this=name)
        return node

    return statement.copy().transform(rename), values


def _placeholder_names(nodes: list[exp.Expression]) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        for placeholder in node.find_all(exp.Placeholder):
            name = placeholder.this
            if name:
                names.add(str(name))
    return names


def _params_for(
    nodes: list[exp.Expression], values: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The subset of `values` a preview query's placeholders need."""
    if values is None:
        return None
    names = _placeholder_names(nodes)
    return {name: values[name] for name in names if name in values}


def _execute(cursor: Any, sql: str, params: dict[str, Any] | None) -> Any:
    return cursor.execute(sql, params) if params else cursor.execute(sql)


def _affected(cursor: Any, sql: str, params: dict[str, Any] | None) -> int:
    row = _execute(cursor, sql, params).fetchone()
    return int(row[0] or 0) if row else 0


def _sql(node: exp.Expression) -> str:
    return node.sql(dialect="duckdb")


# ---------------------------------------------------------------------------
# UPDATE ... FROM
# ---------------------------------------------------------------------------


def _update_from(update: exp.Update) -> exp.Expression | None:
    """Return the ``FROM`` source of an ``UPDATE`` (as parsed from DuckDB SQL)."""
    from_node = update.args.get("from_") or update.args.get("from")
    if isinstance(from_node, exp.From):
        return from_node.this
    return from_node


def _table_reference(table: exp.Table) -> exp.Identifier:
    """The identifier a table is referenced by in its own statement (alias if present)."""
    alias = table.args.get("alias")
    if isinstance(alias, exp.TableAlias) and isinstance(alias.this, exp.Identifier):
        return alias.this.copy()
    name = table.this
    return name.copy() if isinstance(name, exp.Identifier) else exp.Identifier(this=table.name)


def _rowid_shadowed(cursor: Any, target_sql: str) -> bool:
    """True when the target has a real ``rowid`` column, hiding DuckDB's pseudocolumn."""
    cursor.execute(f"SELECT * FROM {target_sql} LIMIT 0")
    return any(col[0].lower() == "rowid" for col in cursor.description or [])


def _update_preview(
    update: exp.Update,
    cursor: Any,
    bind_params: list[Any] | dict[str, Any] | None,
) -> list[int] | None:
    """Count the rows an ``UPDATE ... FROM`` updates and how many joined more than once.

    Snowflake calls a target row that matches several ``FROM`` rows a *multi-joined row*:
    it is updated once, and reported once in each of its two count columns. DuckDB
    instead reports one update per source match, so both counts come from grouping the
    join by the target's row identity (``rowid``). A target that has its own ``rowid``
    column hides that identity; such an update falls back to DuckDB's count.
    """
    statement, values = _retag_positions(update, bind_params)
    assert isinstance(statement, exp.Update)
    target = statement.this
    from_node = _update_from(statement)
    where_node = statement.args.get("where")
    target_sql = _sql(target)
    if _rowid_shadowed(cursor, target_sql):
        return None
    from_sql = f", {_sql(from_node)}" if from_node is not None else ""
    where_sql = f" {_sql(where_node)}" if where_node is not None else ""
    row_id = exp.Column(this=exp.Identifier(this="rowid"), table=_table_reference(target))
    inner = (
        f"SELECT COUNT(*) AS n FROM {target_sql}{from_sql}{where_sql} "
        f"GROUP BY {_sql(row_id)}"
    )
    sql = (
        "SELECT COUNT(*) AS rows_updated, "
        "COALESCE(SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END), 0) AS multi_joined "
        f"FROM ({inner}) AS _sf_update"
    )
    nodes = [node for node in (from_node, where_node) if node is not None]
    row = _execute(cursor, sql, _params_for(nodes, values)).fetchone()
    if not row:
        return [0, 0]
    return [int(row[0] or 0), int(row[1] or 0)]


# ---------------------------------------------------------------------------
# MERGE
# ---------------------------------------------------------------------------


def _branch_class(when: exp.When) -> str:
    """Which rows a ``WHEN`` clause competes for; order only matters within a class."""
    if when.args.get("matched"):
        return "matched"
    if when.args.get("source"):
        return "not_matched_by_source"
    return "not_matched"


def _merge_preview(
    merge: exp.Merge,
    cursor: Any,
    bind_params: list[Any] | dict[str, Any] | None,
) -> list[int]:
    """Count each ``MERGE`` action before DuckDB replaces the split with one total."""
    statement, values = _retag_positions(merge, bind_params)
    assert isinstance(statement, exp.Merge)
    whens = statement.args.get("whens")
    clauses = list(whens.expressions) if isinstance(whens, exp.Whens) else []
    target = statement.this
    source = statement.args.get("using")
    on = statement.args.get("on")
    if not isinstance(target, exp.Table) or source is None:
        return []
    if on is None:
        on = exp.true()

    target_sql = _sql(target)
    source_sql = _sql(source)
    on_sql = _sql(on)
    base_nodes = [target, source, on]

    totals = {"insert": 0, "update": 0, "delete": 0}
    # Conditions of earlier clauses in each class, so a later clause only counts the rows
    # the first matching clause left for it (Snowflake applies only the first match).
    previous: dict[str, list[tuple[exp.Expression, str]]] = {
        "matched": [],
        "not_matched_by_source": [],
        "not_matched": [],
    }
    closed: set[str] = set()

    for when in clauses:
        assert isinstance(when, exp.When)
        action = _action_of(when.args.get("then"))
        branch = _branch_class(when)
        condition = when.args.get("condition")
        condition_sql = _sql(condition) if condition is not None else None

        count = 0
        if branch not in closed:
            conditions: list[tuple[exp.Expression | None, str]] = []
            if condition is not None and condition_sql is not None:
                conditions.append((condition, condition_sql))
            earlier = previous[branch]
            if earlier:
                exclusions = " OR ".join(
                    f"COALESCE(({sql}), FALSE)" for _, sql in earlier
                )
                conditions.append((None, f"NOT ({exclusions})"))
            count = _count_branch(
                cursor,
                branch,
                target_sql,
                source_sql,
                on_sql,
                conditions,
                [*base_nodes, *(node for node, _ in conditions if node is not None)],
                values,
            )

        if action is not None:
            totals[action] += count
        if condition is None:
            closed.add(branch)
        else:
            assert condition_sql is not None
            previous[branch].append((condition, condition_sql))

    return [totals[action] for action in merge_actions(statement)]


def _count_branch(
    cursor: Any,
    branch: str,
    target_sql: str,
    source_sql: str,
    on_sql: str,
    conditions: list[tuple[exp.Expression | None, str]],
    nodes: list[exp.Expression],
    values: dict[str, Any] | None,
) -> int:
    """Count the rows one ``WHEN`` clause acts on, given the state before the merge."""
    if branch == "matched":
        from_sql = f"{target_sql} JOIN {source_sql} ON {on_sql}"
    elif branch == "not_matched_by_source":
        missing = f"NOT EXISTS (SELECT 1 FROM {source_sql} WHERE {on_sql})"
        conditions = [(None, missing), *conditions]
        from_sql = target_sql
    else:
        missing = f"NOT EXISTS (SELECT 1 FROM {target_sql} WHERE {on_sql})"
        conditions = [(None, missing), *conditions]
        from_sql = source_sql
    where_sql = (
        " WHERE " + " AND ".join(f"({sql})" for _, sql in conditions)
        if conditions
        else ""
    )
    params = _params_for(nodes, values)
    return _affected(cursor, f"SELECT COUNT(*) FROM {from_sql}{where_sql}", params)
