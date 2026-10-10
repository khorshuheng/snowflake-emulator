"""Execute (translated) Snowflake SQL against the shared DuckDB backend."""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlglot import exp

from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.errors import classify_exception
from snowflake_emulator.file_formats import (
    FileFormatError,
    get_file_format_manager,
    resolve_file_format,
)
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.statement_types import (
    number_of_binds,
    statement_type_id,
)
from snowflake_emulator.stages import (
    StageError,
    build_copy_sql,
    copy_stage_source,
    create_stage,
    drop_stage,
    get_stage_manager,
    match_stage_command,
    match_stage_ddl,
    parse_get,
    parse_list,
    parse_put,
    parse_remove,
    show_stages,
)
from snowflake_emulator.translator import (
    TranslationError,
    UseStatement,
    is_alter_session,
    is_transaction_control,
    match_use_statement,
    split_statements,
    transpile_to_duckdb,
)
from snowflake_emulator.type_mapping import (
    duckdb_type_to_snowflake,
    duckdb_type_to_snowflake_sql_type,
)


class ExecutionError(RuntimeError):
    """Raised when a translated statement fails to execute against DuckDB.

    Carries the Snowflake ``errno``/``sqlstate`` pair so routers can answer with the
    same error classification Snowflake clients branch on. Defaults to the generic
    compilation-error pair for failures the emulator raises itself (stages, duplicate
    databases).
    """

    def __init__(
        self,
        message: str,
        *,
        errno: str = "100132",
        sqlstate: str = "42000",
    ) -> None:
        super().__init__(message)
        self.errno = errno
        self.sqlstate = sqlstate


def _as_execution_error(exc: BaseException) -> ExecutionError:
    """Wrap a failure as an ``ExecutionError`` with a Snowflake error classification."""
    if isinstance(exc, TranslationError):
        # sqlglot could not parse or transpile the statement: a SQL compilation error.
        return ExecutionError(str(exc), errno="001003", sqlstate="42000")
    errno, sqlstate = classify_exception(exc)
    return ExecutionError(str(exc), errno=errno, sqlstate=sqlstate)


@dataclass
class ColumnMeta:
    name: str
    type: str
    duckdb_type: str = ""


@dataclass
class FileTransfer:
    """Wire-protocol description of a connector-side PUT/GET file transfer."""

    kind: str  # "UPLOAD" | "DOWNLOAD"
    src_locations: list[str]
    stage_location: str
    local_location: str | None = None
    overwrite: bool = False


@dataclass
class ExecutionResult:
    handle: str
    created_on: int
    row_type: list[ColumnMeta]
    rows: list[list[Any]]
    message: str = "Statement executed successfully."
    file_transfer: FileTransfer | None = None
    statement_type_id: int = 0
    number_of_binds: int = 0


def execute_sql(
    manager: DuckDBManager,
    session: SessionContext,
    raw_sql: str,
    bind_params: list[Any] | dict[str, Any] | None = None,
    bind_batch: list[tuple[Any, ...]] | None = None,
) -> ExecutionResult:
    """Execute one or more (semicolon-separated) Snowflake statements.

    Session-scoped ``USE ...`` statements update `session` in place instead of being
    sent to DuckDB. Only the result of the final statement is returned, matching
    Snowflake's own multi-statement behaviour. `bind_params` (a positional list or a
    named mapping) and `bind_batch` (one tuple per row, for an array binding) are only
    applied when `raw_sql` contains a single statement.
    """
    manager.ensure_namespace(session.database, session.schema)
    cursor = manager.cursor()
    cursor.execute(f'USE "{session.database}"."{session.schema}"')

    stage_command = match_stage_command(raw_sql)
    if stage_command is not None:
        return _execute_stage_command(manager, session, raw_sql, stage_command)

    try:
        statements = split_statements(raw_sql)
    except TranslationError as exc:
        raise _as_execution_error(exc) from exc

    row_type: list[ColumnMeta] = []
    rows: list[list[Any]] = []
    message = "Statement executed successfully."
    last_statement: exp.Expression | None = None

    for statement in statements:
        last_statement = statement

        scan_id = _result_scan_query_id(statement, session)
        if scan_id is not None:
            cached = _lookup_cached_result(scan_id)
            if cached is None:
                raise ExecutionError(
                    f"Statement {scan_id} not found", errno="0709", sqlstate="02000"
                )
            row_type = list(cached.row_type)
            rows = [list(row) for row in cached.rows]
            message = "Statement executed successfully."
            continue

        # LAST_QUERY_ID() outside RESULT_SCAN resolves to the previous handle, or fails
        # clearly when the session has not run a query yet.
        statement = statement.transform(lambda node: _resolve_last_query_id(node, session))

        set_result = _execute_set_variable(statement, session)
        if set_result is not None:
            row_type, rows = set_result
            message = "Statement executed successfully."
            continue

        show_variables = _execute_show_variables(statement, session)
        if show_variables is not None:
            row_type, rows = show_variables
            message = "Statement executed successfully."
            continue

        statement = _substitute_variables(statement, session)

        stmt_sql = statement.sql(dialect="snowflake")

        use_stmt = match_use_statement(stmt_sql)
        if use_stmt is not None:
            _apply_use_statement(manager, session, cursor, use_stmt)
            row_type, rows = [], []
            message = "Statement executed successfully."
            continue

        if is_alter_session(stmt_sql) or is_transaction_control(statement):
            # ALTER SESSION SET/UNSET <param> and BEGIN/COMMIT/ROLLBACK have no
            # meaning under the emulator's autocommit execution model; accept them
            # as no-ops (schemachange sets QUERY_TAG and commits around every script).
            row_type, rows = [], []
            message = "Statement executed successfully."
            continue

        stage_ddl = match_stage_ddl(statement)
        if stage_ddl is not None:
            row_type, rows = _execute_stage_ddl(manager, session, statement, stage_ddl)
            continue

        file_format_ddl = _execute_file_format_ddl(session, statement)
        if file_format_ddl is not None:
            row_type, rows = file_format_ddl
            continue

        create_db = _execute_create_database(manager, statement)
        if create_db is not None:
            row_type, rows = create_db
            continue

        show_result = _execute_snowflake_show(cursor, session, statement)
        if show_result is not None:
            row_type, rows = show_result
            continue

        show_columns = _execute_show_columns(cursor, session, statement)
        if show_columns is not None:
            row_type, rows = show_columns
            continue

        show_keys = _execute_show_keys(cursor, session, statement)
        if show_keys is not None:
            row_type, rows = show_keys
            continue

        show_misc = _execute_show_misc(cursor, session, statement)
        if show_misc is not None:
            row_type, rows = show_misc
            continue

        describe_result = _execute_snowflake_describe(cursor, session, statement)
        if describe_result is not None:
            row_type, rows = describe_result
            continue

        try:
            duckdb_sql = transpile_to_duckdb(statement)
        except TranslationError as exc:
            raise _as_execution_error(exc) from exc

        stage_copy = None
        if isinstance(statement, exp.Copy):
            stage_copy = copy_stage_source(statement)

        batch = bind_batch if len(statements) == 1 else None
        batch_affected = 0
        try:
            if stage_copy is not None:
                table_sql, stage_ref, file_format = stage_copy
                file_format = resolve_file_format(session, file_format)
                duckdb_sql = build_copy_sql(session, table_sql, stage_ref, file_format)
            if batch is not None:
                # An array binding (executemany) runs the statement once per row.
                for row in batch:
                    cursor.execute(duckdb_sql, list(row))
                    if cursor.description:
                        fetched = cursor.fetchone()
                        if fetched and isinstance(fetched[0], int):
                            batch_affected += fetched[0]
            elif bind_params is not None and len(statements) == 1:
                cursor.execute(duckdb_sql, bind_params)
            else:
                cursor.execute(duckdb_sql)
        except (StageError, FileFormatError) as exc:
            raise ExecutionError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed execution error
            raise _as_execution_error(exc) from exc

        if batch is not None:
            # Report the total affected rows across the batch, as Snowflake does for a
            # multi-row DML, instead of the last row's count.
            row_type = [ColumnMeta(name="Count", type="fixed", duckdb_type="BIGINT")]
            rows = [[batch_affected]]
        elif cursor.description:
            description = cursor.description
            if _is_ddl_status_artifact(statement, description):
                # DuckDB DDL (CREATE/DROP/ALTER) reports a bare Count/Success column;
                # Snowflake returns a plain status, so report it that way too.
                row_type = [ColumnMeta(name="status", type="text")]
                rows = [["Statement executed successfully."]]
            else:
                row_type = [
                    ColumnMeta(
                        name=col[0],
                        type=duckdb_type_to_snowflake(str(col[1])),
                        duckdb_type=str(col[1]),
                    )
                    for col in description
                ]
                rows = [list(row) for row in cursor.fetchall()]
                if isinstance(statement, exp.Describe):
                    rows = _convert_describe_types(rows)
        else:
            row_type = [ColumnMeta(name="status", type="text")]
            affected = cursor.fetchone()
            count = affected[0] if affected else 0
            rows = [[f"Statement executed successfully, {count} rows affected."]]

    return ExecutionResult(
        handle=str(uuid.uuid4()),
        created_on=int(time.time() * 1000),
        row_type=row_type,
        rows=rows,
        message=message,
        statement_type_id=statement_type_id(last_statement),
        number_of_binds=number_of_binds(last_statement),
    )


# Columns Snowflake reports when asked to describe a statement that produces no result
# set, without running it. Only the names and types matter.
_DESCRIBE_STATUS = [("status", "VARCHAR")]
_DESCRIBE_COPY = [
    ("file", "VARCHAR"),
    ("status", "VARCHAR"),
    ("rows_parsed", "BIGINT"),
    ("rows_loaded", "BIGINT"),
    ("error_limit", "BIGINT"),
    ("errors_seen", "BIGINT"),
    ("first_error", "VARCHAR"),
    ("first_error_line", "BIGINT"),
    ("first_error_character", "BIGINT"),
    ("first_error_column_name", "VARCHAR"),
]
_DESCRIBE_SYNTHETIC: dict[type[exp.Expression], list[tuple[str, str]]] = {
    exp.Create: _DESCRIBE_STATUS,
    exp.Drop: _DESCRIBE_STATUS,
    exp.Alter: _DESCRIBE_STATUS,
    exp.TruncateTable: _DESCRIBE_STATUS,
    # Session/transaction control must not run during a describe; they are
    # read-only to observe and would otherwise mutate the session.
    exp.Use: _DESCRIBE_STATUS,
    exp.Transaction: _DESCRIBE_STATUS,
    exp.Commit: _DESCRIBE_STATUS,
    exp.Rollback: _DESCRIBE_STATUS,
    exp.Insert: [("number of rows inserted", "BIGINT")],
    exp.Update: [
        ("number of rows updated", "BIGINT"),
        ("number of multi-joined rows updated", "BIGINT"),
    ],
    exp.Delete: [("number of rows deleted", "BIGINT")],
    # MERGE reports one count per operation; both are listed (a MERGE that omits a
    # WHEN clause would report fewer, but describing must never execute the write).
    exp.Merge: [
        ("number of rows inserted", "BIGINT"),
        ("number of rows updated", "BIGINT"),
    ],
    exp.Copy: _DESCRIBE_COPY,
}


def _describe_result(
    statement: exp.Expression, columns: list[tuple[str, str]]
) -> ExecutionResult:
    """Build a metadata-only result for a statement described without executing."""
    return ExecutionResult(
        handle=str(uuid.uuid4()),
        created_on=int(time.time() * 1000),
        row_type=[
            ColumnMeta(
                name=name,
                type=duckdb_type_to_snowflake(duckdb_type),
                duckdb_type=duckdb_type,
            )
            for name, duckdb_type in columns
        ],
        rows=[],
        message="Describe statement executed successfully.",
        statement_type_id=statement_type_id(statement),
        number_of_binds=number_of_binds(statement),
    )


def describe_sql(
    manager: DuckDBManager,
    session: SessionContext,
    raw_sql: str,
) -> ExecutionResult | None:
    """Describe a statement's result columns without executing it.

    Snowflake answers a ``describeOnly`` request with result metadata and no rows, so
    a client can prepare a statement without running it. DML/DDL get Snowflake's
    synthetic result shape; a SELECT is described directly with DuckDB ``DESCRIBE``
    (placeholders replaced with NULL so it stays valid without bindings). Anything
    else returns ``None`` and the caller falls back to executing it.
    """
    manager.ensure_namespace(session.database, session.schema)
    try:
        statement = split_statements(raw_sql)[-1]
    except TranslationError:
        return None
    if type(statement) in _DESCRIBE_SYNTHETIC:
        return _describe_result(statement, _DESCRIBE_SYNTHETIC[type(statement)])
    if not isinstance(statement, (exp.Select, exp.Union, exp.Subquery)):
        return None
    binds = number_of_binds(statement)
    statement = statement.transform(
        lambda node: exp.Null()
        if isinstance(node, (exp.Placeholder, exp.Parameter))
        else node
    )
    cursor = manager.cursor()
    cursor.execute(f'USE "{session.database}"."{session.schema}"')
    try:
        duckdb_sql = transpile_to_duckdb(statement)
        cursor.execute(f"DESCRIBE {duckdb_sql}")
    except Exception:  # noqa: BLE001 - only described by running; fall back
        return None
    if not cursor.description:
        return None
    described = cursor.fetchall()
    return ExecutionResult(
        handle=str(uuid.uuid4()),
        created_on=int(time.time() * 1000),
        row_type=[
            ColumnMeta(
                name=row[0],
                type=duckdb_type_to_snowflake(str(row[1])),
                duckdb_type=str(row[1]),
            )
            for row in described
        ],
        rows=[],
        message="Describe statement executed successfully.",
        statement_type_id=statement_type_id(statement),
        number_of_binds=binds,
    )


def _lookup_cached_result(handle: str) -> ExecutionResult | None:
    # Imported lazily: statement_store imports ExecutionResult from this module.
    from snowflake_emulator.statement_store import get_statement_store

    return get_statement_store().get(handle)


def _result_scan_query_id(
    statement: exp.Expression, session: SessionContext
) -> str | None:
    """Return the query id a ``SELECT * FROM TABLE(RESULT_SCAN(...))`` refers to.

    Returns ``None`` when the statement is not a RESULT_SCAN select. Mirrors Snowflake's
    simple form: the argument is a query-id literal or ``LAST_QUERY_ID()``.
    """
    if not isinstance(statement, exp.Select):
        return None
    inner = getattr(statement.args.get("from_"), "this", None)
    if not isinstance(inner, exp.TableFromRows):
        return None
    call = inner.this
    if not (
        isinstance(call, exp.Anonymous)
        and call.name.upper() == "RESULT_SCAN"
        and call.expressions
    ):
        return None
    projections = statement.expressions
    if not (len(projections) == 1 and isinstance(projections[0], exp.Star)):
        raise ExecutionError("RESULT_SCAN is only supported with SELECT * by the emulator.")
    arg = call.expressions[0]
    if isinstance(arg, exp.Literal) and arg.is_string:
        return arg.this
    if isinstance(arg, exp.Anonymous) and arg.name.upper() == "LAST_QUERY_ID":
        if session.last_query_id is None:
            raise ExecutionError(
                "There is no previous query in this session.", errno="0709", sqlstate="02000"
            )
        return session.last_query_id
    raise ExecutionError("RESULT_SCAN requires a query id literal or LAST_QUERY_ID().")


def _resolve_last_query_id(
    node: exp.Expression, session: SessionContext
) -> exp.Expression:
    """Replace a ``LAST_QUERY_ID()`` call with the previous statement's handle literal."""
    if isinstance(node, exp.Anonymous) and node.name.upper() == "LAST_QUERY_ID":
        if session.last_query_id is None:
            raise ExecutionError(
                "There is no previous query in this session.", errno="0709", sqlstate="02000"
            )
        return exp.Literal.string(session.last_query_id)
    return node


def _variable_value(node: exp.Expression | None) -> object:
    """Convert a ``SET`` value expression to a stored value.

    Scalar literals become Python values; any other expression is kept as an AST node
    so ``$name`` expands to the expression (``SET x = 1 + 1`` then ``$x`` is ``2``),
    not to its quoted source text.
    """
    if node is None:
        return None
    if isinstance(node, exp.Literal):
        if node.is_string:
            return str(node.this)
        text = str(node.this)
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Null):
        return None
    return node


def _variable_literal(value: object) -> exp.Expression:
    """Build the SQL expression a ``$name`` reference expands to."""
    if isinstance(value, exp.Expression):
        return value.copy()
    if value is None:
        return exp.Null()
    if isinstance(value, bool):
        return exp.Boolean(this=value)
    if isinstance(value, (int, float)):
        return exp.Literal.number(value)
    return exp.Literal.string(str(value))


def _variable_display(value: object) -> str:
    """Render a session variable for ``SHOW VARIABLES``."""
    if isinstance(value, exp.Expression):
        return value.sql(dialect="snowflake")
    return str(value)


def _execute_set_variable(
    statement: exp.Expression, session: SessionContext
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle Snowflake ``SET name = value`` by storing a session variable."""
    if not isinstance(statement, exp.Set):
        return None
    for item in statement.expressions:
        assignment = item.this if isinstance(item, exp.SetItem) else item
        if not isinstance(assignment, exp.EQ):
            raise ExecutionError("Unsupported SET form (expected `name = value`).")
        name = assignment.this.name if isinstance(assignment.this, exp.Expression) else None
        if not name:
            raise ExecutionError("Unsupported SET form (expected a variable name).")
        session.variables[name.upper()] = _variable_value(assignment.expression)
    return [ColumnMeta(name="status", type="text")], [["Statement executed successfully."]]


def _execute_show_variables(
    statement: exp.Expression, session: SessionContext
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle ``SHOW VARIABLES`` by listing the session's variables."""
    if not (
        isinstance(statement, exp.Command)
        and str(statement.args.get("this") or "").upper() == "SHOW"
        and "VARIABLES" in str(statement.args.get("expression") or "").upper()
    ):
        return None
    rows = [
        [
            name,
            _variable_display(value),
            "expression" if isinstance(value, exp.Expression) else type(value).__name__,
        ]
        for name, value in sorted(session.variables.items())
    ]
    return (
        [
            ColumnMeta(name="name", type="text"),
            ColumnMeta(name="value", type="text"),
            ColumnMeta(name="type", type="text"),
        ],
        rows,
    )


def _substitute_variables(statement: exp.Expression, session: SessionContext) -> exp.Expression:
    """Expand ``$name`` references to their session variable values.

    A ``$name`` with no matching variable is left untouched so it can still serve as a
    named bind parameter.
    """
    if not session.variables:
        return statement

    def replace(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Parameter):
            value = session.variables.get((node.name or "").upper())
            if value is not None or (node.name or "").upper() in session.variables:
                return _variable_literal(value)
        return node

    return statement.transform(replace)


def _convert_describe_types(rows: list[list[Any]]) -> list[list[Any]]:
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


def _result_columns(cursor: Any) -> tuple[list[ColumnMeta], list[list[Any]]]:
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


def _show_scope(statement: exp.Show, session: SessionContext) -> tuple[str, str] | None:
    """Extract the ``(database, schema)`` a scoped ``SHOW ... IN ...`` targets.

    ``SHOW SCHEMAS IN DATABASE db`` targets a database (schema ``None``);
    ``SHOW OBJECTS IN db.schema`` targets a schema. Returns ``None`` for unscoped
    forms (``SHOW TABLES``), which are left to DuckDB's native SHOW.
    """
    scope = statement.args.get("scope")
    kind = statement.args.get("scope_kind")
    if not isinstance(scope, exp.Table):
        return None
    if kind == "DATABASE":
        return scope.name, None
    if kind == "SCHEMA":
        db = scope.args.get("db")
        database = db.name if isinstance(db, exp.Identifier) else session.database
        return database, scope.name
    return None


def _show_limit(statement: exp.Show) -> int | None:
    """Extract the ``LIMIT n`` from a SHOW statement, if any."""
    limit = statement.args.get("limit")
    if isinstance(limit, exp.Limit):
        expr = limit.args.get("expression")
        if isinstance(expr, exp.Literal):
            try:
                return int(expr.this)
            except (TypeError, ValueError):
                return None
    return None


def _escape_like(text: str) -> str:
    """Escape LIKE wildcards so a ``STARTS WITH`` match is a plain prefix."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _name_prefix_filter(column: str, prefix: str | None) -> str:
    """Return a WHERE fragment restricting ``column`` to a ``STARTS WITH`` prefix."""
    if not prefix:
        return ""
    escaped = _escape_like(prefix).replace("'", "''")
    return f" AND {column} LIKE '{escaped}%' ESCAPE '\\'"


def _execute_snowflake_show(
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Execute Snowflake ``SHOW ... IN ...`` statements with Snowflake-shaped output.

    DuckDB's native SHOW returns its own column names (``schema_name``, ...), but
    Snowflake tools such as dbt expect ``name``/``kind``/``database_name``/...
    These are generated from DuckDB's catalog introspection functions instead.
    Returns ``None`` for SHOW forms this handler doesn't own.
    """
    if isinstance(statement, exp.Command):
        text = (statement.args.get("expression") or "").upper()
        if any(kind in text for kind in ("USER FUNCTIONS IN", "ALL FUNCTIONS IN", "EXTERNAL FUNCTIONS IN")):
            # No user functions are emulated; return the Snowflake SHOW FUNCTIONS
            # shape with zero rows (dbt filters ``is_builtin = 'N'``).
            cursor.execute(
                """
                SELECT
                    NULL::TIMESTAMP AS created_on,
                    NULL::VARCHAR AS name,
                    NULL::VARCHAR AS schema_name,
                    NULL::VARCHAR AS catalog_name,
                    'N'::VARCHAR AS is_builtin,
                    'N'::VARCHAR AS is_secure,
                    NULL::VARCHAR AS arguments,
                    NULL::VARCHAR AS result_type,
                    NULL::VARCHAR AS description
                WHERE FALSE
                """
            )
            return _result_columns(cursor)
        return None
    if not isinstance(statement, exp.Show):
        return None
    obj = str(statement.args.get("this") or "").upper()
    scope = _show_scope(statement, session)
    if scope is None:
        return None
    database, schema = scope

    limit = _show_limit(statement)
    prefix = statement.args.get("starts_with")
    prefix = prefix.this if isinstance(prefix, exp.Literal) else None

    if obj == "SCHEMAS":
        sql = f"""
            SELECT
                NULL::TIMESTAMP AS created_on,
                schema_name AS name,
                'SCHEMA' AS kind,
                database_name AS database_name,
                schema_name AS schema_name,
                comment AS comment,
                NULL::VARCHAR AS options,
                NULL::VARCHAR AS owner,
                NULL::BIGINT AS retention_time
            FROM duckdb_schemas()
            WHERE lower(database_name) = lower('{database}')
              AND lower(schema_name) != 'main'
              {_name_prefix_filter('schema_name', prefix)}
            ORDER BY schema_name
        """
    else:
        selects: list[str] = []
        if obj in ("TABLES", "OBJECTS"):
            selects.append(f"""
                SELECT
                    NULL::TIMESTAMP AS created_on,
                    table_name AS name,
                    'TABLE' AS kind,
                    database_name AS database_name,
                    schema_name AS schema_name,
                    comment AS comment,
                    'N' AS is_dynamic,
                    'N' AS is_iceberg
                FROM duckdb_tables()
                WHERE lower(database_name) = lower('{database}')
                  AND lower(schema_name) = lower('{schema}')
                  {_name_prefix_filter('table_name', prefix)}
            """)
        if obj in ("VIEWS", "OBJECTS"):
            selects.append(f"""
                SELECT
                    NULL::TIMESTAMP AS created_on,
                    view_name AS name,
                    'VIEW' AS kind,
                    database_name AS database_name,
                    schema_name AS schema_name,
                    comment AS comment,
                    'N' AS is_dynamic,
                    'N' AS is_iceberg
                FROM duckdb_views()
                WHERE lower(database_name) = lower('{database}')
                  AND lower(schema_name) = lower('{schema}')
                  {_name_prefix_filter('view_name', prefix)}
            """)
        if not selects:
            return None
        sql = " UNION ALL ".join(selects) + " ORDER BY name"

    if limit is not None:
        sql += f" LIMIT {limit}"
    cursor.execute(sql)
    return _result_columns(cursor)


def _execute_show_columns(
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle ``SHOW COLUMNS`` with Snowflake's column names (used by dbt).

    Scopes onto the current schema by default, or a named table/schema/database when
    the statement carries ``IN``. Returns ``None`` for SHOW forms this handler does
    not own.
    """
    if not (
        isinstance(statement, exp.Show)
        and str(statement.args.get("this") or "").upper() == "COLUMNS"
    ):
        return None
    scope_kind = statement.args.get("scope_kind")
    scope = statement.args.get("scope")
    database = session.database
    schema = session.schema
    table_name = None
    if scope_kind == "DATABASE" and isinstance(scope, exp.Table):
        database, schema = scope.name, None
    elif scope_kind == "SCHEMA" and isinstance(scope, exp.Table):
        db = scope.args.get("db")
        database = db.name if isinstance(db, exp.Identifier) else session.database
        schema = scope.name
    elif scope_kind in ("TABLE", "VIEW") and isinstance(scope, exp.Table):
        catalog = scope.args.get("catalog")
        db = scope.args.get("db")
        database = catalog.name if isinstance(catalog, exp.Identifier) else session.database
        schema = db.name if isinstance(db, exp.Identifier) else session.schema
        table_name = scope.name
    elif scope_kind is not None:
        return None

    # Bound as parameters so an identifier containing a quote cannot break the SQL.
    params: list[Any] = [database]
    where = ["lower(database_name) = lower(?)"]
    if schema is not None:
        where.append("lower(schema_name) = lower(?)")
        params.append(schema)
    if table_name is not None:
        where.append("lower(table_name) = lower(?)")
        params.append(table_name)
    cursor.execute(
        f"""
        SELECT
            table_name AS "table_name",
            schema_name AS "schema_name",
            column_name AS "column_name",
            data_type AS "data_type",
            CASE WHEN is_nullable THEN 'Y' ELSE 'N' END AS "null?",
            column_default AS "default",
            'COLUMN' AS "kind",
            NULL::VARCHAR AS "expression",
            comment AS "comment",
            database_name AS "database_name",
            NULL::VARCHAR AS "autoincrement",
            NULL::VARCHAR AS "schema_evolution_record"
        FROM duckdb_columns()
        WHERE {' AND '.join(where)}
        ORDER BY table_name, column_index
        """,
        params,
    )
    row_type, rows = _result_columns(cursor)
    # data_type is column 3 here (unlike DESCRIBE, where it is column 1).
    rows = [
        [*row[:3], duckdb_type_to_snowflake_sql_type(str(row[3])), *row[4:]] for row in rows
    ]
    return row_type, rows


def _show_key_kind(statement: exp.Expression) -> str | None:
    """Classify a SHOW <PRIMARY|UNIQUE|FOREIGN|IMPORTED> KEYS statement."""
    kinds = {
        "PRIMARY KEYS": "PRIMARY",
        "UNIQUE KEYS": "UNIQUE",
        "IMPORTED KEYS": "FOREIGN",
    }
    if isinstance(statement, exp.Show):
        return kinds.get(str(statement.args.get("this") or "").upper())
    # sqlglot cannot parse SHOW FOREIGN KEYS, so it arrives as a Command.
    if isinstance(statement, exp.Command) and str(statement.args.get("this") or "").upper() == "SHOW":
        if "FOREIGN KEYS" in str(statement.args.get("expression") or "").upper():
            return "FOREIGN"
    return None


def _show_key_scope(
    statement: exp.Expression, session: SessionContext
) -> tuple[str, str | None, str | None]:
    """Return the ``(database, schema, table)`` a scoped SHOW KEYS statement targets."""
    scope_kind = statement.args.get("scope_kind")
    scope = statement.args.get("scope")
    database: str = session.database
    schema: str | None = session.schema
    table_name: str | None = None
    if scope_kind == "DATABASE" and isinstance(scope, exp.Table):
        database, schema = scope.name, None
    elif scope_kind == "SCHEMA" and isinstance(scope, exp.Table):
        db = scope.args.get("db")
        database = db.name if isinstance(db, exp.Identifier) else session.database
        schema = scope.name
    elif scope_kind == "TABLE" and isinstance(scope, exp.Table):
        catalog = scope.args.get("catalog")
        db = scope.args.get("db")
        database = catalog.name if isinstance(catalog, exp.Identifier) else session.database
        schema = db.name if isinstance(db, exp.Identifier) else session.schema
        table_name = scope.name
    elif isinstance(statement, exp.Command):
        # SHOW FOREIGN KEYS is not parseable by sqlglot, so read the ``IN TABLE x``
        # target from the raw text.
        match = re.search(r"\bIN\s+(?:TABLE\s+)?([\w.\"]+)", str(statement.args.get("expression") or ""), re.IGNORECASE)
        if match:
            parts = [part.strip('"') for part in match.group(1).split(".")]
            table_name = parts[-1]
            if len(parts) >= 2:
                schema = parts[-2]
            if len(parts) >= 3:
                database = parts[-3]
    return database, schema, table_name


def _execute_show_keys(
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle SHOW PRIMARY/UNIQUE/FOREIGN/IMPORTED KEYS via ``duckdb_constraints()``."""
    kind = _show_key_kind(statement)
    if kind is None:
        return None
    database, schema, table_name = _show_key_scope(statement, session)

    where = ["lower(database_name) = lower(?)"]
    params: list[Any] = [database]
    if schema is not None:
        where.append("lower(schema_name) = lower(?)")
        params.append(schema)
    if table_name is not None:
        where.append(f'lower(table_name) = lower(?)')
        params.append(table_name)

    if kind in ("PRIMARY", "UNIQUE"):
        params.insert(0, "PRIMARY KEY" if kind == "PRIMARY" else "UNIQUE")
        cursor.execute(
            f"""
            SELECT
                NULL::TIMESTAMP AS created_on,
                database_name AS database_name,
                schema_name AS schema_name,
                table_name AS table_name,
                u.column_name AS column_name,
                u.key_sequence AS key_sequence,
                constraint_name AS constraint_name,
                NULL::VARCHAR AS rely,
                NULL::VARCHAR AS comment
            FROM duckdb_constraints(), UNNEST(constraint_column_names) WITH ORDINALITY AS u(column_name, key_sequence)
            WHERE constraint_type = ? AND {' AND '.join(where)}
            ORDER BY table_name, constraint_name, key_sequence
            """,
            params,
        )
    else:
        cursor.execute(
            f"""
            SELECT
                NULL::TIMESTAMP AS created_on,
                database_name AS pk_database_name,
                schema_name AS pk_schema_name,
                referenced_table AS pk_table_name,
                referenced_column_names[u.key_sequence] AS pk_column_name,
                database_name AS fk_database_name,
                schema_name AS fk_schema_name,
                table_name AS fk_table_name,
                u.column_name AS fk_column_name,
                u.key_sequence AS key_sequence,
                NULL::VARCHAR AS update_rule,
                NULL::VARCHAR AS delete_rule,
                constraint_name AS fk_name,
                NULL::VARCHAR AS pk_name,
                NULL::VARCHAR AS deferrability,
                NULL::VARCHAR AS rely,
                NULL::VARCHAR AS comment
            FROM duckdb_constraints(), UNNEST(constraint_column_names) WITH ORDINALITY AS u(column_name, key_sequence)
            WHERE constraint_type = 'FOREIGN KEY' AND {' AND '.join(where)}
            ORDER BY table_name, constraint_name, key_sequence
            """,
            params,
        )
    return _result_columns(cursor)


def _execute_show_misc(
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle the simpler SHOW variants (databases, warehouses, sequences, ...)."""
    obj: str | None = None
    if isinstance(statement, exp.Show):
        obj = str(statement.args.get("this") or "").upper()
    elif isinstance(statement, exp.Command) and str(statement.args.get("this") or "").upper() == "SHOW":
        # sqlglot cannot parse SHOW PARAMETERS.
        if str(statement.args.get("expression") or "").upper().strip().startswith("PARAMETERS"):
            obj = "PARAMETERS"

    if obj == "DATABASES":
        cursor.execute(
            """
            SELECT
                NULL::TIMESTAMP AS created_on,
                database_name AS name,
                'N' AS is_default,
                CASE WHEN lower(database_name) = lower(?) THEN 'Y' ELSE 'N' END AS is_current,
                NULL::VARCHAR AS origin,
                NULL::VARCHAR AS owner,
                comment AS comment,
                NULL::VARCHAR AS options,
                NULL::BIGINT AS retention_time
            FROM duckdb_databases()
            WHERE lower(database_name) NOT IN ('system', 'temp', 'memory')
            ORDER BY name
            """,
            [session.database],
        )
    elif obj == "WAREHOUSES":
        cursor.execute(
            "SELECT ? AS name, 'STARTED' AS state, 'STANDARD' AS type, 'X-SMALL' AS size, "
            "'N' AS running, 'N' AS queued, NULL::TIMESTAMP AS created_on, NULL::VARCHAR AS owner, "
            "NULL::VARCHAR AS comment",
            [session.warehouse],
        )
    elif obj == "SEQUENCES":
        cursor.execute(
            """
            SELECT
                NULL::TIMESTAMP AS created_on,
                sequence_name AS name,
                database_name AS database_name,
                schema_name AS schema_name,
                last_value AS next_value,
                increment_by AS interval,
                comment AS comment
            FROM duckdb_sequences()
            WHERE lower(database_name) = lower(?)
              AND lower(schema_name) = lower(?)
            ORDER BY name
            """,
            [session.database, session.schema],
        )
    elif obj == "PARAMETERS":
        cursor.execute(
            """
            SELECT name, value, "default", level FROM (VALUES
                ('AUTOCOMMIT', 'true', 'true', 'SESSION'),
                ('TIMEZONE', 'UTC', 'UTC', 'SESSION'),
                ('CLIENT_PREFETCH_THREADS', '4', '4', 'SESSION')
            ) AS t(name, value, "default", level)
            """
        )
    elif obj == "PROCEDURES":
        cursor.execute(
            "SELECT NULL::VARCHAR AS name, NULL::VARCHAR AS schema_name, NULL::VARCHAR AS catalog_name WHERE FALSE"
        )
    elif obj == "FUNCTIONS":
        cursor.execute(
            "SELECT NULL::VARCHAR AS name, NULL::VARCHAR AS schema_name, NULL::VARCHAR AS catalog_name WHERE FALSE"
        )
    elif obj == "USERS":
        cursor.execute("SELECT NULL::VARCHAR AS name, NULL::TIMESTAMP AS created_on WHERE FALSE")
    else:
        return None
    return _result_columns(cursor)


def _execute_snowflake_describe(
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Execute ``DESCRIBE TABLE`` with Snowflake's output shape.

    dbt's ``get_columns_in_relation`` reads the ``name``/``type`` columns of the
    DESCRIBE result; DuckDB's own DESCRIBE returns ``column_name``/``column_type``.
    """
    if not (isinstance(statement, exp.Describe) and statement.args.get("kind") == "TABLE"):
        return None
    table = statement.args.get("this")
    if not isinstance(table, exp.Table):
        return None
    db_ident = table.args.get("catalog")
    schema_ident = table.args.get("db")
    database = db_ident.name if isinstance(db_ident, exp.Identifier) else session.database
    schema = schema_ident.name if isinstance(schema_ident, exp.Identifier) else session.schema
    table_name = table.name

    cursor.execute(
        f"""
        SELECT
            column_name AS "name",
            data_type AS "type",
            'COLUMN' AS "kind",
            CASE WHEN is_nullable THEN 'Y' ELSE 'N' END AS "null?",
            column_default AS "default",
            NULL::VARCHAR AS "primary key",
            NULL::VARCHAR AS "unique key",
            NULL::VARCHAR AS "check",
            NULL::VARCHAR AS "expression",
            comment AS "comment",
            NULL::VARCHAR AS "policy name"
        FROM duckdb_columns()
        WHERE lower(database_name) = lower('{database}')
          AND lower(schema_name) = lower('{schema}')
          AND lower(table_name) = lower('{table_name}')
        ORDER BY column_index
        """
    )
    row_type, rows = _result_columns(cursor)
    rows = _convert_describe_types(rows)
    return row_type, rows


def _is_ddl_status_artifact(statement: exp.Expression, description: list[Any]) -> bool:
    """Return True when DuckDB reports DDL as a single ``Count``/``Success`` column.

    DuckDB's Python API reports CREATE/DROP/ALTER as a one-column result named
    ``Count`` or ``Success`` with no rows. Snowflake returns a plain status instead,
    so those statements are routed to the standard status result.
    """
    if len(description) != 1 or description[0][0] not in ("Count", "Success"):
        return False
    return isinstance(statement, (exp.Create, exp.Drop, exp.Alter))


def _execute_create_database(
    manager: DuckDBManager,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Execute ``CREATE DATABASE`` by attaching a fresh catalog (DuckDB has no
    ``CREATE DATABASE``). Returns ``None`` for non-database CREATE statements.
    """
    if not (isinstance(statement, exp.Create) and str(statement.args.get("kind") or "").upper() == "DATABASE"):
        return None
    if isinstance(statement.args.get("clone"), exp.Clone):
        # Snowflake can clone a database; the emulator cannot rebuild the catalog's
        # contents, so it fails loudly instead of creating an empty database.
        raise ExecutionError(
            "CREATE DATABASE ... CLONE is not supported by the emulator "
            "(only CREATE TABLE ... CLONE is)."
        )
    this = statement.args.get("this")
    ident = this.this if isinstance(this, exp.Table) else this
    if not isinstance(ident, exp.Identifier) or not ident.name:
        return None
    # Unquoted identifiers are uppercased (Snowflake semantics); quoted preserved.
    name = ident.name if ident.args.get("quoted") else ident.name.upper()

    if statement.args.get("replace"):
        manager.replace_database(name)
    else:
        if not statement.args.get("exists") and manager.has_database(name):
            raise ExecutionError(f"Database {name} already exists.")
        manager.ensure_namespace(name, "PUBLIC")
    return (
        [ColumnMeta(name="status", type="text")],
        [[f"Database {name} successfully created."]],
    )



def _file_format_options(statement: exp.Expression) -> dict[str, str]:
    """Extract the ``TYPE = CSV, SKIP_HEADER = 1, ...`` options from a CREATE node."""
    options: dict[str, str] = {}
    properties = statement.args.get("properties")
    for prop in properties.expressions if properties else []:
        if not isinstance(prop, exp.Property):
            continue
        value = prop.args.get("value")
        if value is None:
            continue
        if isinstance(value, exp.Literal) and value.is_string:
            options[str(prop.this).upper()] = str(value.this)
        else:
            options[str(prop.this).upper()] = value.sql(dialect="snowflake")
    return options


def _execute_file_format_ddl(
    session: SessionContext, statement: exp.Expression
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle CREATE/DROP FILE FORMAT and SHOW FILE FORMATS."""
    kind = str(statement.args.get("kind") or "").upper()
    if isinstance(statement, exp.Create) and kind == "FILE FORMAT":
        name = statement.this.name if isinstance(statement.this, exp.Table) else ""
        if not name:
            return None
        get_file_format_manager().create(
            session,
            name,
            _file_format_options(statement),
            replace=bool(statement.args.get("replace")),
            if_not_exists=bool(statement.args.get("exists")),
        )
        return [ColumnMeta(name="status", type="text")], [
            [f"File format {name.upper()} successfully created."]
        ]
    if isinstance(statement, exp.Drop) and kind == "FILE FORMAT":
        name = statement.this.name if isinstance(statement.this, exp.Table) else ""
        if not name:
            return None
        get_file_format_manager().drop(session, name, if_exists=bool(statement.args.get("exists")))
        return [ColumnMeta(name="status", type="text")], [
            [f"File format {name.upper()} successfully dropped."]
        ]
    if isinstance(statement, exp.Show) and str(statement.args.get("this") or "").upper() == "FILE FORMATS":
        rows = [
            [fmt.name, fmt.database, fmt.schema, fmt.options.get("TYPE", "")]
            for fmt in get_file_format_manager().list(session)
        ]
        return (
            [
                ColumnMeta(name="name", type="text"),
                ColumnMeta(name="database_name", type="text"),
                ColumnMeta(name="schema_name", type="text"),
                ColumnMeta(name="type", type="text"),
            ],
            rows,
        )
    return None


def _execute_stage_command(
    manager: DuckDBManager,
    session: SessionContext,
    raw_sql: str,
    command: str,
) -> ExecutionResult:
    """Handle PUT/GET/LIST/REMOVE statements locally (no DuckDB involved)."""
    stages = get_stage_manager()
    try:
        if command == "PUT":
            put = parse_put(raw_sql)
            files, target_dir = stages.put_files(session, put)
            row_type = [
                ColumnMeta(name="source", type="text"),
                ColumnMeta(name="target", type="text"),
                ColumnMeta(name="source_size", type="fixed", duckdb_type="BIGINT"),
                ColumnMeta(name="target_size", type="fixed", duckdb_type="BIGINT"),
                ColumnMeta(name="source_compression", type="text"),
                ColumnMeta(name="target_compression", type="text"),
                ColumnMeta(name="status", type="text"),
                ColumnMeta(name="message", type="text"),
            ]
            rows = [
                [
                    os.path.basename(f.rel_path),
                    f.rel_path,
                    f.size,
                    f.size,
                    "NONE",
                    "NONE",
                    f.status,
                    f.message,
                ]
                for f in files
            ]
            overwrite = put.options.get("OVERWRITE", "FALSE").upper() in (
                "TRUE",
                "1",
                "YES",
                "ON",
            )
            return ExecutionResult(
                handle=str(uuid.uuid4()),
                created_on=int(time.time() * 1000),
                row_type=row_type,
                rows=rows,
                message=f"{len(files)} file(s) staged successfully.",
                file_transfer=FileTransfer(
                    kind="UPLOAD",
                    src_locations=_strip_file_prefixes(put.files),
                    stage_location=target_dir,
                    overwrite=overwrite,
                ),
            )

        if command == "GET":
            get = parse_get(raw_sql)
            files, stage_base = stages.get_files(session, get)
            row_type = [
                ColumnMeta(name="file", type="text"),
                ColumnMeta(name="size", type="fixed", duckdb_type="BIGINT"),
                ColumnMeta(name="status", type="text"),
                ColumnMeta(name="message", type="text"),
            ]
            rows = [
                [f.rel_path, f.size, f.status, f.message] for f in files
            ]
            local_dir = os.path.abspath(os.path.expanduser(get.local_dir))
            return ExecutionResult(
                handle=str(uuid.uuid4()),
                created_on=int(time.time() * 1000),
                row_type=row_type,
                rows=rows,
                message=f"{len(files)} file(s) downloaded.",
                file_transfer=FileTransfer(
                    kind="DOWNLOAD",
                    src_locations=[f.rel_path for f in files],
                    stage_location=stage_base,
                    local_location=local_dir,
                ),
            )

        if command == "LIST":
            files = stages.list_files(session, parse_list(raw_sql))
            row_type = [
                ColumnMeta(name="name", type="text"),
                ColumnMeta(name="size", type="fixed", duckdb_type="BIGINT"),
                ColumnMeta(name="md5", type="text"),
                ColumnMeta(name="last_modified", type="text"),
            ]
            rows = [
                [f.rel_path, f.size, f.md5, f.last_modified] for f in files
            ]
            return ExecutionResult(
                handle=str(uuid.uuid4()),
                created_on=int(time.time() * 1000),
                row_type=row_type,
                rows=rows,
                message=f"{len(files)} file(s) listed.",
            )

        if command == "REMOVE":
            files = stages.remove_files(session, parse_remove(raw_sql))
            row_type = [
                ColumnMeta(name="name", type="text"),
                ColumnMeta(name="result", type="text"),
                ColumnMeta(name="message", type="text"),
            ]
            rows = [[f.rel_path, f.status, f.message] for f in files]
            return ExecutionResult(
                handle=str(uuid.uuid4()),
                created_on=int(time.time() * 1000),
                row_type=row_type,
                rows=rows,
                message=f"{len(files)} file(s) removed.",
            )
    except StageError as exc:
        raise ExecutionError(str(exc)) from exc
    raise ExecutionError(f"Unsupported stage command: {command}")


def _strip_file_prefixes(files: list[str]) -> list[str]:
    """Strip ``file://`` so the connector's local storage client sees plain paths."""
    stripped = []
    for f in files:
        if f.startswith("file://"):
            f = f[len("file://") :]
        stripped.append(f)
    return stripped


def _execute_stage_ddl(
    manager: DuckDBManager,
    session: SessionContext,
    statement: exp.Expression,
    kind: str,
) -> tuple[list[ColumnMeta], list[list[Any]]]:
    """Handle CREATE/DROP/ALTER STAGE and SHOW STAGES without DuckDB."""
    stages = get_stage_manager()
    try:
        if kind == "CREATE_STAGE":
            create_stage(stages, session, statement)
            return (
                [ColumnMeta(name="status", type="text")],
                [[f"Stage {statement.this.name} successfully created."]],
            )
        if kind == "DROP_STAGE":
            drop_stage(stages, session, statement)
            return (
                [ColumnMeta(name="status", type="text")],
                [[f"Stage {statement.this.name} successfully dropped."]],
            )
        if kind == "ALTER_STAGE":
            return (
                [ColumnMeta(name="status", type="text")],
                [["Statement executed successfully."]],
            )
        if kind == "SHOW_STAGES":
            rows = [[name, db, schema, url] for name, db, schema, url in show_stages(stages, session)]
            return (
                [
                    ColumnMeta(name="name", type="text"),
                    ColumnMeta(name="database_name", type="text"),
                    ColumnMeta(name="schema_name", type="text"),
                    ColumnMeta(name="url", type="text"),
                ],
                rows,
            )
    except StageError as exc:
        raise ExecutionError(str(exc)) from exc
    raise ExecutionError(f"Unsupported stage DDL: {kind}")


def _apply_use_statement(
    manager: DuckDBManager, session: SessionContext, cursor: Any, use_stmt: UseStatement
) -> None:
    if use_stmt.kind == "DATABASE":
        session.database = use_stmt.name
        manager.ensure_namespace(session.database, session.schema)
    elif use_stmt.kind == "SCHEMA":
        session.schema = use_stmt.name
        manager.ensure_namespace(session.database, session.schema)
    elif use_stmt.kind == "SCHEMA_QUALIFIED":
        db, _, sch = use_stmt.name.partition(".")
        session.database, session.schema = db, sch
        manager.ensure_namespace(session.database, session.schema)
    elif use_stmt.kind == "WAREHOUSE":
        session.warehouse = use_stmt.name
        return  # No DuckDB equivalent; warehouse is purely cosmetic in the emulator.
    elif use_stmt.kind == "ROLE":
        session.role = use_stmt.name
        return  # No DuckDB equivalent; role is purely cosmetic in the emulator.

    cursor.execute(f'USE "{session.database}"."{session.schema}"')
