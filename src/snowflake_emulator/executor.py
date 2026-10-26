"""Execute (translated) Snowflake SQL against the shared DuckDB backend."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

import duckdb
from sqlglot import exp

from snowflake_emulator.core.constants import INFORMATION_SCHEMA_SCHEMA
from snowflake_emulator.core.errors import ExecutionError
from snowflake_emulator.core.results import (
    ColumnMeta,
    ExecutionResult,
    FileTransfer,  # re-exported for compatibility
)
from snowflake_emulator.catalog import column_type_resolver
from snowflake_emulator.translator import column_types
from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.engine.errors import as_execution_error
from snowflake_emulator.engine.handlers.common import convert_describe_types
from snowflake_emulator.engine.handlers import dml
from snowflake_emulator.engine.handlers.ddl import (
    execute_alter_constraints,
    execute_create_database,
    execute_drop_database,
    execute_file_format_ddl,
    execute_function_ddl,
    is_ddl_status_artifact,
    record_constraints,
    record_declared_columns,
    record_temp_table_owner,
)
from snowflake_emulator.engine.handlers.describe import (
    describe_sql,  # re-exported for the wire-protocol router
    execute_function_describe,
    execute_snowflake_describe,
)
from snowflake_emulator.engine.handlers.show import (
    execute_show_columns,
    execute_show_keys,
    execute_show_misc,
    execute_snowflake_show,
)
from snowflake_emulator.engine.handlers.stage import (
    execute_stage_command,
    execute_stage_ddl,
)
from snowflake_emulator.file_formats import (
    FileFormatError,
    resolve_file_format,
)
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.settings import settings
from snowflake_emulator.statement_store import get_statement_store
from snowflake_emulator.statement_types import (
    number_of_binds,
    statement_type_id,
)
from snowflake_emulator.stages import (
    StageError,
    build_copy_sql,
    copy_stage_source,
    get_stage_manager,
    has_stage_file_table,
    match_stage_command,
    match_stage_ddl,
    preprocess_stage_file_clauses,
    rewrite_stage_file_tables,
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
from snowflake_emulator.translator.parsing import _split_mixed_alter
from snowflake_emulator.type_mapping import duckdb_type_to_snowflake
from snowflake_emulator.user_functions import SessionUserFunctionResolver


_COPY_RESULT_COLUMNS = [
    ColumnMeta(name="file", type="text"),
    ColumnMeta(name="status", type="text"),
    ColumnMeta(name="rows_parsed", type="fixed", duckdb_type="BIGINT"),
    ColumnMeta(name="rows_loaded", type="fixed", duckdb_type="BIGINT"),
    ColumnMeta(name="error_limit", type="fixed", duckdb_type="BIGINT"),
    ColumnMeta(name="errors_seen", type="fixed", duckdb_type="BIGINT"),
    ColumnMeta(name="first_error", type="text"),
    ColumnMeta(name="first_error_line", type="fixed", duckdb_type="BIGINT"),
    ColumnMeta(name="first_error_character", type="fixed", duckdb_type="BIGINT"),
    ColumnMeta(name="first_error_column_name", type="text"),
]


def execute_sql(
    manager: DuckDBManager,
    session: SessionContext,
    raw_sql: str,
    bind_params: list[Any] | dict[str, Any] | None = None,
    bind_batch: list[tuple[Any, ...]] | None = None,
    multi_statement_count: int | None = None,
    timezone: str | None = None,
) -> ExecutionResult:
    """Execute one or more (semicolon-separated) Snowflake statements.

    Session-scoped ``USE ...`` statements update `session` in place instead of being
    sent to DuckDB. Only the result of the final statement is returned, matching
    Snowflake's own multi-statement behaviour. `bind_params` (a positional list or a
    named mapping) and `bind_batch` (one tuple per row, for an array binding) are only
    applied when `raw_sql` contains a single statement. `multi_statement_count` is the SQL
    API's ``MULTI_STATEMENT_COUNT`` (0 = any number, N = exactly N); `timezone` is the
    request's ``TIMEZONE``, which DuckDB's session timezone is set to for this statement.

    The statements of one session are serialized and run on that session's own DuckDB
    connection, so a ``CREATE TEMPORARY TABLE`` is visible to every later statement in
    the session and a session never sees another session's temporary tables.
    """
    with manager.session_scope(session.token):
        return _execute_sql(
            manager,
            session,
            raw_sql,
            bind_params,
            bind_batch,
            multi_statement_count,
            timezone,
        )


def _execute_sql(
    manager: DuckDBManager,
    session: SessionContext,
    raw_sql: str,
    bind_params: list[Any] | dict[str, Any] | None = None,
    bind_batch: list[tuple[Any, ...]] | None = None,
    multi_statement_count: int | None = None,
    timezone: str | None = None,
) -> ExecutionResult:
    if _matches_noop(raw_sql):
        return ExecutionResult(
            handle=str(uuid.uuid4()),
            created_on=int(time.time() * 1000),
            row_type=[ColumnMeta(name="status", type="text")],
            rows=[["Statement executed successfully."]],
        )

    manager.ensure_namespace(session.database, session.schema)
    cursor = manager.cursor(session.token)
    cursor.execute(f'USE "{session.database}"."{session.schema}"')
    # Snowflake's default/session timezone is UTC; DuckDB otherwise inherits the host. A
    # request's TIMEZONE parameter applies to this statement's execution.
    cursor.execute("SET TimeZone = ?", [timezone or "UTC"])

    # Resolves user-defined function calls to the DuckDB macros backing them; loads
    # lazily so a session with no ``CREATE FUNCTION`` never touches the side table.
    user_functions = SessionUserFunctionResolver(manager, session)

    stage_command = match_stage_command(raw_sql)
    if stage_command is not None:
        return execute_stage_command(manager, session, raw_sql, stage_command)

    # `FROM @stage/file (FILE_FORMAT => (...))` is not valid sqlglot syntax; the options
    # are captured here (and decoded again in `rewrite_stage_file_tables`).
    raw_sql = preprocess_stage_file_clauses(raw_sql)

    try:
        statements = split_statements(raw_sql)
    except TranslationError as exc:
        raise as_execution_error(exc) from exc
    _check_multi_statement_count(len(statements), multi_statement_count)
    # A multi-action ``ALTER TABLE`` is one Snowflake statement that sqlglot can only read
    # as a raw command: expand it into one ALTER per action now that the count is checked.
    statements = [
        action
        for statement in statements
        for action in _split_mixed_alter(statement)
    ]

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
        statement = _substitute_session_functions(statement, session, timezone)

        stmt_sql = statement.sql(dialect="snowflake")

        use_stmt = match_use_statement(stmt_sql)
        if use_stmt is not None:
            _apply_use_statement(manager, session, cursor, use_stmt)
            row_type, rows = [], []
            message = "Statement executed successfully."
            continue

        if is_alter_session(stmt_sql):
            # Session parameters are not modelled; accept ALTER SESSION SET/UNSET as a
            # no-op (schemachange sets QUERY_TAG around every script).
            row_type, rows = [], []
            message = "Statement executed successfully."
            continue

        if is_transaction_control(statement):
            # Run real transactions on the session's connection. Snowflake accepts
            # COMMIT/ROLLBACK outside a transaction, while DuckDB raises, so those are
            # treated as the no-op Snowflake makes them.
            try:
                if isinstance(statement, exp.Transaction):
                    cursor.execute("BEGIN TRANSACTION")
                elif isinstance(statement, exp.Commit):
                    cursor.execute("COMMIT")
                elif isinstance(statement, exp.Rollback):
                    cursor.execute("ROLLBACK")
            except duckdb.TransactionException:
                pass
            row_type, rows = [], []
            message = "Statement executed successfully."
            continue

        alter_constraints = execute_alter_constraints(manager, session, statement)
        if alter_constraints is not None:
            row_type, rows = alter_constraints
            message = "Statement executed successfully."
            continue

        stage_ddl = match_stage_ddl(statement)
        if stage_ddl is not None:
            row_type, rows = execute_stage_ddl(manager, session, statement, stage_ddl)
            continue

        file_format_ddl = execute_file_format_ddl(session, statement)
        if file_format_ddl is not None:
            row_type, rows = file_format_ddl
            continue

        create_db = execute_create_database(manager, statement)
        if create_db is not None:
            row_type, rows = create_db
            continue

        drop_db = execute_drop_database(manager, statement)
        if drop_db is not None:
            row_type, rows = drop_db
            continue

        function_ddl = execute_function_ddl(manager, cursor, session, statement)
        if function_ddl is not None:
            row_type, rows = function_ddl
            continue

        show_result = execute_snowflake_show(manager, cursor, session, statement)
        if show_result is not None:
            row_type, rows = show_result
            continue

        show_columns = execute_show_columns(cursor, session, statement)
        if show_columns is not None:
            row_type, rows = show_columns
            continue

        show_keys = execute_show_keys(cursor, session, statement)
        if show_keys is not None:
            row_type, rows = show_keys
            continue

        show_misc = execute_show_misc(manager, cursor, session, statement)
        if show_misc is not None:
            row_type, rows = show_misc
            continue

        describe_result = execute_function_describe(manager, session, statement)
        if describe_result is not None:
            row_type, rows = describe_result
            continue

        describe_result = execute_snowflake_describe(cursor, session, statement)
        if describe_result is not None:
            row_type, rows = describe_result
            continue

        if has_stage_file_table(statement):
            try:
                statement = rewrite_stage_file_tables(
                    statement, session, cursor, get_stage_manager()
                )
            except (StageError, FileFormatError) as exc:
                raise ExecutionError(str(exc)) from exc

        try:
            duckdb_sql = transpile_to_duckdb(
                statement,
                user_functions,
                column_types=column_type_resolver(cursor, session),
            )
        except TranslationError as exc:
            raise as_execution_error(exc) from exc

        stage_copy = None
        if isinstance(statement, exp.Copy):
            stage_copy = copy_stage_source(statement)

        # DML result shapes (Snowflake names/columns instead of DuckDB's ``Count``);
        # only a MERGE split or a multi-joined UPDATE needs counting before it runs.
        executed = (
            dml.parse_executed_statement(duckdb_sql)
            if isinstance(statement, (exp.Update, exp.Merge))
            else None
        )
        dml_columns = dml.result_columns(statement, executed)
        dml_totals = [0] * len(dml_columns) if dml_columns is not None else None
        preview: list[int] | None = None

        batch = bind_batch if len(statements) == 1 else None
        batch_affected = 0
        copy_loaded: list[tuple[str, int]] | None = None
        try:
            if stage_copy is not None:
                table_sql, stage_ref, file_format, pattern, projection = stage_copy
                file_format = resolve_file_format(session, file_format)
                copy_plan = build_copy_sql(
                    session,
                    table_sql,
                    stage_ref,
                    file_format,
                    get_stage_manager(),
                    pattern,
                    projection,
                )
                copy_loaded = []
                for file_label, copy_sql in copy_plan:
                    cursor.execute(copy_sql)
                    loaded = 0
                    if cursor.description:
                        fetched = cursor.fetchone()
                        if fetched and isinstance(fetched[0], int):
                            loaded = fetched[0]
                    copy_loaded.append((file_label, loaded))
            elif batch is not None:
                # An array binding (executemany) runs the statement once per row.
                for row in batch:
                    row_preview = dml.preview_values(executed, cursor, list(row))
                    cursor.execute(duckdb_sql, list(row))
                    affected = _affected_rows(cursor)
                    batch_affected += affected
                    if dml_totals is not None:
                        values = dml.result_values(statement, row_preview, affected)
                        dml_totals = [
                            total + value for total, value in zip(dml_totals, values)
                        ]
            elif bind_params is not None and len(statements) == 1:
                preview = dml.preview_values(executed, cursor, bind_params)
                cursor.execute(duckdb_sql, bind_params)
            else:
                preview = dml.preview_values(executed, cursor, None)
                cursor.execute(duckdb_sql)
        except (StageError, FileFormatError) as exc:
            raise ExecutionError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed execution error
            raise as_execution_error(exc) from exc

        if (
            isinstance(statement, exp.Create)
            and str(statement.args.get("kind") or "").upper() == "TABLE"
        ):
            record_declared_columns(manager, session, statement)
            record_constraints(manager, session, statement)
            record_temp_table_owner(manager, session, statement)

        if dml_totals is not None and batch is None:
            affected = 0 if preview is not None else _affected_rows(cursor)
            dml_totals = dml.result_values(statement, preview, affected)

        if copy_loaded is not None:
            row_type = _COPY_RESULT_COLUMNS
            rows = [
                [
                    file_label,
                    "LOADED",
                    loaded,
                    loaded,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                ]
                for file_label, loaded in copy_loaded
            ]
        elif dml_totals is not None:
            # Snowflake names the affected rows per action instead of DuckDB's ``Count``.
            row_type = [dml.result_column(name) for name in dml_columns or []]
            rows = [list(dml_totals)]
        elif batch is not None:
            # Report the total affected rows across the batch, as Snowflake does for a
            # multi-row DML, instead of the last row's count.
            row_type = [ColumnMeta(name="Count", type="fixed", duckdb_type="BIGINT")]
            rows = [[batch_affected]]
        elif cursor.description:
            description = cursor.description
            if is_ddl_status_artifact(statement, description):
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
                    rows = convert_describe_types(rows)
                else:
                    apply_declared_lengths(cursor, session, statement, row_type)
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


def _affected_rows(cursor: Any) -> int:
    """Read the affected-row count DuckDB reports for a DML statement."""
    if not cursor.description:
        return 0
    fetched = cursor.fetchone()
    if fetched and isinstance(fetched[0], int):
        return fetched[0]
    return 0


_TEXT_LENGTH_RE = re.compile(r"^\s*(VAR)?CHAR\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
_TEXT_TYPES = ("VARCHAR", "CHAR", "NCHAR", "NVARCHAR", "TEXT", "STRING")
_UNBOUNDED_TEXT_LENGTH = 16777216  # Snowflake's maximum VARCHAR length


def _declared_text_length(declared: str | None) -> int | None:
    """The length a declared Snowflake type text gives: ``VARCHAR(5)`` -> 5."""
    if not declared:
        return None
    match = _TEXT_LENGTH_RE.match(declared)
    if match is not None:
        return int(match.group(2))
    name = declared.strip().upper()
    if name == "CHAR":
        return 1
    if name in _TEXT_TYPES:
        return _UNBOUNDED_TEXT_LENGTH
    return None


def _catalog_text_length(
    cursor: Any, session: SessionContext, column: exp.Column, statement: exp.Expression
) -> int | None:
    """Look the declared type of a bare text column up in the emulator's catalog side table."""
    aliases = column_types.statement_table_aliases(statement)
    table = aliases.get(column.table.upper(), column.table) if column.table else None
    database = column.text("db") or None
    schema = database or None  # a bare ``db.column`` is unusual; leave it unresolved
    schema = column.text("schema") if column.args.get("db") else schema
    conditions = ["lower(column_name) = lower(?)", "lower(database_name) <> 'temp'"]
    params: list[Any] = [column.name]
    if table:
        conditions.append("lower(table_name) = lower(?)")
        params.append(table)
    if database:
        conditions.append("lower(database_name) = lower(?)")
        params.append(database)
    candidates = cursor.execute(
        f"SELECT database_name, schema_name, table_name FROM duckdb_columns() "
        f"WHERE {' AND '.join(conditions)}",
        params,
    ).fetchall()
    if not candidates:
        return None
    # Prefer the session's own schema/database, the way name resolution does.
    def rank(row: tuple[str, str, str]) -> tuple[int, int]:
        return (
            0 if str(row[1]).lower() == str(session.schema).lower() else 1,
            0 if str(row[0]).lower() == str(session.database).lower() else 1,
        )

    lengths: set[int | None] = set()
    for db_name, schema_name, table_name in sorted({tuple(row) for row in candidates}, key=rank):
        try:
            declared = cursor.execute(
                f'SELECT declared_type FROM "{db_name}".{INFORMATION_SCHEMA_SCHEMA}._sf_columns_ext '
                "WHERE lower(table_schema) = lower(?) AND lower(table_name) = lower(?) "
                "AND lower(column_name) = lower(?)",
                [schema_name, table_name, column.name],
            ).fetchone()
        except Exception:  # noqa: BLE001 - a database without the side table
            continue
        lengths.add(_declared_text_length(declared[0] if declared else None))
        if len(lengths) > 1:
            return None  # ambiguous: different tables give different lengths
    return lengths.pop() if len(lengths) == 1 else None


def apply_declared_lengths(
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
    row_type: list[ColumnMeta],
) -> None:
    """Fill ``ColumnMeta.length`` for text output columns.

    DuckDB's VARCHAR type carries no length, while Snowflake reports the declared maximum
    in ``rowType.length``: the length of a cast (``CAST(x AS VARCHAR(5))``), of a created
    column the catalog side table recorded, or of a string literal.
    """
    if isinstance(statement, exp.Union):
        select = statement.this
    else:
        select = statement
    if not isinstance(select, exp.Select):
        return
    for item, meta in zip(select.expressions, row_type):
        if meta.type != "text":
            continue
        value = item.this if isinstance(item, exp.Alias) else item
        length: int | None = None
        if isinstance(value, (exp.Cast, exp.TryCast)) and isinstance(value.to, exp.DataType):
            if value.to.this.name in _TEXT_TYPES:
                if value.to.expressions:
                    parameter = value.to.expressions[0]
                    if isinstance(parameter, exp.DataTypeParam):
                        parameter = parameter.this
                    if isinstance(parameter, exp.Literal) and not parameter.is_string:
                        length = int(str(parameter.this))
                if length is None:
                    length = _declared_text_length(value.to.this.name)
        elif isinstance(value, exp.Literal) and value.is_string:
            length = len(str(value.this))
        elif isinstance(value, exp.Column):
            length = _catalog_text_length(cursor, session, value, statement)
        if length is not None:
            meta.length = length


def _substitute_session_functions(
    statement: exp.Expression, session: SessionContext, timezone: str | None = None
) -> exp.Expression:
    """Replace Snowflake's CURRENT_* session functions with their session values.

    ``CURRENT_TIMEZONE()`` reports the zone the statement runs in -- a request's ``TIMEZONE``
    parameter when one was sent, UTC otherwise -- so it cannot contradict the timestamps it
    renders alongside.
    """
    values: dict[type[exp.Expression], str] = {
        exp.CurrentUser: session.user,
        exp.CurrentRole: session.role,
        exp.CurrentWarehouse: session.warehouse,
        exp.CurrentAccount: "EMULATOR_ACCOUNT",
        exp.CurrentRegion: "LOCAL",
        exp.CurrentTimezone: timezone or "UTC",
        exp.CurrentSession: str(abs(hash(session.token)) % 10_000_000),
    }

    def replace(node: exp.Expression) -> exp.Expression:
        value = values.get(type(node))
        return exp.Literal.string(value) if value is not None else node

    return statement.transform(replace)


def _check_multi_statement_count(count: int, allowed: int | None) -> None:
    """Enforce the SQL API's ``MULTI_STATEMENT_COUNT`` against the statements in the request.

    Snowflake's default is 1, so a request that chains statements has to opt in with a count
    of 0 (any number) or the exact number sent; a mismatch is an error rather than a silent
    partial execution.
    """
    if allowed is None:
        return
    if allowed < 0:
        raise ExecutionError(
            "MULTI_STATEMENT_COUNT must not be negative.", errno="001003", sqlstate="42000"
        )
    if allowed == 0:
        return
    if count != allowed:
        raise ExecutionError(
            f"Multi-statement count {allowed} does not match the {count} statement(s) in the "
            "request; use 0 for any number.",
            errno="001003",
            sqlstate="42000",
        )


def _matches_noop(sql: str) -> bool:
    """True when the statement matches a configured no-op regex.

    Lets operators skip statements the emulator does not model (session parameters,
    account-level commands) without hard-failing the client.
    """
    for pattern in settings.noop_regexes:
        try:
            if re.search(pattern, sql, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _lookup_cached_result(handle: str) -> ExecutionResult | None:
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
