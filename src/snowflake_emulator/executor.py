"""Execute (translated) Snowflake SQL against the shared DuckDB backend."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlglot import exp

from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.sessions import SessionContext
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
    """Raised when a translated statement fails to execute against DuckDB."""


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


def execute_sql(
    manager: DuckDBManager,
    session: SessionContext,
    raw_sql: str,
    bind_params: list[Any] | None = None,
) -> ExecutionResult:
    """Execute one or more (semicolon-separated) Snowflake statements.

    Session-scoped ``USE ...`` statements update `session` in place instead of being
    sent to DuckDB. Only the result of the final statement is returned, matching
    Snowflake's own multi-statement behaviour. `bind_params`, when given, are only
    applied when `raw_sql` contains a single statement (positional ``?`` binding).
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
        raise ExecutionError(str(exc)) from exc

    row_type: list[ColumnMeta] = []
    rows: list[list[Any]] = []
    message = "Statement executed successfully."

    for statement in statements:
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

        try:
            duckdb_sql = transpile_to_duckdb(statement)
        except TranslationError as exc:
            raise ExecutionError(str(exc)) from exc

        stage_copy = None
        if isinstance(statement, exp.Copy):
            stage_copy = copy_stage_source(statement)

        try:
            if stage_copy is not None:
                table_sql, stage_ref, file_format = stage_copy
                duckdb_sql = build_copy_sql(session, table_sql, stage_ref, file_format)
            if bind_params is not None and len(statements) == 1:
                cursor.execute(duckdb_sql, bind_params)
            else:
                cursor.execute(duckdb_sql)
        except StageError as exc:
            raise ExecutionError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed execution error
            raise ExecutionError(f"{exc}") from exc

        if cursor.description:
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
    )


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
