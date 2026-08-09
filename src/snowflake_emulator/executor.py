"""Execute (translated) Snowflake SQL against the shared DuckDB backend."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.translator import (
    TranslationError,
    UseStatement,
    is_query,
    match_use_statement,
    split_statements,
    transpile_to_duckdb,
)
from snowflake_emulator.type_mapping import duckdb_type_to_snowflake


class ExecutionError(RuntimeError):
    """Raised when a translated statement fails to execute against DuckDB."""


@dataclass
class ColumnMeta:
    name: str
    type: str
    duckdb_type: str = ""


@dataclass
class ExecutionResult:
    handle: str
    created_on: int
    row_type: list[ColumnMeta]
    rows: list[list[Any]]
    message: str = "Statement executed successfully."


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

    try:
        statements = split_statements(raw_sql)
    except TranslationError as exc:
        raise ExecutionError(str(exc)) from exc

    row_type: list[ColumnMeta] = []
    rows: list[list[Any]] = []
    message = "Statement executed successfully."

    for statement in statements:
        use_stmt = match_use_statement(statement.sql(dialect="snowflake"))
        if use_stmt is not None:
            _apply_use_statement(manager, session, cursor, use_stmt)
            row_type, rows = [], []
            message = "Statement executed successfully."
            continue

        try:
            duckdb_sql = transpile_to_duckdb(statement)
        except TranslationError as exc:
            raise ExecutionError(str(exc)) from exc

        try:
            if bind_params is not None and len(statements) == 1:
                cursor.execute(duckdb_sql, bind_params)
            else:
                cursor.execute(duckdb_sql)
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
