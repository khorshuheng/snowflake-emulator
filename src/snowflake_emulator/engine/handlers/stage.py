"""Handlers for stage commands and stage DDL.

``PUT``/``GET``/``LIST``/``REMOVE`` and stage DDL do not touch DuckDB: they operate on
the local stage tree and build the result shapes Snowflake clients expect.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from sqlglot import exp

from snowflake_emulator.core.errors import ExecutionError
from snowflake_emulator.core.results import ColumnMeta, ExecutionResult, FileTransfer
from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.engine.handlers.common import name_matches, show_name_patterns
from snowflake_emulator.engine.handlers.show import show_limit, show_scope
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.stages import (
    StageError,
    alter_stage,
    create_stage,
    drop_stage,
    get_stage_manager,
    parse_get,
    parse_list,
    parse_put,
    parse_remove,
    show_stages,
)


def execute_stage_command(
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
                    f.source_size or f.size,
                    f.size,
                    f.source_compression,
                    f.target_compression,
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
                    auto_compress=put.options.get("AUTO_COMPRESS", "TRUE").upper()
                    not in ("FALSE", "0", "NO", "OFF"),
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
            rows = [[f.rel_path, f.size, f.status, f.message] for f in files]
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
            rows = [[f.rel_path, f.size, f.md5, f.last_modified] for f in files]
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


def execute_stage_ddl(
    manager: DuckDBManager,
    session: SessionContext,
    statement: exp.Expression,
    kind: str,
) -> tuple[list[ColumnMeta], list[list[Any]]]:
    """Handle CREATE/DROP/ALTER STAGE and SHOW STAGES without DuckDB."""
    stages = get_stage_manager()
    try:
        if kind == "CREATE_STAGE":
            create_stage(manager, stages, session, statement)
            return (
                [ColumnMeta(name="status", type="text")],
                [[f"Stage {statement.this.name} successfully created."]],
            )
        if kind == "DROP_STAGE":
            name = drop_stage(manager, stages, session, statement)
            return (
                [ColumnMeta(name="status", type="text")],
                [[f"Stage {name} successfully dropped."]],
            )
        if kind == "ALTER_STAGE":
            name = alter_stage(manager, session, statement)
            return (
                [ColumnMeta(name="status", type="text")],
                [[f"Stage {name} successfully altered."]],
            )
        if kind == "SHOW_STAGES":
            scope = show_scope(statement, session) if isinstance(statement, exp.Show) else None
            database, schema = scope or (session.database, session.schema)
            rows = [
                [name, db, schema_name, url]
                for name, db, schema_name, url in show_stages(
                    manager, stages, session, database, schema
                )
            ]
            prefix, like = show_name_patterns(statement)
            if prefix or like:
                rows = [row for row in rows if name_matches(str(row[0]), prefix, like)]
            limit = show_limit(statement)
            if limit is not None:
                rows = rows[:limit]
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
