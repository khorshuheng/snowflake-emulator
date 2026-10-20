"""Snowflake-shaped ``DESCRIBE`` handlers and the ``describeOnly`` path."""

from __future__ import annotations

import time
import uuid
from typing import Any

from sqlglot import exp

from snowflake_emulator.catalog import column_type_resolver
from snowflake_emulator.core.constants import INFORMATION_SCHEMA_SCHEMA
from snowflake_emulator.core.errors import ExecutionError
from snowflake_emulator.core.results import ColumnMeta, ExecutionResult
from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.engine.handlers.common import (
    prefer_declared_type,
    result_columns,
    temp_inclusive_scope,
)
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.statement_types import number_of_binds, statement_type_id
from snowflake_emulator.translator import (
    TranslationError,
    normalize_function_signature,
    split_statements,
    transpile_to_duckdb,
)
from snowflake_emulator.type_mapping import (
    duckdb_type_to_snowflake,
    duckdb_type_to_snowflake_sql_type,
)
from snowflake_emulator.user_functions import query_user_functions


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

# ``DESCRIBE FUNCTION`` adds the signature bounds a caller needs to tell overloads
# apart. Snowflake reports both as the number of declared arguments.
_USER_FUNCTION_DESCRIBE_SQL = """
    SELECT
        NULL::TIMESTAMP AS created_on,
        NULL::VARCHAR AS name,
        NULL::VARCHAR AS schema_name,
        NULL::VARCHAR AS catalog_name,
        NULL::VARCHAR AS is_builtin,
        NULL::VARCHAR AS is_aggregate,
        NULL::VARCHAR AS is_ansi,
        NULL::INTEGER AS min_num_arguments,
        NULL::INTEGER AS max_num_arguments,
        NULL::VARCHAR AS arguments,
        NULL::VARCHAR AS description,
        NULL::VARCHAR AS language,
        NULL::VARCHAR AS body
    WHERE FALSE
"""


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
    """Describe a statement's result columns without executing it."""
    with manager.session_scope(session.token):
        return _describe_sql(manager, session, raw_sql)


def _describe_sql(
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
    cursor = manager.cursor(session.token)
    cursor.execute(f'USE "{session.database}"."{session.schema}"')
    try:
        duckdb_sql = transpile_to_duckdb(
            statement, column_types=column_type_resolver(cursor, session)
        )
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


def execute_function_describe(
    manager: DuckDBManager,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Execute ``DESCRIBE FUNCTION`` from the recorded user functions.

    An omitted argument list lists every overload, matching Snowflake's behaviour for a
    function whose name is unambiguous. Returns ``None`` for other DESCRIBE forms.
    """
    if not (
        isinstance(statement, exp.Describe)
        and str(statement.args.get("kind") or "").upper() == "FUNCTION"
    ):
        return None
    this = statement.args.get("this")
    types: list[exp.Expression] = []
    if isinstance(this, exp.Schema):
        types = list(this.expressions)
        this = this.this
    if not isinstance(this, exp.Table):
        return None

    functions = query_user_functions(
        manager,
        session.database,
        session.schema,
        name=this.name,
        signature=normalize_function_signature(types) if types else None,
    )
    if not functions:
        raise ExecutionError(f"Function {this.name.upper()} does not exist.")

    cursor = manager.cursor(session.token)
    cursor.execute(_USER_FUNCTION_DESCRIBE_SQL)
    row_type = result_columns(cursor)[0]
    rows = [
        [
            fn.created_on,
            fn.name,
            fn.schema,
            fn.database,
            "N",  # is_builtin
            "N",  # is_aggregate: aggregate UDFs are not emulated
            "N",  # is_ansi
            fn.arity,
            fn.arity,
            fn.describe_signature,
            None,  # description
            fn.language,
            fn.body,
        ]
        for fn in functions
    ]
    return row_type, rows


def execute_snowflake_describe(
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
            c.column_name AS "name",
            c.data_type AS "type",
            e.declared_type AS "declared_type",
            'COLUMN' AS "kind",
            CASE WHEN c.is_nullable THEN 'Y' ELSE 'N' END AS "null?",
            c.column_default AS "default",
            NULL::VARCHAR AS "primary key",
            NULL::VARCHAR AS "unique key",
            NULL::VARCHAR AS "check",
            NULL::VARCHAR AS "expression",
            c.comment AS "comment",
            NULL::VARCHAR AS "policy name"
        FROM duckdb_columns() c
        LEFT JOIN "{database}".{INFORMATION_SCHEMA_SCHEMA}._sf_columns_ext e
          ON lower(e.table_catalog) = lower(c.database_name)
         AND lower(e.table_schema) = lower(c.schema_name)
         AND lower(e.table_name) = lower(c.table_name)
         AND lower(e.column_name) = lower(c.column_name)
         AND lower(c.database_name) <> 'temp'
        WHERE {temp_inclusive_scope('c', database, schema)}
          AND lower(c.table_name) = lower('{table_name}')
        ORDER BY c.column_index
        """
    )
    row_type, rows = result_columns(cursor)
    # Drop the helper "declared_type" column (index 2) from metadata and rows, and use
    # the declared Snowflake type when present, else map the DuckDB type.
    row_type = [row_type[0], row_type[1], *row_type[3:]]
    converted: list[list[Any]] = []
    for row in rows:
        sf_type = prefer_declared_type(row[2], row[1]) or duckdb_type_to_snowflake_sql_type(
            str(row[1])
        )
        converted.append([row[0], sf_type, *row[3:]])
    return row_type, converted
