"""The core Snowflake SQL API v2 surface: ``/api/v2/statements``."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Header, HTTPException, status

from snowflake_emulator.bindings import decode_bindings
from snowflake_emulator.core.results import ColumnMeta, ExecutionResult
from snowflake_emulator.database import get_manager
from snowflake_emulator.core.errors import error_code_pair
from snowflake_emulator.executor import execute_sql
from snowflake_emulator.schemas import ResultSetMetaData, RowType, StatementRequest, StatementResponse
from snowflake_emulator.sessions import SessionContext, get_session_manager
from snowflake_emulator.statement_store import get_statement_store

router = APIRouter(prefix="/api/v2/statements", tags=["statements"])


def _error_detail(exc: BaseException) -> dict:
    """Build the SQL API v2 error detail with the classified errno/sqlState."""
    errno, sqlstate = error_code_pair(exc)
    return {"message": str(exc), "sqlState": sqlstate, "code": errno}


def _extract_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    if authorization.startswith(prefix):
        return authorization[len(prefix) :].strip()
    return authorization.strip()


def _apply_overrides(session: SessionContext, body: StatementRequest) -> None:
    if body.database:
        session.database = body.database
    if body.schema_:
        session.schema = body.schema_
    if body.warehouse:
        session.warehouse = body.warehouse
    if body.role:
        session.role = body.role


def _json_element(element: Any) -> Any:
    """Decode one element of a ``JSON[]`` cell, recursing into a nested list."""
    if isinstance(element, list):
        return [_json_element(inner) for inner in element]
    if isinstance(element, str):
        try:
            return json.loads(element)
        except json.JSONDecodeError:
            return element
    return element


def _is_json_column(col: ColumnMeta) -> bool:
    """True for a cell DuckDB typed as JSON, i.e. whose text is JSON-serialized.

    Keyed on the DuckDB type rather than the Snowflake one, which both a JSON and a VARIANT
    cell report as ``variant``: a DuckDB ``VARIANT`` cell is already a plain Python value, so
    its text is *not* JSON-serialized and parsing it would read the string ``'42'`` as the
    number 42.
    """
    return (col.duckdb_type or "").upper() == "JSON"


def _json_cell(value: Any, col: ColumnMeta) -> Any:
    """Inline a semi-structured cell as a JSON value, as Snowflake's rowset does.

    DuckDB hands a JSON value to Python as its JSON *text*, so the rowset would otherwise
    carry a quoted JSON string (``'"y"'`` where Snowflake embeds ``y``), a number as a
    string, and each element of a ``JSON[]`` cell quoted. The connector's wire protocol is
    untouched: it represents every cell as a string, which is what it decodes.
    """
    if value is None:
        return None
    if _is_json_column(col) and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, list) and "JSON" in (col.duckdb_type or "").upper():
        return [_json_element(element) for element in value]
    return value


def _to_response(result: ExecutionResult) -> StatementResponse:
    return StatementResponse(
        statementHandle=result.handle,
        statementStatusUrl=f"/api/v2/statements/{result.handle}",
        message=result.message,
        createdOn=result.created_on,
        resultSetMetaData=ResultSetMetaData(
            numRows=len(result.rows),
            rowType=[RowType(name=col.name, type=col.type) for col in result.row_type],
        ),
        data=[
            [_json_cell(value, col) for value, col in zip(row, result.row_type)]
            for row in result.rows
        ],
    )


@router.post("", status_code=status.HTTP_200_OK, response_model=StatementResponse)
def submit_statement(
    body: StatementRequest,
    authorization: str | None = Header(default=None),
) -> StatementResponse:
    token = _extract_token(authorization)
    session = get_session_manager().get_or_create(token)
    _apply_overrides(session, body)

    manager = get_manager(session)
    try:
        decoded = decode_bindings(body.bindings)
        result = execute_sql(
            manager,
            session,
            body.statement,
            bind_params=decoded.positional if decoded.positional is not None else decoded.named,
            bind_batch=decoded.batch,
        )
    except Exception as exc:  # noqa: BLE001 - always answer in Snowflake's error shape
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_error_detail(exc),
        ) from exc

    session.last_query_id = result.handle
    get_statement_store().put(result)
    return _to_response(result)


@router.get("/{handle}", response_model=StatementResponse)
def get_statement_result(handle: str) -> StatementResponse:
    result = get_statement_store().get(handle)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": f"Statement handle not found: {handle}", "sqlState": "02000"},
        )
    return _to_response(result)


@router.post("/{handle}/cancel")
def cancel_statement(handle: str) -> dict:
    # Statements execute synchronously and complete before this endpoint could ever
    # be called, so cancellation is always a no-op success (mirrors real behaviour
    # for already-finished statements).
    return {"success": True, "message": f"Statement {handle} is already complete."}
