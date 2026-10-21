"""The core Snowflake SQL API v2 surface: ``/api/v2/statements``."""

from __future__ import annotations

import json
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse

from snowflake_emulator.bindings import decode_bindings
from snowflake_emulator.core.results import ColumnMeta, ExecutionResult
from snowflake_emulator.database import get_manager
from snowflake_emulator.core.errors import error_code_pair
from snowflake_emulator.executor import execute_sql
from snowflake_emulator.output_formats import (
    OutputFormatError,
    OutputFormats,
    format_cell,
    output_formats_from_parameters,
)
from snowflake_emulator.schemas import (
    PartitionInfo,
    ResultSetMetaData,
    RowType,
    StatementRequest,
    StatementResponse,
)
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


def _request_parameters(parameters: dict[str, Any] | None) -> tuple[int | None, str | None, OutputFormats]:
    """Validate the request's ``parameters`` object.

    Returns ``(multi_statement_count, timezone, output_formats)``. Only the parameters the
    emulator honours are read; an invalid value is a request error, never a silent default.
    """
    if not parameters:
        return None, None, OutputFormats()
    upper = {str(key).upper(): value for key, value in parameters.items()}

    count: int | None = None
    if "MULTI_STATEMENT_COUNT" in upper:
        raw = upper["MULTI_STATEMENT_COUNT"]
        try:
            count = int(raw)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": f"MULTI_STATEMENT_COUNT must be an integer, got '{raw}'.",
                    "sqlState": "42000",
                    "code": "001003",
                },
            ) from None
        if count < 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": "MULTI_STATEMENT_COUNT must not be negative.",
                    "sqlState": "42000",
                    "code": "001003",
                },
            )

    timezone = upper.get("TIMEZONE")
    if timezone is not None:
        timezone = str(timezone)
        if not _is_valid_timezone(timezone):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": f"Invalid TIMEZONE '{timezone}'.",
                    "sqlState": "22023",
                    "code": "100177",
                },
            )

    for name in (
        "DATE_OUTPUT_FORMAT",
        "TIME_OUTPUT_FORMAT",
        "TIMESTAMP_OUTPUT_FORMAT",
        "BINARY_OUTPUT_FORMAT",
    ):
        value = upper.get(name)
        if value is not None and not isinstance(value, str):
            # A number or a list would be coerced into a pattern ("5" renders the year 5),
            # answer with wrong text instead of an error.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": f"{name} must be a string, got {type(value).__name__}.",
                    "sqlState": "22023",
                    "code": "100177",
                },
            )

    try:
        formats = output_formats_from_parameters(parameters)
    except OutputFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"message": str(exc), "sqlState": "22023", "code": "100177"},
        ) from exc
    return count, timezone, formats


def _is_valid_timezone(name: str) -> bool:
    """True for an IANA zone name, the only form DuckDB's ``SET TimeZone`` accepts.

    A numeric offset (``+05:00``) is rejected here rather than passed on, because DuckDB
    answers ``Unknown TimeZone`` and the request would fail with a confusing message.
    """
    try:
        ZoneInfo(name)
    except Exception:  # noqa: BLE001 - any failure means the name is unknown
        return False
    return True


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


def _to_response(
    result: ExecutionResult, partition: int | None = None
) -> StatementResponse:
    formats = result.output_formats or OutputFormats()
    # The emulator materializes the whole result as one partition. A request for a
    # later partition still gets the correct metadata, with an empty rowset.
    rows = result.rows if partition in (None, 0) else []
    return StatementResponse(
        statementHandle=result.handle,
        statementStatusUrl=f"/api/v2/statements/{result.handle}",
        message=result.message,
        createdOn=result.created_on,
        resultSetMetaData=ResultSetMetaData(
            numRows=len(rows),
            rowType=[RowType(name=col.name, type=col.type) for col in result.row_type],
            partitionInfo=[PartitionInfo(rowCount=len(result.rows))],
        ),
        data=[
            [
                _json_cell(format_cell(value, col, formats), col)
                for value, col in zip(row, result.row_type)
            ]
            for row in rows
        ],
    )


@router.post("", status_code=status.HTTP_200_OK, response_model=StatementResponse)
def submit_statement(
    body: StatementRequest,
    authorization: str | None = Header(default=None),
    async_: bool = Query(default=False, alias="async"),
):
    token = _extract_token(authorization)
    session = get_session_manager().get_or_create(token)
    _apply_overrides(session, body)

    manager = get_manager(session)
    count, timezone, formats = _request_parameters(body.parameters)
    try:
        decoded = decode_bindings(body.bindings)
        result = execute_sql(
            manager,
            session,
            body.statement,
            bind_params=decoded.positional if decoded.positional is not None else decoded.named,
            bind_batch=decoded.batch,
            multi_statement_count=1 if count is None else count,
            timezone=timezone,
        )
    except Exception as exc:  # noqa: BLE001 - always answer in Snowflake's error shape
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_error_detail(exc),
        ) from exc

    session.last_query_id = result.handle
    result.output_formats = formats
    get_statement_store().put(result)
    if async_:
        # Execution is synchronous, but the SQL API contract is to return 202 and let
        # the client poll the handle.
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "statementHandle": result.handle,
                "statementStatusUrl": f"/api/v2/statements/{result.handle}",
                "message": result.message,
                "sqlState": "00000",
                "code": "090001",
                "createdOn": result.created_on,
            },
        )
    return _to_response(result)


@router.get("/{handle}", response_model=StatementResponse)
def get_statement_result(
    handle: str, partition: int | None = Query(default=None, ge=0)
) -> StatementResponse:
    result = get_statement_store().get(handle)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": f"Statement handle not found: {handle}", "sqlState": "02000"},
        )
    return _to_response(result, partition)


@router.post("/{handle}/cancel")
def cancel_statement(handle: str) -> dict:
    # Statements execute synchronously and complete before this endpoint could ever
    # be called, so cancellation is always a no-op success (mirrors real behaviour
    # for already-finished statements).
    return {"success": True, "message": f"Statement {handle} is already complete."}
