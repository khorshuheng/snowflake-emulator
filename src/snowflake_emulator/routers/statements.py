"""The core Snowflake SQL API v2 surface: ``/api/v2/statements``."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, status

from snowflake_emulator.database import get_manager
from snowflake_emulator.executor import ExecutionError, ExecutionResult, execute_sql
from snowflake_emulator.schemas import ResultSetMetaData, RowType, StatementRequest, StatementResponse
from snowflake_emulator.sessions import SessionContext, get_session_manager
from snowflake_emulator.statement_store import get_statement_store

router = APIRouter(prefix="/api/v2/statements", tags=["statements"])


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
        data=result.rows,
    )


@router.post("", status_code=status.HTTP_200_OK, response_model=StatementResponse)
def submit_statement(
    body: StatementRequest,
    authorization: str | None = Header(default=None),
) -> StatementResponse:
    token = _extract_token(authorization)
    session = get_session_manager().get_or_create(token)
    _apply_overrides(session, body)

    manager = get_manager()
    try:
        result = execute_sql(manager, session, body.statement)
    except ExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"message": str(exc), "sqlState": "42000", "code": "100132"},
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface any bug as a Snowflake-shaped error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"message": str(exc), "sqlState": "42000", "code": "100132"},
        ) from exc

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
