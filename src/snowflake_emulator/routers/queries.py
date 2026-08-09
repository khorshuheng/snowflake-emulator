"""Snowflake's internal wire-protocol query endpoints, used by the official
``snowflake-connector-python`` driver (as opposed to the simpler SQL API v2)."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Header

from snowflake_emulator.database import get_manager
from snowflake_emulator.executor import ExecutionError, execute_sql
from snowflake_emulator.routers.auth import _extract_token
from snowflake_emulator.sessions import get_session_manager
from snowflake_emulator.statement_store import get_statement_store
from snowflake_emulator.wire_format import column_wire_metadata, encode_rows

router = APIRouter(prefix="/queries", tags=["wire-protocol"])


def _extract_bind_params(bindings: dict[str, Any] | None) -> list[Any] | None:
    """Convert the connector's ``{"1": {"type": ..., "value": ...}}`` bindings into a
    positional list suitable for DuckDB's ``?`` parameter substitution."""
    if not bindings:
        return None
    ordered_keys = sorted(bindings, key=lambda k: int(k))
    return [bindings[k].get("value") for k in ordered_keys]


@router.post("/v1/query-request")
def query_request(
    body: dict[str, Any],
    requestId: str | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    token = _extract_token(authorization)
    session = get_session_manager().get_or_create(token)

    sql_text: str = body.get("sqlText", "")
    bind_params = _extract_bind_params(body.get("bindings"))

    manager = get_manager()
    try:
        result = execute_sql(manager, session, sql_text, bind_params=bind_params)
    except ExecutionError as exc:
        return {
            "success": False,
            "message": str(exc),
            "code": "100132",
            "data": {"sqlState": "42000", "queryId": str(uuid.uuid4())},
        }

    get_statement_store().put(result)

    rowtype = [column_wire_metadata(col) for col in result.row_type]
    rowset = encode_rows(result.rows, result.row_type)

    is_use_statement = sql_text.strip().upper().startswith("USE ")

    return {
        "success": True,
        "message": None,
        "data": {
            "parameters": [],
            "rowtype": rowtype,
            "rowset": rowset,
            "total": len(rowset),
            "returned": len(rowset),
            "queryId": result.handle,
            "sqlState": "00000",
            "queryResultFormat": "json",
            "finalDatabaseName": session.database if is_use_statement else None,
            "finalSchemaName": session.schema if is_use_statement else None,
            "finalWarehouseName": session.warehouse if is_use_statement else None,
            "finalRoleName": session.role if is_use_statement else None,
        },
    }


@router.post("/v1/abort-request")
def abort_request(body: dict[str, Any] | None = None) -> dict:
    # Statements execute synchronously and complete before an abort could ever be
    # dispatched, so this is always a no-op success.
    return {"success": True, "data": {}}


@router.get("/{query_id}/result")
def get_query_result(query_id: str) -> dict:
    result = get_statement_store().get(query_id)
    if result is None:
        return {"success": False, "message": f"Query not found: {query_id}", "data": {}}

    rowtype = [column_wire_metadata(col) for col in result.row_type]
    rowset = encode_rows(result.rows, result.row_type)
    return {
        "success": True,
        "message": None,
        "data": {
            "rowtype": rowtype,
            "rowset": rowset,
            "total": len(rowset),
            "returned": len(rowset),
            "queryId": result.handle,
            "sqlState": "00000",
            "queryResultFormat": "json",
        },
    }
