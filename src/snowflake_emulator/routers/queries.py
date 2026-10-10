"""Snowflake's internal wire-protocol query endpoints, used by the official
``snowflake-connector-python`` driver (as opposed to the simpler SQL API v2)."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Header

from snowflake_emulator import statement_types
from snowflake_emulator.bindings import decode_bindings
from snowflake_emulator.database import get_manager
from snowflake_emulator.errors import error_code_pair
from snowflake_emulator.executor import describe_sql, execute_sql
from snowflake_emulator.routers.auth import _extract_token
from snowflake_emulator.sessions import get_session_manager
from snowflake_emulator.statement_store import get_statement_store
from snowflake_emulator.wire_format import column_wire_metadata, encode_rows

router = APIRouter(prefix="/queries", tags=["wire-protocol"])
monitoring_router = APIRouter(tags=["monitoring"])


def _final_names(session) -> dict:
    """Session context echoed on every query response, as Snowflake does."""
    return {
        "finalDatabaseName": session.database,
        "finalSchemaName": session.schema,
        "finalWarehouseName": session.warehouse,
        "finalRoleName": session.role,
    }


def _describe_only_response(session, result) -> dict:
    """Answer a ``describeOnly`` request with metadata and no rows."""
    return {
        "success": True,
        "message": None,
        "data": {
            "parameters": [],
            "rowtype": [column_wire_metadata(col) for col in result.row_type],
            "rowset": [],
            "total": 0,
            "returned": 0,
            "queryId": result.handle,
            "sqlState": "00000",
            "queryResultFormat": "json",
            "statementTypeId": result.statement_type_id,
            "numberOfBinds": result.number_of_binds,
            "arrayBindSupported": result.statement_type_id == statement_types.INSERT,
            **_final_names(session),
        },
    }


def _error_response(exc: BaseException) -> dict:
    """Build a Snowflake-shaped failure payload carrying the classified errno/sqlState."""
    errno, sqlstate = error_code_pair(exc)
    return {
        "success": False,
        "message": str(exc),
        "code": errno,
        "data": {"errorCode": errno, "sqlState": sqlstate, "queryId": str(uuid.uuid4())},
    }


@router.post("/v1/query-request")
def query_request(
    body: dict[str, Any],
    requestId: str | None = None,
    authorization: str | None = Header(default=None),
) -> dict:
    token = _extract_token(authorization)
    session = get_session_manager().get_or_create(token)

    sql_text: str = body.get("sqlText", "")

    manager = get_manager(session)
    try:
        if body.get("describeOnly"):
            described = describe_sql(manager, session, sql_text)
            if described is not None:
                session.last_query_id = described.handle
                get_statement_store().put(described)
                return _describe_only_response(session, described)
        decoded = decode_bindings(body.get("bindings"))
        result = execute_sql(
            manager,
            session,
            sql_text,
            bind_params=decoded.positional if decoded.positional is not None else decoded.named,
            bind_batch=decoded.batch,
        )
    except Exception as exc:  # noqa: BLE001 - always answer in Snowflake's error shape
        # Without this, an unexpected exception (e.g. an invalid identifier) escapes
        # as an HTTP 500 and clients retry indefinitely instead of failing fast.
        return _error_response(exc)

    session.last_query_id = result.handle
    get_statement_store().put(result)

    if result.file_transfer is not None:
        # PUT/GET: hand the connector a local storage location and let its
        # file-transfer agent copy the files itself (SnowflakeLocalStorageClient).
        transfer = result.file_transfer
        data: dict[str, Any] = {
            "command": transfer.kind,
            "queryId": result.handle,
            "sqlState": "00000",
            "src_locations": transfer.src_locations,
            "stageInfo": {
                "locationType": "LOCAL_FS",
                "location": transfer.stage_location,
                "creds": {},
            },
            "parallel": 4,
            "overwrite": transfer.overwrite,
            "autoCompress": False,
            "sourceCompression": "auto_detect",
            "threshold": 67108864,
        }
        if transfer.kind == "DOWNLOAD" and transfer.local_location:
            data["localLocation"] = transfer.local_location
        return {"success": True, "message": None, "data": data}

    rowtype = [column_wire_metadata(col) for col in result.row_type]
    rowset = encode_rows(result.rows, result.row_type)

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
            "statementTypeId": result.statement_type_id,
            **_final_names(session),
        },
    }


@monitoring_router.get("/monitoring/queries/{sfqid}")
def monitoring_query(sfqid: str) -> dict:
    """Statement-status endpoint the connector polls for an in-flight query.

    Statements execute synchronously, so a known query is always already done.
    Intentionally unauthenticated: the emulator is a local development server, not a
    shared service, so this read-only status endpoint requires no session token.
    """
    known = get_statement_store().get(sfqid) is not None
    return {
        "data": {"queries": [{"status": "SUCCESS"}] if known else []},
        "success": True,
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
