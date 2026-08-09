"""Auth-adjacent endpoints: obtain and discard a bearer token for a session.

``/session/v1/login-request`` and ``/session?delete=true`` also implement the shapes
expected by the real ``snowflake-connector-python`` driver (which talks Snowflake's
internal wire protocol, distinct from the simpler SQL API v2), so the same emulator
can back both integration styles.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Header

from snowflake_emulator.sessions import SessionContext, get_session_manager

router = APIRouter(tags=["session"])


def _extract_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.startswith("Bearer "):
        return authorization[len("Bearer ") :].strip()
    if authorization.startswith("Snowflake Token="):
        return authorization[len("Snowflake Token=") :].strip().strip('"')
    return authorization.strip()


def _apply_query_overrides(
    session: SessionContext,
    database: str | None,
    schema: str | None,
    warehouse: str | None,
    role: str | None,
) -> None:
    if database:
        session.database = database
    if schema:
        session.schema = schema
    if warehouse:
        session.warehouse = warehouse
    if role:
        session.role = role


@router.post("/session/v1/login-request")
async def login(
    databaseName: str | None = None,
    schemaName: str | None = None,
    warehouse: str | None = None,
    roleName: str | None = None,
) -> dict:
    """Mimic Snowflake's login endpoint: always succeeds and issues a fresh token.

    Real Snowflake credential material is not validated -- this emulator is intended
    for local development and testing only. The response covers both what the SQL
    API v2 client and the ``snowflake-connector-python`` driver expect to see.
    """
    session = get_session_manager().create()
    _apply_query_overrides(session, databaseName, schemaName, warehouse, roleName)

    return {
        "success": True,
        "message": None,
        "data": {
            "token": session.token,
            "masterToken": session.token,
            "sessionId": abs(hash(session.token)) % 10_000_000,
            "serverVersion": "snowflake-emulator",
            "sessionInfo": {
                "databaseName": session.database,
                "schemaName": session.schema,
                "warehouseName": session.warehouse,
                "roleName": session.role,
            },
            "parameters": [
                {"name": "AUTOCOMMIT", "value": True},
                {"name": "TIMEZONE", "value": "UTC"},
                {"name": "CLIENT_PREFETCH_THREADS", "value": 4},
            ],
        },
    }


@router.post("/session")
def close_session(delete: bool = False, authorization: str | None = Header(default=None)) -> dict:
    """Handles ``POST /session?delete=true``, used by the connector to log out."""
    if not delete:
        return {"success": True}
    token = _extract_token(authorization)
    closed = get_session_manager().close(token) if token else False
    return {"success": True, "data": {"closed": closed}}


@router.post("/session/logout-request")
def logout(token: str) -> dict:
    closed = get_session_manager().close(token)
    return {"success": closed}


@router.post("/session/token")
def issue_token() -> dict:
    """Convenience endpoint (non-Snowflake) for quickly grabbing a bearer token."""
    session = get_session_manager().create()
    return {"token": session.token, "createdOn": int(time.time() * 1000)}
