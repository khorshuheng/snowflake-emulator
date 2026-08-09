"""Pydantic request/response models mirroring the Snowflake SQL API v2 shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StatementRequest(BaseModel):
    """Body of ``POST /api/v2/statements``."""

    statement: str
    database: str | None = None
    schema_: str | None = Field(default=None, alias="schema")
    warehouse: str | None = None
    role: str | None = None
    timeout: int | None = None
    bindings: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}


class RowType(BaseModel):
    """Metadata describing a single result column."""

    name: str
    type: str
    nullable: bool = True
    length: int | None = None
    precision: int | None = None
    scale: int | None = None


class ResultSetMetaData(BaseModel):
    numRows: int
    format: str = "jsonv2"
    rowType: list[RowType]


class StatementResponse(BaseModel):
    """Body returned by both the synchronous execute call and the status/result poll."""

    statementHandle: str
    statementStatusUrl: str
    message: str
    sqlState: str = "00000"
    code: str = "090001"
    createdOn: int
    resultSetMetaData: ResultSetMetaData
    data: list[list[Any]]


class ErrorResponse(BaseModel):
    message: str
    sqlState: str = "42000"
    code: str = "100132"
    statementHandle: str | None = None
