"""Result types shared by the execution engine and the protocol encoders.

These live outside ``executor`` so the encoding and storage modules
(``arrow``, ``wire_format``, ``statement_store``) do not have to import the
execution orchestrator just to name its result types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ColumnMeta:
    """Metadata for one result column."""

    name: str
    type: str
    duckdb_type: str = ""
    # Declared maximum length of a text column (Snowflake's ``length`` in ``rowType``),
    # which DuckDB's VARCHAR type does not carry.
    length: int | None = None


@dataclass
class FileTransfer:
    """Wire-protocol description of a connector-side PUT/GET file transfer."""

    kind: str  # "UPLOAD" | "DOWNLOAD"
    src_locations: list[str]
    stage_location: str
    local_location: str | None = None
    overwrite: bool = False


@dataclass
class ExecutionResult:
    """One statement's result, independent of the protocol it will be encoded for."""

    handle: str
    created_on: int
    row_type: list[ColumnMeta]
    rows: list[list[Any]]
    message: str = "Statement executed successfully."
    file_transfer: FileTransfer | None = None
    statement_type_id: int = 0
    # The output formats the request asked for, replayed when a later GET fetches the result.
    output_formats: Any | None = None
    number_of_binds: int = 0
