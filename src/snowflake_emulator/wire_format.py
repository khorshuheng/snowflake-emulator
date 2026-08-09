"""Encode DuckDB result rows using Snowflake's internal "query-request" wire format.

This format (used by the official ``snowflake-connector-python`` driver, as opposed
to the simpler SQL API v2 JSON body) represents every cell as a string (or ``None``),
using type-specific encodings that the connector's ``SnowflakeConverter`` decodes on
the client side -- e.g. dates as epoch-day integers, timestamps as
``"<epoch-seconds>.<microseconds>"``, and booleans as ``"1"``/``"0"``.
"""

from __future__ import annotations

import datetime
import decimal
import json
from typing import Any

from snowflake_emulator.executor import ColumnMeta
from snowflake_emulator.type_mapping import parse_precision_scale

_EPOCH_DATE = datetime.date(1970, 1, 1)
_EPOCH_DATETIME = datetime.datetime(1970, 1, 1)


def column_wire_metadata(col: ColumnMeta) -> dict[str, Any]:
    """Build the ``rowtype`` entry the connector expects for a single column."""
    precision, scale = parse_precision_scale(col.duckdb_type)
    if col.type in ("timestamp_ntz", "timestamp_tz", "timestamp_ltz"):
        scale = 6
    return {
        "name": col.name,
        "type": col.type,
        "length": None,
        "precision": precision,
        "scale": scale,
        "nullable": True,
    }


def encode_cell(value: Any, sf_type: str, scale: int) -> str | None:
    """Encode a single Python value as the wire-format string for its Snowflake type."""
    if value is None:
        return None

    if sf_type == "boolean":
        return "1" if value else "0"

    if sf_type == "date":
        if isinstance(value, datetime.datetime):
            value = value.date()
        return str((value - _EPOCH_DATE).days)

    if sf_type == "time":
        total_seconds = (
            value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000
        )
        return repr(total_seconds)

    if sf_type in ("timestamp_ntz", "timestamp_tz", "timestamp_ltz"):
        dt = value
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        delta = dt - _EPOCH_DATETIME
        seconds = delta.days * 86400 + delta.seconds
        encoded = f"{seconds}.{delta.microseconds:06d}"
        if sf_type == "timestamp_tz":
            # No real timezone tracking in the emulator; report everything as UTC.
            encoded = f"{encoded} 1440"
        return encoded

    if sf_type == "fixed":
        if scale and scale > 0:
            return str(value)
        return str(int(value))

    if sf_type == "real":
        return repr(float(value))

    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)

    if isinstance(value, decimal.Decimal):
        return str(value)

    return str(value)


def encode_rows(rows: list[list[Any]], columns: list[ColumnMeta]) -> list[list[str | None]]:
    """Encode every cell of every row per its column's Snowflake wire type."""
    scales = [parse_precision_scale(col.duckdb_type)[1] for col in columns]
    encoded: list[list[str | None]] = []
    for row in rows:
        encoded.append(
            [encode_cell(value, col.type, scale) for value, col, scale in zip(row, columns, scales)]
        )
    return encoded
