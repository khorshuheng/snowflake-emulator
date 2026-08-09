"""Map DuckDB column types to Snowflake's SQL API ``rowType`` type names."""

from __future__ import annotations

import re

_TYPE_MAP: dict[str, str] = {
    "BIGINT": "fixed",
    "HUGEINT": "fixed",
    "INTEGER": "fixed",
    "SMALLINT": "fixed",
    "TINYINT": "fixed",
    "UBIGINT": "fixed",
    "UINTEGER": "fixed",
    "USMALLINT": "fixed",
    "UTINYINT": "fixed",
    "DOUBLE": "real",
    "FLOAT": "real",
    "DECIMAL": "fixed",
    "BOOLEAN": "boolean",
    "VARCHAR": "text",
    "BLOB": "binary",
    "DATE": "date",
    "TIME": "time",
    "TIMESTAMP": "timestamp_ntz",
    "TIMESTAMP WITH TIME ZONE": "timestamp_tz",
    "TIMESTAMP_TZ": "timestamp_tz",
    "INTERVAL": "text",
    "JSON": "variant",
}


def duckdb_type_to_snowflake(duckdb_type: str) -> str:
    """Best-effort mapping of a DuckDB type name to a Snowflake SQL API type string."""
    normalized = duckdb_type.upper()
    if normalized.startswith("DECIMAL"):
        return "fixed"
    if normalized.startswith("STRUCT") or normalized.startswith("MAP"):
        return "object"
    if normalized.endswith("[]") or normalized.startswith("LIST") or normalized.startswith("ARRAY"):
        return "array"
    return _TYPE_MAP.get(normalized, "text")


_DECIMAL_RE = re.compile(r"^DECIMAL\((\d+),\s*(\d+)\)$")


def parse_precision_scale(duckdb_type: str) -> tuple[int | None, int]:
    """Extract ``(precision, scale)`` from a DuckDB type name, e.g. ``DECIMAL(10,2)``.

    Non-decimal types default to ``(None, 0)``.
    """
    m = _DECIMAL_RE.match(duckdb_type.upper())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, 0
