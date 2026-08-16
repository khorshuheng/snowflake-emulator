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


# Fixed-width DuckDB integer types that Snowflake represents as an untyped ``NUMBER``.
_INTEGER_TYPES = {
    "BIGINT",
    "HUGEINT",
    "INTEGER",
    "SMALLINT",
    "TINYINT",
    "UBIGINT",
    "UINTEGER",
    "USMALLINT",
    "UTINYINT",
}

_SQL_TYPE_MAP: dict[str, str] = {
    "DOUBLE": "FLOAT",
    "FLOAT": "FLOAT",
    "BOOLEAN": "BOOLEAN",
    "VARCHAR": "VARCHAR(16777216)",
    "BLOB": "BINARY",
    "DATE": "DATE",
    "TIME": "TIME(9)",
    "TIMESTAMP": "TIMESTAMP_NTZ(9)",
    "TIMESTAMP WITH TIME ZONE": "TIMESTAMP_TZ(9)",
    "TIMESTAMP_TZ": "TIMESTAMP_TZ(9)",
    "INTERVAL": "VARCHAR(16777216)",
    "JSON": "VARIANT",
}


def duckdb_type_to_snowflake_sql(duckdb_type: str) -> str:
    """Map a DuckDB column type name to the SQL type name Snowflake would report.

    Unlike `duckdb_type_to_snowflake`, which produces the coarse ``rowType`` category
    used in the SQL API's result metadata (``fixed``, ``text``, ...), this returns the
    fully-qualified type name (e.g. ``NUMBER(38,0)``, ``VARCHAR(16777216)``) as it
    would appear in ``DESCRIBE TABLE``/``SHOW COLUMNS`` output.
    """
    normalized = duckdb_type.upper().strip()
    if normalized in _INTEGER_TYPES:
        return "NUMBER(38,0)"
    if normalized.startswith("DECIMAL"):
        precision, scale = parse_precision_scale(normalized)
        if precision is not None:
            return f"NUMBER({precision},{scale})"
        return "NUMBER(38,0)"
    if normalized.startswith("STRUCT") or normalized.startswith("MAP"):
        return "OBJECT"
    if normalized.endswith("[]") or normalized.startswith("LIST") or normalized.startswith("ARRAY"):
        return "ARRAY"
    return _SQL_TYPE_MAP.get(normalized, normalized)
