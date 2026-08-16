"""Map DuckDB column types to Snowflake's SQL API ``rowType`` type names."""

from __future__ import annotations

import re

# Exact-match table for scalar DuckDB types with a direct Snowflake equivalent.
_TYPE_MAP: dict[str, str] = {
    # Integers -> Snowflake NUMBER (reported as ``fixed`` in rowType).
    "BIGINT": "fixed",
    "HUGEINT": "fixed",
    "INTEGER": "fixed",
    "SMALLINT": "fixed",
    "TINYINT": "fixed",
    "UBIGINT": "fixed",
    "UHUGEINT": "fixed",
    "UINTEGER": "fixed",
    "USMALLINT": "fixed",
    "UTINYINT": "fixed",
    # Floating point.
    "DOUBLE": "real",
    "FLOAT": "real",
    "BOOLEAN": "boolean",
    # Strings / binary.
    "VARCHAR": "text",
    "BLOB": "binary",
    "BIT": "binary",
    # Temporal. DuckDB's TIMESTAMP_S/MS/NS are all timestamp-without-tz aliases.
    "DATE": "date",
    "TIME": "time",
    "TIME WITH TIME ZONE": "time",
    "TIMETZ": "time",
    "TIMESTAMP": "timestamp_ntz",
    "TIMESTAMP_S": "timestamp_ntz",
    "TIMESTAMP_MS": "timestamp_ntz",
    "TIMESTAMP_NS": "timestamp_ntz",
    "TIMESTAMP WITH TIME ZONE": "timestamp_tz",
    "TIMESTAMP_TZ": "timestamp_tz",
    "TIMESTAMPTZ": "timestamp_tz",
    "INTERVAL": "text",
    # Semi-structured.
    "JSON": "variant",
    # Snowflake has no native UUID/enum/union types; text is the closest match.
    "UUID": "text",
    "NULL": "text",
}

# Fixed-size DuckDB arrays are reported as e.g. ``FLOAT[2]`` / ``INTEGER[2]``
# (as opposed to variable-length ``FLOAT[]`` which is already caught by the
# ``endswith("[]")`` check below).
_FIXED_ARRAY_RE = re.compile(r"^.+\[\d+\]$")


def duckdb_type_to_snowflake(duckdb_type: str) -> str:
    """Best-effort mapping of a DuckDB type name to a Snowflake SQL API type string."""
    normalized = duckdb_type.strip().upper()
    # Array detection comes first: DECIMAL(10,2)[] is an array of decimals.
    if normalized.startswith(("LIST", "ARRAY")):
        return "array"
    if normalized.endswith("[]") or _FIXED_ARRAY_RE.match(normalized):
        return "array"
    if normalized.startswith("DECIMAL"):
        return "fixed"
    if normalized.startswith(("STRUCT", "MAP", "UNION")):
        return "object"
    if normalized.startswith("ENUM"):
        return "text"
    return _TYPE_MAP.get(normalized, "text")


# DuckDB type name -> Snowflake SQL data type name, as shown in the ``type`` column
# of ``DESCRIBE TABLE`` output. Unlike the SQL API ``rowType`` names above ("fixed",
# "text", ...), DESCRIBE reports the actual SQL types a client would see on Snowflake.
_SQL_TYPE_MAP: dict[str, str] = {
    "BIGINT": "NUMBER(38,0)",
    "HUGEINT": "NUMBER(38,0)",
    "INTEGER": "NUMBER(38,0)",
    "SMALLINT": "NUMBER(38,0)",
    "TINYINT": "NUMBER(38,0)",
    "UBIGINT": "NUMBER(38,0)",
    "UINTEGER": "NUMBER(38,0)",
    "USMALLINT": "NUMBER(38,0)",
    "UTINYINT": "NUMBER(38,0)",
    "DOUBLE": "FLOAT",
    "FLOAT": "FLOAT",
    "REAL": "FLOAT",
    "BOOLEAN": "BOOLEAN",
    "VARCHAR": "VARCHAR",
    "BLOB": "BINARY",
    "BINARY": "BINARY",
    "DATE": "DATE",
    "TIME": "TIME",
    "TIME WITH TIME ZONE": "TIME",
    "TIMESTAMP": "TIMESTAMP_NTZ",
    "TIMESTAMP WITH TIME ZONE": "TIMESTAMP_TZ",
    "TIMESTAMP_TZ": "TIMESTAMP_TZ",
    "TIMESTAMP WITH LOCAL TIME ZONE": "TIMESTAMP_LTZ",
    "TIMESTAMP_LTZ": "TIMESTAMP_LTZ",
    "INTERVAL": "VARCHAR",
    "JSON": "VARIANT",
    "UUID": "VARCHAR",
    "ENUM": "VARCHAR",
    "BIT": "VARCHAR",
}

_DESCRIBE_DECIMAL_RE = re.compile(r"^DECIMAL\((\d+),\s*(\d+)\)$")


def duckdb_type_to_snowflake_sql_type(duckdb_type: str) -> str:
    """Map a DuckDB type name to the Snowflake SQL type shown by ``DESCRIBE TABLE``.

    DuckDB's ``DESCRIBE`` output embeds DuckDB type names (e.g. ``INTEGER``,
    ``TIMESTAMP WITH TIME ZONE``, ``VARCHAR[]``) in its ``column_type`` column. Snowflake
    clients expect Snowflake type names instead, so the emulator converts them before
    returning the result set.
    """
    normalized = duckdb_type.upper().strip()
    if normalized.startswith("DECIMAL"):
        m = _DESCRIBE_DECIMAL_RE.match(normalized)
        if m:
            return f"NUMBER({m.group(1)},{m.group(2)})"
        return "NUMBER"
    if normalized.startswith("STRUCT") or normalized.startswith("MAP"):
        return "OBJECT"
    if normalized.startswith("LIST") or normalized.startswith("ARRAY") or normalized.endswith("[]"):
        return "ARRAY"
    return _SQL_TYPE_MAP.get(normalized, "VARCHAR")


_DECIMAL_RE = re.compile(r"^DECIMAL\((\d+),\s*(\d+)\)$")


def parse_precision_scale(duckdb_type: str) -> tuple[int | None, int]:
    """Extract ``(precision, scale)`` from a DuckDB type name, e.g. ``DECIMAL(10,2)``.

    Non-decimal types default to ``(None, 0)``.
    """
    m = _DECIMAL_RE.match(duckdb_type.upper())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, 0
