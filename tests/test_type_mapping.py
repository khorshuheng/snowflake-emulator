"""Tests for the DuckDB -> Snowflake column type mapping."""

from __future__ import annotations

import pytest

from snowflake_emulator.type_mapping import duckdb_type_to_snowflake, parse_precision_scale


@pytest.mark.parametrize(
    ("duckdb_type", "expected"),
    [
        # Integers -> fixed
        ("TINYINT", "fixed"),
        ("SMALLINT", "fixed"),
        ("INTEGER", "fixed"),
        ("BIGINT", "fixed"),
        ("HUGEINT", "fixed"),
        ("UTINYINT", "fixed"),
        ("USMALLINT", "fixed"),
        ("UINTEGER", "fixed"),
        ("UBIGINT", "fixed"),
        ("UHUGEINT", "fixed"),
        ("DECIMAL(10,2)", "fixed"),
        ("DECIMAL(38, 0)", "fixed"),
        # Floating point -> real
        ("FLOAT", "real"),
        ("DOUBLE", "real"),
        # Strings / binary
        ("VARCHAR", "text"),
        ("BLOB", "binary"),
        ("BIT", "binary"),
        # Temporal
        ("DATE", "date"),
        ("TIME", "time"),
        ("TIME WITH TIME ZONE", "time"),
        ("TIMETZ", "time"),
        ("TIMESTAMP", "timestamp_ntz"),
        ("TIMESTAMP_S", "timestamp_ntz"),
        ("TIMESTAMP_MS", "timestamp_ntz"),
        ("TIMESTAMP_NS", "timestamp_ntz"),
        ("TIMESTAMP WITH TIME ZONE", "timestamp_tz"),
        ("TIMESTAMP_TZ", "timestamp_tz"),
        ("TIMESTAMPTZ", "timestamp_tz"),
        ("INTERVAL", "text"),
        # Semi-structured
        ("JSON", "variant"),
        ("STRUCT(a INTEGER)", "object"),
        ('STRUCT("weird name" INTEGER)', "object"),
        ("MAP(INTEGER, VARCHAR)", "object"),
        ("UNION(a INTEGER)", "object"),
        # Arrays: variable-length and fixed-size
        ("INTEGER[]", "array"),
        ("VARCHAR[]", "array"),
        ("DECIMAL(10,2)[]", "array"),
        ("INTEGER[2]", "array"),
        ("FLOAT[2]", "array"),
        ("VARCHAR[3]", "array"),
        # No native Snowflake equivalent
        ("UUID", "text"),
        ("ENUM('a','b')", "text"),
        ("NULL", "text"),
        # Unknown types degrade to text rather than leaking a DuckDB name
        ("SOMETHING_EXOTIC", "text"),
    ],
)
def test_duckdb_type_to_snowflake(duckdb_type, expected):
    assert duckdb_type_to_snowflake(duckdb_type) == expected


def test_mapping_is_case_insensitive():
    assert duckdb_type_to_snowflake("timestamp with time zone") == "timestamp_tz"
    assert duckdb_type_to_snowflake("float[2]") == "array"
    assert duckdb_type_to_snowflake("  varchar  ") == "text"


@pytest.mark.parametrize(
    ("duckdb_type", "expected"),
    [
        ("DECIMAL(10,2)", (10, 2)),
        ("decimal(38, 0)", (38, 0)),
        ("INTEGER", (None, 0)),
        ("TIMESTAMP", (None, 0)),
        ("DECIMAL(10,2)[]", (None, 0)),
    ],
)
def test_parse_precision_scale(duckdb_type, expected):
    assert parse_precision_scale(duckdb_type) == expected
