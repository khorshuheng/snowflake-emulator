"""Tests for the DuckDB -> Snowflake column type mapping."""

from __future__ import annotations

import pytest

from snowflake_emulator.type_mapping import (
    duckdb_timestamp_scale,
    duckdb_type_to_snowflake,
    duckdb_type_to_snowflake_sql_type,
    parse_precision_scale,
)


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
        # Integers -> Snowflake NUMBER
        ("INTEGER", "NUMBER(38,0)"),
        ("BIGINT", "NUMBER(38,0)"),
        ("UHUGEINT", "NUMBER(38,0)"),
        ("DECIMAL(10,2)", "NUMBER(10,2)"),
        ("DECIMAL(38, 0)", "NUMBER(38,0)"),
        # Floating point
        ("FLOAT", "FLOAT"),
        ("DOUBLE", "FLOAT"),
        # Strings / binary
        ("VARCHAR", "VARCHAR"),
        ("BLOB", "BINARY"),
        ("BIT", "VARCHAR"),
        # Temporal
        ("DATE", "DATE"),
        ("TIME", "TIME"),
        ("TIME WITH TIME ZONE", "TIME"),
        ("TIMETZ", "TIME"),
        ("TIMESTAMP", "TIMESTAMP_NTZ(6)"),
        ("TIMESTAMP_S", "TIMESTAMP_NTZ(0)"),
        ("TIMESTAMP_MS", "TIMESTAMP_NTZ(3)"),
        ("TIMESTAMP_US", "TIMESTAMP_NTZ(6)"),
        ("TIMESTAMP_NS", "TIMESTAMP_NTZ(9)"),
        ("TIMESTAMP WITH TIME ZONE", "TIMESTAMP_TZ(6)"),
        ("TIMESTAMPTZ", "TIMESTAMP_TZ(6)"),
        ("TIMESTAMP_LTZ", "TIMESTAMP_LTZ(6)"),
        # Semi-structured
        ("JSON", "VARIANT"),
        ("STRUCT(a INTEGER)", "OBJECT"),
        ("MAP(INTEGER, VARCHAR)", "OBJECT"),
        ("UNION(a INTEGER)", "OBJECT"),
        # Arrays
        ("INTEGER[]", "ARRAY"),
        ("INTEGER[2]", "ARRAY"),
        ("FLOAT[2]", "ARRAY"),
        ("DECIMAL(10,2)[]", "ARRAY"),
        # No native Snowflake equivalent -> VARCHAR
        ("UUID", "VARCHAR"),
        ("ENUM('a','b')", "VARCHAR"),
        ("INTERVAL", "VARCHAR"),
    ],
)
def test_duckdb_type_to_snowflake_sql_type(duckdb_type, expected):
    assert duckdb_type_to_snowflake_sql_type(duckdb_type) == expected


def test_describe_sql_type_mapping_is_case_insensitive():
    assert duckdb_type_to_snowflake_sql_type("timestamp with time zone") == "TIMESTAMP_TZ(6)"
    assert duckdb_type_to_snowflake_sql_type("integer[2]") == "ARRAY"
    assert duckdb_type_to_snowflake_sql_type("  varchar  ") == "VARCHAR"


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


@pytest.mark.parametrize(
    ("duckdb_type", "expected"),
    [
        # DuckDB's real timestamp granularity, not the declared Snowflake precision.
        ("TIMESTAMP_S", 0),
        ("TIMESTAMP_MS", 3),
        ("TIMESTAMP", 6),
        ("TIMESTAMP_US", 6),
        ("TIMESTAMP_NS", 9),
        ("TIMESTAMPTZ", 6),
        ("TIMESTAMP WITH LOCAL TIME ZONE", 6),
        # Non-timestamp types have no timestamp scale.
        ("INTEGER", None),
        ("DECIMAL(10,2)", None),
        ("VARCHAR", None),
    ],
)
def test_duckdb_timestamp_scale(duckdb_type, expected):
    assert duckdb_timestamp_scale(duckdb_type) == expected
