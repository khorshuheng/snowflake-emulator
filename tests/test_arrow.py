"""Arrow result encoding: Snowflake's timestamp struct must retain NULL rows."""

from __future__ import annotations

import base64
import datetime
import decimal

import pyarrow as pa
import pytest

from snowflake_emulator.arrow import encode_arrow
from snowflake_emulator.core.results import ColumnMeta


def test_arrow_timestamp_struct_keeps_null_validity():
    """A NULL timestamp is a NULL struct, not a struct with NULL children."""
    columns = [
        ColumnMeta(name="TS", type="timestamp_ntz", duckdb_type="TIMESTAMP"),
        ColumnMeta(name="TZ", type="timestamp_tz", duckdb_type="TIMESTAMPTZ"),
    ]
    value = datetime.datetime(2020, 1, 2, 3, 4, 5, 123456)
    table = pa.ipc.open_stream(
        base64.b64decode(
            encode_arrow(
                columns,
                [[None, None], [value, value.replace(tzinfo=datetime.timezone.utc)]],
            )
        )
    ).read_all()

    for name in ("TS", "TZ"):
        struct = table.column(name).chunk(0)
        assert struct.is_null().to_pylist() == [True, False], name
        assert struct[0].as_py() is None, name
        decoded = struct[1].as_py()
        assert decoded["epoch"] == 1577934245, name
        assert decoded["fraction"] == 123456000, name


def _arrow_table(columns, rows):
    return pa.ipc.open_stream(base64.b64decode(encode_arrow(columns, rows))).read_all()


def test_arrow_scale_zero_number_uses_decimal128_when_int64_cannot_hold_it():
    """A wide NUMBER is DECIMAL128, as Snowflake sends it; int64 would wrap in pandas."""
    from snowflake_emulator.arrow import _arrow_type

    for duckdb_type, expected in (
        ("DECIMAL(18,0)", pa.int64()),
        ("BIGINT", pa.int64()),
        ("DECIMAL(19,0)", pa.decimal128(19, 0)),
        ("DECIMAL(38,0)", pa.decimal128(38, 0)),
        ("HUGEINT", pa.decimal128(38, 0)),
        ("UBIGINT", pa.decimal128(20, 0)),
        ("DECIMAL(38,2)", pa.decimal128(38, 2)),
    ):
        column = ColumnMeta(name="N", type="fixed", duckdb_type=duckdb_type)
        assert _arrow_type(column) == expected, duckdb_type

    value = decimal.Decimal("9" * 38)
    table = _arrow_table(
        [ColumnMeta(name="N", type="fixed", duckdb_type="DECIMAL(38,0)")], [[value]]
    )
    assert table.schema.field("N").type == pa.decimal128(38, 0)
    assert table.column("N")[0].as_py() == value


def test_arrow_value_beyond_the_snowflake_type_is_a_data_error():
    """A 39-digit DuckDB HUGEINT is beyond NUMBER(38,0): report a data error, not a 500."""
    from snowflake_emulator.core.errors import ExecutionError

    with pytest.raises(ExecutionError) as error:
        encode_arrow(
            [ColumnMeta(name="H", type="fixed", duckdb_type="HUGEINT")], [[2**127 - 1]]
        )
    assert (error.value.errno, error.value.sqlstate) == ("100038", "22000")
