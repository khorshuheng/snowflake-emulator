"""Arrow result encoding: Snowflake's timestamp struct must retain NULL rows."""

from __future__ import annotations

import base64
import datetime

import pyarrow as pa

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
