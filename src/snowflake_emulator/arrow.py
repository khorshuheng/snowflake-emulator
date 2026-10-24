"""Encode query results as Apache Arrow, the format Snowflake uses for columnar results.

The connector takes the Arrow path when the response sets ``queryResultFormat`` to
``"arrow"``. Snowflake's Arrow dialect is not plain Arrow: every field carries
``logicalType``/``precision``/``scale``/``charLength`` metadata, timestamps are
``struct<epoch, fraction[, timezone]>`` and ``TIME`` is nanoseconds since midnight.
This mirrors ``fakesnow.arrow``, which is written against the same connector.
"""

from __future__ import annotations

import base64
import datetime
import decimal
import json
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from snowflake_emulator.core.errors import ExecutionError
from snowflake_emulator.core.results import ColumnMeta
from snowflake_emulator.type_mapping import parse_precision_scale


def _arrow_type(col: ColumnMeta) -> pa.DataType:
    """The Arrow type a column's values are stored as before Snowflake metadata."""
    precision, scale = parse_precision_scale(col.duckdb_type)
    sf_type = col.type
    if sf_type == "fixed":
        # A scale-0 NUMBER becomes an int64 only when the declared precision guarantees the
        # value fits; Snowflake sends a wide NUMBER as DECIMAL128, and an int64 column also
        # wraps silently in pandas once a sum passes 2**63. DuckDB integer types without a
        # declared precision keep int64, except the unsigned/128-bit ones that do not fit.
        if scale and scale > 0:
            return pa.decimal128(precision or 38, scale)
        if precision is not None and precision <= 18:
            return pa.int64()
        if precision is None:
            declared = (col.duckdb_type or "").strip().upper()
            if declared in ("BIGINT", "INTEGER", "SMALLINT", "TINYINT", "UINTEGER",
                            "USMALLINT", "UTINYINT"):
                return pa.int64()
            if declared == "UBIGINT":
                return pa.decimal128(20, 0)
        return pa.decimal128(precision or 38, 0)
    if sf_type == "real":
        return pa.float64()
    if sf_type == "boolean":
        return pa.bool_()
    if sf_type == "binary":
        return pa.binary()
    if sf_type == "date":
        return pa.date32()
    if sf_type == "time":
        return pa.time64("us")
    if sf_type == "timestamp_ntz":
        return pa.timestamp("us")
    if sf_type in ("timestamp_tz", "timestamp_ltz"):
        return pa.timestamp("us", tz="UTC")
    # text / variant / object / array: JSON text for the semi-structured types.
    return pa.string()


def _cell(value: Any, col: ColumnMeta) -> Any:
    if value is None:
        return None
    if col.type == "variant" and (col.duckdb_type or "").strip().upper() == "VARIANT":
        # As in the wire encoder: a DuckDB VARIANT is a plain value that must reach the
        # connector as its JSON text, compact the way Snowflake sends it. A JSON cell
        # already is JSON text.
        return json.dumps(value, default=str, separators=(",", ":"))
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, separators=(",", ":"))
    if col.type == "time" and isinstance(value, datetime.time):
        return (
            (value.hour * 3600 + value.minute * 60 + value.second) * 1_000_000_000
            + value.microsecond * 1000
        )
    if col.type == "fixed" and isinstance(value, float) and not isinstance(value, decimal.Decimal):
        return decimal.Decimal(str(value))
    return value


def _to_sf_schema(schema: pa.Schema, row_type: list[ColumnMeta]) -> pa.Schema:
    """Attach Snowflake's per-field metadata and reshape timestamp/time columns."""

    def sf_field(field: pa.Field, col: ColumnMeta) -> pa.Field:
        if isinstance(field.type, pa.TimestampType):
            fields = [
                pa.field("epoch", pa.int64(), nullable=False),
                pa.field("fraction", pa.int32(), nullable=False),
            ]
            if field.type.tz:
                fields.append(pa.field("timezone", pa.int32(), nullable=False))
            field = field.with_type(pa.struct(fields))
        elif isinstance(field.type, pa.Time64Type):
            field = field.with_type(pa.int64())
        elif pa.types.is_uint64(field.type):
            field = field.with_type(pa.int64())
        precision, scale = parse_precision_scale(col.duckdb_type)
        return field.with_metadata(
            {
                "logicalType": col.type.upper(),
                "precision": str(precision or 38),
                "scale": str(scale or 0),
                "charLength": "0",
            }
        )

    return pa.schema([sf_field(schema.field(i), col) for i, col in enumerate(row_type)])


def _timestamp_to_sf_struct(ts: pa.ChunkedArray) -> pa.Array:
    combined = ts.combine_chunks()
    without_subseconds = pc.floor_temporal(combined, unit="second")
    epoch = pc.divide(without_subseconds.cast(pa.int64()), 1_000_000)
    fraction = pc.round(pc.multiply(pc.subsecond(combined), 1_000_000_000)).cast(pa.int32())
    if combined.type.tz:
        timezone = pa.array([1440] * len(combined), type=pa.int32())
        return pa.StructArray.from_arrays(
            arrays=[epoch, fraction, timezone],
            fields=[
                pa.field("epoch", pa.int64(), nullable=False),
                pa.field("fraction", pa.int32(), nullable=False),
                pa.field("timezone", pa.int32(), nullable=False),
            ],
            mask=combined.is_null(),
        )
    return pa.StructArray.from_arrays(
        arrays=[epoch, fraction],
        fields=[
            pa.field("epoch", pa.int64(), nullable=False),
            pa.field("fraction", pa.int32(), nullable=False),
        ],
        mask=combined.is_null(),
    )


def _to_sf(table: pa.Table, row_type: list[ColumnMeta]) -> pa.Table:
    def to_sf_col(col: pa.ChunkedArray) -> pa.ChunkedArray:
        if pa.types.is_timestamp(col.type):
            return _timestamp_to_sf_struct(col)  # type: ignore[return-value]
        if pa.types.is_time(col.type):
            return pc.multiply(col.cast(pa.int64()), 1000)  # microseconds -> nanoseconds
        return col

    return pa.Table.from_arrays(
        [to_sf_col(column) for column in table.columns],
        schema=_to_sf_schema(table.schema, row_type),
    )


def to_ipc_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def encode_arrow(row_type: list[ColumnMeta], rows: list[list[Any]]) -> str:
    """Return base64 of the Arrow IPC stream for a result set.

    An empty result returns the empty string: that is how Snowflake signals "no rows",
    and the connector's Arrow iterator does not accept a serialized zero-row batch.
    """
    if not rows:
        return ""
    arrays = []
    fields = []
    for index, col in enumerate(row_type):
        arrow_type = _arrow_type(col)
        try:
            arrays.append(
                pa.array([_cell(row[index], col) for row in rows], type=arrow_type)
            )
        except (pa.ArrowException, OverflowError, ValueError) as exc:
            # A value the Snowflake type cannot represent (a 39-digit DuckDB HUGEINT is
            # beyond NUMBER(38,0)) must not surface as an unclassified HTTP 500 that the
            # client retries; report Snowflake's numeric-out-of-range pair instead.
            raise ExecutionError(
                f"A value in column {col.name!r} does not fit its Snowflake type "
                f"{col.duckdb_type or col.type}: {exc}",
                errno="100038",
                sqlstate="22000",
            ) from exc
        fields.append(pa.field(col.name, arrow_type))
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    return base64.b64encode(to_ipc_bytes(_to_sf(table, row_type))).decode("ascii")
