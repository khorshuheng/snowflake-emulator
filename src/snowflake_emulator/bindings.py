"""Decode the connector's wire-protocol bind parameters.

The Snowflake connector sends bindings as ``{"<key>": {"type": ..., "value": ...}}``
where the key is an ordinal (``"1"``, ``"2"``) for positional binds or a name for
named binds, and ``value`` is a string in the type's wire encoding. An array binding
(``executemany``) instead carries a *list* in each ``value``, with one entry per row.

Decoding to native Python types before handing them to DuckDB keeps comparisons and
casts faithful (e.g. a ``FIXED`` bind is an ``int``, not the string ``"1"``), and
lets array bindings be transposed into one execution per row.
"""

from __future__ import annotations

import binascii
import datetime
from dataclasses import dataclass
from typing import Any


@dataclass
class DecodedBindings:
    """Bindings split by the shape the request used.

    Exactly one of the three is populated for a non-empty binding set.
    """

    positional: list[Any] | None = None
    named: dict[str, Any] | None = None
    batch: list[tuple[Any, ...]] | None = None


# A semi-structured bind value is a single value even when it is a list; only an
# array-binding column (one list per placeholder, sent by ``executemany``) is
# transposed into rows.
_SEMI_STRUCTURED_TYPES = frozenset({"VARIANT", "OBJECT", "ARRAY"})


def _is_array_binding_column(binding: dict[str, Any]) -> bool:
    """True when this binding is one column of an array binding (``executemany``)."""
    if not isinstance(binding.get("value"), list):
        return False
    return str(binding.get("type", "")).upper() not in _SEMI_STRUCTURED_TYPES


def from_binding(binding: dict[str, Any]) -> Any:
    """Decode one ``{"type": ..., "value": ...}`` binding to a Python value."""
    type_ = str(binding.get("type", "")).upper()
    value = binding.get("value")
    if value is None:
        return None
    if type_ == "FIXED":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if type_ == "BINARY":
        return binascii.unhexlify(value)
    if type_ == "BOOLEAN":
        return str(value).lower() == "true"
    if type_ == "DATE":
        return _from_date(value)
    if type_ == "TIME":
        return _from_time(value)
    if type_ == "TIMESTAMP_NTZ":
        # The connector encodes a naive wall-clock timestamp as a UTC-based epoch, so
        # decode to a *naive* datetime; an aware one would let DuckDB shift it into the
        # session timezone (e.g. 10:20:30 arriving as 18:20:30 at UTC+8).
        return _from_datetime_ntz(value)
    if type_ == "TIMESTAMP_LTZ":
        # The JDBC driver binds setTimestamp as TIMESTAMP_LTZ; it is an instant.
        return _from_datetime(value)
    if type_ == "TIMESTAMP_TZ":
        return _from_datetime_tz(value)
    # VARIANT/OBJECT/ARRAY/TEXT and anything unrecognised arrive as plain values.
    return value


def _from_date(value: Any) -> datetime.date:
    """Snowflake encodes DATE binds as milliseconds since the epoch."""
    seconds = int(value) / 1000
    return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).date()


def _from_time(value: Any) -> datetime.time:
    """Snowflake encodes TIME binds as nanoseconds since midnight."""
    nanoseconds = int(value)
    microseconds = nanoseconds / 1000
    moment = datetime.datetime.fromtimestamp(microseconds / 1_000_000, datetime.timezone.utc)
    return moment.replace(microsecond=int(microseconds % 1_000_000)).time()


def _from_datetime(value: Any) -> datetime.datetime:
    """Snowflake encodes timestamp binds as nanoseconds since the epoch."""
    nanoseconds = int(value)
    microseconds = nanoseconds / 1000
    moment = datetime.datetime.fromtimestamp(microseconds / 1_000_000, datetime.timezone.utc)
    return moment.replace(microsecond=int(microseconds % 1_000_000))


def _from_datetime_ntz(value: Any) -> datetime.datetime:
    """Decode a TIMESTAMP_NTZ bind to a naive wall-clock datetime.

    The connector encodes the wall clock as a UTC-based epoch, so dropping the
    timezone after decoding against UTC reproduces exactly what the client sent.
    """
    microseconds = int(value) / 1000
    moment = datetime.datetime.fromtimestamp(microseconds / 1_000_000, datetime.timezone.utc)
    return moment.replace(tzinfo=None, microsecond=int(microseconds % 1_000_000))


def _from_datetime_tz(value: Any) -> datetime.datetime:
    """Decode a TIMESTAMP_TZ bind: ``"<epoch ns> <utc offset minutes + 1440>"``."""
    nanoseconds, _, offset_minutes = str(value).partition(" ")
    moment = _from_datetime(nanoseconds)
    if not offset_minutes:
        return moment
    offset = datetime.timedelta(minutes=int(offset_minutes) - 1440)
    return moment.astimezone(datetime.timezone(offset))


def decode_bindings(bindings: dict[str, Any] | None) -> DecodedBindings:
    """Split and decode a request's ``bindings`` object.

    Numeric keys are positional; any other key means the bindings are named. If any
    value is a list the request is an array binding and the columns are transposed
    into one tuple per row (mismatched column lengths raise ``ValueError``).
    """
    if not bindings:
        return DecodedBindings()

    if not all(key.isdigit() for key in bindings):
        return DecodedBindings(named={name: from_binding(b) for name, b in bindings.items()})

    ordered = [bindings[key] for key in sorted(bindings, key=int)]
    array_columns = [_is_array_binding_column(binding) for binding in ordered]
    if any(array_columns):
        if not all(array_columns):
            # A partial list would silently expand a scalar string into one row per
            # character; reject the malformed request instead.
            raise ValueError("Array binding must provide a list value for every column")
        columns = [
            [from_binding({**binding, "value": item}) for item in binding["value"]]
            for binding in ordered
        ]
        try:
            batch = [tuple(row) for row in zip(*columns, strict=True)]
        except ValueError as exc:  # column arrays of differing length
            raise ValueError(f"Array binding columns have mismatched lengths: {exc}") from exc
        return DecodedBindings(batch=batch)

    return DecodedBindings(positional=[from_binding(binding) for binding in ordered])
