"""Snowflake's output-format session parameters for the SQL API rowset.

The SQL API renders every cell as text, and ``DATE_OUTPUT_FORMAT``, ``TIME_OUTPUT_FORMAT``,
``TIMESTAMP_OUTPUT_FORMAT`` and ``BINARY_OUTPUT_FORMAT`` decide that text. Snowflake's defaults
are ``YYYY-MM-DD``, ``HH24:MI:SS``, ``YYYY-MM-DD HH24:MI:SS.FF3`` and ``HEX``; the DATE and
BINARY defaults were what the emulator already produced, while a TIME and TIMESTAMP cell was
rendered as ISO-8601 (``2024-03-04T05:06:07``), which Snowflake never emits.

Only the pattern tokens that map onto ``strftime`` are translated -- the ones Snowflake's
documentation shows -- and an unknown token is rejected rather than emitted literally, so a
format the emulator cannot honour fails the request instead of returning wrong text.
"""

from __future__ import annotations

import base64
import datetime
from dataclasses import dataclass, field
from typing import Any

from snowflake_emulator.core.results import ColumnMeta

DEFAULT_DATE_FORMAT = "YYYY-MM-DD"
DEFAULT_TIME_FORMAT = "HH24:MI:SS"
DEFAULT_TIMESTAMP_FORMAT = "YYYY-MM-DD HH24:MI:SS.FF3"
DEFAULT_BINARY_FORMAT = "HEX"

# ``FF<n>`` keeps that many fraction digits, which strftime cannot express, so the pattern
# carries this sentinel and the digits are substituted at render time.
_FF = "\x01"

# Snowflake's pattern tokens, in the order they must be tried ("MM" before "M" would be a
# substring trap, so the table is longest-first).
_TOKENS: tuple[tuple[str, str], ...] = (
    ("YYYY", "%Y"),
    ("YY", "%y"),
    ("HH24", "%H"),
    ("HH12", "%I"),
    ("FF9", _FF),
    ("FF6", _FF),
    ("FF3", _FF),
    ("FF0", ""),
    ("MM", "%m"),
    ("DD", "%d"),
    ("MI", "%M"),
    ("SS", "%S"),
    ("AM", "%p"),
    ("PM", "%p"),
)

_BINARY_FORMATS = ("HEX", "BASE64")

# ``FF<n>`` keeps that many fraction digits, which strftime cannot express, so the pattern
# carries this sentinel and the digits are substituted at render time.
_FF = "\x01"


class OutputFormatError(ValueError):
    """A request parameter the emulator cannot honour."""


@dataclass
class OutputFormats:
    """The four output formats a request asked for (Snowflake's defaults otherwise)."""

    date: str = DEFAULT_DATE_FORMAT
    time: str = DEFAULT_TIME_FORMAT
    timestamp: str = DEFAULT_TIMESTAMP_FORMAT
    binary: str = DEFAULT_BINARY_FORMAT
    prepared: dict[str, tuple[str, int | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prepared = {
            "date": _strftime_pattern("DATE_OUTPUT_FORMAT", self.date),
            "time": _strftime_pattern("TIME_OUTPUT_FORMAT", self.time),
            "timestamp": _strftime_pattern("TIMESTAMP_OUTPUT_FORMAT", self.timestamp),
        }
        if self.binary.upper() not in _BINARY_FORMATS:
            raise OutputFormatError(
                f"BINARY_OUTPUT_FORMAT must be one of {', '.join(_BINARY_FORMATS)}, "
                f"got '{self.binary}'"
            )
        self.binary = self.binary.upper()


def _strftime_pattern(name: str, pattern: str) -> tuple[str, int | None]:
    """Translate a Snowflake date/time pattern into a ``strftime`` one.

    A token the emulator does not know fails the request: emitting the pattern's letters
    literally would answer with text Snowflake never produces.
    """
    result: list[str] = []
    fraction: int | None = None
    index = 0
    upper = pattern.upper()
    while index < len(pattern):
        for token, replacement in _TOKENS:
            if upper.startswith(token, index):
                if replacement == _FF:
                    fraction = int(token[2:])
                result.append(replacement)
                index += len(token)
                break
        else:
            character = pattern[index]
            if character.isalpha():
                raise OutputFormatError(
                    f"{name} contains the unsupported pattern token "
                    f"'{pattern[index:].split()[0]}'"
                )
            result.append(character)
            index += 1
    return "".join(result), fraction


def output_formats_from_parameters(parameters: dict[str, Any] | None) -> OutputFormats:
    """Build the formats from a request's ``parameters`` object, ignoring unrelated keys."""
    if not parameters:
        return OutputFormats()
    upper = {str(key).upper(): value for key, value in parameters.items()}
    return OutputFormats(
        date=str(upper.get("DATE_OUTPUT_FORMAT", DEFAULT_DATE_FORMAT)),
        time=str(upper.get("TIME_OUTPUT_FORMAT", DEFAULT_TIME_FORMAT)),
        timestamp=str(upper.get("TIMESTAMP_OUTPUT_FORMAT", DEFAULT_TIMESTAMP_FORMAT)),
        binary=str(upper.get("BINARY_OUTPUT_FORMAT", DEFAULT_BINARY_FORMAT)),
    )


def format_cell(value: Any, col: ColumnMeta, formats: OutputFormats) -> Any:
    """Render one cell the way the requested output formats describe it.

    Only the temporal and binary types have a format; every other value is returned unchanged
    for the rowset encoder.
    """
    if value is None:
        return None
    kind = (col.type or "").upper()
    if isinstance(value, datetime.datetime):
        return _format_temporal(value, *formats.prepared["timestamp"])
    if isinstance(value, datetime.time):
        return _format_temporal(value, *formats.prepared["time"])
    if isinstance(value, datetime.date):
        return value.strftime(formats.prepared["date"][0])
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if formats.binary == "BASE64":
            return base64.b64encode(raw).decode("ascii")
        return raw.hex().upper()
    if kind == "BINARY" and isinstance(value, str):
        # The executor already decoded a BINARY column to text (hex); re-encode for BASE64.
        try:
            raw = bytes.fromhex(value)
        except ValueError:
            return value
        if formats.binary == "BASE64":
            return base64.b64encode(raw).decode("ascii")
        return raw.hex().upper()
    return value


def _format_temporal(
    value: datetime.datetime | datetime.time, pattern: str, fraction: int | None
) -> str:
    """Apply a translated pattern, filling in the ``FF<n>`` digits it asks for."""
    if fraction is None:
        return value.strftime(pattern)
    digits = f"{getattr(value, 'microsecond', 0):06d}"[:fraction]
    return value.strftime(pattern).replace(_FF, digits)
