"""Snowflake scalar semantics that the plain sqlglot translation gets wrong.

Each rewrite restores one documented rule of a *function* (as opposed to the
semi-structured rewrites in ``semi_structured.py``):

* ``DATEADD``/``DATE_TRUNC`` keep a DATE input a DATE (doc: "If date_or_time_part is
  day or larger ... the function returns a DATE value" / "The returned value is the
  same type as the input value"). DuckDB's date arithmetic and ``date_trunc`` answer
  TIMESTAMP, so the result is cast back.
* ``LPAD``/``RPAD`` with an empty pad string inserts no padding but still truncates
  (doc: "If the pad argument is empty, no padding is inserted, but the result is
  still truncated to length_expr"). DuckDB raises "Insufficient padding".
* division by zero raises Snowflake's ``100051 (22012): Division by zero``; DuckDB
  answers NULL, which silently reports a result Snowflake rejects.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from snowflake_emulator.number_format import NumberFormatError, validate_model
from snowflake_emulator.translator import column_types
from snowflake_emulator.translator.arrays import _is_json_value, _is_variant_value
from snowflake_emulator.translator.constants import WRITE_DIALECT
from snowflake_emulator.translator.errors import TranslationError

# The types Snowflake applies a format model to. A numeric model applies to everything else.
_TEMPORAL_TYPES = (
    exp.DataType.Type.DATE,
    exp.DataType.Type.DATETIME,
    exp.DataType.Type.TIME,
    exp.DataType.Type.TIMETZ,
    exp.DataType.Type.TIMESTAMP,
    exp.DataType.Type.TIMESTAMPTZ,
    exp.DataType.Type.TIMESTAMPLTZ,
    exp.DataType.Type.TIMESTAMPNTZ,
)

# Snowflake date parts that make DATEADD/DATE_TRUNC answer a DATE for a DATE input.
# Every other part (hour, minute, second, millisecond, ...) answers TIMESTAMP_NTZ.
_DATE_OR_LARGER = frozenset(
    {
        "YEAR", "Y", "YY", "YYY", "YYYY", "YR", "YEARS", "YRS",
        "QUARTER", "Q", "QQ", "QTR", "QUARTERS", "QTRS",
        "MONTH", "MM", "MON", "MONS", "MONTHS",
        "WEEK", "W", "WW", "WK", "WEEKOFYEAR", "WEEKS", "WKS",
        "DAY", "D", "DD", "DDD", "DAYS", "DAYOFMONTH", "DAYOFWEEK", "DAYOFYEAR",
        "DW", "WEEKDAY",
    }
)  # fmt: skip

_DIVISION_BY_ZERO = "Division by zero"


def _sql(node: exp.Expression | None) -> str:
    """Serialize an operand of a call this module rebuilds from SQL text."""
    return node.sql(dialect=WRITE_DIALECT) if node is not None else "NULL"

# Marks the safe division this module builds, so a second pass over the same tree does not wrap
# it again (the guard would grow on every pass).
_DIVISION_GUARD_META = "sfemu_division_guard"


def _unit_name(node: exp.Expression) -> str:
    """The upper-cased date-part name of a ``DATEADD``/``DATE_TRUNC`` call."""
    unit = node.args.get("unit")
    if isinstance(unit, exp.Var):
        return (unit.name or "").upper()
    if isinstance(unit, exp.Literal):
        return str(unit.this).upper()
    return ""


def _is_date_typed(node: exp.Expression | None) -> bool:
    """True when `node` is a DATE: a cast to DATE, or a column the catalog says is one."""
    if isinstance(node, exp.Paren):
        return _is_date_typed(node.this)
    if (
        isinstance(node, exp.Cast)
        and isinstance(node.to, exp.DataType)
        and node.to.this == exp.DataType.Type.DATE
    ):
        return True
    declared = column_types.declared_column_type(node)
    return declared is not None and declared.strip().upper() == "DATE"


def _is_temporal(node: exp.Expression | None) -> bool:
    """True for a DATE/TIME/TIMESTAMP value: the input a TO_CHAR format model applies to."""
    if isinstance(node, exp.Paren):
        return _is_temporal(node.this)
    if isinstance(node, exp.Cast) and isinstance(node.to, exp.DataType):
        return node.to.this in _TEMPORAL_TYPES
    declared = column_types.declared_column_type(node)
    return declared is not None and declared.strip().upper().startswith(("DATE", "TIME"))


def _rewrite_operand(node: exp.Expression) -> exp.Expression:
    """Put an operand in its final Snowflake form before it is serialized.

    The statement pass is pre-order, so a nested ``col:path`` has not had its JSON cast
    applied when this node is visited; entering the pass on the operand fixes that.
    """
    from snowflake_emulator.translator.semi_structured import (
        _rewrite_snowflake_specifics,
    )

    return node.transform(_rewrite_snowflake_specifics)


def _text_of(node: exp.Expression) -> str:
    """The text Snowflake coerces the value to before a format model applies.

    A semi-structured value uses its text form (a string stored in a VARIANT without its
    JSON quotes), which is what the VARIANT -> VARCHAR conversion produces; anything else
    is just rendered as text.
    """
    from snowflake_emulator.translator.semi_structured import _text_operand

    if (
        _is_json_value(node)
        or _is_variant_value(node)
        or column_types.is_semi_structured_typed(node)
    ):
        converted = _text_operand(node)
        if converted is not None:
            return _sql(converted)
    return f"CAST({_sql(node)} AS VARCHAR)"


def _format_model(node: exp.Expression) -> str:
    """The string literal model of a ``TO_CHAR``/``TO_NUMBER`` call, validated up front."""
    model = node.args.get("format")
    if not (isinstance(model, exp.Literal) and model.is_string):
        raise TranslationError(
            "A numeric format model must be a string literal, so the emulator can apply it."
        )
    text = str(model.this)
    try:
        validate_model(text)
    except NumberFormatError as exc:
        raise TranslationError(str(exc)) from exc
    return text


def _rewrite_to_char_format(node: exp.ToChar) -> exp.Expression:
    """``TO_CHAR(number, '<model>')``: apply Snowflake's numeric format model.

    sqlglot drops the format for a DuckDB target, so the call is rebuilt onto the
    ``sfemu_format_number`` helper. A DATE/TIME/TIMESTAMP operand keeps sqlglot's own
    translation of its date format model.
    """
    if _is_temporal(node.this):
        return node
    model = _format_model(node)
    operand = _rewrite_operand(node.this) if isinstance(node.this, exp.Expression) else node.this
    return sqlglot.parse_one(
        f"sfemu_format_number({_text_of(operand)}, "
        f"{exp.Literal.string(model).sql(dialect=WRITE_DIALECT)})",
        read=WRITE_DIALECT,
    )


def _rewrite_to_number_format(node: exp.ToNumber) -> exp.Expression:
    """``TO_NUMBER(<text>, '<model>' [, precision, scale])``: parse with the model.

    The parsed value is cast to the requested ``DECIMAL(p, s)``; without a precision the
    emulator keeps the same DOUBLE shape its format-less ``TO_NUMBER`` already returns.
    ``TRY_TO_NUMBER`` passes the safe flag, so unparseable input is NULL, not an error.
    """
    model = _format_model(node)
    operand = _rewrite_operand(node.this) if isinstance(node.this, exp.Expression) else node.this
    safe = "TRUE" if node.args.get("safe") else "FALSE"
    parsed = (
        f"sfemu_parse_number({_text_of(operand)}, "
        f"{exp.Literal.string(model).sql(dialect=WRITE_DIALECT)}, {safe})"
    )
    precision = node.args.get("precision")
    scale = node.args.get("scale")
    if precision is not None and scale is not None:
        target = f"DECIMAL({_sql(precision)}, {_sql(scale)})"
    elif precision is not None:
        target = f"DECIMAL({_sql(precision)}, 0)"
    else:
        target = "DOUBLE"
    return sqlglot.parse_one(f"CAST({parsed} AS {target})", read=WRITE_DIALECT)


def _rewrite_date_add(node: exp.DateAdd) -> exp.Expression:
    """``DATEADD(<day or larger>, n, <DATE>)`` answers a DATE, not a TIMESTAMP."""
    if _unit_name(node) not in _DATE_OR_LARGER or not _is_date_typed(node.this):
        return node
    return exp.cast(node, exp.DataType.build("DATE"), dialect=WRITE_DIALECT)


def _rewrite_date_trunc(node: exp.Expression) -> exp.Expression:
    """``DATE_TRUNC`` keeps a DATE input's type: DATE for a day-or-larger part.

    Handles both parses sqlglot produces -- ``DateTrunc`` for a DATE-typed argument and
    ``TimestampTrunc`` for one whose type it does not know. Rebuilt in the DuckDB dialect so
    the generator's own DATE cast (which it adds for a Snowflake DATE_TRUNC) does not turn a
    sub-day truncation into a DATE: Snowflake answers TIMESTAMP_NTZ there.
    """
    if not _is_date_typed(node.this):
        return node
    part = _unit_name(node)
    if not part:
        return node
    inner = node.this.sql(dialect=WRITE_DIALECT)
    truncated = f"DATE_TRUNC('{part}', CAST({inner} AS TIMESTAMP))"
    if part in _DATE_OR_LARGER:
        truncated = f"CAST({truncated} AS DATE)"
    return sqlglot.parse_one(truncated, read=WRITE_DIALECT)


def _rewrite_pad(node: exp.Pad) -> exp.Expression:
    """``LPAD``/``RPAD`` with an empty pad string: no padding, but still truncated.

    ``SUBSTR(base, 1, n)`` is exactly that (it returns the whole string when n is
    larger), and it avoids DuckDB's "Insufficient padding" error. A pad string that is
    only known at run time is left alone.
    """
    fill = node.args.get("fill_pattern")
    if not (isinstance(fill, exp.Literal) and fill.is_string and fill.this == ""):
        return node
    length = node.args.get("expression")
    if length is None:
        return node
    return exp.Substring(
        this=node.this, start=exp.Literal.number(1), length=length
    )


def _rewrite_division(node: exp.Div) -> exp.Expression:
    """``a / b`` raises Snowflake's division-by-zero error; DuckDB answers NULL.

    The guard skips a NULL dividend (`NULL / 0` is NULL in Snowflake) and the division
    itself uses ``NULLIF`` so the value never divides by zero. ``DIV0``/``DIV0NULL``
    expand to a CASE that answers before this branch runs, and a NULL divisor keeps
    returning NULL.
    """
    if node.args.get("safe") or node.meta.get(_DIVISION_GUARD_META):
        return node
    divisor = node.args.get("expression")
    dividend = node.this
    if not isinstance(divisor, exp.Expression) or isinstance(divisor, exp.Null):
        return node
    dividend_sql = dividend.sql(dialect=WRITE_DIALECT) if isinstance(dividend, exp.Expression) else "NULL"
    divisor_sql = divisor.sql(dialect=WRITE_DIALECT)
    guarded = sqlglot.parse_one(
        f"CASE WHEN ({divisor_sql}) = 0 AND NOT ({dividend_sql}) IS NULL "
        f"THEN error('{_DIVISION_BY_ZERO}') "
        f"ELSE ({dividend_sql}) / NULLIF({divisor_sql}, 0) END",
        read=WRITE_DIALECT,
    )
    for inner in guarded.find_all(exp.Div):
        inner.meta[_DIVISION_GUARD_META] = True
    return guarded


def rewrite_scalar_semantics(node: exp.Expression) -> exp.Expression:
    """Rewrite one scalar node whose DuckDB form differs from Snowflake's documented one."""
    if isinstance(node, exp.ToChar) and node.args.get("format") is not None:
        return _rewrite_to_char_format(node)
    if isinstance(node, exp.ToNumber) and node.args.get("format") is not None:
        return _rewrite_to_number_format(node)
    if isinstance(node, exp.DateAdd):
        return _rewrite_date_add(node)
    if isinstance(node, (exp.DateTrunc, exp.TimestampTrunc)):
        return _rewrite_date_trunc(node)
    if isinstance(node, exp.Pad):
        return _rewrite_pad(node)
    if isinstance(node, exp.Div):
        return _rewrite_division(node)
    return node
