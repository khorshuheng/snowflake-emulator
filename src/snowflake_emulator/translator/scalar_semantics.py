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

from snowflake_emulator.translator import column_types
from snowflake_emulator.translator.constants import WRITE_DIALECT

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
    if isinstance(node, exp.DateAdd):
        return _rewrite_date_add(node)
    if isinstance(node, (exp.DateTrunc, exp.TimestampTrunc)):
        return _rewrite_date_trunc(node)
    if isinstance(node, exp.Pad):
        return _rewrite_pad(node)
    if isinstance(node, exp.Div):
        return _rewrite_division(node)
    return node
