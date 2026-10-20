"""Snowflake aggregate and window functions DuckDB does not provide.

Two functions live here, both documented on docs.snowflake.com: ``HASH_AGG`` (a signed 64-bit
fingerprint that never returns NULL, hashes empty input to 0, and is influenced by NULLs and
duplicate rows) and ``RATIO_TO_REPORT`` (a row's value over its partition's sum).
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from snowflake_emulator.translator.constants import WRITE_DIALECT
from snowflake_emulator.translator.errors import TranslationError

# Range of a signed 64-bit value; the sum of row hashes is folded into it.
_HASH_AGG_MODULUS = 9223372036854775808


def _rewrite_hash_agg(node: exp.Expression) -> exp.Expression:
    """``HASH_AGG(...)`` -> a folded ``SUM(HASH(...))``.

    DuckDB has no ``hash_agg`` aggregate. ``SUM(HASH(...))`` is order-independent, counts
    duplicate rows, and hashes NULLs too (DuckDB's ``HASH`` never returns NULL), which is the
    contract Snowflake documents. The sum is folded into a signed 64-bit value, and an empty
    input -- whose sum is NULL -- becomes the documented ``0``.
    """
    if isinstance(node.parent, exp.Window):
        raise TranslationError(
            "HASH_AGG as a window function is not supported by the emulator; use it as a "
            "GROUP BY aggregate."
        )
    arguments = [node.this, *(node.args.get("expressions") or [])]
    if any(isinstance(argument, exp.Star) for argument in arguments):
        raise TranslationError(
            "HASH_AGG(*) is not supported by the emulator; name the columns to hash."
        )
    rendered = ", ".join(
        argument.sql(dialect=WRITE_DIALECT)
        for argument in arguments
        if isinstance(argument, exp.Expression)
    )
    return sqlglot.parse_one(
        f"COALESCE(CAST(SUM(HASH({rendered})) % {_HASH_AGG_MODULUS} AS BIGINT), 0)",
        read=WRITE_DIALECT,
    )


def _is_ratio_to_report(value: exp.Expression | None) -> bool:
    """True when a window's function is ``RATIO_TO_REPORT`` (any other window is not ours)."""
    return (
        isinstance(value, exp.Anonymous)
        and str(value.name).upper() == "RATIO_TO_REPORT"
    )


def _rewrite_ratio_to_report(node: exp.Expression) -> exp.Expression:
    """``RATIO_TO_REPORT(x) OVER (PARTITION BY ...)`` -> ``x / SUM(x) OVER (...)``.

    Snowflake documents the result as the row's value divided by the partition's sum, and says
    an ``ORDER BY`` inside ``OVER`` does not affect it, so the window is rebuilt with the same
    partitioning and no ordering. A partition that sums to zero yields NULL rather than a
    division error.
    """
    if not isinstance(node, exp.Window) or not _is_ratio_to_report(node.this):
        return node
    inner = node.this
    if len(inner.expressions) != 1:
        raise TranslationError("RATIO_TO_REPORT takes a single expression.")
    value_sql = inner.expressions[0].sql(dialect=WRITE_DIALECT)
    partition = [
        part.sql(dialect=WRITE_DIALECT)
        for part in (node.args.get("partition_by") or [])
    ]
    over = f"PARTITION BY {', '.join(partition)}" if partition else ""
    return sqlglot.parse_one(
        f"{value_sql} / NULLIF(SUM({value_sql}) OVER ({over}), 0)",
        read=WRITE_DIALECT,
    )


def _rewrite_aggregate_functions(node: exp.Expression) -> exp.Expression:
    """Rewrite the Snowflake aggregates and windows sqlglot leaves to DuckDB."""
    if isinstance(node, exp.HashAgg):
        return _rewrite_hash_agg(node)
    if isinstance(node, exp.Window):
        return _rewrite_ratio_to_report(node)
    return node
