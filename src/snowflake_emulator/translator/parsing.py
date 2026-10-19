"""Snowflake SQL parsing: statement splitting and the ARRAY_FLATTEN marker."""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp
from sqlglot.tokens import TokenType

from snowflake_emulator import sqlglot_patch  # noqa: F401  (global parser settings)
from snowflake_emulator.translator.constants import (
    READ_DIALECT,
    _ARRAY_FLATTEN_NAME,
    _FLATTEN_MARKER,
)
from snowflake_emulator.translator.errors import TranslationError


def preprocess_array_flatten(sql: str) -> str:
    """Rename ``ARRAY_FLATTEN(x, depth)`` so sqlglot's one-argument limit accepts it.

    sqlglot's ``ArrayFlatten`` node takes a single argument, so the documented
    two-argument Snowflake form fails to parse before any rewrite can run. Only the call's
    name is renamed, located by token position so a literal that mentions it is untouched;
    the translator then rewrites the renamed call (``_rewrite_array_flatten``).
    """
    tokens = sqlglot.Dialect.get_or_raise(READ_DIALECT).tokenize(sql)
    spans = [
        (token.start, token.end)
        for index, token in enumerate(tokens[:-1])
        if token.text.upper() == _ARRAY_FLATTEN_NAME
        and tokens[index + 1].token_type == TokenType.L_PAREN
    ]
    if not spans:
        return sql
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(sql[cursor:start])
        parts.append(_FLATTEN_MARKER)
        cursor = end + 1  # Token.end is the index of the token's last character
    parts.append(sql[cursor:])
    return "".join(parts)


# sqlglot underlines the offending token with ANSI escapes, which say nothing in an HTTP
# error body.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

# Snowflake orders a SHOW's clauses as ``SHOW <object> [LIKE '<pattern>'] [IN <scope>]
# [STARTS WITH '<name>']``, so the MySQL order (``... IN SCHEMA s LIKE 'x'``) does not
# parse. The hint is confined to one statement, so a multi-statement body cannot
# trigger it on an unrelated failure.
_SHOW_CLAUSE_ORDER_RE = re.compile(
    r"^\s*SHOW\b[^;]*\bIN\b[^;]*\bLIKE\b", re.IGNORECASE | re.DOTALL
)


def _parse_error_message(sql: str, exc: Exception) -> str:
    """Explain a parse failure, naming the Snowflake clause order where that is the cause."""
    message = _ANSI_ESCAPE_RE.sub("", str(exc))
    if _SHOW_CLAUSE_ORDER_RE.search(sql):
        message += (
            "\nSnowflake puts LIKE before IN, e.g. "
            "SHOW TABLES LIKE '<pattern>' IN SCHEMA <database>.<schema>."
        )
    return message


def split_statements(sql: str) -> list[exp.Expression]:
    """Split a (possibly multi-statement) Snowflake SQL string into parsed expressions.

    Returning parsed `exp.Expression` objects (rather than re-serialized SQL text)
    avoids a lossy parse -> Snowflake-text -> re-parse -> DuckDB-text round trip,
    which can drop information that doesn't survive an intermediate Snowflake-dialect
    print (e.g. `VARCHAR[]` collapsing to a bare `ARRAY`).
    """
    try:
        expressions = sqlglot.parse(preprocess_array_flatten(sql), read=READ_DIALECT)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a translation error
        raise TranslationError(f"Failed to parse SQL: {_parse_error_message(sql, exc)}") from exc

    statements = [expr for expr in expressions if expr is not None]
    if not statements:
        raise TranslationError("No SQL statements found.")
    return statements
