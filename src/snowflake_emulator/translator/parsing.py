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


def _array_to_string_argument_count(tokens: list, index: int) -> int | None:
    """Number of comma-separated arguments of the call whose ``(`` is at `index`.

    Counted from the tokens rather than the AST because sqlglot drops an extra
    ``ARRAY_TO_STRING`` argument silently, so the parsed call cannot reveal it. None when the
    parentheses never close: the statement is malformed, and the parser's own error is the
    useful one.
    """
    depth = 0
    arguments = 1
    for token in tokens[index:]:
        if token.token_type == TokenType.L_PAREN:
            depth += 1
        elif token.token_type == TokenType.R_PAREN:
            depth -= 1
            if depth == 0:
                return arguments
        elif token.token_type == TokenType.COMMA and depth == 1:
            arguments += 1
    return None


_CREATE_FUNCTION_RE = re.compile(
    r"(?is)^\s*CREATE\s+(OR\s+REPLACE\s+)?(GLOBAL\s+|LOCAL\s+|TEMPORARY\s+|SECURE\s+)*"
    r"FUNCTION\b"
)


def _in_create_function_header(sql: str, position: int) -> bool:
    """True when `position` falls inside a ``CREATE [OR REPLACE] [SECURE] FUNCTION`` header."""
    statement_start = sql.rfind(";", 0, position) + 1
    return bool(_CREATE_FUNCTION_RE.match(sql[statement_start:position]))


def _reject_extra_array_to_string_arguments(tokens: list, sql: str) -> None:
    """Reject a third ``ARRAY_TO_STRING`` argument instead of letting sqlglot drop it.

    Snowflake documents ``ARRAY_TO_STRING(array, separator)``: a NULL element becomes an
    empty string and there is no null-replacement parameter, so a third argument is not
    Snowflake SQL. sqlglot parses it away, which would turn an error into a silently
    different answer.
    """
    for index, token in enumerate(tokens[:-1]):
        if token.text.upper() != "ARRAY_TO_STRING":
            continue
        if tokens[index + 1].token_type != TokenType.L_PAREN:
            continue
        if _in_create_function_header(sql, token.start):
            # A ``CREATE FUNCTION array_to_string(a, b, c)`` signature is not a call: Snowflake
            # lets a user-defined function shadow a built-in name.
            continue
        arguments = _array_to_string_argument_count(tokens, index + 1)
        if arguments is not None and arguments > 2:
            raise TranslationError(
                "ARRAY_TO_STRING takes an array and a separator; Snowflake's documented "
                "signature has no null-replacement argument (a NULL element already becomes "
                "an empty string)."
            )


def preprocess_array_calls(sql: str) -> str:
    """Prepare the Snowflake array calls sqlglot cannot round-trip, before parsing.

    ``ARRAY_FLATTEN(x, depth)`` is renamed, because sqlglot's ``ArrayFlatten`` node takes a
    single argument and the emulator's legacy two-argument form would otherwise fail to parse
    before any rewrite can run. Only the call's name is renamed, located by token position so
    a literal that mentions it is untouched; the translator then rewrites the renamed call
    (``_rewrite_array_flatten``). An ``ARRAY_TO_STRING`` call with more than the documented two
    arguments is rejected here for the same reason: sqlglot drops the extra argument silently.
    """
    tokens = sqlglot.Dialect.get_or_raise(READ_DIALECT).tokenize(sql)
    _reject_extra_array_to_string_arguments(tokens, sql)
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
        expressions = sqlglot.parse(preprocess_array_calls(sql), read=READ_DIALECT)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a translation error
        raise TranslationError(f"Failed to parse SQL: {_parse_error_message(sql, exc)}") from exc

    statements = [expr for expr in expressions if expr is not None]
    if not statements:
        raise TranslationError("No SQL statements found.")
    return statements
