"""Execution-layer error wrapping shared by the statement handlers."""

from __future__ import annotations

from snowflake_emulator.core.errors import ExecutionError, classify_exception
from snowflake_emulator.translator import TranslationError


def as_execution_error(exc: BaseException) -> ExecutionError:
    """Wrap a failure as an ``ExecutionError`` with a Snowflake error classification."""
    if isinstance(exc, TranslationError):
        # Most translation failures are sqlglot parse/transpile errors, but a rewrite may
        # carry the specific Snowflake compile-time condition it reproduces.
        return ExecutionError(
            str(exc),
            errno=exc.errno or "001003",
            sqlstate=exc.sqlstate or "42000",
        )
    errno, sqlstate = classify_exception(exc)
    return ExecutionError(str(exc), errno=errno, sqlstate=sqlstate)
