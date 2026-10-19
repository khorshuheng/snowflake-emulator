"""Execution-layer error wrapping shared by the statement handlers."""

from __future__ import annotations

from snowflake_emulator.core.errors import ExecutionError, classify_exception
from snowflake_emulator.translator import TranslationError


def as_execution_error(exc: BaseException) -> ExecutionError:
    """Wrap a failure as an ``ExecutionError`` with a Snowflake error classification."""
    if isinstance(exc, TranslationError):
        # sqlglot could not parse or transpile the statement: a SQL compilation error.
        return ExecutionError(str(exc), errno="001003", sqlstate="42000")
    errno, sqlstate = classify_exception(exc)
    return ExecutionError(str(exc), errno=errno, sqlstate=sqlstate)
