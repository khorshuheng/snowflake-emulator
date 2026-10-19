"""Backward-compatible import location for error classification helpers.

The implementation lives in :mod:`snowflake_emulator.core.errors`; this module keeps
existing imports (``from snowflake_emulator.errors import error_code_pair``) working.
"""

from snowflake_emulator.core.errors import (
    DEFAULT_ERRNO,
    DEFAULT_SQLSTATE,
    ExecutionError,
    classify_exception,
    error_code_pair,
)

__all__ = [
    "DEFAULT_ERRNO",
    "DEFAULT_SQLSTATE",
    "ExecutionError",
    "classify_exception",
    "error_code_pair",
]
