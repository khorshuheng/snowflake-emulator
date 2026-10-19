"""Snowflake-shaped error types and DuckDB failure classification.

Clients branch on the ``code``/``sqlState`` pair returned with a failed statement
-- dbt's ``retry_on_database_errors``, "already exists" detection in migration
tooling, connector retry logic. DuckDB raises a typed exception hierarchy, but
the emulator previously flattened every failure to ``100132``/``42000``, which
destroyed that signal. This module preserves it.

Only mappings with an established Snowflake equivalent are listed; anything else
falls back to the generic compilation-error pair the emulator used before.
"""

from __future__ import annotations

import duckdb

# Fallback for an unclassified failure: Snowflake's generic "SQL compilation error".
DEFAULT_ERRNO = "100132"
DEFAULT_SQLSTATE = "42000"

# Snowflake error codes / SQLSTATEs. The values mirror what Snowflake itself
# reports for the equivalent condition (see the SQL reference error list). Snowflake
# codes are always six-digit zero-padded strings (e.g. ``002003``, ``001003``), and
# the connector parses ``int(code)`` then re-formats with ``{:06d}``, so padding here
# keeps raw string comparisons (JDBC, telemetry) faithful too.
_CATALOG = ("002003", "42S02")  # object does not exist or not authorized
_BINDER = ("002043", "02000")  # invalid or missing object reference
_PARSER = ("001003", "42000")  # SQL compilation error (syntax)
_CONNECTION = ("250002", "08003")  # connection does not exist
_TRANSACTION = ("250001", "25000")  # invalid transaction state
_NOT_IMPLEMENTED = ("100132", "0A000")  # feature not supported
_DATA = ("100038", "22000")  # numeric value out of range / data exception


class ExecutionError(RuntimeError):
    """Raised when a translated statement fails to execute against DuckDB.

    Carries the Snowflake ``errno``/``sqlstate`` pair so routers can answer with the
    same error classification Snowflake clients branch on. Defaults to the generic
    compilation-error pair for failures the emulator raises itself (stages, duplicate
    databases).
    """

    def __init__(
        self,
        message: str,
        *,
        errno: str = DEFAULT_ERRNO,
        sqlstate: str = DEFAULT_SQLSTATE,
    ) -> None:
        super().__init__(message)
        self.errno = errno
        self.sqlstate = sqlstate


def classify_exception(exc: BaseException) -> tuple[str, str]:
    """Return the ``(errno, sqlstate)`` Snowflake reports for ``exc``.

    ``BinderException`` and ``CatalogException`` are both DuckDB ``ProgrammingError``
    subclasses, so they are tested before the generic fallback. A duplicate JSON key
    inside ``OBJECT_CONSTRUCT`` surfaces as a ``BinderException`` whose message names
    the struct, which Snowflake reports as a data error rather than a bad reference.
    """
    if isinstance(exc, duckdb.BinderException):
        if "Duplicate struct entry name" in str(exc):
            return "100103", "22000"
        return _BINDER
    if isinstance(exc, duckdb.CatalogException):
        return _CATALOG
    if isinstance(exc, (duckdb.ParserException, duckdb.SyntaxException)):
        return _PARSER
    if isinstance(exc, duckdb.ConnectionException):
        return _CONNECTION
    if isinstance(exc, duckdb.TransactionException):
        return _TRANSACTION
    if isinstance(exc, duckdb.NotImplementedException):
        return _NOT_IMPLEMENTED
    if isinstance(
        exc,
        (
            duckdb.ConversionException,
            duckdb.OutOfRangeException,
            duckdb.TypeMismatchException,
        ),
    ):
        return _DATA
    return DEFAULT_ERRNO, DEFAULT_SQLSTATE


def error_code_pair(exc: BaseException) -> tuple[str, str]:
    """Return the ``(errno, sqlstate)`` for any failure surfaced by the emulator.

    An ``ExecutionError`` already carries its own classified pair; everything else is
    classified from its type. Routers use this so both the wire-protocol and SQL API
    error shapes are built the same way.
    """
    errno = getattr(exc, "errno", None)
    sqlstate = getattr(exc, "sqlstate", None)
    if isinstance(errno, str) and isinstance(sqlstate, str):
        return errno, sqlstate
    return classify_exception(exc)
