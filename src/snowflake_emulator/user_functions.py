"""User-defined functions (``CREATE FUNCTION``) emulated as DuckDB macros.

DuckDB has no SQL-bodied ``CREATE FUNCTION``; it has ``CREATE MACRO``, a bind-time
inlined expression. Macros differ from Snowflake UDFs in two ways this module has to
work around:

* A macro name cannot be overloaded -- a second ``CREATE MACRO f(...)`` under the same
  name is a "Macro Function with name f already exists" error -- while Snowflake
  resolves UDFs by full signature. Every function is therefore stored under a macro name
  mangled with its arity (see ``translator.user_function_macro_name``) and call sites are
  rewritten to match. Two functions that differ only by *argument type* cannot be
  emulated and are rejected loudly rather than resolved to the wrong one.
* A macro carries no Snowflake-side metadata. The declared signature, return type,
  language and body are kept in the per-catalog ``_sf_functions_ext`` side table (the
  same mechanism ``_sf_columns_ext`` uses for declared column types), so
  ``SHOW FUNCTIONS`` / ``DESCRIBE FUNCTION`` can report them and the mapping survives a
  restart against a persistent database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.translator import (
    UserFunctionSpec,
    render_create_macro,
    render_drop_macro,
)


class UserFunctionError(ValueError):
    """Raised for user-defined function DDL the emulator cannot carry out."""


@dataclass(frozen=True)
class StoredFunction:
    """A user-defined function as recorded in a catalog's side table."""

    name: str
    database: str
    schema: str
    arity: int
    macro_name: str
    arguments: tuple[tuple[str, str], ...]  # (parameter name, declared Snowflake type)
    return_type: str
    language: str
    body: str
    created_on: datetime | None

    @property
    def argument_types(self) -> tuple[str, ...]:
        return tuple(declared for _, declared in self.arguments)

    @property
    def show_signature(self) -> str:
        """The ``arguments`` column of ``SHOW FUNCTIONS`` (types only, no names)."""
        return f"{self.name}({', '.join(self.argument_types)}) RETURN {self.return_type}"

    @property
    def describe_signature(self) -> str:
        """The ``arguments`` column of ``DESCRIBE FUNCTION`` (named parameters)."""
        named = ", ".join(f"{name} {declared}" for name, declared in self.arguments)
        return f"{self.name}({named}) RETURN {self.return_type}"


def _signature(spec: UserFunctionSpec) -> tuple[tuple[str, str], ...]:
    return tuple(
        (argument.name, argument.declared_type or "VARCHAR") for argument in spec.arguments
    )


def _stored_function(row: tuple[Any, ...]) -> StoredFunction:
    raw = row[5]
    pairs = json.loads(raw) if raw else []
    return StoredFunction(
        name=str(row[0]),
        schema=str(row[1]),
        database=str(row[2]),
        arity=int(row[3]),
        macro_name=str(row[4]),
        arguments=tuple((str(pair[0]), str(pair[1])) for pair in pairs),
        return_type=str(row[6]),
        language=str(row[7]),
        body=str(row[8]),
        created_on=row[9] if isinstance(row[9], datetime) else None,
    )


def query_user_functions(
    manager: DuckDBManager,
    database: str,
    schema: str | None = None,
    name: str | None = None,
    like: str | None = None,
) -> list[StoredFunction]:
    """Return the recorded user-defined functions matching the given filters."""
    return [
        _stored_function(row)
        for row in manager.query_functions(database, schema=schema, name=name, like=like)
    ]


def execute_create_function(
    manager: DuckDBManager,
    cursor: Any,
    session: SessionContext,
    spec: UserFunctionSpec,
) -> str:
    """Create a user-defined function: the DuckDB macro plus its recorded metadata.

    Validated before anything is executed, so a rejected definition leaves no macro
    behind. Returns the Snowflake status message for the statement.
    """
    database = spec.database or session.database
    schema = spec.schema or session.schema
    displayed = spec.name.upper()
    existing = query_user_functions(manager, database, schema, name=spec.name)
    same_arity = [fn for fn in existing if fn.arity == spec.arity]

    if same_arity and same_arity[0].arguments != _signature(spec):
        raise UserFunctionError(
            f"Function {displayed} already exists with {spec.arity} argument(s) but a "
            "different signature. The emulator cannot overload a function by argument "
            "type, because a DuckDB macro is resolved by name alone."
        )
    if same_arity:
        if not spec.replace:
            if spec.if_not_exists:
                return f"Function {displayed} already exists."
            raise UserFunctionError(f"Function {displayed} already exists.")

    manager.ensure_namespace(database, schema)
    try:
        cursor.execute(render_create_macro(spec, database, schema))
    except Exception as exc:  # noqa: BLE001 - reported as a Snowflake execution error
        raise UserFunctionError(f"Failed to create function {displayed}: {exc}") from exc

    manager.save_function(
        database,
        schema,
        spec.name,
        spec.arity,
        spec.macro_name,
        json.dumps([list(pair) for pair in _signature(spec)]),
        spec.return_type,
        spec.language,
        spec.body,
    )
    return f"Function {displayed} successfully created."


def execute_drop_function(
    manager: DuckDBManager,
    cursor: Any,
    session: SessionContext,
    name: str,
    arity: int | None = None,
    *,
    if_exists: bool = False,
) -> str:
    """Drop a user-defined function and the DuckDB macro backing it.

    Without an argument count every overload of `name` is dropped, matching Snowflake's
    ``DROP FUNCTION <name>``. The signature's *types* are ignored: they are rendered
    differently by Snowflake and DuckDB (``STRING`` vs ``VARCHAR``), so matching on them
    would reject valid drops.
    """
    displayed = name.upper()
    database = session.database
    schema = session.schema
    removed = manager.delete_functions(database, schema, name, arity)
    if not removed:
        if if_exists:
            return f"Function {displayed} does not exist."
        raise UserFunctionError(f"Function {displayed} does not exist.")
    for _, _, macro_name in removed:
        # The recorded metadata is the source of truth, so a macro that is already gone
        # (e.g. its catalog was replaced) must not turn a drop into a failure.
        cursor.execute(render_drop_macro(macro_name, database, schema, if_exists=True))
    return f"Function {displayed} successfully dropped."


class SessionUserFunctionResolver:
    """Resolves user-defined function calls against a session's current namespace.

    sqlglot drops the qualifier from an unqualified-looking call (``db.schema.f(x)`` is
    parsed exactly like ``f(x)``), so resolution is always against the session's current
    database and schema -- which is also where ``CREATE FUNCTION`` without a qualified
    name records the function.
    """

    def __init__(self, manager: DuckDBManager, session: SessionContext) -> None:
        self._manager = manager
        self._session = session
        self._functions: dict[str, dict[int, str]] | None = None

    def resolve(self, name: str, arity: int) -> str | None:
        if self._functions is None:
            self._functions = self._manager.function_arities(
                self._session.database, self._session.schema
            )
        return self._functions.get(name.upper(), {}).get(arity)
