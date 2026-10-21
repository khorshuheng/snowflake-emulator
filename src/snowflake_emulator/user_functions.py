"""User-defined functions (``CREATE FUNCTION``) emulated as DuckDB macros.

DuckDB has no SQL-bodied ``CREATE FUNCTION``; it has ``CREATE MACRO``, a bind-time
inlined expression. Three differences from Snowflake UDFs drive this module:

* **A macro name cannot be overloaded.** A second ``CREATE MACRO f(...)`` under the same
  name is a "Macro Function with name f already exists" error, while Snowflake resolves a
  UDF by full signature. Two functions that differ only by *arity* are handled by mangling
  the name with the arity and rewriting call sites, which the translator does. Two that
  differ only by *argument type* cannot be overloaded that way either, so they are
  dispatched at run time -- see below.
* **A macro carries no Snowflake-side metadata.** The declared signature, return type,
  language and body are kept in the per-catalog ``_sf_functions_ext`` side table (the same
  mechanism ``_sf_columns_ext`` uses for declared column types), so ``SHOW FUNCTIONS`` /
  ``DESCRIBE FUNCTION`` can report them and the mapping survives a restart.
* **A macro body is not overloadable at bind time.** ``CASE WHEN typeof(a) = 'VARCHAR'
  THEN upper(a) ELSE list_reverse(a) END`` cannot work: DuckDB binds *every* branch, and
  each one fails to bind for the other overload's type (``upper(LIST)`` and
  ``list_reverse(VARCHAR)`` are both errors). An explicit ``CAST`` inside a branch does
  not help either, since ``CAST(... AS INTEGER[])`` is itself rejected.

Type overloading therefore goes through a Python function. ``create_function`` accepts a
generic signature (``parameters=None``), and the callback receives the real Python
values, so which overload applies is decided from the argument types at call time. Each
overload's body is stored as a separate macro and the dispatcher runs the matching one.

The argument *types* are passed in, not inferred from the values, because a value cannot
carry them: the emulator stores a Snowflake ``ARRAY`` as a DuckDB ``JSON`` column, and
DuckDB hands that to a Python callback as plain text -- indistinguishable from a string,
so the ``STRING`` overload would be chosen for an array. The translator therefore appends
``typeof(arg)`` for each argument of a dispatched call, giving the callback each
argument's SQL type (``JSON``, ``VARCHAR[]``, ``VARCHAR``, and ``"NULL"`` for an untyped
NULL). Two consequences of the JSON representation are handled in the dispatcher:

* a JSON argument is cast to ``VARCHAR[]`` before running an ARRAY-typed overload, and
  every array argument is normalized the same way, since ``JSON[] = VARCHAR[]`` fails;
* a ``NULL`` in a typed column still reports that column's type, so it selects that
  column's overload; only an untyped ``NULL`` is a wildcard, and it is accepted when the
  overloads it matches agree on the answer.

The trade-offs, all deliberate:

* the dispatcher is a Python callback plus a nested query per row, not an inlined macro;
* all overloads of one (name, arity) must share a return type, because the native
  function needs one concrete declared return type (``create_function`` has no ``ANY``);
* a Python function is not stored in the DuckDB catalog, so it is re-registered on first
  use after a restart from the side table (which is the source of truth for bodies);
* a function becomes reachable through a dispatcher once it is overloaded and stays that
  way for the life of the process, because a registered native function cannot be
  unregistered;
* a *single* overload keeps using a plain macro; an overloaded call goes through the
  dispatcher, which is told each argument's SQL type and normalizes an ARRAY argument to
  ``VARCHAR[]`` so the body sees an ``ARRAY`` column and an ``ARRAY_CONSTRUCT`` literal the
  same way.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import duckdb
from duckdb.func import FunctionNullHandling

from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.translator import (
    UserFunctionArgument,
    UserFunctionCall,
    UserFunctionSpec,
    render_create_macro,
    render_drop_macro,
    render_qualified_name,
    _duckdb_type_text,
    translate_function_body,
    user_function_dispatch_name,
    user_function_macro_name,
    user_function_sub_macro_name,
)
from snowflake_emulator.type_mapping import duckdb_type_to_snowflake_sql_type


class UserFunctionError(ValueError):
    """Raised for user-defined function DDL the emulator cannot carry out."""


def type_kind(declared_type: str) -> str:
    """Reduce a declared argument type to the category dispatch compares on.

    Two overloads in the same category (``NUMBER`` vs ``FLOAT``) cannot be told apart
    from a Python value, so they are reported as ambiguous rather than guessed at.
    """
    name = declared_type.strip().upper()
    if name.endswith("[]") or name.startswith(("ARRAY", "LIST")):
        return "ARRAY"
    if name.startswith(("VARCHAR", "CHAR", "TEXT", "STRING", "ENUM")):
        return "STRING"
    if name.startswith(
        (
            "NUMBER",
            "DECIMAL",
            "NUMERIC",
            "INT",
            "BIGINT",
            "SMALLINT",
            "TINYINT",
            "HUGEINT",
            "FLOAT",
            "DOUBLE",
            "REAL",
        )
    ):
        return "NUMBER"
    if name.startswith("BOOL"):
        return "BOOLEAN"
    if name.startswith(("TIMESTAMP", "DATE", "TIME")):
        return "TEMPORAL"
    if name.startswith(("BINARY", "VARBINARY", "BLOB")):
        return "BINARY"
    if name.startswith(("VARIANT", "OBJECT", "JSON", "MAP", "STRUCT")):
        return "OBJECT"
    return "OTHER"


def value_kind(value: Any) -> str | None:
    """Reduce a runtime Python value to a dispatch category; ``None`` is a wildcard.

    A SQL ``NULL`` carries no type, so it matches any overload -- unless it leaves more
    than one candidate, which is resolved only if the candidates agree.
    """
    if value is None:
        return None
    # bool is a subclass of int, so it has to be tested first.
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, (list, tuple)):
        return "ARRAY"
    if isinstance(value, str):
        return "STRING"
    if isinstance(value, (int, float, Decimal)):
        return "NUMBER"
    if isinstance(value, (datetime, date, time, timedelta)):
        return "TEMPORAL"
    if isinstance(value, (bytes, bytearray)):
        return "BINARY"
    if isinstance(value, dict):
        return "OBJECT"
    return "OTHER"


def describe_kind(kind: str | None) -> str:
    """Name a dispatch category for an error message.

    Deliberately not the category's own name: ``STRING`` would be read as the SQL type
    ``STRING``, which is not what a declared ``VARCHAR`` overload is compared against.
    """
    return {
        "STRING": "text",
        "NUMBER": "number",
        "ARRAY": "array",
        "BOOLEAN": "boolean",
        "TEMPORAL": "date/time",
        "BINARY": "binary",
        "OBJECT": "object",
        "JSON": "json",
        "OTHER": "unknown type",
    }.get(kind or "", "NULL")


@dataclass(frozen=True)
class StoredFunction:
    """A user-defined function as recorded in a catalog's side table."""

    name: str
    database: str
    schema: str
    arity: int
    signature: str
    callable_name: str
    sub_macro_name: str | None
    arguments: tuple[tuple[str, str], ...]  # (parameter name, declared Snowflake type)
    return_type: str
    return_type_duckdb: str | None
    language: str
    body: str
    created_on: datetime | None

    @property
    def argument_types(self) -> tuple[str, ...]:
        return tuple(declared for _, declared in self.arguments)

    @property
    def argument_kinds(self) -> tuple[str, ...]:
        return tuple(type_kind(declared) for declared in self.argument_types)

    @property
    def show_signature(self) -> str:
        """The ``arguments`` column of ``SHOW FUNCTIONS`` (types only, no names)."""
        return f"{self.name}({', '.join(self.argument_types)}) RETURN {self.return_type}"

    @property
    def describe_signature(self) -> str:
        """The ``arguments`` column of ``DESCRIBE FUNCTION`` (named parameters)."""
        named = ", ".join(f"{name} {declared}" for name, declared in self.arguments)
        return f"{self.name}({named}) RETURN {self.return_type}"

    def to_spec(self) -> UserFunctionSpec:
        """Rebuild the spec a macro for this function can be (re)rendered from.

        The body is re-translated rather than cached as DuckDB SQL, so the stored
        Snowflake source stays the single source of truth.
        """
        return UserFunctionSpec(
            name=self.name,
            arguments=tuple(
                UserFunctionArgument(name=name, declared_type=declared)
                for name, declared in self.arguments
            ),
            return_type=self.return_type,
            return_type_duckdb=self.return_type_duckdb or self.return_type,
            language=self.language,
            body=self.body,
            body_sql=translate_function_body(
                self.body,
                [
                    (name, _duckdb_type_text(declared))
                    for name, declared in self.arguments
                ],
            ),
            database=self.database,
            schema=self.schema,
            replace=True,
        )


def kind_from_duckdb_type(duckdb_type: str) -> str | None:
    """Reduce a DuckDB type name (as ``typeof()`` reports it) to a dispatch category.

    This is the exact signal, and it is a *static* one: the emulator stores a Snowflake
    ``ARRAY`` as a DuckDB ``JSON`` column, and DuckDB hands that value to a Python
    callback as plain text, so the value cannot say whether the argument was an array.
    Its SQL type can -- and it stays ``JSON`` even for a NULL, so a NULL in a typed column
    still selects that column's overload rather than becoming a wildcard.
    """
    # DuckDB reports an untyped NULL as the quoted "NULL", and a concrete type otherwise.
    name = duckdb_type.strip().strip('"').upper()
    if name in ("NULL", "SQLNULL", "UNKNOWN"):
        return None
    if name == "JSON":
        return "JSON"
    return type_kind(duckdb_type_to_snowflake_sql_type(name))


def kinds_match(declared: str, actual: str | None) -> bool:
    """Return True when a runtime category satisfies a declared overload's category."""
    if actual is None:
        return True
    if declared == actual:
        return True
    # ARRAY and OBJECT are both stored as DuckDB JSON, so a JSON argument cannot be told
    # apart from either; it is accepted by whichever of the two is declared.
    return actual == "JSON" and declared in ("ARRAY", "OBJECT")


@dataclass(frozen=True)
class Overload:
    """One overload behind a dispatcher: the types it declares and its body's macro."""

    declared_types: tuple[str, ...]
    macro: str


@dataclass
class DispatchGroup:
    """The run-time dispatcher shared by every overload of one (name, arity)."""

    callable_name: str
    database: str
    schema: str
    function_name: str
    arity: int
    overloads: dict[tuple[str, ...], Overload] = field(default_factory=dict)

    def matching(self, kinds: tuple[str | None, ...]) -> list[tuple[str, ...]]:
        """Return the overload keys whose argument categories accept `kinds`.

        A lone overload needs no dispatch, which also keeps it as forgiving as the macro
        path: DuckDB gets to coerce the arguments rather than this rejecting a value of
        an unexpected type.
        """
        if len(self.overloads) == 1:
            return list(self.overloads)
        return [
            declared
            for declared in self.overloads
            if all(
                kinds_match(wanted, actual)
                for wanted, actual in zip(declared, kinds)
            )
        ]

    def ambiguity(self, kinds: tuple[str | None, ...]) -> Exception:
        """Build the error for a call no single overload can be chosen for."""
        signatures = ", ".join(
            f"({', '.join(overload.declared_types)})" for overload in self.overloads.values()
        )
        rendered = ", ".join(describe_kind(kind) for kind in kinds)
        if not self.matching(kinds):
            return UserFunctionError(
                f"{self.function_name} has no overload matching ({rendered}). "
                f"Declared overloads: {signatures}."
            )
        return UserFunctionError(
            f"{self.function_name} is ambiguous for ({rendered}): it matches the "
            f"overloads {signatures}. A NULL carries no type, so pass a non-NULL value "
            "of the type you mean to select an overload."
        )


def _signature(spec: UserFunctionSpec) -> tuple[tuple[str, str], ...]:
    return tuple(
        (argument.name, argument.declared_type or "VARCHAR") for argument in spec.arguments
    )


def _stored_function(row: tuple[Any, ...]) -> StoredFunction:
    raw = row[7]
    pairs = json.loads(raw) if raw else []
    return StoredFunction(
        name=str(row[0]),
        schema=str(row[1]),
        database=str(row[2]),
        arity=int(row[3]),
        signature=str(row[4] or ""),
        callable_name=str(row[5]),
        sub_macro_name=None if row[6] is None else str(row[6]),
        arguments=tuple((str(pair[0]), str(pair[1])) for pair in pairs),
        return_type=str(row[8]),
        return_type_duckdb=None if row[9] is None else str(row[9]),
        language=str(row[10]),
        body=str(row[11]),
        created_on=row[12] if isinstance(row[12], datetime) else None,
    )


def query_user_functions(
    manager: DuckDBManager,
    database: str,
    schema: str | None = None,
    name: str | None = None,
    like: str | None = None,
    signature: str | None = None,
) -> list[StoredFunction]:
    """Return the recorded user-defined functions matching the given filters."""
    return [
        _stored_function(row)
        for row in manager.query_functions(
            database, schema=schema, name=name, like=like, signature=signature
        )
    ]


def _dispatch_groups(manager: DuckDBManager) -> dict[str, DispatchGroup]:
    return manager.dispatch_groups


def _make_dispatcher(manager: DuckDBManager, callable_name: str):
    """Build the Python callback registered for a dispatched (name, arity).

    The callback reads the current overload mapping from the manager at call time, so a
    dispatcher that is already registered keeps working when overloads are added or
    dropped (a native function cannot be re-registered under the same name).
    """

    def dispatch(*args: Any) -> Any:
        group = _dispatch_groups(manager).get(callable_name)
        if group is None:
            raise UserFunctionError(f"{callable_name} is no longer defined.")
        # The call site appends ``typeof(arg)`` for each argument; fall back to the Python
        # values if it did not, so the dispatcher still works when called directly.
        arity = group.arity
        values = args[:arity]
        reported = args[arity:] if len(args) == arity * 2 else ()
        kinds = tuple(
            kind_from_duckdb_type(str(name)) for name in reported
        ) if reported else tuple(value_kind(value) for value in values)

        matches = group.matching(kinds)
        if len(matches) == 1:
            overload = group.overloads[matches[0]]
            return _evaluate(manager, overload, values, kinds)
        if not matches:
            raise group.ambiguity(kinds)
        # A NULL argument matches every overload, since a NULL carries no type. Run them
        # all and accept the outcome only when they agree, so a NULL-safe function keeps
        # working on NULL columns instead of failing outright.
        results: list[Any] = []
        for match in matches:
            try:
                results.append(
                    _evaluate(manager, group.overloads[match], values, kinds)
                )
            except Exception:  # noqa: BLE001 - a body may legitimately reject a NULL
                continue
        if results and all(_same_result(result, results[0]) for result in results):
            return results[0]
        raise group.ambiguity(kinds)

    return dispatch


def _evaluate(
    manager: DuckDBManager,
    overload: Overload,
    values: tuple[Any, ...],
    kinds: tuple[str | None, ...],
) -> Any:
    """Run one overload's macro over the given argument values.

    A Snowflake ``ARRAY`` reaches the emulator as a JSON value, while the body expects a
    DuckDB LIST. The cast is applied here, where the argument's type is known, and it has
    to normalize *every* array argument the same way: a JSON value compared against a
    plain ``VARCHAR[]`` operand fails, so both sides are presented as ``VARCHAR[]`` (which
    is also what an ``ARRAY_CONSTRUCT`` literal is). An OBJECT-typed parameter is left as
    JSON, since that is already its representation.
    """
    placeholders = ", ".join(
        "CAST(? AS VARCHAR[])"
        if kind == "JSON" and type_kind(declared) == "ARRAY"
        else "?"
        for declared, kind in zip(overload.declared_types, kinds)
    )
    cursor = manager.connection.cursor()
    return cursor.execute(
        f"SELECT {overload.macro}({placeholders})", list(values)
    ).fetchone()[0]


def _same_result(left: Any, right: Any) -> bool:
    """Compare two overload results, tolerating values that cannot be compared."""
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001 - incomparable types simply disagree
        return False


def _sub_macro_index(signature: str) -> int:
    """Return a stable sub-macro index for an overload's signature.

    Derived from the signature rather than its position, so adding or dropping one
    overload never renames the others' macros (which would strand the old ones).
    """
    return zlib.crc32(signature.encode("utf-8"))


def materialize_dispatch(
    manager: DuckDBManager, overloads: list[StoredFunction]
) -> str:
    """Create/refresh the dispatcher and sub-macros for one overloaded (name, arity).

    Returns the dispatcher's name. Idempotent, so it doubles as restart recovery: the
    sub-macro bodies are re-rendered from the stored Snowflake source and the native
    function is (re)registered on this process's connection.
    """
    first = overloads[0]
    callable_name = user_function_dispatch_name(
        first.database, first.schema, first.name, first.arity
    )
    missing = [fn.signature for fn in overloads if not fn.return_type_duckdb]
    if missing:
        raise UserFunctionError(
            f"{first.name} cannot be dispatched because the DuckDB return type of "
            f"{', '.join(missing)} is not recorded. Recreate the function to overload it."
        )
    return_types = {fn.return_type_duckdb for fn in overloads}
    if len(return_types) > 1:
        raise UserFunctionError(
            f"All overloads of {first.name} must share a return type to be dispatched, "
            f"but {', '.join(sorted(return_types))} were declared."
        )

    indices: dict[int, str] = {}
    overloads_by_kinds: dict[tuple[str, ...], Overload] = {}
    sub_macros: dict[str, str] = {}
    for fn in sorted(overloads, key=lambda fn: fn.signature):
        index = _sub_macro_index(fn.signature)
        if index in indices:
            raise UserFunctionError(
                f"Overloads {indices[index]} and {fn.signature} of {first.name} collide on "
                "their internal macro name; rename one of them."
            )
        indices[index] = fn.signature
        sub_macro = user_function_sub_macro_name(
            first.database, first.schema, first.name, first.arity, index
        )
        sub_macros[fn.signature] = sub_macro
        manager.connection.cursor().execute(
            render_create_macro(
                fn.to_spec(), first.database, first.schema, macro_name=sub_macro
            )
        )
        overloads_by_kinds[fn.argument_kinds] = Overload(
            declared_types=fn.argument_types,
            macro=render_qualified_name(first.database, first.schema, sub_macro),
        )

    group = _dispatch_groups(manager).get(callable_name)
    if group is None:
        group = DispatchGroup(
            callable_name=callable_name,
            database=first.database,
            schema=first.schema,
            function_name=first.name,
            arity=first.arity,
        )
        _dispatch_groups(manager)[callable_name] = group
        # Point the dispatcher at the overloads before it can be called, then register
        # it once for the life of the process.
        group.overloads = overloads_by_kinds
        manager.connection.create_function(
            callable_name,
            _make_dispatcher(manager, callable_name),
            None,
            duckdb.sqltype(first.return_type_duckdb),
            # NULLs have to reach the callback: a NULL-safe UDF (EQUAL_NULL semantics)
            # only works when it is given the NULLs, and the default handling returns
            # NULL without ever calling it.
            null_handling=FunctionNullHandling.SPECIAL,
        )
    else:
        group.overloads = overloads_by_kinds
    # The side table is the source of truth for how the group is reached, and a restart
    # has to be able to rebuild this mapping, so it is written from the same dict the
    # dispatcher was just given.
    manager.set_function_dispatch(
        first.database,
        first.schema,
        first.name,
        first.arity,
        callable_name,
        sub_macros,
    )
    return callable_name


def ensure_dispatchers(manager: DuckDBManager, database: str, schema: str) -> None:
    """Register the run-time dispatchers for a schema's overloaded functions.

    Native functions live only for the life of the process, so this runs before a
    statement is translated, which is also what makes a persisted catalog usable again
    after a restart.
    """
    grouped: dict[tuple[str, int], list[StoredFunction]] = {}
    for fn in query_user_functions(manager, database, schema):
        grouped.setdefault((fn.name, fn.arity), []).append(fn)
    for overloads in grouped.values():
        if len(overloads) > 1:
            materialize_dispatch(manager, overloads)


def execute_create_function(
    manager: DuckDBManager,
    cursor: Any,
    session: SessionContext,
    spec: UserFunctionSpec,
) -> str:
    """Create a user-defined function: the DuckDB macro (or dispatcher) plus metadata.

    Validated before anything is executed, so a rejected definition leaves no macro
    behind. Returns the Snowflake status message for the statement.
    """
    database = spec.database or session.database
    schema = spec.schema or session.schema
    displayed = spec.name.upper()
    existing = query_user_functions(manager, database, schema, name=spec.name)
    same_arity = [fn for fn in existing if fn.arity == spec.arity]
    exact = [fn for fn in same_arity if fn.signature == spec.type_signature]

    if exact and not spec.replace:
        if spec.if_not_exists:
            return f"Function {displayed} already exists."
        raise UserFunctionError(f"Function {displayed} already exists.")

    manager.ensure_namespace(database, schema)
    # A name/arity backed by one overload is a plain macro; adding a differently typed
    # overload promotes the whole group to a run-time dispatcher.
    overloaded = bool([fn for fn in same_arity if fn.signature != spec.type_signature])
    if overloaded:
        conflicts = [fn for fn in same_arity if fn.return_type != spec.return_type]
        if conflicts:
            raise UserFunctionError(
                f"Function {displayed} already exists with {spec.arity} argument(s) "
                f"returning {conflicts[0].return_type}. Overloads that differ only by "
                f"argument type must share a return type, because the emulator "
                f"dispatches them through one DuckDB function; this one returns "
                f"{spec.return_type}."
            )

    try:
        if overloaded:
            # Drop the single-overload macro the group was reachable through before it
            # became a dispatcher, then rebuild everything from the stored metadata.
            for fn in same_arity:
                if fn.sub_macro_name is None:
                    cursor.execute(
                        render_drop_macro(fn.callable_name, database, schema, if_exists=True)
                    )
        else:
            cursor.execute(render_create_macro(spec, database, schema))
    except Exception as exc:  # noqa: BLE001 - reported as a Snowflake execution error
        raise UserFunctionError(f"Failed to create function {displayed}: {exc}") from exc

    manager.save_function(
        database,
        schema,
        name=spec.name,
        num_arguments=spec.arity,
        signature=spec.type_signature,
        callable_name=(
            spec.macro_name
            if not overloaded
            else user_function_dispatch_name(database, schema, spec.name, spec.arity)
        ),
        sub_macro_name=None,
        arguments=json.dumps([list(pair) for pair in _signature(spec)]),
        return_type=spec.return_type,
        return_type_duckdb=spec.return_type_duckdb,
        language=spec.language,
        body=spec.body,
    )
    if overloaded:
        materialize_dispatch(
            manager, query_user_functions(manager, database, schema, name=spec.name)
        )
    return f"Function {displayed} successfully created."


def execute_drop_function(
    manager: DuckDBManager,
    cursor: Any,
    session: SessionContext,
    name: str,
    arity: int | None = None,
    *,
    signature: str | None = None,
    if_exists: bool = False,
) -> str:
    """Drop a user-defined function and the DuckDB macro backing it.

    Without an argument count every overload of `name` is dropped, matching Snowflake's
    ``DROP FUNCTION <name>``. `signature` narrows it to one overload of a name/arity that
    carries several.
    """
    displayed = name.upper()
    database = session.database
    schema = session.schema
    removed = manager.delete_functions(database, schema, name, arity, signature)
    if not removed:
        if if_exists:
            return f"Function {displayed} does not exist."
        raise UserFunctionError(f"Function {displayed} does not exist.")

    for _, removed_arity, sub_macro_name in removed:
        # For a dispatched group the overload's body lives in its own sub-macro; for a
        # single overload the macro *is* the callable. A macro that is already gone must
        # not turn a drop into a failure, since the recorded metadata is the source of
        # truth.
        target = sub_macro_name or user_function_macro_name(name, removed_arity)
        cursor.execute(render_drop_macro(target, database, schema, if_exists=True))

    # A group that is already dispatched keeps its dispatcher (a registered native
    # function cannot be unregistered), but has to forget the overload just dropped, so
    # it is rebuilt from whatever remains.
    grouped: dict[tuple[str, int], list[StoredFunction]] = {}
    for fn in query_user_functions(manager, database, schema, name=name):
        grouped.setdefault((fn.name, fn.arity), []).append(fn)
    for overloads in grouped.values():
        if any(fn.sub_macro_name for fn in overloads):
            materialize_dispatch(manager, overloads)
    return f"Function {displayed} successfully dropped."


class SessionUserFunctionResolver:
    """Resolves user-defined function calls against a session's current namespace.

    sqlglot drops the qualifier from a call (``db.schema.f(x)`` is parsed exactly like
    ``f(x)``), so resolution is always against the session's current database and schema
    -- which is also where ``CREATE FUNCTION`` without a qualified name records the
    function.
    """

    def __init__(self, manager: DuckDBManager, session: SessionContext) -> None:
        self._manager = manager
        self._session = session
        self._functions: dict[str, dict[int, str]] | None = None

    def resolve(self, name: str, arity: int) -> UserFunctionCall | None:
        if self._functions is None:
            # Native dispatchers are per-process, so make sure this schema's overloaded
            # functions are callable before any statement referring to them runs.
            ensure_dispatchers(
                self._manager, self._session.database, self._session.schema
            )
            self._functions = self._manager.function_arities(
                self._session.database, self._session.schema
            )
        callable_name = self._functions.get(name.upper(), {}).get(arity)
        if callable_name is None:
            return None
        return UserFunctionCall(
            callable_name=callable_name,
            dispatch=callable_name in self._manager.dispatch_groups,
        )
