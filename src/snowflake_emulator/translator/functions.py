"""Snowflake SQL user-defined function translation and DuckDB macro rendering."""

from __future__ import annotations

import sqlglot
from dataclasses import dataclass
from typing import Protocol, Sequence

from sqlglot import exp

from snowflake_emulator.translator.constants import (
    READ_DIALECT,
    WRITE_DIALECT,
    _SIMPLE_IDENTIFIER_RE,
)
from snowflake_emulator.translator.errors import TranslationError
from snowflake_emulator.translator.parsing import preprocess_array_calls
from snowflake_emulator.translator.transforms import _apply_transforms
from snowflake_emulator.type_mapping import duckdb_type_to_snowflake_sql_type


USER_FUNCTION_MACRO_PREFIX = "__SFUDF_"


def user_function_macro_name(name: str, arity: int) -> str:
    """Return the DuckDB macro name backing a function with a single overload."""
    return f"{USER_FUNCTION_MACRO_PREFIX}{name}__{arity}"


def user_function_dispatch_name(database: str, schema: str, name: str, arity: int) -> str:
    """Return the name of the Python function dispatching an overloaded (name, arity).

    A DuckDB function registered through the Python API is *not* schema-scoped, unlike a
    macro, so the database and schema are part of the name to keep two schemas that each
    define the same function from colliding.
    """
    return f"{USER_FUNCTION_MACRO_PREFIX}{database}_{schema}_{name}__{arity}"


def user_function_sub_macro_name(
    database: str, schema: str, name: str, arity: int, index: int
) -> str:
    """Return the macro name holding one overload's body behind a dispatcher."""
    return f"{user_function_dispatch_name(database, schema, name, arity)}__{index}"


def _identifier(name: str) -> exp.Identifier:
    """Build an identifier node, quoting it only when it is not a plain identifier."""
    if _SIMPLE_IDENTIFIER_RE.match(name):
        return exp.to_identifier(name)
    return exp.to_identifier(name, quoted=True)


def _snowflake_name(identifier: exp.Identifier) -> str:
    """Return an identifier's name in Snowflake casing (unquoted names are uppercased)."""
    return identifier.name if identifier.args.get("quoted") else identifier.name.upper()


def _render_identifier(name: str) -> str:
    """Render an identifier as SQL text, quoted only when it needs to be."""
    return _identifier(name).sql(dialect=WRITE_DIALECT)


def render_qualified_name(*parts: str) -> str:
    """Render a dotted, per-part-quoted name for use in a DuckDB statement."""
    return ".".join(_render_identifier(part) for part in parts)


@dataclass(frozen=True)
class UserFunctionArgument:
    """A user-defined function's parameter, in Snowflake identifier casing.

    ``name`` is the *effective* name after Snowflake's identifier casing is applied
    (unquoted names are uppercased), so it matches the names the translated body refers
    to.
    """

    name: str
    declared_type: str | None
    #: The same declared type in DuckDB's spelling, used to coerce a call's argument inside
    #: the macro (Snowflake coerces every argument to the parameter's declared type).
    duckdb_type: str | None = None


@dataclass(frozen=True)
class UserFunctionSpec:
    """A Snowflake SQL UDF, translated into the DuckDB macro that emulates it."""

    name: str
    arguments: tuple[UserFunctionArgument, ...]
    return_type: str
    #: The declared return type in DuckDB's spelling, which is what the Python API needs
    #: to register the dispatcher for an overloaded function.
    return_type_duckdb: str
    language: str
    body: str
    body_sql: str
    database: str | None = None
    schema: str | None = None
    replace: bool = False
    if_not_exists: bool = False

    @property
    def arity(self) -> int:
        return len(self.arguments)

    @property
    def macro_name(self) -> str:
        return user_function_macro_name(self.name, self.arity)

    @property
    def type_signature(self) -> str:
        """The declared argument types, identifying one overload of a name/arity."""
        types = [argument.declared_type or "VARCHAR" for argument in self.arguments]
        return "|".join(types)

    @property
    def show_signature(self) -> str:
        """The argument list as Snowflake's ``SHOW FUNCTIONS`` reports it."""
        types = [argument.declared_type or "VARCHAR" for argument in self.arguments]
        return f"{self.name}({', '.join(types)}) RETURN {self.return_type}"


def _function_properties(statement: exp.Create) -> tuple[exp.DataType | None, str, bool]:
    """Return a ``CREATE FUNCTION``'s declared ``(return type, language, is_table)``."""
    returns: exp.DataType | None = None
    language = "SQL"
    is_table = False
    properties = statement.args.get("properties")
    for prop in properties.expressions if properties else []:
        if isinstance(prop, exp.ReturnsProperty):
            is_table = bool(prop.args.get("is_table"))
            if isinstance(prop.this, exp.DataType):
                returns = prop.this
        elif isinstance(prop, exp.LanguageProperty) and prop.this is not None:
            language = (prop.this.name or "SQL").upper() or "SQL"
    return returns, language, is_table


def _function_body_text(statement: exp.Create) -> str:
    """Return a ``CREATE FUNCTION``'s body as written.

    Snowflake accepts the body either dollar-quoted (``AS $$ ... $$``, parsed as a
    ``RawString``) or as a plain string literal (``AS 'x + 1'``).
    """
    body = statement.args.get("expression")
    if isinstance(body, exp.RawString):
        return str(body.this)
    if isinstance(body, exp.Literal) and body.is_string:
        return str(body.this)
    raise TranslationError("CREATE FUNCTION requires a body (AS $$ ... $$).")


def normalize_function_signature(type_nodes: list[exp.Expression]) -> str:
    """Normalize a ``DROP``/``DESCRIBE FUNCTION`` argument list to the stored signature.

    The types in a signature are written the way the client spelled them (``STRING``,
    ``NUMBER(10,2)``) and may parse as a ``DataType`` or a bare ``Identifier``, so they
    are pushed through the same Snowflake-to-DuckDB normalization used when the function
    was recorded. Without this, ``DROP FUNCTION f(STRING, STRING)`` would not match the
    stored ``VARCHAR|VARCHAR`` and would silently drop the wrong overload -- or none.
    """
    parts: list[str] = []
    for node in type_nodes:
        if isinstance(node, exp.DataType):
            duckdb_type = node.sql(dialect=WRITE_DIALECT)
        else:
            # ``DESCRIBE FUNCTION f(ARRAY, ARRAY)`` parses the types as identifiers.
            built = exp.DataType.build(node.sql(dialect=READ_DIALECT), dialect=READ_DIALECT)
            duckdb_type = built.sql(dialect=WRITE_DIALECT)
        parts.append(duckdb_type_to_snowflake_sql_type(duckdb_type))
    return "|".join(parts)


def _duckdb_type_text(declared_type: str | None) -> str | None:
    """A Snowflake type spelling rendered in DuckDB's (``NUMBER(38,0)`` -> ``DECIMAL(38,0)``)."""
    if not declared_type:
        return None
    try:
        return exp.DataType.build(declared_type, dialect=READ_DIALECT).sql(
            dialect=WRITE_DIALECT
        )
    except Exception:  # noqa: BLE001 - an unparsable persisted type is simply not coerced
        return None


def _cast_parameters(
    expression: exp.Expression, parameters: Sequence[tuple[str, str | None]]
) -> exp.Expression:
    """Coerce each parameter reference to its declared type, as Snowflake does on a call.

    A DuckDB macro is untyped, so the argument keeps whatever type it was called with: a
    ``VARCHAR`` parameter would see a DATE from ``shout(CURRENT_DATE())`` and ``UPPER`` then
    failed with a binder error, where Snowflake has already coerced the DATE to VARCHAR.
    """
    types = {
        name.upper(): duckdb_type
        for name, duckdb_type in parameters
        if duckdb_type
    }
    if not types:
        return expression

    def build(duckdb_type: str) -> exp.DataType | None:
        # An unparameterised Snowflake ARRAY/OBJECT renders as ``[]`` in DuckDB's dialect,
        # which is not a type: such a parameter keeps the caller's value instead of failing
        # the whole CREATE FUNCTION.
        try:
            kind = exp.DataType.build(duckdb_type, dialect=WRITE_DIALECT)
        except Exception:  # noqa: BLE001 - an unusable spelling simply means "do not coerce"
            return None
        return kind if isinstance(kind, exp.DataType) and kind.this is not None else None

    def replace(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Column) or node.table:
            return node
        duckdb_type = types.get(node.name.upper())
        if duckdb_type is None:
            return node
        parent = node.parent
        if isinstance(parent, exp.Cast) and parent.to.sql(dialect=WRITE_DIALECT) == duckdb_type:
            return node  # already coerced
        kind = build(duckdb_type)
        if kind is None:
            return node
        return exp.cast(node.copy(), kind, dialect=WRITE_DIALECT)

    return expression.transform(replace)


def translate_function_body(
    body: str, parameters: Sequence[tuple[str, str | None]] = ()
) -> str:
    """Translate a SQL UDF body into a DuckDB scalar expression.

    Snowflake allows the body to be either a bare expression (``AS $$ x + 1 $$``) or a
    single-column ``SELECT`` (``AS $$ SELECT x + 1 $$``). A body selecting more than one
    column is a table function, which is rejected here because the emulator only
    emulates scalar UDFs.
    """
    if not body.strip():
        raise TranslationError("CREATE FUNCTION requires a non-empty body.")
    try:
        parsed = sqlglot.parse_one(preprocess_array_calls(body), read=READ_DIALECT)
    except Exception as exc:  # noqa: BLE001 - reported as a translation error
        raise TranslationError(
            f"Failed to parse the function body as SQL: {exc}"
        ) from exc
    if parsed is None:
        raise TranslationError("CREATE FUNCTION requires a non-empty body.")
    if isinstance(parsed, exp.Select):
        if len(parsed.expressions) != 1:
            raise TranslationError(
                "A SQL UDF body selecting more than one column is a table function, "
                "which the emulator does not support (only scalar UDFs are)."
            )
        parsed = parsed.expressions[0]
        if isinstance(parsed, exp.Alias):
            parsed = parsed.this
    return _cast_parameters(_apply_transforms(parsed), parameters).sql(
        dialect=WRITE_DIALECT
    )


def translate_create_function(statement: exp.Expression) -> UserFunctionSpec:
    """Translate a ``CREATE [OR REPLACE] FUNCTION`` into a DuckDB-macro spec.

    Raises :class:`TranslationError` for anything the emulator cannot emulate
    faithfully, rather than letting DuckDB create a function that silently returns the
    body text (DuckDB reads ``CREATE FUNCTION f(x) AS 'x + 1'`` as a reference to a
    native function named ``x + 1``).
    """
    if not (
        isinstance(statement, exp.Create)
        and str(statement.args.get("kind") or "").upper() == "FUNCTION"
    ):
        raise TranslationError("Not a CREATE FUNCTION statement.")
    udf = statement.this
    if not isinstance(udf, exp.UserDefinedFunction) or not isinstance(udf.this, exp.Table):
        raise TranslationError("Unsupported CREATE FUNCTION form.")
    table = udf.this

    returns, language, is_table = _function_properties(statement)
    if language != "SQL":
        raise TranslationError(
            f"{language} UDFs are not supported by the emulator "
            f"(no {language.lower()} runtime available)."
        )
    if is_table:
        raise TranslationError(
            "Table-returning user-defined functions (RETURNS TABLE ...) are not "
            "supported by the emulator (only scalar SQL UDFs are)."
        )
    if returns is None:
        raise TranslationError("CREATE FUNCTION requires a RETURNS <type> clause.")

    arguments: list[UserFunctionArgument] = []
    for argument in udf.expressions:
        kind = argument.args.get("kind") if isinstance(argument, exp.ColumnDef) else None
        identifier = argument.this
        if not isinstance(identifier, exp.Identifier):
            raise TranslationError("Unsupported CREATE FUNCTION argument list.")
        name = _snowflake_name(identifier)
        # Rendered through DuckDB first so the reported argument types match the
        # reported return type (``NUMBER`` -> ``DECIMAL(38, 0)`` -> ``NUMBER(38,0)``).
        declared = (
            duckdb_type_to_snowflake_sql_type(kind.sql(dialect=WRITE_DIALECT))
            if isinstance(kind, exp.DataType)
            else None
        )
        arguments.append(
            UserFunctionArgument(
                name=name,
                declared_type=declared,
                duckdb_type=kind.sql(dialect=WRITE_DIALECT)
                if isinstance(kind, exp.DataType)
                else None,
            )
        )

    catalog = table.args.get("catalog")
    db = table.args.get("db")
    # The function name (and any qualifier) needs Snowflake's identifier casing here:
    # ``CREATE FUNCTION`` is handled before the generic transpile path, so the
    # uppercase-identifiers transform never sees this statement.
    function_name = (
        _snowflake_name(table.this)
        if isinstance(table.this, exp.Identifier)
        else table.name
    )
    body = _function_body_text(statement)
    return UserFunctionSpec(
        name=function_name,
        arguments=tuple(arguments),
        # Report the return type the way ``DESCRIBE``/``SHOW FUNCTIONS`` do for tables,
        # normalising DuckDB's spelling back to Snowflake's (DECIMAL -> NUMBER).
        return_type=duckdb_type_to_snowflake_sql_type(returns.sql(dialect=WRITE_DIALECT)),
        return_type_duckdb=returns.sql(dialect=WRITE_DIALECT),
        language=language,
        body=body,
        body_sql=translate_function_body(
            body, [(argument.name, argument.duckdb_type) for argument in arguments]
        ),
        database=_snowflake_name(catalog) if isinstance(catalog, exp.Identifier) else None,
        schema=_snowflake_name(db) if isinstance(db, exp.Identifier) else None,
        replace=bool(statement.args.get("replace")),
        if_not_exists=bool(statement.args.get("exists")),
    )


def render_create_macro(
    spec: UserFunctionSpec, database: str, schema: str, macro_name: str | None = None
) -> str:
    """Render the DuckDB ``CREATE MACRO`` statement holding a SQL UDF's body.

    The macro is catalog/schema-qualified so it lands in the function's Snowflake
    namespace regardless of what the executing cursor's current schema happens to be.
    `macro_name` overrides the name the macro is created under, which is how one
    overload of a dispatched function is stored behind its dispatcher.
    """
    clauses = ""
    if spec.replace:
        clauses = " OR REPLACE"
    elif spec.if_not_exists:
        clauses = " IF NOT EXISTS"
    params = ", ".join(_render_identifier(argument.name) for argument in spec.arguments)
    name = render_qualified_name(database, schema, macro_name or spec.macro_name)
    return f"CREATE{clauses} MACRO {name}({params}) AS ({spec.body_sql})"


def render_drop_macro(
    macro_name: str, database: str, schema: str, *, if_exists: bool = False
) -> str:
    """Render the DuckDB ``DROP MACRO`` statement removing a user-defined function."""
    clause = " IF EXISTS" if if_exists else ""
    return f"DROP MACRO{clause} {render_qualified_name(database, schema, macro_name)}"


@dataclass(frozen=True)
class UserFunctionCall:
    """How a resolved Snowflake UDF call is reached in DuckDB."""

    callable_name: str
    #: A Python dispatcher needs each argument's DuckDB type appended to the call, since
    #: a generic callback receives values only -- and cannot, for instance, tell a JSON
    #: array column from a text value (both arrive as ``str``).
    dispatch: bool


class UserFunctionResolver(Protocol):
    """Resolves a Snowflake UDF call to the DuckDB callable that emulates it."""

    def resolve(self, name: str, arity: int) -> UserFunctionCall | None:
        """Return the callable for `name` called with `arity` arguments."""
        ...


def _rewrite_user_function_calls(
    statement: exp.Expression, resolver: UserFunctionResolver
) -> exp.Expression:
    """Point calls to user-defined functions at the macro or dispatcher backing them.

    Only ``Anonymous`` calls are considered: a UDF whose name matches a function sqlglot
    models natively (``COALESCE``, ``ARRAY_SIZE``, ...) is parsed into that node instead
    and keeps its built-in behaviour.
    """
    for call in list(statement.find_all(exp.Anonymous)):
        resolved = resolver.resolve(call.name, len(call.expressions))
        if resolved is None:
            continue
        if resolved.dispatch:
            arguments = list(call.expressions)
            call.set(
                "expressions",
                [
                    *arguments,
                    *(
                        exp.Anonymous(this="typeof", expressions=[argument.copy()])
                        for argument in arguments
                    ),
                ],
            )
        call.set("this", _identifier(resolved.callable_name))
    return statement


def _reject_unsupported_function(statement: exp.Expression) -> None:
    """Raise for ``CREATE FUNCTION`` statements the emulator can't faithfully emulate.

    DuckDB has no JavaScript/Python/Java/Scala runtime, so a Snowflake UDF with a
    non-SQL ``LANGUAGE`` clause would be silently translated into a broken function
    (the body becomes a string literal). Reject it loudly instead.

    Kept as a guard for the generic transpile path: ``CREATE FUNCTION`` is normally
    handled by :func:`translate_create_function` before reaching it.
    """
    if not (isinstance(statement, exp.Create) and statement.args.get("kind") == "FUNCTION"):
        return
    props = statement.args.get("properties")
    if props is None:
        return
    for prop in props.expressions:
        if isinstance(prop, exp.LanguageProperty) and prop.this is not None:
            lang = (prop.this.name or "").upper()
            if lang and lang != "SQL":
                raise TranslationError(
                    f"{lang} UDFs are not supported by the emulator "
                    f"(no {lang.lower()} runtime available)."
                )
