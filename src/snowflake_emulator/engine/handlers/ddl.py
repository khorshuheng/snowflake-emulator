"""Handlers for object DDL that need Snowflake semantics.

``CREATE DATABASE``, file formats and user-defined functions are not passed straight
to DuckDB: the first is emulated with an attached catalog, the other two need side
metadata. Table DDL stays on the normal DuckDB path.
"""

from __future__ import annotations

from typing import Any

from sqlglot import exp

from snowflake_emulator.core.ast import drop_target_table
from snowflake_emulator.core.errors import ExecutionError
from snowflake_emulator.core.results import ColumnMeta
from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.engine.errors import as_execution_error
from snowflake_emulator.engine.handlers.common import name_matches, show_name_patterns
from snowflake_emulator.engine.handlers.show import show_limit
from snowflake_emulator.file_formats import get_file_format_manager
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.translator import (
    TranslationError,
    normalize_function_signature,
    translate_create_function,
)
from snowflake_emulator.user_functions import (
    UserFunctionError,
    execute_create_function,
    execute_drop_function,
)


def execute_create_database(
    manager: DuckDBManager,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Execute ``CREATE DATABASE`` by attaching a fresh catalog.

    DuckDB has no ``CREATE DATABASE``. Returns ``None`` for non-database CREATE
    statements.
    """
    if not (
        isinstance(statement, exp.Create)
        and str(statement.args.get("kind") or "").upper() == "DATABASE"
    ):
        return None
    if isinstance(statement.args.get("clone"), exp.Clone):
        # Snowflake can clone a database; the emulator cannot rebuild the catalog's
        # contents, so it fails loudly instead of creating an empty database.
        raise ExecutionError(
            "CREATE DATABASE ... CLONE is not supported by the emulator "
            "(only CREATE TABLE ... CLONE is)."
        )
    this = statement.args.get("this")
    ident = this.this if isinstance(this, exp.Table) else this
    if not isinstance(ident, exp.Identifier) or not ident.name:
        return None
    # Unquoted identifiers are uppercased (Snowflake semantics); quoted preserved.
    name = ident.name if ident.args.get("quoted") else ident.name.upper()

    if statement.args.get("replace"):
        manager.replace_database(name)
    else:
        if not statement.args.get("exists") and manager.has_database(name):
            raise ExecutionError(f"Database {name} already exists.")
        manager.ensure_namespace(name, "PUBLIC")
    return (
        [ColumnMeta(name="status", type="text")],
        [[f"Database {name} successfully created."]],
    )


def _file_format_options(statement: exp.Expression) -> dict[str, str]:
    """Extract the ``TYPE = CSV, SKIP_HEADER = 1, ...`` options from a CREATE node."""
    options: dict[str, str] = {}
    properties = statement.args.get("properties")
    for prop in properties.expressions if properties else []:
        if not isinstance(prop, exp.Property):
            continue
        value = prop.args.get("value")
        if value is None:
            continue
        if isinstance(value, exp.Literal) and value.is_string:
            options[str(prop.this).upper()] = str(value.this)
        else:
            options[str(prop.this).upper()] = value.sql(dialect="snowflake")
    return options


def execute_file_format_ddl(
    session: SessionContext, statement: exp.Expression
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle CREATE/DROP FILE FORMAT and SHOW FILE FORMATS."""
    kind = str(statement.args.get("kind") or "").upper()
    if isinstance(statement, exp.Create) and kind == "FILE FORMAT":
        name = statement.this.name if isinstance(statement.this, exp.Table) else ""
        if not name:
            return None
        get_file_format_manager().create(
            session,
            name,
            _file_format_options(statement),
            replace=bool(statement.args.get("replace")),
            if_not_exists=bool(statement.args.get("exists")),
        )
        return [ColumnMeta(name="status", type="text")], [
            [f"File format {name.upper()} successfully created."]
        ]
    if isinstance(statement, exp.Drop) and kind == "FILE FORMAT":
        target = drop_target_table(statement)
        name = target.name if target is not None else ""
        if not name:
            return None
        get_file_format_manager().drop(
            session, name, if_exists=bool(statement.args.get("exists"))
        )
        return [ColumnMeta(name="status", type="text")], [
            [f"File format {name.upper()} successfully dropped."]
        ]
    if (
        isinstance(statement, exp.Show)
        and str(statement.args.get("this") or "").upper() == "FILE FORMATS"
    ):
        rows = [
            [fmt.name, fmt.database, fmt.schema, fmt.options.get("TYPE", "")]
            for fmt in get_file_format_manager().list(session)
        ]
        prefix, like = show_name_patterns(statement)
        if prefix or like:
            rows = [row for row in rows if name_matches(str(row[0]), prefix, like)]
        limit = show_limit(statement)
        if limit is not None:
            rows = rows[:limit]
        return (
            [
                ColumnMeta(name="name", type="text"),
                ColumnMeta(name="database_name", type="text"),
                ColumnMeta(name="schema_name", type="text"),
                ColumnMeta(name="type", type="text"),
            ],
            rows,
        )
    return None


def execute_function_ddl(
    manager: DuckDBManager,
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle ``CREATE FUNCTION`` / ``DROP FUNCTION`` for user-defined functions.

    Snowflake UDFs are emulated as DuckDB macros, with the Snowflake-side metadata kept
    in the catalog's side table (see ``user_functions``). Returns ``None`` for other DDL.
    """
    kind = str(statement.args.get("kind") or "").upper()
    status = [ColumnMeta(name="status", type="text")]
    if isinstance(statement, exp.Create) and kind == "FUNCTION":
        try:
            spec = translate_create_function(statement)
            message = execute_create_function(manager, cursor, session, spec)
        except (TranslationError, UserFunctionError) as exc:
            raise as_execution_error(exc) from exc
        return status, [[message]]
    if isinstance(statement, exp.Drop) and kind == "FUNCTION":
        this = drop_target_table(statement)
        if this is None:
            return None
        # `DROP FUNCTION f(a NUMBER, b NUMBER)` names one overload; the types are
        # normalized the same way they were recorded so the drop cannot match the wrong
        # one. Without a signature every overload of the name is dropped.
        types = list(statement.expressions)
        try:
            message = execute_drop_function(
                manager,
                cursor,
                session,
                this.name,
                len(types) if types else None,
                signature=normalize_function_signature(types) if types else None,
                if_exists=bool(statement.args.get("exists")),
            )
        except UserFunctionError as exc:
            raise as_execution_error(exc) from exc
        return status, [[message]]
    return None


def created_table_target(
    session: SessionContext, statement: exp.Create
) -> tuple[str, str, str] | None:
    """Return the ``(database, schema, table)`` a CREATE TABLE targets, else None."""
    schema = statement.this
    if not (isinstance(schema, exp.Schema) and isinstance(schema.this, exp.Table)):
        return None
    table = schema.this
    catalog = table.args.get("catalog")
    db = table.args.get("db")
    database = catalog.name if isinstance(catalog, exp.Identifier) else session.database
    table_schema = db.name if isinstance(db, exp.Identifier) else session.schema
    return database, table_schema, table.name


def is_temporary_table_create(statement: exp.Create) -> bool:
    """Return True for a ``CREATE TEMPORARY|VOLATILE TABLE``.

    ``VOLATILE`` is checked as written: it reaches here before the translator rewrites it
    into the ``TEMPORARY`` property.
    """
    properties = statement.args.get("properties")
    if properties is None:
        return False
    return any(
        isinstance(prop, (exp.TemporaryProperty, exp.VolatileProperty))
        for prop in properties.expressions
    )


def record_declared_columns(
    manager: DuckDBManager, session: SessionContext, statement: exp.Create
) -> None:
    """Persist the declared Snowflake column types for a freshly created table.

    DuckDB drops ``VARCHAR`` lengths, so the type as written is kept in the
    per-catalog side table so ``DESCRIBE`` / ``INFORMATION_SCHEMA.COLUMNS`` can report it.

    Temporary tables are skipped: that side table is keyed by (catalog, schema, table),
    which two sessions' same-named temporary tables would share, and a temporary table
    shadowing a permanent one would overwrite -- and so silently lose -- the permanent
    table's declared types. A temporary table therefore reports DuckDB's own type.
    """
    target = created_table_target(session, statement)
    if target is None or is_temporary_table_create(statement):
        return
    database, table_schema, table_name = target
    columns: list[tuple[str, str | None, str | None]] = []
    for column in statement.this.expressions:
        if isinstance(column, exp.ColumnDef):
            kind = column.args.get("kind")
            declared = kind.sql(dialect="snowflake") if kind is not None else None
            columns.append((column.this.name, declared, None))
    if columns:
        manager.record_columns(database, table_schema, table_name, columns)


def record_temp_table_owner(
    manager: DuckDBManager, session: SessionContext, statement: exp.Create
) -> None:
    """Remember which database a session's temporary table was created in.

    ``CREATE OR REPLACE DATABASE`` uses this to drop the temporary tables it owned, the
    way Snowflake drops them with the schema they belong to (DuckDB keeps them in a
    separate connection-local catalog that a database replacement does not touch).
    """
    if not is_temporary_table_create(statement):
        return
    target = created_table_target(session, statement)
    if target is None:
        return
    database, _, table_name = target
    manager.record_temp_table(session.token, table_name, database)


def is_ddl_status_artifact(statement: exp.Expression, description: list[Any]) -> bool:
    """Return True when DuckDB reports DDL as a single ``Count``/``Success`` column.

    DuckDB's Python API reports CREATE/DROP/ALTER as a one-column result named
    ``Count`` or ``Success`` with no rows. Snowflake returns a plain status instead,
    so those statements are routed to the standard status result.
    """
    if len(description) != 1 or description[0][0] not in ("Count", "Success"):
        return False
    return isinstance(statement, (exp.Create, exp.Drop, exp.Alter))
