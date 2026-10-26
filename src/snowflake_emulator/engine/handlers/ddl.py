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


def execute_drop_database(
    manager: DuckDBManager, statement: exp.Expression
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Execute ``DROP [IF EXISTS] DATABASE`` by detaching and removing its catalog."""
    if not (
        isinstance(statement, exp.Drop)
        and str(statement.args.get("kind") or "").upper() == "DATABASE"
    ):
        return None
    target = drop_target_table(statement)
    if target is None or not target.name:
        return None
    name = _sf_name(target.this) if isinstance(target.this, exp.Identifier) else target.name
    exists = manager.has_database(name)
    if not exists:
        if statement.args.get("exists"):
            return [ColumnMeta(name="status", type="text")], [
                [f"Database {name} does not exist."]
            ]
        raise ExecutionError(f"Database {name} does not exist.")
    manager.drop_database(name)
    return [ColumnMeta(name="status", type="text")], [
        [f"Database {name} successfully dropped."]
    ]


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


def _object_name(table: exp.Table | None, session: SessionContext) -> str:
    """The name of a DDL target, in Snowflake's casing (quoted names keep their case)."""
    if table is None:
        return ""
    identifier = table.this
    if isinstance(identifier, exp.Identifier) and not identifier.args.get("quoted"):
        return identifier.name.upper()
    return identifier.name if isinstance(identifier, exp.Identifier) else table.name


def _object_namespace(
    table: exp.Table | None, session: SessionContext
) -> tuple[str, str]:
    """The ``(database, schema)`` a DDL target names, defaulting to the session's."""
    if table is None:
        return session.database, session.schema
    return (
        table.text("catalog") or session.database,
        table.text("db") or session.schema,
    )


def execute_file_format_ddl(
    session: SessionContext, statement: exp.Expression
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle CREATE/DROP FILE FORMAT and SHOW FILE FORMATS."""
    kind = str(statement.args.get("kind") or "").upper()
    if isinstance(statement, exp.Create) and kind == "FILE FORMAT":
        table = statement.this if isinstance(statement.this, exp.Table) else None
        name = _object_name(table, session)
        if not name:
            return None
        database, schema = _object_namespace(table, session)
        get_file_format_manager().create(
            session,
            name,
            _file_format_options(statement),
            replace=bool(statement.args.get("replace")),
            if_not_exists=bool(statement.args.get("exists")),
            database=database,
            schema=schema,
        )
        return [ColumnMeta(name="status", type="text")], [
            [f"File format {name.upper()} successfully created."]
        ]
    if isinstance(statement, exp.Drop) and kind == "FILE FORMAT":
        target = drop_target_table(statement)
        name = _object_name(target, session)
        if not name:
            return None
        database, schema = _object_namespace(target, session)
        get_file_format_manager().drop(
            session,
            name,
            if_exists=bool(statement.args.get("exists")),
            database=database,
            schema=schema,
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
    """Return True when DuckDB reports a statement without a result set as a bare column.

    DuckDB's Python API reports DDL (CREATE/DROP/ALTER/TRUNCATE) as a one-column result
    named ``Count`` or ``Success``. Snowflake returns a plain status for those, so they are
    routed to the standard status result.
    """
    if len(description) != 1 or description[0][0] not in ("Count", "Success"):
        return False
    return isinstance(statement, (exp.Create, exp.Drop, exp.Alter, exp.TruncateTable))


def _sf_name(identifier: exp.Identifier) -> str:
    return identifier.name if identifier.args.get("quoted") else identifier.name.upper()


def _display_table_parts(
    session: SessionContext, table: exp.Table
) -> tuple[str, str, str]:
    """Return Snowflake-cased ``(database, schema, table)`` parts for a table node."""
    catalog = table.args.get("catalog")
    db = table.args.get("db")
    name = table.this
    database = _sf_name(catalog) if isinstance(catalog, exp.Identifier) else session.database
    schema = _sf_name(db) if isinstance(db, exp.Identifier) else session.schema
    table_name = _sf_name(name) if isinstance(name, exp.Identifier) else table.name
    return database, schema, table_name


def _reference_info(reference: exp.Reference) -> tuple[str | None, list[str]]:
    """Return the ``(table, columns)`` a REFERENCES clause points at."""
    target = reference.this
    if isinstance(target, exp.Schema):
        table = (
            _sf_name(target.this.this)
            if isinstance(target.this, exp.Table)
            and isinstance(target.this.this, exp.Identifier)
            else None
        )
        columns = [
            _sf_name(item) for item in target.expressions if isinstance(item, exp.Identifier)
        ]
        return table, columns
    if isinstance(target, exp.Table):
        return _sf_name(target.this) if isinstance(target.this, exp.Identifier) else None, []
    return None, []


def _constraint_rows_from_items(
    items: list[exp.Expression], table_name: str
) -> list[tuple[str, str, str, int, str | None, str | None]]:
    """Extract primary/unique/foreign-key rows from CREATE/ALTER constraint items."""
    rows: list[tuple[str, str, str, int, str | None, str | None]] = []
    unnamed = 0

    def add_table_constraint(node: exp.Expression, name: str | None) -> None:
        nonlocal unnamed
        unnamed += 1
        if isinstance(node, exp.PrimaryKey):
            columns = [_sf_name(item) for item in node.expressions if isinstance(item, exp.Identifier)]
            constraint_name = name or f"{table_name}_pkey"
            for sequence, column in enumerate(columns, start=1):
                rows.append((constraint_name, "PRIMARY KEY", column, sequence, None, None))
        elif isinstance(node, exp.UniqueColumnConstraint):
            target = node.this
            columns = (
                [_sf_name(item) for item in target.expressions if isinstance(item, exp.Identifier)]
                if isinstance(target, exp.Schema)
                else []
            )
            constraint_name = name or f"{table_name}_key_{unnamed}"
            for sequence, column in enumerate(columns, start=1):
                rows.append((constraint_name, "UNIQUE", column, sequence, None, None))
        elif isinstance(node, exp.ForeignKey):
            columns = [_sf_name(item) for item in node.expressions if isinstance(item, exp.Identifier)]
            reference = node.args.get("reference")
            ref_table, ref_columns = (
                _reference_info(reference) if isinstance(reference, exp.Reference) else (None, [])
            )
            constraint_name = name or f"{table_name}_fkey_{unnamed}"
            for sequence, column in enumerate(columns, start=1):
                ref_column = ref_columns[sequence - 1] if sequence <= len(ref_columns) else None
                rows.append(
                    (constraint_name, "FOREIGN KEY", column, sequence, ref_table, ref_column)
                )

    for item in items:
        if isinstance(item, exp.ColumnDef):
            column = _sf_name(item.this) if isinstance(item.this, exp.Identifier) else item.this.name
            for constraint in item.args.get("constraints") or []:
                kind = constraint.kind
                if isinstance(kind, exp.PrimaryKeyColumnConstraint):
                    rows.append(
                        (f"{table_name}_{column}_pkey", "PRIMARY KEY", column, 1, None, None)
                    )
                elif isinstance(kind, exp.UniqueColumnConstraint):
                    rows.append((f"{table_name}_{column}_key", "UNIQUE", column, 1, None, None))
                elif isinstance(kind, exp.Reference):
                    ref_table, ref_columns = _reference_info(kind)
                    rows.append(
                        (
                            f"{table_name}_{column}_fkey",
                            "FOREIGN KEY",
                            column,
                            1,
                            ref_table,
                            ref_columns[0] if ref_columns else None,
                        )
                    )
        elif isinstance(item, exp.Constraint):
            name = _sf_name(item.this) if isinstance(item.this, exp.Identifier) else None
            for inner in item.expressions:
                add_table_constraint(inner, name)
        else:
            add_table_constraint(item, None)
    return rows


def _constraint_rows(
    schema: exp.Schema, table_name: str
) -> list[tuple[str, str, str, int, str | None, str | None]]:
    return _constraint_rows_from_items(schema.expressions, table_name)


def record_constraints(
    manager: DuckDBManager, session: SessionContext, statement: exp.Create
) -> None:
    """Persist a CREATE TABLE's declared constraints without enforcing them."""
    target = created_table_target(session, statement)
    if target is None or is_temporary_table_create(statement):
        return
    database, table_schema, table_name = target
    schema = statement.this
    if not isinstance(schema, exp.Schema) or not isinstance(schema.this, exp.Table):
        return
    display_database, display_schema, display_table = _display_table_parts(
        session, schema.this
    )
    manager.record_constraints(
        display_database,
        display_schema,
        display_table,
        _constraint_rows(schema, display_table),
    )


def execute_alter_constraints(
    manager: DuckDBManager, session: SessionContext, statement: exp.Expression
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Record ``ALTER TABLE ... ADD CONSTRAINT`` and skip DuckDB execution.

    Snowflake does not enforce the added constraint. Only an ALTER whose sole actions
    are ADD CONSTRAINT is handled here; mixed ALTERs keep their normal DuckDB path.
    """
    if not (isinstance(statement, exp.Alter) and statement.args.get("kind") == "TABLE"):
        return None
    actions = statement.args.get("actions") or []
    if not actions or not all(isinstance(action, exp.AddConstraint) for action in actions):
        return None
    table = statement.this
    if not isinstance(table, exp.Table):
        return None
    database, schema, table_name = _display_table_parts(session, table)
    rows: list[tuple[str, str, str, int, str | None, str | None]] = []
    for action in actions:
        rows.extend(_constraint_rows_from_items(action.expressions, table_name))
    if rows:
        manager.record_constraints(database, schema, table_name, rows, replace=False)
    return (
        [ColumnMeta(name="status", type="text")],
        [["Statement executed successfully."]],
    )
