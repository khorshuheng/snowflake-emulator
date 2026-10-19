"""Snowflake-shaped ``SHOW`` handlers.

DuckDB's native SHOW returns its own column names, while Snowflake tools such as dbt
expect ``name``/``kind``/``database_name``/... These handlers build the result from
DuckDB's catalog introspection functions instead.
"""

from __future__ import annotations

import re
from typing import Any

from sqlglot import exp

from snowflake_emulator.core.constants import INFORMATION_SCHEMA_SCHEMA
from snowflake_emulator.core.results import ColumnMeta
from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.engine.handlers.common import (
    name_matches,
    prefer_declared_type,
    result_columns,
    show_name_patterns,
    temp_inclusive_scope,
)
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.type_mapping import duckdb_type_to_snowflake_sql_type
from snowflake_emulator.user_functions import query_user_functions


def show_scope(
    statement: exp.Show, session: SessionContext
) -> tuple[str, str | None] | None:
    """Extract the ``(database, schema)`` a scoped ``SHOW ... IN ...`` targets.

    ``SHOW SCHEMAS IN DATABASE db`` targets a database (schema ``None``);
    ``SHOW OBJECTS IN db.schema`` targets a schema. An unscoped ``SHOW TABLES``,
    ``SHOW VIEWS``, ``SHOW OBJECTS`` or ``SHOW SCHEMAS`` is scoped to the session's own
    database/schema, so that its ``LIKE``/``STARTS WITH``/``LIMIT`` clause is applied
    rather than dropped (DuckDB's native SHOW, which these forms used to fall through
    to, ignores them). Returns ``None`` for the SHOW forms handled elsewhere.
    """
    scope = statement.args.get("scope")
    kind = statement.args.get("scope_kind")
    if isinstance(scope, exp.Table):
        if kind == "DATABASE":
            return scope.name, None
        if kind == "SCHEMA":
            db = scope.args.get("db")
            database = db.name if isinstance(db, exp.Identifier) else session.database
            return database, scope.name
        return None
    obj = str(statement.args.get("this") or "").upper()
    if obj == "SCHEMAS":
        return session.database, None
    if obj in ("TABLES", "VIEWS", "OBJECTS", "SEQUENCES", "STAGES"):
        return session.database, session.schema
    return None


def show_limit(statement: exp.Show) -> int | None:
    """Extract the ``LIMIT n`` from a SHOW statement, if any."""
    limit = statement.args.get("limit")
    if isinstance(limit, exp.Limit):
        expr = limit.args.get("expression")
        if isinstance(expr, exp.Literal):
            try:
                return int(expr.this)
            except (TypeError, ValueError):
                return None
    return None


def escape_like(text: str) -> str:
    """Escape LIKE wildcards so a ``STARTS WITH`` match is a plain prefix."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def name_prefix_filter(column: str, prefix: str | None) -> str:
    """Return a WHERE fragment restricting ``column`` to a ``STARTS WITH`` prefix."""
    if not prefix:
        return ""
    escaped = escape_like(prefix).replace("'", "''")
    return f" AND {column} LIKE '{escaped}%' ESCAPE '\\'"


def name_like_filter(column: str, pattern: str | None) -> str:
    """Return a WHERE fragment matching ``column`` against a ``SHOW ... LIKE`` pattern.

    Unlike ``STARTS WITH`` the pattern keeps its ``%``/``_`` wildcards. Matching is
    case-insensitive because object names are upper-cased unless they were quoted.
    """
    if not pattern:
        return ""
    escaped = pattern.replace("\\", "\\\\").replace("'", "''")
    return f" AND {column} ILIKE '{escaped}' ESCAPE '\\'"


def execute_snowflake_show(
    manager: DuckDBManager,
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Execute Snowflake ``SHOW ... IN ...`` statements with Snowflake-shaped output.

    Returns ``None`` for SHOW forms this handler doesn't own.
    """
    if isinstance(statement, exp.Command):
        text = str(statement.args.get("expression") or "")
        upper = text.upper()
        if any(
            kind in upper
            for kind in ("USER FUNCTIONS", "ALL FUNCTIONS", "EXTERNAL FUNCTIONS")
        ):
            # sqlglot cannot parse ``SHOW USER FUNCTIONS IN ...``, so this form arrives as
            # a raw command and its scope/LIKE clause is parsed from the text.
            database, schema = function_show_scope(text, session)
            return user_function_show(
                manager, cursor, database, schema, show_like_pattern(text)
            )
        return None
    if not isinstance(statement, exp.Show):
        return None
    obj = str(statement.args.get("this") or "").upper()
    scope = show_scope(statement, session)
    if scope is None:
        return None
    database, schema = scope

    limit = show_limit(statement)
    prefix, like = show_name_patterns(statement)

    if obj == "SCHEMAS":
        sql = f"""
            SELECT
                NULL::TIMESTAMP AS created_on,
                schema_name AS name,
                'SCHEMA' AS kind,
                database_name AS database_name,
                schema_name AS schema_name,
                comment AS comment,
                NULL::VARCHAR AS options,
                NULL::VARCHAR AS owner,
                NULL::BIGINT AS retention_time
            FROM duckdb_schemas()
            WHERE lower(database_name) = lower('{database}')
              AND lower(schema_name) != 'main'
              AND lower(schema_name) != lower('{INFORMATION_SCHEMA_SCHEMA}')
              {name_prefix_filter('schema_name', prefix)}
              {name_like_filter('schema_name', like)}
            ORDER BY schema_name
        """
    else:
        selects: list[str] = []
        if obj in ("TABLES", "OBJECTS"):
            selects.append(f"""
                SELECT
                    NULL::TIMESTAMP AS created_on,
                    t.table_name AS name,
                    'TABLE' AS kind,
                    CASE WHEN lower(t.database_name) = 'temp'
                         THEN current_database() ELSE t.database_name END AS database_name,
                    CASE WHEN lower(t.database_name) = 'temp'
                         THEN current_schema() ELSE t.schema_name END AS schema_name,
                    t.comment AS comment,
                    'N' AS is_dynamic,
                    'N' AS is_iceberg
                FROM duckdb_tables() t
                WHERE {temp_inclusive_scope('t', database, schema)}
                  {name_prefix_filter('t.table_name', prefix)}
                  {name_like_filter('t.table_name', like)}
            """)
        if obj in ("VIEWS", "OBJECTS"):
            selects.append(f"""
                SELECT
                    NULL::TIMESTAMP AS created_on,
                    v.view_name AS name,
                    'VIEW' AS kind,
                    v.database_name AS database_name,
                    v.schema_name AS schema_name,
                    v.comment AS comment,
                    'N' AS is_dynamic,
                    'N' AS is_iceberg
                FROM duckdb_views() v
                WHERE {temp_inclusive_scope('v', database, schema, 'view_name')}
                  {name_prefix_filter('v.view_name', prefix)}
                  {name_like_filter('v.view_name', like)}
            """)
        if not selects:
            return None
        sql = " UNION ALL ".join(selects) + " ORDER BY name"

    if limit is not None:
        sql += f" LIMIT {limit}"
    cursor.execute(sql)
    return result_columns(cursor)


def execute_show_columns(
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle ``SHOW COLUMNS`` with Snowflake's column names (used by dbt).

    Scopes onto the current schema by default, or a named table/schema/database when
    the statement carries ``IN``. Returns ``None`` for SHOW forms this handler does
    not own.
    """
    if not (
        isinstance(statement, exp.Show)
        and str(statement.args.get("this") or "").upper() == "COLUMNS"
    ):
        return None
    scope_kind = statement.args.get("scope_kind")
    scope = statement.args.get("scope")
    database = session.database
    schema = session.schema
    table_name = None
    if scope_kind == "DATABASE" and isinstance(scope, exp.Table):
        database, schema = scope.name, None
    elif scope_kind == "SCHEMA" and isinstance(scope, exp.Table):
        db = scope.args.get("db")
        database = db.name if isinstance(db, exp.Identifier) else session.database
        schema = scope.name
    elif scope_kind in ("TABLE", "VIEW") and isinstance(scope, exp.Table):
        catalog = scope.args.get("catalog")
        db = scope.args.get("db")
        database = catalog.name if isinstance(catalog, exp.Identifier) else session.database
        schema = db.name if isinstance(db, exp.Identifier) else session.schema
        table_name = scope.name
    elif scope_kind is not None:
        return None

    # Bound as parameters so an identifier containing a quote cannot break the SQL. The
    # second branch of each test includes the session's own temporary tables, which
    # DuckDB reports in the connection-local ``temp`` catalog.
    params: list[Any] = [database, database]
    where = [
        "((lower(c.database_name) = lower(?)"
        " AND NOT EXISTS (SELECT 1 FROM duckdb_tables() sh"
        " WHERE lower(sh.database_name) = 'temp'"
        " AND lower(sh.table_name) = lower(c.table_name)))"
        " OR (lower(c.database_name) = 'temp'"
        " AND lower(current_database()) = lower(?)))"
    ]
    if schema is not None:
        where.append(
            "((lower(c.schema_name) = lower(?))"
            " OR (lower(c.database_name) = 'temp' AND lower(current_schema()) = lower(?)))"
        )
        params.extend([schema, schema])
    where.append("(lower(c.database_name) = 'temp' OR lower(c.schema_name) <> lower(?))")
    params.append(INFORMATION_SCHEMA_SCHEMA)
    if table_name is not None:
        where.append("lower(c.table_name) = lower(?)")
        params.append(table_name)
    cursor.execute(
        f"""
        SELECT
            c.table_name AS "table_name",
            CASE WHEN lower(c.database_name) = 'temp'
                 THEN current_schema() ELSE c.schema_name END AS "schema_name",
            c.column_name AS "column_name",
            c.data_type AS "data_type",
            e.declared_type AS "declared_type",
            CASE WHEN c.is_nullable THEN 'Y' ELSE 'N' END AS "null?",
            c.column_default AS "default",
            'COLUMN' AS "kind",
            NULL::VARCHAR AS "expression",
            c.comment AS "comment",
            CASE WHEN lower(c.database_name) = 'temp'
                 THEN current_database() ELSE c.database_name END AS "database_name",
            NULL::VARCHAR AS "autoincrement",
            NULL::VARCHAR AS "schema_evolution_record"
        FROM duckdb_columns() c
        LEFT JOIN "{database}".{INFORMATION_SCHEMA_SCHEMA}._sf_columns_ext e
          ON lower(e.table_catalog) = lower(c.database_name)
         AND lower(e.table_schema) = lower(c.schema_name)
         AND lower(e.table_name) = lower(c.table_name)
         AND lower(e.column_name) = lower(c.column_name)
         AND lower(c.database_name) <> 'temp'
        WHERE {' AND '.join(where)}
        ORDER BY c.table_name, c.column_index
        """,
        params,
    )
    row_type, rows = result_columns(cursor)
    # Drop the helper "declared_type" column (index 4) and prefer the declared type.
    row_type = [*row_type[:4], *row_type[5:]]
    converted: list[list[Any]] = []
    for row in rows:
        sf_type = prefer_declared_type(row[4], row[3]) or duckdb_type_to_snowflake_sql_type(
            str(row[3])
        )
        converted.append([*row[:3], sf_type, *row[5:]])
    prefix, like = show_name_patterns(statement)
    if prefix or like:
        converted = [row for row in converted if name_matches(row[2], prefix, like)]
    limit = show_limit(statement)
    if limit is not None:
        converted = converted[:limit]
    return row_type, converted


def show_key_kind(statement: exp.Expression) -> str | None:
    """Classify a SHOW <PRIMARY|UNIQUE|FOREIGN|IMPORTED> KEYS statement."""
    kinds = {
        "PRIMARY KEYS": "PRIMARY",
        "UNIQUE KEYS": "UNIQUE",
        "IMPORTED KEYS": "FOREIGN",
    }
    if isinstance(statement, exp.Show):
        return kinds.get(str(statement.args.get("this") or "").upper())
    # sqlglot cannot parse SHOW FOREIGN KEYS, so it arrives as a Command.
    if isinstance(statement, exp.Command) and str(statement.args.get("this") or "").upper() == "SHOW":
        if "FOREIGN KEYS" in str(statement.args.get("expression") or "").upper():
            return "FOREIGN"
    return None


def show_key_scope(
    statement: exp.Expression, session: SessionContext
) -> tuple[str, str | None, str | None]:
    """Return the ``(database, schema, table)`` a scoped SHOW KEYS statement targets."""
    scope_kind = statement.args.get("scope_kind")
    scope = statement.args.get("scope")
    database: str = session.database
    schema: str | None = session.schema
    table_name: str | None = None
    if scope_kind == "DATABASE" and isinstance(scope, exp.Table):
        database, schema = scope.name, None
    elif scope_kind == "SCHEMA" and isinstance(scope, exp.Table):
        db = scope.args.get("db")
        database = db.name if isinstance(db, exp.Identifier) else session.database
        schema = scope.name
    elif scope_kind == "TABLE" and isinstance(scope, exp.Table):
        catalog = scope.args.get("catalog")
        db = scope.args.get("db")
        database = catalog.name if isinstance(catalog, exp.Identifier) else session.database
        schema = db.name if isinstance(db, exp.Identifier) else session.schema
        table_name = scope.name
    elif isinstance(statement, exp.Command):
        # SHOW FOREIGN KEYS is not parseable by sqlglot, so read the ``IN TABLE x``
        # target from the raw text.
        match = re.search(
            r"\bIN\s+(?:TABLE\s+)?([\w.\"]+)",
            str(statement.args.get("expression") or ""),
            re.IGNORECASE,
        )
        if match:
            parts = [part.strip('"') for part in match.group(1).split(".")]
            table_name = parts[-1]
            if len(parts) >= 2:
                schema = parts[-2]
            if len(parts) >= 3:
                database = parts[-3]
    return database, schema, table_name


def execute_show_keys(
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle SHOW PRIMARY/UNIQUE/FOREIGN/IMPORTED KEYS via ``duckdb_constraints()``."""
    kind = show_key_kind(statement)
    if kind is None:
        return None
    database, schema, table_name = show_key_scope(statement, session)

    where = ["lower(database_name) = lower(?)"]
    params: list[Any] = [database]
    if schema is not None:
        where.append("lower(schema_name) = lower(?)")
        params.append(schema)
    if table_name is not None:
        where.append("lower(table_name) = lower(?)")
        params.append(table_name)

    if kind in ("PRIMARY", "UNIQUE"):
        params.insert(0, "PRIMARY KEY" if kind == "PRIMARY" else "UNIQUE")
        cursor.execute(
            f"""
            SELECT
                NULL::TIMESTAMP AS created_on,
                database_name AS database_name,
                schema_name AS schema_name,
                table_name AS table_name,
                u.column_name AS column_name,
                u.key_sequence AS key_sequence,
                constraint_name AS constraint_name,
                NULL::VARCHAR AS rely,
                NULL::VARCHAR AS comment
            FROM duckdb_constraints(), UNNEST(constraint_column_names) WITH ORDINALITY AS u(column_name, key_sequence)
            WHERE constraint_type = ? AND {' AND '.join(where)}
            ORDER BY table_name, constraint_name, key_sequence
            """,
            params,
        )
    else:
        cursor.execute(
            f"""
            SELECT
                NULL::TIMESTAMP AS created_on,
                database_name AS pk_database_name,
                schema_name AS pk_schema_name,
                referenced_table AS pk_table_name,
                referenced_column_names[u.key_sequence] AS pk_column_name,
                database_name AS fk_database_name,
                schema_name AS fk_schema_name,
                table_name AS fk_table_name,
                u.column_name AS fk_column_name,
                u.key_sequence AS key_sequence,
                NULL::VARCHAR AS update_rule,
                NULL::VARCHAR AS delete_rule,
                constraint_name AS fk_name,
                NULL::VARCHAR AS pk_name,
                NULL::VARCHAR AS deferrability,
                NULL::VARCHAR AS rely,
                NULL::VARCHAR AS comment
            FROM duckdb_constraints(), UNNEST(constraint_column_names) WITH ORDINALITY AS u(column_name, key_sequence)
            WHERE constraint_type = 'FOREIGN KEY' AND {' AND '.join(where)}
            ORDER BY table_name, constraint_name, key_sequence
            """,
            params,
        )
    return result_columns(cursor)


def execute_show_misc(
    manager: DuckDBManager,
    cursor: Any,
    session: SessionContext,
    statement: exp.Expression,
) -> tuple[list[ColumnMeta], list[list[Any]]] | None:
    """Handle the simpler SHOW variants (databases, warehouses, sequences, ...)."""
    obj: str | None = None
    if isinstance(statement, exp.Show):
        obj = str(statement.args.get("this") or "").upper()
    elif isinstance(statement, exp.Command) and str(statement.args.get("this") or "").upper() == "SHOW":
        # sqlglot cannot parse SHOW PARAMETERS.
        if str(statement.args.get("expression") or "").upper().strip().startswith("PARAMETERS"):
            obj = "PARAMETERS"

    if obj == "DATABASES":
        cursor.execute(
            """
            SELECT
                NULL::TIMESTAMP AS created_on,
                database_name AS name,
                'N' AS is_default,
                CASE WHEN lower(database_name) = lower(?) THEN 'Y' ELSE 'N' END AS is_current,
                NULL::VARCHAR AS origin,
                NULL::VARCHAR AS owner,
                comment AS comment,
                NULL::VARCHAR AS options,
                NULL::BIGINT AS retention_time
            FROM duckdb_databases()
            WHERE lower(database_name) NOT IN ('system', 'temp', 'memory')
            ORDER BY name
            """,
            [session.database],
        )
    elif obj == "WAREHOUSES":
        cursor.execute(
            "SELECT ? AS name, 'STARTED' AS state, 'STANDARD' AS type, 'X-SMALL' AS size, "
            "'N' AS running, 'N' AS queued, NULL::TIMESTAMP AS created_on, NULL::VARCHAR AS owner, "
            "NULL::VARCHAR AS comment",
            [session.warehouse],
        )
    elif obj == "SEQUENCES":
        scope = show_scope(statement, session) if isinstance(statement, exp.Show) else None
        database, schema = scope or (session.database, session.schema)
        where = ["lower(database_name) = lower(?)"]
        params: list[Any] = [database]
        if schema is not None:
            where.append("lower(schema_name) = lower(?)")
            params.append(schema)
        cursor.execute(
            f"""
            SELECT
                NULL::TIMESTAMP AS created_on,
                sequence_name AS name,
                database_name AS database_name,
                schema_name AS schema_name,
                last_value AS next_value,
                increment_by AS interval,
                comment AS comment
            FROM duckdb_sequences()
            WHERE {' AND '.join(where)}
            ORDER BY name
            """,
            params,
        )
    elif obj == "PARAMETERS":
        cursor.execute(
            """
            SELECT name, value, "default", level FROM (VALUES
                ('AUTOCOMMIT', 'true', 'true', 'SESSION'),
                ('TIMEZONE', 'UTC', 'UTC', 'SESSION'),
                ('CLIENT_PREFETCH_THREADS', '4', '4', 'SESSION')
            ) AS t(name, value, "default", level)
            """
        )
    elif obj == "PROCEDURES":
        cursor.execute(
            "SELECT NULL::VARCHAR AS name, NULL::VARCHAR AS schema_name, NULL::VARCHAR AS catalog_name WHERE FALSE"
        )
    elif obj == "FUNCTIONS":
        scope = show_scope(statement, session) if isinstance(statement, exp.Show) else None
        database, schema = scope or (session.database, session.schema)
        like = None
        if isinstance(statement, exp.Show):
            pattern = statement.args.get("like")
            if isinstance(pattern, exp.Literal) and pattern.is_string:
                like = str(pattern.this)
        return user_function_show(manager, cursor, database, schema, like)
    elif obj == "USERS":
        cursor.execute("SELECT NULL::VARCHAR AS name, NULL::TIMESTAMP AS created_on WHERE FALSE")
    else:
        return None
    row_type, rows = result_columns(cursor)
    prefix, like = show_name_patterns(statement)
    if prefix or like:
        name_index = {
            "DATABASES": 1,
            "WAREHOUSES": 0,
            "SEQUENCES": 1,
            "PARAMETERS": 0,
            "PROCEDURES": 0,
            "USERS": 0,
        }.get(obj)
        if name_index is not None:
            rows = [
                row
                for row in rows
                if name_matches(str(row[name_index]), prefix, like)
            ]
    limit = show_limit(statement) if isinstance(statement, exp.Show) else None
    if limit is not None:
        rows = rows[:limit]
    return row_type, rows


# Snowflake's ``SHOW FUNCTIONS`` column set. The metadata is derived from a typed,
# empty query so the declared Snowflake types (and the DuckDB types the wire encoders
# use) match the rest of the SHOW surface; the rows are built from recorded functions.
_USER_FUNCTION_SHOW_SQL = """
    SELECT
        NULL::TIMESTAMP AS created_on,
        NULL::VARCHAR AS name,
        NULL::VARCHAR AS schema_name,
        NULL::VARCHAR AS catalog_name,
        NULL::VARCHAR AS is_builtin,
        NULL::VARCHAR AS is_secure,
        NULL::VARCHAR AS arguments,
        NULL::VARCHAR AS result_type,
        NULL::VARCHAR AS description
    WHERE FALSE
"""

# ``SHOW USER FUNCTIONS IN <scope>`` is a raw command (sqlglot cannot parse it), so its
# optional ``LIKE``/``IN`` clauses are matched out of the statement text.
_FUNCTION_SHOW_LIKE_RE = re.compile(r"LIKE\s+'((?:[^']|'')*)'", re.IGNORECASE)
_FUNCTION_SHOW_SCOPE_RE = re.compile(
    r"\bIN\s+(?:DATABASE\s+|SCHEMA\s+)?([A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*)?)",
    re.IGNORECASE,
)
_FUNCTION_SHOW_DATABASE_RE = re.compile(r"\bIN\s+DATABASE\b", re.IGNORECASE)


def show_like_pattern(text: str) -> str | None:
    """Extract the ``LIKE '<pattern>'`` filter from a raw SHOW statement."""
    match = _FUNCTION_SHOW_LIKE_RE.search(text)
    return match.group(1).replace("''", "'") if match else None


def function_show_scope(
    text: str, session: SessionContext
) -> tuple[str, str | None]:
    """Extract the ``(database, schema)`` a raw ``SHOW ... FUNCTIONS IN ...`` targets.

    ``None`` for the schema means every schema in the database
    (``SHOW FUNCTIONS IN DATABASE db``); an unscoped statement uses the session's current
    database and schema.
    """
    match = _FUNCTION_SHOW_SCOPE_RE.search(text)
    if match is None:
        return session.database, session.schema
    parts = match.group(1).split(".")
    if _FUNCTION_SHOW_DATABASE_RE.search(text):
        return parts[0], None
    if len(parts) == 2:
        return parts[0], parts[1]
    return session.database, parts[0]


def user_function_show(
    manager: DuckDBManager,
    cursor: Any,
    database: str,
    schema: str | None,
    like: str | None = None,
) -> tuple[list[ColumnMeta], list[list[Any]]]:
    """Build Snowflake-shaped ``SHOW FUNCTIONS`` output from the recorded functions."""
    cursor.execute(_USER_FUNCTION_SHOW_SQL)
    row_type = result_columns(cursor)[0]
    rows = [
        [
            fn.created_on,
            fn.name,
            fn.schema,
            fn.database,
            "N",  # is_builtin: only user functions are recorded
            "N",  # is_secure: SECURE is accepted but not enforced
            fn.show_signature,
            fn.return_type,
            None,  # description
        ]
        for fn in query_user_functions(manager, database, schema, like=like)
    ]
    return row_type, rows
