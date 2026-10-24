"""Snowflake-shaped ``SHOW`` handlers.

DuckDB's native SHOW returns its own column names, while Snowflake tools such as dbt
expect ``name``/``kind``/``database_name``/... These handlers build the result from
DuckDB's catalog introspection functions instead.
"""

from __future__ import annotations

import re
from typing import Any

from sqlglot import exp

from snowflake_emulator.core.constants import (
    EMULATED_ACCOUNT_NAME,
    INFORMATION_SCHEMA_SCHEMA,
)
from snowflake_emulator.core.results import ColumnMeta
from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.engine.handlers.common import (
    name_matches,
    prefer_declared_type,
    result_columns,
    show_name_patterns,
    temp_inclusive_scope,
)
from snowflake_emulator.output_formats import (
    DEFAULT_BINARY_FORMAT,
    DEFAULT_DATE_FORMAT,
    DEFAULT_TIME_FORMAT,
    DEFAULT_TIMESTAMP_FORMAT,
)
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.type_mapping import duckdb_type_to_snowflake_sql_type
from snowflake_emulator.user_functions import query_user_functions

# Snowflake's system roles; the emulator keeps the account-role hierarchy static.
_SYSTEM_ROLES = ("ACCOUNTADMIN", "SYSADMIN", "PUBLIC")

# The session parameters the emulator models (Snowflake's list is much longer). ``key`` /
# ``value`` / ``default`` / ``level`` / ``description`` is Snowflake's SHOW PARAMETERS shape.
_SESSION_PARAMETERS: tuple[tuple[str, str, str, str], ...] = (
    ("AUTOCOMMIT", "true", "true", "If true, statements are committed automatically."),
    ("BINARY_OUTPUT_FORMAT", DEFAULT_BINARY_FORMAT, DEFAULT_BINARY_FORMAT,
     "Format used for BINARY output: HEX, BASE64 or UTF-8."),
    ("CLIENT_PREFETCH_THREADS", "4", "4", "Number of threads used by the client to prefetch result chunks."),
    ("CLIENT_RESULT_CHUNK_SIZE", "160", "160", "Size (in MB) of each result chunk sent to the client."),
    ("CLIENT_SESSION_KEEP_ALIVE", "false", "false", "Whether the client keeps the session alive."),
    ("DATE_OUTPUT_FORMAT", DEFAULT_DATE_FORMAT, DEFAULT_DATE_FORMAT, "Format of DATE output."),
    ("MULTI_STATEMENT_COUNT", "1", "1", "Number of statements a SQL API request may carry (0 = any)."),
    ("QUOTED_IDENTIFIERS_IGNORE_CASE", "false", "false",
     "Whether quoted identifiers are matched case-insensitively."),
    ("STATEMENT_TIMEOUT_IN_SECONDS", "172800", "172800", "Statement timeout; 0 disables it."),
    ("TIMEZONE", "UTC", "UTC", "Session time zone."),
    ("TIME_OUTPUT_FORMAT", DEFAULT_TIME_FORMAT, DEFAULT_TIME_FORMAT, "Format of TIME output."),
    ("TIMESTAMP_DAY_IS_ALWAYS_29TH", "false", "false", "Whether DAY is always the 29th of the month."),
    ("TIMESTAMP_LTZ_OUTPUT_FORMAT", DEFAULT_TIMESTAMP_FORMAT, DEFAULT_TIMESTAMP_FORMAT,
     "Format of TIMESTAMP_LTZ output."),
    ("TIMESTAMP_NTZ_OUTPUT_FORMAT", DEFAULT_TIMESTAMP_FORMAT, DEFAULT_TIMESTAMP_FORMAT,
     "Format of TIMESTAMP_NTZ output."),
    ("TIMESTAMP_OUTPUT_FORMAT", DEFAULT_TIMESTAMP_FORMAT, DEFAULT_TIMESTAMP_FORMAT,
     "Format of TIMESTAMP output."),
    ("TIMESTAMP_TZ_OUTPUT_FORMAT", DEFAULT_TIMESTAMP_FORMAT, DEFAULT_TIMESTAMP_FORMAT,
     "Format of TIMESTAMP_TZ output."),
    ("USE_CACHED_RESULT", "true", "true", "Whether a repeated query may reuse a cached result."),
    ("WEEK_OF_YEAR_POLICY", "0", "0", "How the first week of the year is determined."),
    ("WEEK_START", "0", "0", "Day the week starts on (0 = legacy Monday behaviour)."),
)

# A curated catalogue of the built-in functions the emulator translates. SHOW FUNCTIONS
# lists built-ins, user-defined functions and external functions (docs); SHOW USER
# FUNCTIONS lists only the user-defined ones, and INFORMATION_SCHEMA.FUNCTIONS is
# documented as UDF-only, so it stays as it is.
_BUILTIN_FUNCTIONS: tuple[tuple[str, str, str, str], ...] = (
    # (name, signature, is_aggregate, short description)
    ("ABS", "(NUMBER) -> NUMBER", "N", "Absolute value."),
    ("ARRAY_AGG", "(VARCHAR) -> ARRAY", "Y", "Aggregates values into an array."),
    ("ARRAY_APPEND", "(ARRAY, VARIANT) -> ARRAY", "N", "Appends an element to an array."),
    ("ARRAY_CAT", "(ARRAY, ARRAY) -> ARRAY", "N", "Concatenates two arrays."),
    ("ARRAY_COMPACT", "(ARRAY) -> ARRAY", "N", "Removes NULL elements."),
    ("ARRAY_CONTAINS", "(VARIANT, ARRAY) -> BOOLEAN", "N", "Whether an array holds a value."),
    ("ARRAY_CONSTRUCT", "(VARIANT, ...) -> ARRAY", "N", "Builds an array from its arguments."),
    ("ARRAY_DISTINCT", "(ARRAY) -> ARRAY", "N", "Removes duplicate elements."),
    ("ARRAY_EXCEPT", "(ARRAY, ARRAY) -> ARRAY", "N", "Elements of the first array not in the second."),
    ("ARRAY_FLATTEN", "(ARRAY) -> ARRAY", "N", "Flattens one level of nested arrays."),
    ("ARRAY_GENERATE_RANGE", "(NUMBER, NUMBER, NUMBER) -> ARRAY", "N", "Builds a numeric range array."),
    ("ARRAY_INSERT", "(ARRAY, NUMBER, VARIANT) -> ARRAY", "N", "Inserts an element at a position."),
    ("ARRAY_INTERSECTION", "(ARRAY, ARRAY) -> ARRAY", "N", "Elements present in both arrays."),
    ("ARRAY_POSITION", "(VARIANT, ARRAY) -> NUMBER", "N", "Position of a value in an array."),
    ("ARRAY_PREPEND", "(VARIANT, ARRAY) -> ARRAY", "N", "Prepends an element to an array."),
    ("ARRAY_REMOVE", "(ARRAY, VARIANT) -> ARRAY", "N", "Removes a value from an array."),
    ("ARRAY_REMOVE_AT", "(ARRAY, NUMBER) -> ARRAY", "N", "Removes an element by position."),
    ("ARRAY_SIZE", "(ARRAY) -> NUMBER", "N", "Number of elements in an array."),
    ("ARRAY_SLICE", "(ARRAY, NUMBER, NUMBER) -> ARRAY", "N", "A range of array elements."),
    ("ARRAY_SORT", "(ARRAY, BOOLEAN, BOOLEAN) -> ARRAY", "N", "Sorts an array."),
    ("ARRAY_TO_STRING", "(ARRAY, VARCHAR) -> VARCHAR", "N", "Joins array elements as text."),
    ("ARRAY_UNIQUE", "(ARRAY) -> ARRAY", "N", "Distinct elements of an array."),
    ("AVG", "(NUMBER) -> NUMBER", "Y", "Average of the values."),
    ("BOOL_AND", "(BOOLEAN) -> BOOLEAN", "Y", "TRUE when every value is TRUE."),
    ("BOOL_OR", "(BOOLEAN) -> BOOLEAN", "Y", "TRUE when any value is TRUE."),
    ("CAST", "(VARIANT) -> VARIANT", "N", "Converts a value to another data type."),
    ("CEIL", "(NUMBER) -> NUMBER", "N", "Rounds up."),
    ("CHARINDEX", "(VARCHAR, VARCHAR, NUMBER) -> NUMBER", "N", "Position of a substring."),
    ("COALESCE", "(VARIANT, ...) -> VARIANT", "N", "First non-NULL argument."),
    ("CONCAT", "(VARCHAR, ...) -> VARCHAR", "N", "Concatenates its arguments."),
    ("CONTAINS", "(VARCHAR, VARCHAR) -> BOOLEAN", "N", "Whether a string holds a substring."),
    ("COUNT", "(VARIANT) -> NUMBER", "Y", "Number of rows."),
    ("COUNT_IF", "(BOOLEAN) -> NUMBER", "Y", "Number of rows where the condition holds."),
    ("CURRENT_DATE", "() -> DATE", "N", "Current date."),
    ("CURRENT_TIMESTAMP", "() -> TIMESTAMP_LTZ", "N", "Current timestamp."),
    ("DATEADD", "(VARCHAR, NUMBER, TIMESTAMP) -> TIMESTAMP", "N", "Adds an interval to a date or time."),
    ("DATEDIFF", "(VARCHAR, TIMESTAMP, TIMESTAMP) -> NUMBER", "N", "Difference between two dates."),
    ("DATE_TRUNC", "(VARCHAR, TIMESTAMP) -> TIMESTAMP", "N", "Truncates a date or time."),
    ("DIV0", "(NUMBER, NUMBER) -> NUMBER", "N", "Division that returns 0 for a zero divisor."),
    ("DIV0NULL", "(NUMBER, NUMBER) -> NUMBER", "N", "Division that returns 0 for a zero or NULL divisor."),
    ("EQUAL_NULL", "(VARIANT, VARIANT) -> BOOLEAN", "N", "NULL-safe equality."),
    ("FLATTEN", "(VARIANT) -> TABLE", "N", "Flattens a compound value into rows."),
    ("FLOOR", "(NUMBER) -> NUMBER", "N", "Rounds down."),
    ("GET", "(VARIANT, VARCHAR) -> VARIANT", "N", "Extracts a value from semi-structured data."),
    ("GET_IGNORE_CASE", "(VARIANT, VARCHAR) -> VARIANT", "N", "Case-insensitive GET."),
    ("GET_PATH", "(VARIANT, VARCHAR) -> VARIANT", "N", "Extracts a value by path."),
    ("GREATEST", "(VARIANT, ...) -> VARIANT", "N", "Largest of its arguments."),
    ("IFF", "(BOOLEAN, VARIANT, VARIANT) -> VARIANT", "N", "Conditional expression."),
    ("INITCAP", "(VARCHAR) -> VARCHAR", "N", "Capitalizes each word."),
    ("IS_ARRAY", "(VARIANT) -> BOOLEAN", "N", "Whether the value is an ARRAY."),
    ("IS_DATE", "(VARIANT) -> BOOLEAN", "N", "Whether the value is a DATE."),
    ("IS_INTEGER", "(VARIANT) -> BOOLEAN", "N", "Whether the value is an integer."),
    ("IS_NULL_VALUE", "(VARIANT) -> BOOLEAN", "N", "Whether the value is a JSON null."),
    ("IS_OBJECT", "(VARIANT) -> BOOLEAN", "N", "Whether the value is an OBJECT."),
    ("IS_VARCHAR", "(VARIANT) -> BOOLEAN", "N", "Whether the value is a string."),
    ("LAG", "(VARIANT, NUMBER, VARIANT) -> VARIANT", "N", "Value from a preceding row."),
    ("LEAD", "(VARIANT, NUMBER, VARIANT) -> VARIANT", "N", "Value from a following row."),
    ("LEAST", "(VARIANT, ...) -> VARIANT", "N", "Smallest of its arguments."),
    ("LEFT", "(VARCHAR, NUMBER) -> VARCHAR", "N", "Leftmost characters."),
    ("LENGTH", "(VARCHAR) -> NUMBER", "N", "Number of characters."),
    ("LISTAGG", "(VARCHAR, VARCHAR) -> VARCHAR", "Y", "Concatenates values in order."),
    ("LOWER", "(VARCHAR) -> VARCHAR", "N", "Converts to lower case."),
    ("LPAD", "(VARCHAR, NUMBER, VARCHAR) -> VARCHAR", "N", "Left-pads a string."),
    ("LTRIM", "(VARCHAR, VARCHAR) -> VARCHAR", "N", "Removes leading characters."),
    ("MAX", "(VARIANT) -> VARIANT", "Y", "Largest value."),
    ("MD5", "(VARCHAR) -> VARCHAR", "N", "MD5 hash as hex text."),
    ("MEDIAN", "(NUMBER) -> NUMBER", "Y", "Median of the values."),
    ("MIN", "(VARIANT) -> VARIANT", "Y", "Smallest value."),
    ("MOD", "(NUMBER, NUMBER) -> NUMBER", "N", "Remainder of a division."),
    ("NULLIF", "(VARIANT, VARIANT) -> VARIANT", "N", "NULL when the arguments are equal."),
    ("OBJECT_CONSTRUCT", "(VARCHAR, VARIANT, ...) -> OBJECT", "N", "Builds an OBJECT."),
    ("OBJECT_DELETE", "(OBJECT, VARCHAR, ...) -> OBJECT", "N", "Removes key-value pairs."),
    ("OBJECT_INSERT", "(OBJECT, VARCHAR, VARIANT, BOOLEAN) -> OBJECT", "N", "Inserts or updates a pair."),
    ("OBJECT_KEYS", "(OBJECT) -> ARRAY", "N", "Keys of an object."),
    ("OBJECT_PICK", "(OBJECT, VARCHAR, ...) -> OBJECT", "N", "Keeps the given key-value pairs."),
    ("PARSE_JSON", "(VARCHAR) -> VARIANT", "N", "Parses JSON text into a VARIANT."),
    ("POW", "(NUMBER, NUMBER) -> NUMBER", "N", "Raises a number to a power."),
    ("RANK", "() -> NUMBER", "N", "Rank of a row within its window."),
    ("REGEXP_LIKE", "(VARCHAR, VARCHAR, VARCHAR) -> BOOLEAN", "N", "Regular-expression match."),
    ("REGEXP_REPLACE", "(VARCHAR, VARCHAR, VARCHAR) -> VARCHAR", "N", "Regular-expression replacement."),
    ("REGEXP_SUBSTR", "(VARCHAR, VARCHAR) -> VARCHAR", "N", "Regular-expression extraction."),
    ("REPLACE", "(VARCHAR, VARCHAR, VARCHAR) -> VARCHAR", "N", "Replaces every occurrence."),
    ("REVERSE", "(VARCHAR) -> VARCHAR", "N", "Reverses a string."),
    ("RIGHT", "(VARCHAR, NUMBER) -> VARCHAR", "N", "Rightmost characters."),
    ("ROUND", "(NUMBER, NUMBER) -> NUMBER", "N", "Rounds half away from zero."),
    ("ROW_NUMBER", "() -> NUMBER", "N", "Row number within its window."),
    ("RPAD", "(VARCHAR, NUMBER, VARCHAR) -> VARCHAR", "N", "Right-pads a string."),
    ("RTRIM", "(VARCHAR, VARCHAR) -> VARCHAR", "N", "Removes trailing characters."),
    ("SHA2", "(VARCHAR, NUMBER) -> VARCHAR", "N", "SHA-2 hash as hex text."),
    ("SPLIT", "(VARCHAR, VARCHAR) -> ARRAY", "N", "Splits a string on a separator."),
    ("SPLIT_PART", "(VARCHAR, VARCHAR, NUMBER) -> VARCHAR", "N", "One part of a split string."),
    ("SQRT", "(NUMBER) -> NUMBER", "N", "Square root."),
    ("STARTSWITH", "(VARCHAR, VARCHAR) -> BOOLEAN", "N", "Whether a string starts with a prefix."),
    ("STDDEV", "(NUMBER) -> NUMBER", "Y", "Sample standard deviation."),
    ("SUBSTR", "(VARCHAR, NUMBER, NUMBER) -> VARCHAR", "N", "Substring by position."),
    ("SUM", "(NUMBER) -> NUMBER", "Y", "Sum of the values."),
    ("TO_ARRAY", "(VARIANT) -> ARRAY", "N", "Converts a value to an ARRAY."),
    ("TO_CHAR", "(VARIANT, VARCHAR) -> VARCHAR", "N", "Converts a value to text."),
    ("TO_DATE", "(VARCHAR, VARCHAR) -> DATE", "N", "Converts a value to a DATE."),
    ("TO_JSON", "(VARIANT) -> VARCHAR", "N", "JSON text of a VARIANT."),
    ("TO_NUMBER", "(VARCHAR, VARCHAR, NUMBER, NUMBER) -> NUMBER", "N", "Converts a value to NUMBER."),
    ("TO_TIMESTAMP", "(VARCHAR, VARCHAR) -> TIMESTAMP", "N", "Converts a value to a timestamp."),
    ("TO_VARCHAR", "(VARIANT, VARCHAR) -> VARCHAR", "N", "Converts a value to text."),
    ("TO_VARIANT", "(VARIANT) -> VARIANT", "N", "Wraps a value in a VARIANT."),
    ("TRANSLATE", "(VARCHAR, VARCHAR, VARCHAR) -> VARCHAR", "N", "Replaces characters one-for-one."),
    ("TRIM", "(VARCHAR, VARCHAR) -> VARCHAR", "N", "Removes leading and trailing characters."),
    ("TRUNC", "(NUMBER, NUMBER) -> NUMBER", "N", "Truncates a number."),
    ("TRY_PARSE_JSON", "(VARCHAR) -> VARIANT", "N", "PARSE_JSON that answers NULL on failure."),
    ("TRY_TO_NUMBER", "(VARCHAR, VARCHAR, NUMBER, NUMBER) -> NUMBER", "N", "TO_NUMBER that answers NULL on failure."),
    ("TYPEOF", "(VARIANT) -> VARCHAR", "N", "Type of the value stored in a VARIANT."),
    ("UPPER", "(VARCHAR) -> VARCHAR", "N", "Converts to upper case."),
    ("ZEROIFNULL", "(NUMBER) -> NUMBER", "N", "0 when the value is NULL."),
)

# The table functions in the catalogue above.
_TABLE_FUNCTIONS = {"FLATTEN"}


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
            # a raw command and its scope/LIKE clause is parsed from the text. Only
            # ``SHOW FUNCTIONS``/``SHOW ALL FUNCTIONS`` list the built-ins; the USER and
            # EXTERNAL variants list user-defined functions only (docs).
            database, schema = function_show_scope(text, session)
            return user_function_show(
                manager,
                cursor,
                database,
                schema,
                show_like_pattern(text),
                builtins="ALL FUNCTIONS" in upper,
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

    where = ["lower(table_catalog) = lower(?)"]
    params: list[Any] = [database]
    if schema is not None:
        where.append("lower(table_schema) = lower(?)")
        params.append(schema)
    if table_name is not None:
        where.append("lower(table_name) = lower(?)")
        params.append(table_name)
    source = f'"{database}".{INFORMATION_SCHEMA_SCHEMA}._sf_constraints_ext'

    if kind in ("PRIMARY", "UNIQUE"):
        params.insert(0, "PRIMARY KEY" if kind == "PRIMARY" else "UNIQUE")
        cursor.execute(
            f"""
            SELECT
                NULL::TIMESTAMP AS created_on,
                table_catalog AS database_name,
                table_schema AS schema_name,
                table_name AS table_name,
                column_name AS column_name,
                key_sequence AS key_sequence,
                constraint_name AS constraint_name,
                NULL::VARCHAR AS rely,
                NULL::VARCHAR AS comment
            FROM {source}
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
                table_catalog AS pk_database_name,
                table_schema AS pk_schema_name,
                referenced_table AS pk_table_name,
                referenced_column AS pk_column_name,
                table_catalog AS fk_database_name,
                table_schema AS fk_schema_name,
                table_name AS fk_table_name,
                column_name AS fk_column_name,
                key_sequence AS key_sequence,
                NULL::VARCHAR AS update_rule,
                NULL::VARCHAR AS delete_rule,
                constraint_name AS fk_name,
                NULL::VARCHAR AS pk_name,
                NULL::VARCHAR AS deferrability,
                NULL::VARCHAR AS rely,
                NULL::VARCHAR AS comment
            FROM {source}
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
    command_text = ""
    if isinstance(statement, exp.Show):
        obj = str(statement.args.get("this") or "").upper()
    elif isinstance(statement, exp.Command) and str(statement.args.get("this") or "").upper() == "SHOW":
        # sqlglot cannot parse these SHOW variants, so the object is read from the raw text.
        command_text = str(statement.args.get("expression") or "").strip()
        command_upper = command_text.upper()
        if command_upper.startswith("TERSE "):
            command_upper = command_upper[len("TERSE "):]
            command_text = command_text[len("TERSE "):]
        for keyword in ("PARAMETERS", "ROLES", "GRANTS", "USER FUNCTIONS"):
            if command_upper.startswith(keyword):
                obj = keyword
                break

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
        values = ",\n                ".join(
            f"('{key}', '{value}', '{default}', 'SESSION', '{description}')"
            for key, value, default, description in _SESSION_PARAMETERS
        )
        cursor.execute(
            f"""
            SELECT key, value, "default", level, description FROM (VALUES
                {values}
            ) AS t(key, value, "default", level, description)
            """
        )
    elif obj == "ROLES":
        cursor.execute(
            """
            SELECT * FROM (VALUES
                (NULL::TIMESTAMP, 'ACCOUNTADMIN', ?, ?, 'N', 1, 0, 2, '', ''),
                (NULL::TIMESTAMP, 'SYSADMIN', 'N', 'N', 'N', 0, 0, 0, '', ''),
                (NULL::TIMESTAMP, 'PUBLIC', 'N', 'N', 'N', 0, 3, 0, '', '')
            ) AS t(created_on, name, is_default, is_current, is_inherited,
                   assigned_to_users, granted_to_roles, granted_roles, owner, comment)
            """,
            ["Y" if session.role.upper() == "ACCOUNTADMIN" else "N"] * 2,
        )
    elif obj == "GRANTS":
        return show_grants(cursor, session, command_text)
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
    if command_text:
        like = like or show_like_pattern(command_text)
        match = re.search(r"STARTS\s+WITH\s+'((?:[^']|'')*)'", command_text, re.IGNORECASE)
        if match is not None:
            prefix = prefix or match.group(1).replace("''", "'")
    if prefix or like:
        name_index = {
            "DATABASES": 1,
            "WAREHOUSES": 0,
            "SEQUENCES": 1,
            "PARAMETERS": 0,
            "ROLES": 1,
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
    if limit is None and command_text:
        match = re.search(r"\bLIMIT\s+(\d+)", command_text, re.IGNORECASE)
        if match is not None:
            limit = int(match.group(1))
    if limit is not None:
        rows = rows[:limit]
    return row_type, rows


# Snowflake's ``SHOW GRANTS`` shape: created_on, privilege, granted_on, name, granted_to,
# grantee_name, grant_option, granted_by (and granted_by_role_type for the ``ON`` variants).
_GRANT_COLUMNS = [
    ColumnMeta(name="created_on", type="timestamp_ntz", duckdb_type="TIMESTAMP"),
    ColumnMeta(name="privilege", type="text", duckdb_type="VARCHAR"),
    ColumnMeta(name="granted_on", type="text", duckdb_type="VARCHAR"),
    ColumnMeta(name="name", type="text", duckdb_type="VARCHAR"),
    ColumnMeta(name="granted_to", type="text", duckdb_type="VARCHAR"),
    ColumnMeta(name="grantee_name", type="text", duckdb_type="VARCHAR"),
    ColumnMeta(name="grant_option", type="boolean", duckdb_type="BOOLEAN"),
    ColumnMeta(name="granted_by", type="text", duckdb_type="VARCHAR"),
]

# The static role hierarchy the emulator reports: ACCOUNTADMIN inherits SYSADMIN, and every
# role inherits PUBLIC.
_ROLE_PARENTS = {"ACCOUNTADMIN": ("SYSADMIN",), "SYSADMIN": (), "PUBLIC": ()}


def _emulator_databases(cursor: Any) -> list[str]:
    """The user databases the emulator has, for OWNERSHIP/USAGE grant rows."""
    return [
        str(row[0])
        for row in cursor.execute(
            "SELECT database_name FROM duckdb_databases() "
            "WHERE lower(database_name) NOT IN ('system', 'temp', 'memory') ORDER BY 1"
        ).fetchall()
    ]


def _role_grant_rows(role: str, granted_to: str) -> list[list[Any]]:
    """The roles granted to `role` (or ``PUBLIC``), as SHOW GRANTS TO ROLE rows."""
    rows: list[list[Any]] = []
    inherited = _ROLE_PARENTS.get(role.upper(), ())
    for parent in (*inherited, *(("PUBLIC",) if role.upper() != "PUBLIC" else ())):
        rows.append([None, None, "ROLE", parent, granted_to, role, False, "ACCOUNTADMIN"])
    return rows


def _object_grant_rows(databases: list[str], role: str, granted_to: str) -> list[list[Any]]:
    """The privileges `role` holds on the emulator's databases."""
    rows: list[list[Any]] = []
    for database in databases:
        if role.upper() == "ACCOUNTADMIN":
            rows.append(
                [None, "OWNERSHIP", "DATABASE", database, granted_to, role, False, "ACCOUNTADMIN"]
            )
        elif role.upper() == "PUBLIC":
            rows.append(
                [None, "USAGE", "DATABASE", database, granted_to, role, False, "ACCOUNTADMIN"]
            )
    return rows


def show_grants(
    cursor: Any, session: SessionContext, text: str
) -> tuple[list[ColumnMeta], list[list[Any]]]:
    """Build Snowflake-shaped ``SHOW GRANTS`` output from the emulator's static model.

    Snowflake's privilege model is not emulated (everything is owned by the session role),
    so the rows describe the system role hierarchy, OWNERSHIP of each database, USAGE of
    each database for PUBLIC, and the account-level privileges ACCOUNTADMIN holds.
    """
    databases = _emulator_databases(cursor)
    upper = text.upper()
    words = text.split()
    upper_words = [word.upper() for word in words]
    rows: list[list[Any]] = []
    columns = list(_GRANT_COLUMNS)
    if "ON" in upper_words:
        index = upper_words.index("ON")
        target = words[index + 1] if index + 1 < len(words) else ""
        if target.upper() == "ACCOUNT":
            for privilege in ("CREATE DATABASE", "CREATE ROLE", "CREATE WAREHOUSE"):
                rows.append(
                    [None, privilege, "ACCOUNT", EMULATED_ACCOUNT_NAME, "ROLE",
                     "ACCOUNTADMIN", False, "ACCOUNTADMIN"]
                )
            columns.append(ColumnMeta(name="granted_by_role_type", type="text", duckdb_type="VARCHAR"))
            rows = [row + ["ROLE"] for row in rows]
        else:
            name = words[index + 2] if index + 2 < len(words) else ""
            if target and name:
                rows.append(
                    [None, "OWNERSHIP", target.upper(), name, "ROLE", session.role,
                     False, session.role]
                )
                if target.upper() == "DATABASE":
                    rows.append(
                        [None, "USAGE", target.upper(), name, "ROLE", "PUBLIC", False, session.role]
                    )
            columns.append(ColumnMeta(name="granted_by_role_type", type="text", duckdb_type="VARCHAR"))
            rows = [row + ["ROLE"] for row in rows]
        return columns, rows
    if "TO" in upper_words:
        index = upper_words.index("TO")
        kind = upper_words[index + 1] if index + 1 < len(upper_words) else ""
        name = words[index + 2] if index + 2 < len(words) else session.user
        if kind == "ROLE":
            rows.extend(_role_grant_rows(name, "ROLE"))
            rows.extend(_object_grant_rows(databases, name, "ROLE"))
        else:
            rows.append([None, None, "ROLE", "ACCOUNTADMIN", "USER", name, False, "SECURITYADMIN"])
        return columns, rows
    # Bare SHOW GRANTS is documented as SHOW GRANTS TO USER <current_user>.
    rows.append(
        [None, None, "ROLE", "ACCOUNTADMIN", "USER", session.user, False, "SECURITYADMIN"]
    )
    return columns, rows


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
    builtins: bool = True,
) -> tuple[list[ColumnMeta], list[list[Any]]]:
    """Build Snowflake-shaped ``SHOW FUNCTIONS`` output.

    ``SHOW FUNCTIONS`` lists built-ins as well as user-defined functions (docs); ``SHOW USER
    FUNCTIONS`` passes ``builtins=False``. INFORMATION_SCHEMA.FUNCTIONS stays UDF-only, which
    is what Snowflake documents for that view.
    """
    cursor.execute(_USER_FUNCTION_SHOW_SQL)
    row_type = result_columns(cursor)[0]
    rows = [
        [
            fn.created_on,
            fn.name,
            fn.schema,
            fn.database,
            "N",  # is_builtin: a recorded user function
            "N",  # is_secure: SECURE is accepted but not enforced
            fn.show_signature,
            fn.return_type,
            None,  # description
        ]
        for fn in query_user_functions(manager, database, schema, like=like)
    ]
    if builtins:
        for name, signature, _aggregate, description in _BUILTIN_FUNCTIONS:
            if like and not name_matches(name, None, like):
                continue
            arguments, _, result = signature.partition(" -> ")
            rows.append(
                [
                    None,  # created_on
                    name,
                    None,  # schema_name: NULL for a built-in
                    None,  # catalog_name: NULL for a built-in
                    "Y",
                    "N",
                    arguments,
                    result,
                    description,
                ]
            )
    return row_type, rows
