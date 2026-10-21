"""Translate Snowflake SQL into DuckDB-compatible SQL using sqlglot."""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from snowflake_emulator import sqlglot_patch  # noqa: F401  (global parser settings)
from snowflake_emulator.core.ast import drop_target_table  # re-exported for compatibility
from snowflake_emulator.core.constants import INFORMATION_SCHEMA_SCHEMA
from snowflake_emulator.translator.arrays import ColumnTypeResolver
from snowflake_emulator.translator.ast_helpers import _quoted_identifier
from snowflake_emulator.translator.constants import (
    EMULATED_SNOWFLAKE_VERSION,
    READ_DIALECT,
    WRITE_DIALECT,
    _SIMPLE_IDENTIFIER_RE,
)
from snowflake_emulator.translator.errors import TranslationError
from snowflake_emulator.translator.functions import (
    UserFunctionArgument,
    UserFunctionCall,
    UserFunctionResolver,
    UserFunctionSpec,
    _reject_unsupported_function,
    _rewrite_user_function_calls,
    normalize_function_signature,
    render_create_macro,
    render_drop_macro,
    render_qualified_name,
    translate_create_function,
    _duckdb_type_text,
    translate_function_body,
    user_function_dispatch_name,
    user_function_macro_name,
    user_function_sub_macro_name,
)
from snowflake_emulator.translator.parsing import (
    preprocess_array_calls,
    split_statements,
)
from snowflake_emulator.translator.transforms import _apply_transforms


_USE_RE = re.compile(
    r"^\s*USE\s+(DATABASE|SCHEMA|WAREHOUSE|ROLE)\s+(.+?)\s*;?\s*$",
    re.IGNORECASE,
)

# ``USE DATABASE IDENTIFIER('MY_DB')`` — Snowflake's IDENTIFIER() resolves its string
# literal argument to a bare identifier. sqlglot keeps the wrapper when re-serialized
# with the Snowflake dialect, so unwrap it before using the name.
_IDENTIFIER_FN_RE = re.compile(
    r"^IDENTIFIER\(\s*(['\"])(.*?)\1\s*\)$",
    re.IGNORECASE | re.DOTALL,
)

# Session-scoped ``ALTER SESSION ...`` statements have no DuckDB equivalent.
_ALTER_SESSION_RE = re.compile(r"^\s*ALTER\s+SESSION\b", re.IGNORECASE)


def _unwrap_identifier_fn(name: str) -> str:
    """Return the bare identifier wrapped by ``IDENTIFIER('literal')``, if any."""
    m = _IDENTIFIER_FN_RE.match(name)
    return m.group(2) if m else name



def _upper_if_simple(name: str) -> str:
    """Uppercase a name only when it is a plain (unquoted-style) identifier.

    Snowflake resolves ``IDENTIFIER('simple_literal')`` like an unquoted identifier
    (uppercased), but a literal that needs quoting stays case-sensitive.
    """
    return name.upper() if _SIMPLE_IDENTIFIER_RE.match(name) else name


def _normalize_use_name(name: str) -> str:
    """Normalize a ``USE`` target to Snowflake identifier casing.

    Unquoted names (and ``IDENTIFIER('simple_literal')``) are uppercased; quoted
    names preserve their exact case. Handles dotted ``USE SCHEMA db.schema`` paths.
    """
    parts: list[str] = []
    for raw in name.split("."):
        raw = raw.strip()
        unwrapped = _unwrap_identifier_fn(raw)
        if unwrapped != raw:
            raw = _upper_if_simple(unwrapped)
        elif raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            raw = raw[1:-1]
        else:
            raw = raw.upper()
        parts.append(raw)
    return ".".join(parts)


def is_alter_session(sql: str) -> bool:
    """Detect Snowflake session-scoped ``ALTER SESSION ...`` statements."""
    return _ALTER_SESSION_RE.match(sql) is not None


def is_transaction_control(statement: exp.Expression) -> bool:
    """Detect transaction-control statements (BEGIN/COMMIT/ROLLBACK).

    sqlglot parses ``BEGIN``/``BEGIN TRANSACTION`` as ``exp.Transaction`` and
    ``COMMIT``/``ROLLBACK`` (with or without ``WORK``) as ``exp.Commit``/
    ``exp.Rollback``. ``START TRANSACTION`` is not valid Snowflake, so it is not
    handled (sqlglot mis-parses it as ``START AS TRANSACTION`` anyway).
    """
    return isinstance(statement, (exp.Transaction, exp.Commit, exp.Rollback))


@dataclass(frozen=True)
class UseStatement:
    """A parsed ``USE <OBJECT> <NAME>`` session command."""

    kind: str  # DATABASE | SCHEMA | WAREHOUSE | ROLE | SCHEMA_QUALIFIED
    name: str



def match_use_statement(sql: str) -> UseStatement | None:
    """Detect Snowflake session-scoped ``USE ...`` statements that DuckDB has no concept of."""
    m = _USE_RE.match(sql)
    if not m:
        return None
    kind, name = m.group(1).upper(), m.group(2).strip()
    name = _normalize_use_name(name)
    if "." in name and kind == "SCHEMA":
        # `USE SCHEMA db.schema` is also valid Snowflake syntax.
        return UseStatement(kind="SCHEMA_QUALIFIED", name=name)
    return UseStatement(kind=kind, name=name)






def _auto_increment_constraint(col: exp.ColumnDef) -> exp.ColumnConstraint | None:
    """Return the AUTOINCREMENT/IDENTITY constraint on `col`, if any."""
    for constraint in col.args.get("constraints") or []:
        if isinstance(
            constraint.kind,
            (exp.AutoIncrementColumnConstraint, exp.GeneratedAsIdentityColumnConstraint),
        ):
            return constraint
    return None


def _auto_increment_bounds(
    constraint: exp.ColumnConstraint,
) -> tuple[exp.Expression, exp.Expression]:
    """Extract (start, increment) values from an auto-increment constraint."""
    kind = constraint.kind
    if isinstance(kind, exp.AutoIncrementColumnConstraint):
        return exp.Literal.number(1), exp.Literal.number(1)
    start = kind.args.get("start")
    increment = kind.args.get("increment")
    return (
        start if start is not None else exp.Literal.number(1),
        increment if increment is not None else exp.Literal.number(1),
    )


def _table_name_parts(table: exp.Table) -> list[str]:
    """Return [catalog?, schema?, table] identifier parts for a CREATE TABLE target."""
    parts: list[str] = []
    if table.args.get("catalog") is not None:
        parts.append(table.args["catalog"].name)
    if table.args.get("db") is not None:
        parts.append(table.args["db"].name)
    parts.append(table.name)
    return parts


def _make_drop_table(table: exp.Table) -> exp.Drop:
    """Build a ``DROP TABLE IF EXISTS <table>`` for ``table``.

    sqlglot models a ``DROP`` target as a ``tables`` list (``this`` before 30.18, which
    the project no longer supports).
    """
    return exp.Drop(tables=[table.copy()], kind="TABLE", exists=True)


def _make_sequence_create(
    seq_parts: list[str],
    start: exp.Expression,
    increment: exp.Expression,
    *,
    replace: bool,
) -> exp.Create:
    """Build a DuckDB CREATE SEQUENCE statement for an auto-increment column."""
    this = exp.Identifier(this=seq_parts[-1], quoted=True)
    db = exp.Identifier(this=seq_parts[-2], quoted=True) if len(seq_parts) > 1 else None
    catalog = exp.Identifier(this=seq_parts[-3], quoted=True) if len(seq_parts) > 2 else None
    return exp.Create(
        this=exp.Table(this=this, db=db, catalog=catalog),
        kind="SEQUENCE",
        replace=replace,
        exists=not replace,
        properties=exp.Properties(
            expressions=[exp.SequenceProperties(start=start, increment=increment)]
        ),
    )


def _nextval_default(seq_parts: list[str]) -> exp.ColumnConstraint:
    """Build a ``DEFAULT nextval('...')`` column constraint for a sequence."""
    seq_sql_name = ".".join(_quoted_identifier(part) for part in seq_parts)
    return exp.ColumnConstraint(
        kind=exp.DefaultColumnConstraint(
            this=exp.func("nextval", exp.Literal.string(seq_sql_name))
        )
    )


def _rewrite_autoincrement_create(
    statement: exp.Create,
) -> tuple[list[exp.Expression], exp.Create]:
    """Rewrite a CREATE TABLE so AUTOINCREMENT/IDENTITY columns use DuckDB sequences.

    DuckDB has no identity-column support, so each Snowflake AUTOINCREMENT/IDENTITY
    column is backed by an explicit ``CREATE SEQUENCE`` plus a column default of
    ``nextval(...)``.
    """
    if statement.args.get("kind") != "TABLE" or not isinstance(statement.this, exp.Schema):
        return [], statement

    table = statement.this.this
    if not isinstance(table, exp.Table):
        return [], statement

    has_auto_increment = any(
        isinstance(col, exp.ColumnDef) and _auto_increment_constraint(col) is not None
        for col in statement.this.expressions
    )
    if not has_auto_increment:
        return [], statement

    table_parts = _table_name_parts(table)
    replace = bool(statement.args.get("replace"))
    pre_statements: list[exp.Expression] = []

    # ``CREATE OR REPLACE SEQUENCE`` cannot replace a sequence that an existing
    # table depends on, so drop the old table first when the CREATE is a replace.
    if replace:
        pre_statements.append(_make_drop_table(table))

    columns: list[exp.Expression] = []
    for col in statement.this.expressions:
        if not isinstance(col, exp.ColumnDef):
            columns.append(col)
            continue

        constraint = _auto_increment_constraint(col)
        if constraint is None:
            columns.append(col.copy())
            continue

        seq_parts = table_parts[:-1] + [f"{table.name}_{col.this.name}_seq"]
        start, increment = _auto_increment_bounds(constraint)
        pre_statements.append(
            _make_sequence_create(seq_parts, start, increment, replace=replace)
        )

        remaining = [
            c for c in (col.args.get("constraints") or []) if c is not constraint
        ]
        remaining.append(_nextval_default(seq_parts))
        new_col = col.copy()
        new_col.set("constraints", remaining)
        columns.append(new_col)

    schema = statement.this.copy()
    schema.set("expressions", columns)
    rewritten = statement.copy()
    rewritten.set("this", schema)
    if replace:
        rewritten.set("replace", False)
    return pre_statements, rewritten




def _rewrite_clone(statement: exp.Expression) -> exp.Expression:
    """Rewrite ``CREATE TABLE b CLONE a`` as ``CREATE TABLE b AS SELECT * FROM a``.

    Snowflake's CLONE is a zero-copy metadata operation and DuckDB has no equivalent,
    so the rows are copied instead (functionally equivalent for local testing). A
    time-travel clause (``AT``/``BEFORE``) is rejected: the emulator keeps no
    historical state, so it cannot rebuild a past version of the source.
    """
    if not isinstance(statement, exp.Create):
        return statement
    clone = statement.args.get("clone")
    if not isinstance(clone, exp.Clone):
        return statement
    kind = str(statement.args.get("kind") or "").upper()
    if kind != "TABLE":
        # Rejected loudly rather than silently creating an empty object.
        raise TranslationError(
            f"CREATE {kind} ... CLONE is not supported by the emulator "
            "(only CREATE TABLE ... CLONE is)."
        )
    source = clone.this
    if not isinstance(source, exp.Table):
        return statement
    if source.args.get("when") is not None:
        raise TranslationError(
            "CREATE TABLE ... CLONE ... AT/BEFORE (time travel) is not supported "
            "by the emulator."
        )
    rewritten = statement.copy()
    rewritten.set("clone", None)
    rewritten.set("expression", exp.select("*").from_(source.copy()))
    return rewritten




def _strip_table_constraints(statement: exp.Expression) -> exp.Expression:
    """Remove constraints DuckDB would enforce but Snowflake would not.

    Snowflake records primary, unique, foreign-key and check constraints without
    enforcing them (``NOT NULL`` is the exception). DuckDB enforces all of them, so the
    declarations are stripped here; their metadata is recorded separately for SHOW.
    """
    if not (
        isinstance(statement, exp.Create)
        and str(statement.args.get("kind") or "").upper() == "TABLE"
        and isinstance(statement.this, exp.Schema)
    ):
        return statement
    rewritten = statement.copy()
    schema = rewritten.this.copy()
    expressions: list[exp.Expression] = []
    for item in schema.expressions:
        if isinstance(item, exp.ColumnDef):
            constraints = item.args.get("constraints") or []
            kept = [
                constraint
                for constraint in constraints
                if not isinstance(
                    constraint.kind,
                    (
                        exp.PrimaryKeyColumnConstraint,
                        exp.UniqueColumnConstraint,
                        exp.CheckColumnConstraint,
                        exp.Reference,
                    ),
                )
            ]
            if len(kept) != len(constraints):
                item = item.copy()
                item.set("constraints", kept)
            expressions.append(item)
            continue
        if isinstance(
            item,
            (
                exp.Constraint,
                exp.PrimaryKey,
                exp.UniqueColumnConstraint,
                exp.ForeignKey,
                exp.CheckColumnConstraint,
            ),
        ):
            continue
        expressions.append(item)
    schema.set("expressions", expressions)
    rewritten.set("this", schema)
    return rewritten


def _reject_unsupported_ddl(statement: exp.Expression) -> None:
    """Reject DDL the emulator cannot emulate faithfully instead of downgrading it."""
    if isinstance(statement, exp.Command):
        text = str(statement.args.get("expression") or "").lstrip().upper()
        if text.startswith("RECURSIVE VIEW"):
            raise TranslationError(
                "CREATE RECURSIVE VIEW is not supported by the emulator."
            )
        return
    if not isinstance(statement, exp.Create):
        return
    kind = str(statement.args.get("kind") or "").upper()
    if kind == "INDEX":
        raise TranslationError("CREATE INDEX is not supported by Snowflake or the emulator.")
    if kind == "VIEW":
        properties = statement.args.get("properties")
        if properties is not None and any(
            isinstance(prop, exp.MaterializedProperty)
            for prop in properties.expressions
        ):
            raise TranslationError(
                "CREATE MATERIALIZED VIEW is not supported by the emulator."
            )


def transpile_to_duckdb(
    statement: exp.Expression,
    user_functions: UserFunctionResolver | None = None,
    column_types: ColumnTypeResolver | None = None,
) -> str:
    """Transpile a single parsed Snowflake statement into its DuckDB equivalent.

    `user_functions` resolves calls to user-defined functions (``CREATE FUNCTION``) to
    the DuckDB macro backing them; pass ``None`` when no functions can be in scope.
    `column_types` resolves a bare column's DuckDB type, which an array rewrite needs to tell
    a real DuckDB LIST from a VARIANT/JSON array; pass ``None`` when no catalog is in scope.
    """
    try:
        _reject_unsupported_function(statement)
        _reject_unsupported_ddl(statement)
        statement = _rewrite_clone(statement)
        statement = _apply_transforms(statement, column_types)
        if user_functions is not None:
            statement = _rewrite_user_function_calls(statement, user_functions)

        # Snowflake's ``INSERT OVERWRITE`` clears the table then inserts; DuckDB has
        # no such statement (it would be mis-serialized as ``INSERT OVERWRITE TABLE``).
        if isinstance(statement, exp.Insert) and statement.args.get("overwrite"):
            statement = statement.copy()
            statement.set("overwrite", False)
            target = statement.args.get("this")
            table = target.this if isinstance(target, exp.Schema) else target
            if isinstance(table, exp.Table):
                delete = exp.Delete(this=table.copy())
                return f"{delete.sql(dialect=WRITE_DIALECT)};\n{statement.sql(dialect=WRITE_DIALECT)}"

        if isinstance(statement, exp.Create) and statement.args.get("kind") == "TABLE":
            pre_statements, create = _rewrite_autoincrement_create(statement)
            create = _strip_table_constraints(create)
            if pre_statements:
                parts = [seq.sql(dialect=WRITE_DIALECT) for seq in pre_statements]
                parts.append(create.sql(dialect=WRITE_DIALECT))
                return ";\n".join(parts)
            statement = create
        return statement.sql(dialect=WRITE_DIALECT)
    except TranslationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TranslationError(f"Failed to transpile SQL: {exc}") from exc


def statement_kind(sql: str) -> str:
    """Best-effort classification of a statement (SELECT, INSERT, CREATE, ...)."""
    try:
        parsed = sqlglot.parse_one(sql, read=READ_DIALECT)
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    return type(parsed).__name__.upper() if parsed else "UNKNOWN"


def is_query(sql: str) -> bool:
    """Return True if the statement returns rows (SELECT/SHOW-like) rather than a status."""
    try:
        parsed = sqlglot.parse_one(sql, read=READ_DIALECT)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(parsed, (exp.Select, exp.Union, exp.Subquery))
