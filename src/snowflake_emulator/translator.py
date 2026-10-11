"""Translate Snowflake SQL into DuckDB-compatible SQL using sqlglot."""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

READ_DIALECT = "snowflake"
WRITE_DIALECT = "duckdb"

# Snowflake array functions whose array operand the emulator must present to DuckDB as a
# LIST rather than as the JSON it stores a VARIANT array as (see
# ``_rewrite_json_array_function``).
_JSON_ARRAY_FUNCTIONS = (exp.ArrayContains, exp.ArraySize, exp.ArraySlice)

# DuckDB reserves the ``information_schema`` schema name, so Snowflake's
# ``INFORMATION_SCHEMA`` is exposed under a non-reserved name and references are
# rewritten at the AST level (see ``_rewrite_information_schema``). The name is
# kept uppercase so it matches the emulator's Snowflake-style identifier casing.
INFORMATION_SCHEMA_SCHEMA = "SF_INFORMATION_SCHEMA"

# Snowflake's ``CURRENT_VERSION()`` reports the server's own release version. DuckDB's
# ``VERSION()`` returns ``v1.5.5`` (with a leading ``v``), which breaks clients such as
# snowflake-sqlalchemy that parse each dot-separated component as an ``int``. Answer
# with a Snowflake-compatible dotted-numeric version instead.
EMULATED_SNOWFLAKE_VERSION = "8.23.1"

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


_SIMPLE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


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


class TranslationError(ValueError):
    """Raised when a Snowflake statement cannot be parsed or transpiled."""


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


def split_statements(sql: str) -> list[exp.Expression]:
    """Split a (possibly multi-statement) Snowflake SQL string into parsed expressions.

    Returning parsed `exp.Expression` objects (rather than re-serialized SQL text)
    avoids a lossy parse -> Snowflake-text -> re-parse -> DuckDB-text round trip,
    which can drop information that doesn't survive an intermediate Snowflake-dialect
    print (e.g. `VARCHAR[]` collapsing to a bare `ARRAY`).
    """
    try:
        expressions = sqlglot.parse(sql, read=READ_DIALECT)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a translation error
        raise TranslationError(f"Failed to parse SQL: {exc}") from exc

    statements = [expr for expr in expressions if expr is not None]
    if not statements:
        raise TranslationError("No SQL statements found.")
    return statements


def _quoted_identifier(name: str) -> str:
    """Quote an identifier for use inside a DuckDB SQL string literal."""
    return '"' + name.replace('"', '""') + '"'


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
        pre_statements.append(exp.Drop(this=table.copy(), kind="TABLE", exists=True))

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


def _flatten_subquery(input_expr: exp.Expression) -> exp.Subquery:
    """Build the DuckDB subquery equivalent of Snowflake's ``FLATTEN(input => X)``.

    Snowflake's FLATTEN exposes SEQ/KEY/PATH/INDEX/VALUE/THIS. DuckDB's
    ``json_each()`` exposes key/value/fullkey/...; we map those onto Snowflake's
    names. INDEX is approximated as the array index (``TRY_CAST(key AS BIGINT)``),
    which is NULL for object keys — matching Snowflake's object behaviour.
    """
    x = input_expr.sql(dialect=WRITE_DIALECT)
    select = sqlglot.parse_one(
        f"SELECT ROW_NUMBER() OVER () - 1 AS seq, key, fullkey AS path, "
        f"TRY_CAST(key AS BIGINT) AS index, value, {x} AS this "
        f"FROM json_each({x})",
        read=WRITE_DIALECT,
    )
    return exp.Subquery(this=select)


def _uppercase_unquoted_identifiers(node: exp.Expression) -> exp.Expression:
    """Uppercase unquoted identifiers to match Snowflake's identifier casing.

    Snowflake uppercases unquoted identifiers (quoted ones stay case-sensitive).
    DuckDB preserves whatever case the client typed, so without this, table/column/
    alias names round-trip in the as-written case instead of Snowflake's uppercase
    canonical form. Quoted identifiers are left untouched.
    """
    if isinstance(node, exp.Identifier):
        if not node.args.get("quoted") and isinstance(node.this, str):
            node.set("this", node.this.upper())
    return node


def _is_information_schema(ident: exp.Identifier) -> bool:
    """Return True if ``ident`` names Snowflake's system ``INFORMATION_SCHEMA``.

    Unquoted references are case-insensitive (Snowflake uppercases them), while a
    quoted reference only resolves to the system schema when spelled exactly
    ``INFORMATION_SCHEMA``. Quoted mixed-case spellings name a different, unrelated
    schema and are left alone.
    """
    name = ident.name
    if not name or name.upper() != "INFORMATION_SCHEMA":
        return False
    if ident.args.get("quoted"):
        return name == "INFORMATION_SCHEMA"
    return True


def _rewrite_information_schema(node: exp.Expression) -> exp.Expression:
    """Map Snowflake's ``INFORMATION_SCHEMA`` onto the emulator's compat schema.

    Applied on the parsed AST (rather than rewriting raw SQL) so string literals and
    comments mentioning INFORMATION_SCHEMA are left untouched. Only the schema slot
    of a qualified name (``{db}.INFORMATION_SCHEMA.<obj>``) is rewritten.
    """
    if isinstance(node, (exp.Table, exp.Column)):
        db = node.args.get("db")
        if isinstance(db, exp.Identifier) and _is_information_schema(db):
            node.set("db", exp.Identifier(this=INFORMATION_SCHEMA_SCHEMA))
    return node


def _casts_to(node: exp.Expression, types: tuple[exp.DataType.Type, ...]) -> bool:
    """True when `node` is a ``CAST`` straight to one of `types`."""
    return (
        isinstance(node, exp.Cast)
        and isinstance(node.to, exp.DataType)
        and node.to.this in types
    )


def _is_json_value(node: exp.Expression) -> bool:
    """True for expressions the emulator evaluates to DuckDB's JSON type.

    Snowflake's OBJECT/ARRAY casts are rewritten to JSON before this runs, so a cast to
    either of those counts too.
    """
    if isinstance(node, (exp.JSONExtract, exp.JSONExtractScalar, exp.ParseJSON)):
        return True
    return _casts_to(node, (exp.DataType.Type.JSON,))


def _is_variant_value(node: exp.Expression) -> bool:
    """True for expressions the emulator evaluates to DuckDB's VARIANT type."""
    if isinstance(node, exp.ToVariant):
        return True
    return _casts_to(node, (exp.DataType.Type.VARIANT,))


def _is_json_list(node: exp.Expression) -> bool:
    """True for a ``CAST(... AS JSON[])``, the list form these rewrites produce."""
    return (
        isinstance(node, exp.Cast)
        and isinstance(node.to, exp.DataType)
        and node.to.this == exp.DataType.Type.ARRAY
        and any(
            isinstance(inner, exp.DataType) and inner.this == exp.DataType.Type.JSON
            for inner in node.to.expressions
        )
    )


def _is_json_array_value(node: exp.Expression) -> bool:
    """True for expressions the emulator evaluates to DuckDB's JSON[] type.

    Covers an array stored as JSON, and an ``ARRAY_SLICE`` over one -- which this module
    turns into a JSON[] call, so an enclosing call must treat it as an array too.
    """
    if _is_json_value(node) or _is_json_list(node):
        return True
    return isinstance(node, exp.ArraySlice) and _is_json_array_value(node.this)


def _duckdb_slice_bound(bound: exp.Expression) -> exp.Expression:
    """Translate one Snowflake ``ARRAY_SLICE`` bound into DuckDB's 1-based indexing.

    Snowflake's bounds are 0-based and inclusive, DuckDB's are 1-based and inclusive, and
    both count a negative bound from the end -- so only a non-negative bound moves, by one.
    DuckDB clamps an out-of-range bound and returns an empty list when the start is past
    the end, which is what Snowflake does too.
    """
    if _is_json_value(bound) or _is_variant_value(bound):
        # Snowflake casts a boundary to an integer for us; DuckDB will not add a JSON value.
        bound = exp.cast(bound, exp.DataType.build("INTEGER"), dialect=WRITE_DIALECT)
    if isinstance(bound, exp.Literal) and not bound.is_string:
        try:
            value = int(bound.name)
        except ValueError:
            pass
        else:
            return exp.Literal.number(value + 1 if value >= 0 else value)
    if isinstance(bound, exp.Placeholder) and bound.this is None:
        # The sign adjustment has to name the bound twice, so a positional `?` would need
        # two values for one placeholder. Say so instead of failing on a parameter count.
        raise TranslationError(
            "ARRAY_SLICE with a '?' index is not supported by the emulator; pass a literal "
            "or a column."
        )
    # A bound only known at run time needs the same adjustment. This names the bound twice,
    # which a bound column tolerates (sqlglot's own rendering of a start bound names it
    # three times).
    bound_sql = bound.sql(dialect=WRITE_DIALECT)
    return sqlglot.parse_one(
        f"{bound_sql} + CAST({bound_sql} >= 0 AS INTEGER)", read=WRITE_DIALECT
    )


def _rewrite_array_slice(node: exp.Expression) -> exp.Expression:
    """Rewrite ``ARRAY_SLICE`` onto DuckDB's 1-based, inclusive ``array_slice``.

    Applied to every slice, not just a JSON one: sqlglot's own rendering moves only the
    start bound, so a native list is off by one at the end as well
    (``ARRAY_SLICE(['a','b','c'], 0, 1)`` returned one element instead of two), while a
    JSON array silently became a slice of the JSON *text* (``'['``). A JSON array operand
    is cast to ``JSON[]`` so DuckDB binds its list function.
    """
    start, end = node.args.get("start"), node.args.get("end")
    if start is None or end is None:
        # Snowflake requires both bounds; without one, DuckDB would return NULL instead.
        raise TranslationError("ARRAY_SLICE requires both a start and an end index.")
    operand = _rewrite_json_array_functions(node.this)
    if _is_json_array_value(operand) and not _is_json_list(operand):
        operand = exp.cast(operand, exp.DataType.build("JSON[]"), dialect=WRITE_DIALECT)
    return exp.func(
        "array_slice",
        operand,
        _duckdb_slice_bound(start),
        _duckdb_slice_bound(end),
    )


def _rewrite_json_array_function(node: exp.Expression) -> exp.Expression:
    """Make an array function work over a Snowflake VARIANT/JSON array.

    ``ARRAY_CONTAINS(v, col:path)``, ``ARRAY_SIZE(col:path)`` and ``ARRAY_SLICE(col:path,
    from, to)`` reach DuckDB as calls over JSON, which its list functions reject: they take
    a LIST, while the emulator stores a semi-structured array as JSON. The array operand is
    therefore cast to ``JSON[]``; a searched value that is neither a VARIANT nor already
    JSON is encoded with ``TO_JSON`` so both sides compare as VARIANT values, the way
    Snowflake compares them; and slice bounds are moved from Snowflake's 0-based indexing
    to DuckDB's 1-based one. A bare VARIANT column (whose array-ness is only known at run
    time) is left alone: DuckDB raises, where casting it could silently compare unequal
    values.

    Called only once the operands have had their own rewrites applied (see
    ``_rewrite_json_array_functions``), because the replacement is built from their SQL.
    """
    if isinstance(node, exp.ArraySlice):
        return _rewrite_array_slice(node)
    if not _is_json_array_value(node.this):
        return node
    operand = _rewrite_json_array_functions(node.this)
    array = (
        operand
        if _is_json_list(operand)
        else exp.cast(operand, exp.DataType.build("JSON[]"), dialect=WRITE_DIALECT)
    )
    if isinstance(node, exp.ArraySize):
        node.set("this", array)
        return node

    value = node.args.get("expression")
    if value is not None and not (_is_json_value(value) or _is_variant_value(value)):
        value = exp.func("to_json", value)
    # sqlglot renders this as `CASE WHEN value IS NULL THEN ... ELSE ARRAY_CONTAINS(...)`,
    # which evaluates the value expression twice -- a bound `?` then needs two values.
    # This form is equivalent (checked against that CASE for a found/absent value, a NULL
    # value, and an array holding NULL) while naming the value once.
    array_sql = array.sql(dialect=WRITE_DIALECT)
    value_sql = value.sql(dialect=WRITE_DIALECT) if value is not None else "NULL"
    return sqlglot.parse_one(
        f"COALESCE(ARRAY_CONTAINS({array_sql}, {value_sql}), "
        f"NULLIF(ARRAY_LENGTH({array_sql}) <> LIST_COUNT({array_sql}), FALSE))",
        read=WRITE_DIALECT,
    )


def _rewrite_json_array_functions(statement: exp.Expression) -> exp.Expression:
    """Apply ``_rewrite_json_array_function`` to every array call in `statement`.

    A separate pass over a snapshot, because ``transform`` visits a parent before its
    children: rebuilding a call from its operands' SQL requires those operands to have
    been rewritten first, so an operand that is itself a rewritten call is handled
    recursively (``_rewrite_json_array_function``). A node detached by an ancestor's
    replacement is skipped.
    """
    # Snowflake documents SLICE as an alias of ARRAY_SLICE, but sqlglot leaves it as an
    # uninterpreted call that DuckDB rejects. Nested calls are converted by the loop too:
    # a converted call holds the live child nodes, so an inner SLICE is still in the tree.
    for alias in list(statement.find_all(exp.Anonymous)):
        if alias.name.upper() == "SLICE" and len(alias.expressions) == 3:
            array, start, end = alias.expressions
            alias.replace(exp.ArraySlice(this=array, start=start, end=end, zero_based=True))

    root = statement
    for node in list(statement.find_all(*_JSON_ARRAY_FUNCTIONS)):
        if node.parent is None and node is not root:
            continue  # already replaced as part of an enclosing call
        replacement = _rewrite_json_array_function(node)
        if replacement is node:
            continue
        if node is not root:
            node.replace(replacement)
        elif root.parent is not None:
            node.replace(replacement)  # keep the enclosing node's operand consistent
            root = replacement
        else:
            root = replacement
    return root


def _rewrite_snowflake_specifics(node: exp.Expression) -> exp.Expression:
    """Rewrite Snowflake nodes that sqlglot can't fully transpile to DuckDB."""
    # ``CURRENT_VERSION()`` -> a Snowflake-style dotted-numeric version literal.
    if isinstance(node, exp.CurrentVersion):
        return exp.Literal.string(EMULATED_SNOWFLAKE_VERSION)

    # Snowflake JSON path extraction (``col:path``, ``GET_PATH``) transpiles to
    # DuckDB's ``->``/``->>``, which parses its operand as JSON on the fly. A VALUE
    # bound from a Python dict is stored by DuckDB in a Python-repr form (e.g.
    # ``{'a': 1}``) that is not valid JSON, so the extraction raises. Casting the
    # operand to JSON first makes it round-trip through real JSON instead of the
    # VARIANT/VARCHAR repr. ``exp.cast`` skips an operand that is already cast to JSON.
    if isinstance(node, (exp.JSONExtract, exp.JSONExtractScalar)):
        if not isinstance(node.this, exp.ParseJSON):
            node.set(
                "this",
                exp.cast(node.this, exp.DataType.build("JSON"), dialect=WRITE_DIALECT),
            )
        return node

    # ``LATERAL FLATTEN(input => X)`` -> DuckDB json_each-based lateral subquery.
    if (
        isinstance(node, exp.Lateral)
        and isinstance(node.this, exp.Explode)
        and isinstance(node.this.this, exp.Kwarg)
    ):
        key = node.this.this.args.get("this")
        if getattr(key, "name", None) and key.name.upper() == "INPUT":
            new = node.copy()
            new.set("this", _flatten_subquery(node.this.this.expression))
            alias = new.args.get("alias")
            if alias is not None:
                alias.set("columns", [])
            return new

    # ``TABLE(FLATTEN(input => X))`` -> the same subquery, unwrapped from TABLE(...).
    if (
        isinstance(node, exp.TableFromRows)
        and isinstance(node.this, exp.Explode)
        and isinstance(node.this.this, exp.Kwarg)
    ):
        key = node.this.this.args.get("this")
        if getattr(key, "name", None) and key.name.upper() == "INPUT":
            sub = _flatten_subquery(node.this.this.expression)
            sub.set("alias", node.args.get("alias"))
            return sub

    # Snowflake sequence pseudo-column ``seq.NEXTVAL`` -> ``nextval('seq')``.
    if (
        isinstance(node, exp.Column)
        and node.name
        and node.name.upper() == "NEXTVAL"
        and node.args.get("table") is not None
    ):
        parts = [
            node.args.get("catalog"),
            node.args.get("db"),
            node.args.get("table"),
        ]
        seq_name = ".".join(_quoted_identifier(p.name) for p in parts if p is not None)
        return exp.func("nextval", exp.Literal.string(seq_name))

    # Snowflake semi-structured column types. VARIANT is already accepted by DuckDB;
    # bare OBJECT/ARRAY have no DuckDB DDL equivalent, so map them to JSON. An ARRAY
    # with an element type (e.g. VARCHAR[]) already transpiles to DuckDB's ``TEXT[]``.
    if isinstance(node, exp.DataType):
        if node.this == exp.DataType.Type.OBJECT:
            return exp.DataType.build("JSON")
        if node.this == exp.DataType.Type.ARRAY and not node.expressions:
            return exp.DataType.build("JSON")
        # Snowflake's TIMESTAMP_TZ(p)/TIMESTAMP_LTZ(p) accept a precision parameter,
        # but DuckDB's ``TIMESTAMP WITH [LOCAL] TIME ZONE`` takes none -- sqlglot
        # renders ``TIMESTAMP_LTZ(3)`` as ``TIMESTAMPTZ(3)``, which DuckDB rejects.
        # Drop the modifier so the DDL/CAST is accepted; DuckDB stores microsecond
        # precision either way (reported honestly via the timestamp scale mapping).
        if (
            node.this
            in (exp.DataType.Type.TIMESTAMPTZ, exp.DataType.Type.TIMESTAMPLTZ)
            and node.expressions
        ):
            stripped = node.copy()
            stripped.set("expressions", [])
            return stripped

    # ``MERGE ... UPDATE SET t.col = ...`` -> unqualified target column. DuckDB rejects
    # the target-table qualifier on the left side of a SET assignment.
    if isinstance(node, exp.Update):
        for eq in node.expressions:
            if isinstance(eq, exp.EQ) and isinstance(eq.this, exp.Column):
                for key in ("table", "db", "catalog"):
                    eq.this.args.pop(key, None)
        return node

    return node


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


def _reject_unsupported_function(statement: exp.Expression) -> None:
    """Raise for ``CREATE FUNCTION`` statements the emulator can't faithfully emulate.

    DuckDB has no JavaScript/Python/Java/Scala runtime, so a Snowflake UDF with a
    non-SQL ``LANGUAGE`` clause would be silently translated into a broken function
    (the body becomes a string literal). Reject it loudly instead.
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


def transpile_to_duckdb(statement: exp.Expression) -> str:
    """Transpile a single parsed Snowflake statement into its DuckDB equivalent."""
    try:
        _reject_unsupported_function(statement)
        statement = _rewrite_clone(statement)
        statement = statement.transform(_uppercase_unquoted_identifiers)
        statement = statement.transform(_rewrite_information_schema)
        statement = statement.transform(_rewrite_snowflake_specifics)
        # After the rewrites above: an array call is rebuilt from its operands' SQL, so
        # those operands must already be in their final DuckDB form.
        statement = _rewrite_json_array_functions(statement)

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
            if pre_statements:
                parts = [seq.sql(dialect=WRITE_DIALECT) for seq in pre_statements]
                parts.append(create.sql(dialect=WRITE_DIALECT))
                return ";\n".join(parts)
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
