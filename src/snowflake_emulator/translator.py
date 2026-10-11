"""Translate Snowflake SQL into DuckDB-compatible SQL using sqlglot."""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.tokens import TokenType

READ_DIALECT = "snowflake"
WRITE_DIALECT = "duckdb"

# ``ARRAY_FLATTEN`` is renamed before parsing: sqlglot's node for it accepts one argument,
# so the documented two-argument Snowflake form fails to parse outright.
_FLATTEN_MARKER = "SFEMU_ARRAY_FLATTEN"
_ARRAY_FLATTEN_NAME = "ARRAY_FLATTEN"
# Each level of ARRAY_FLATTEN is a separate DuckDB call, so a deep flatten is bounded.
_ARRAY_FLATTEN_MAX_DEPTH = 10

# Snowflake array functions this module rewrites. sqlglot renders these faithfully for a
# DuckDB LIST, so a JSON array operand mainly needs the cast to ``JSON[]`` -- see
# ``_rewrite_json_array_function``, which also rebuilds the ones sqlglot gets wrong.
_JSON_ARRAY_FUNCTIONS = (
    exp.ArrayAgg,
    exp.ArrayAppend,
    exp.ArrayCompact,
    exp.ArrayConcat,
    exp.ArrayContains,
    exp.ArrayDistinct,
    exp.ArrayExcept,
    exp.ArrayFilter,
    exp.ArrayInsert,
    exp.ArrayIntersect,
    exp.ArrayMax,
    exp.ArrayMin,
    exp.ArrayOverlaps,
    exp.ArrayPosition,
    exp.ArrayPrepend,
    exp.ArrayRemove,
    exp.ArrayRemoveAt,
    exp.ArraySize,
    exp.ArraySlice,
    exp.ArrayToString,
    exp.Reduce,
    exp.SortArray,
    exp.Transform,
)
# Calls sqlglot leaves uninterpreted, or maps onto a DuckDB function with different
# semantics, so they are recognised by name.
_ARRAY_CALL_NAMES = ("ARRAY_UNIQUE", "ARRAY_REPEAT", _FLATTEN_MARKER)
# Calls whose second operand is a value searched for in, or inserted into, the array.
_ARRAY_VALUE_FUNCTIONS = (exp.ArrayAppend, exp.ArrayInsert, exp.ArrayRemove)

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


def preprocess_array_flatten(sql: str) -> str:
    """Rename ``ARRAY_FLATTEN(x, depth)`` so sqlglot's one-argument limit accepts it.

    sqlglot's ``ArrayFlatten`` node takes a single argument, so the documented
    two-argument Snowflake form fails to parse before any rewrite can run. Only the call's
    name is renamed, located by token position so a literal that mentions it is untouched;
    the translator then rewrites the renamed call (``_rewrite_array_flatten``).
    """
    tokens = sqlglot.Dialect.get_or_raise(READ_DIALECT).tokenize(sql)
    spans = [
        (token.start, token.end)
        for index, token in enumerate(tokens[:-1])
        if token.text.upper() == _ARRAY_FLATTEN_NAME
        and tokens[index + 1].token_type == TokenType.L_PAREN
    ]
    if not spans:
        return sql
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(sql[cursor:start])
        parts.append(_FLATTEN_MARKER)
        cursor = end + 1  # Token.end is the index of the token's last character
    parts.append(sql[cursor:])
    return "".join(parts)


def split_statements(sql: str) -> list[exp.Expression]:
    """Split a (possibly multi-statement) Snowflake SQL string into parsed expressions.

    Returning parsed `exp.Expression` objects (rather than re-serialized SQL text)
    avoids a lossy parse -> Snowflake-text -> re-parse -> DuckDB-text round trip,
    which can drop information that doesn't survive an intermediate Snowflake-dialect
    print (e.g. `VARCHAR[]` collapsing to a bare `ARRAY`).
    """
    try:
        expressions = sqlglot.parse(preprocess_array_flatten(sql), read=READ_DIALECT)
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

    The operand is rewritten first and cast to JSON: a VARIANT column is not JSON to
    DuckDB (``json_each`` rejects it), and this pass runs before the operand's own
    ``col:path`` rewrite, so the cast has to be applied here.
    """
    operand = exp.cast(
        input_expr.transform(_rewrite_snowflake_specifics),
        exp.DataType.build("JSON"),
        dialect=WRITE_DIALECT,
    )
    x = operand.sql(dialect=WRITE_DIALECT)
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


def _json_list(node: exp.Expression | None) -> exp.Expression | None:
    """Present a Snowflake array to DuckDB as a LIST: a JSON value becomes ``JSON[]``.

    Anything else (a native DuckDB list, or a bare column whose type is only known at run
    time) is returned unchanged.
    """
    if node is None or _is_json_list(node) or not _is_json_value(node):
        return node
    return exp.cast(node, exp.DataType.build("JSON[]"), dialect=WRITE_DIALECT)


def _json_element(node: exp.Expression | None) -> exp.Expression | None:
    """Encode a searched value so it compares as a VARIANT against ``JSON[]`` elements."""
    if node is None or _is_json_value(node) or _is_variant_value(node):
        return node
    return exp.func("to_json", node)


def _sql(node: exp.Expression | None, default: str = "NULL") -> str:
    """Serialize an operand of a call this module rebuilds from SQL text."""
    return node.sql(dialect=WRITE_DIALECT) if node is not None else default


def _array_call_name(node: exp.Expression) -> str | None:
    """The Snowflake name of a call sqlglot left uninterpreted, when it is one we rewrite."""
    if isinstance(node, exp.Anonymous) and isinstance(node.this, str):
        name = node.this.upper()
        return name if name in _ARRAY_CALL_NAMES else None
    return None


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


def _rewrite_array_contains(node: exp.Expression) -> exp.Expression:
    """``ARRAY_CONTAINS(v, col:path)``: compare as VARIANT values, naming the value once.

    sqlglot renders this as ``CASE WHEN value IS NULL THEN ... ELSE ARRAY_CONTAINS(...)``,
    which evaluates the value expression twice -- a bound `?` then needs two values. The
    form below is equivalent (checked against that CASE for a found/absent value, a NULL
    value, and an array holding NULL) while naming the value once.
    """
    array = _json_list(_rewrite_json_array_functions(node.this))
    value = _rewrite_json_array_functions(node.args.get("expression"))
    if _is_json_array_value(array):
        # Only a JSON element needs encoding: against a native list the value keeps its type.
        value = _json_element(value)
    array_sql = _sql(array)
    return sqlglot.parse_one(
        f"COALESCE(ARRAY_CONTAINS({array_sql}, {_sql(value)}), "
        f"NULLIF(ARRAY_LENGTH({array_sql}) <> LIST_COUNT({array_sql}), FALSE))",
        read=WRITE_DIALECT,
    )


def _rewrite_array_size(node: exp.Expression) -> exp.Expression:
    """``ARRAY_SIZE(col:path)``: DuckDB needs a LIST, not the JSON the array is stored as."""
    node.set("this", _json_list(_rewrite_json_array_functions(node.this)))
    return node


def _rewrite_array_position(node: exp.Expression) -> exp.Expression:
    """``ARRAY_POSITION`` is 0-based, and 0 rather than NULL when the value is absent."""
    array = _rewrite_json_array_functions(node.this)
    value = _rewrite_json_array_functions(node.args.get("expression"))
    if _is_json_array_value(array):
        value = _json_element(value)
    return sqlglot.parse_one(
        f"COALESCE(ARRAY_POSITION({_sql(_json_list(array))}, {_sql(value)}) - 1, 0)",
        read=WRITE_DIALECT,
    )


def _rewrite_array_prepend(node: exp.Expression) -> exp.Expression:
    """sqlglot's ``LIST_PREPEND`` does not bind in DuckDB; build it from ``LIST_CONCAT``."""
    array = _rewrite_json_array_functions(node.args.get("expression"))
    value = _rewrite_json_array_functions(node.this)
    if _is_json_array_value(array):
        value = _json_element(value)
    return sqlglot.parse_one(
        f"LIST_CONCAT([{_sql(value)}], {_sql(_json_list(array))})", read=WRITE_DIALECT
    )


def _rewrite_array_unique(node: exp.Expression) -> exp.Expression:
    """DuckDB's ``array_unique`` returns a *count*; Snowflake's returns the distinct array."""
    if len(node.expressions) != 1:
        raise TranslationError("ARRAY_UNIQUE takes a single array argument.")
    array = _json_list(_rewrite_json_array_functions(node.expressions[0]))
    return exp.func("list_distinct", array)


def _rewrite_array_set_operation(node: exp.Expression) -> exp.Expression:
    """``ARRAY_EXCEPT``/``ARRAYS_OVERLAP``: sqlglot's rendering does not bind a JSON[].

    A native pair keeps sqlglot's rendering (and with it the multiset semantics it models);
    only a JSON operand is rebuilt, where DuckDB has no equivalent for the ``&&`` operator
    or the zip-based except.
    """
    left = _json_list(_rewrite_json_array_functions(node.this))
    right = _json_list(_rewrite_json_array_functions(node.args.get("expression")))
    if not (_is_json_list(left) or _is_json_list(right)):
        return node
    if isinstance(node, exp.ArrayOverlaps):
        return exp.func("list_has_any", left, right)
    return sqlglot.parse_one(
        f"LIST_FILTER({_sql(left)}, _sfemu_element -> "
        f"NOT LIST_CONTAINS({_sql(right)}, _sfemu_element))",
        read=WRITE_DIALECT,
    )


def _rewrite_array_repeat(node: exp.Expression) -> exp.Expression:
    """``ARRAY_REPEAT`` has no DuckDB function; build it from ``RANGE`` and a transform."""
    if len(node.expressions) != 2:
        raise TranslationError("ARRAY_REPEAT takes a value and a count.")
    value, count = (_rewrite_json_array_functions(argument) for argument in node.expressions)
    if _is_json_value(count) or _is_variant_value(count):
        # A count read out of a VARIANT (``j:count``) needs to be a number for RANGE.
        count = exp.cast(count, exp.DataType.build("INTEGER"), dialect=WRITE_DIALECT)
    return sqlglot.parse_one(
        f"LIST_TRANSFORM(RANGE(1, {_sql(count)} + 1), _sfemu_repeat -> {_sql(value)})",
        read=WRITE_DIALECT,
    )


def _rewrite_array_flatten(node: exp.Expression) -> exp.Expression:
    """``ARRAY_FLATTEN(arr, depth)`` -> one DuckDB ``FLATTEN`` per level.

    DuckDB cannot flatten a ``JSON[]`` whose elements are JSON arrays, so each level casts
    the elements back to a list. A level whose elements are no longer arrays is left alone
    for a JSON array (DuckDB's ``FLATTEN`` would raise, where Snowflake stops flattening);
    a native list keeps DuckDB's own behaviour.
    """
    if len(node.expressions) != 2:
        raise TranslationError("ARRAY_FLATTEN takes an array and a depth.")
    array, depth = node.expressions
    if not (isinstance(depth, exp.Literal) and not depth.is_string and depth.name.isdigit()):
        raise TranslationError("ARRAY_FLATTEN needs a constant depth.")
    levels = int(depth.name)
    if not 1 <= levels <= _ARRAY_FLATTEN_MAX_DEPTH:
        raise TranslationError(
            f"ARRAY_FLATTEN depth must be between 1 and {_ARRAY_FLATTEN_MAX_DEPTH}."
        )
    array = _json_list(_rewrite_json_array_functions(array))
    sql = _sql(array)
    for _ in range(levels):
        if not _is_json_list(array):
            sql = f"FLATTEN({sql})"
            continue
        # Only flatten a level whose elements are still arrays: casting a scalar element to
        # a list would raise, where Snowflake simply stops flattening (the depth is past the
        # nesting).
        all_arrays = (
            f"LIST_COUNT(LIST_FILTER({sql}, _sfemu_outer -> "
            f"json_type(_sfemu_outer) = 'ARRAY')) = LIST_COUNT({sql})"
        )
        flattened = (
            f"FLATTEN(LIST_TRANSFORM({sql}, _sfemu_inner -> CAST(_sfemu_inner AS JSON[])))"
        )
        sql = f"CASE WHEN {all_arrays} THEN {flattened} ELSE {sql} END"
    return sqlglot.parse_one(sql, read=WRITE_DIALECT)


def _rewrite_array_to_string(node: exp.Expression) -> exp.Expression:
    """sqlglot casts each element with ``CAST(x AS TEXT)``, keeping a JSON string's quotes."""
    array = _rewrite_json_array_functions(node.this)
    known = {"this", "expression", "null_is_empty", "null_delim_is_null"}
    if not _is_json_array_value(array) or set(node.args) - known:
        return node
    return sqlglot.parse_one(
        f"ARRAY_TO_STRING(LIST_TRANSFORM({_sql(_json_list(array))}, "
        f"_sfemu_element -> COALESCE(_sfemu_element ->> '$', '')), "
        f"{_sql(node.args.get('expression'), chr(39) + ',' + chr(39))})",
        read=WRITE_DIALECT,
    )


def _cast_array_operands(node: exp.Expression) -> exp.Expression:
    """Cast the array operand(s) of a call sqlglot otherwise renders faithfully.

    A searched value is encoded with ``TO_JSON`` when the array it is compared against is
    JSON, so the comparison happens between VARIANT values, as Snowflake does.
    """
    array_is_json = _is_json_array_value(node.this)
    if isinstance(node.this, exp.Expression):
        node.set("this", _json_list(_rewrite_json_array_functions(node.this)))
    if isinstance(node, _ARRAY_VALUE_FUNCTIONS):
        value = node.args.get("expression")
        if isinstance(value, exp.Expression):
            encoded = _json_element(_rewrite_json_array_functions(value))
            node.set("expression", encoded if array_is_json else value)
    if node.args.get("expressions"):
        node.set(
            "expressions",
            [
                _json_list(_rewrite_json_array_functions(e))
                if isinstance(e, exp.Expression)
                else e
                for e in node.args["expressions"]
            ],
        )
    return node


def _rewrite_json_array_function(node: exp.Expression) -> exp.Expression:
    """Make a Snowflake array function work over a VARIANT/JSON array.

    A VARIANT array (``col:path``, ``$1:path``, ``PARSE_JSON('[...]')``, ``x::VARIANT``) is
    stored as DuckDB JSON, which its list functions reject: they take a LIST. The array
    operand is therefore cast to ``JSON[]``, a searched value is encoded with ``TO_JSON`` so
    both sides compare as VARIANT values the way Snowflake compares them, and the functions
    whose sqlglot mapping is wrong for *any* operand (indexing, ``SLICE``, ``ARRAY_UNIQUE``,
    ``ARRAY_REPEAT``, ``ARRAY_FLATTEN``) are rebuilt. A bare VARIANT column (whose array-ness
    is only known at run time) is left alone: DuckDB raises, where casting it could silently
    compare unequal values.

    Called only once the operands have had their own rewrites applied (see
    ``_rewrite_json_array_functions``), because the replacement is built from their SQL.
    """
    if isinstance(node, exp.ArraySlice):
        return _rewrite_array_slice(node)
    if isinstance(node, exp.ArrayPosition):
        return _rewrite_array_position(node)
    if isinstance(node, exp.ArrayPrepend):
        return _rewrite_array_prepend(node)
    if isinstance(node, exp.ArrayToString):
        return _rewrite_array_to_string(node)
    if isinstance(node, (exp.ArrayExcept, exp.ArrayOverlaps)):
        return _rewrite_array_set_operation(node)
    name = _array_call_name(node)
    if name == "ARRAY_UNIQUE":
        return _rewrite_array_unique(node)
    if name == "ARRAY_REPEAT":
        return _rewrite_array_repeat(node)
    if name == _FLATTEN_MARKER:
        return _rewrite_array_flatten(node)
    if isinstance(node, exp.ArrayContains):
        return _rewrite_array_contains(node)
    if isinstance(node, exp.ArraySize):
        return _rewrite_array_size(node)
    if isinstance(node, _JSON_ARRAY_FUNCTIONS):
        return _cast_array_operands(node)
    # Anything else -- including an uninterpreted call such as a Snowflake function sqlglot
    # does not know -- is left for DuckDB, whose error names it.
    return node


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
    for node in list(statement.find_all(*_JSON_ARRAY_FUNCTIONS, exp.Anonymous)):
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
