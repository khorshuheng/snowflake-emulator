"""Semi-structured and Snowflake-specific AST rewrites."""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from snowflake_emulator.core.constants import INFORMATION_SCHEMA_SCHEMA
from snowflake_emulator.translator.arrays import _is_json_value, _is_variant_value
from snowflake_emulator.translator.ast_helpers import _quoted_identifier
from snowflake_emulator.translator import column_types
from snowflake_emulator.translator.constants import (
    EMULATED_SNOWFLAKE_VERSION,
    WRITE_DIALECT,
    _SIMPLE_IDENTIFIER_RE,
)
from snowflake_emulator.translator.errors import TranslationError


def _flatten_kwargs(explode: exp.Explode) -> dict[str, exp.Expression]:
    """Return the named arguments of a FLATTEN call."""
    kwargs: dict[str, exp.Expression] = {}
    for item in explode.expressions or []:
        if isinstance(item, exp.Kwarg) and isinstance(item.this, exp.Var):
            kwargs[item.this.name.upper()] = item.expression
    return kwargs


def _flatten_bool(node: exp.Expression | None) -> bool:
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Literal) and node.is_string:
        return str(node.this).strip().upper() in ("TRUE", "1", "YES", "ON")
    return False


def _flatten_arguments(explode: exp.Explode) -> tuple[exp.Expression, bool, str | None]:
    """Return ``(input, outer, path)`` for a FLATTEN call, rejecting unsupported options."""
    kwargs = _flatten_kwargs(explode)
    if "RECURSIVE" in kwargs and _flatten_bool(kwargs["RECURSIVE"]):
        raise TranslationError(
            "FLATTEN recursive => TRUE is not supported by the emulator."
        )
    path: str | None = None
    path_node = kwargs.get("PATH")
    if path_node is not None:
        if not (isinstance(path_node, exp.Literal) and path_node.is_string):
            raise TranslationError("FLATTEN path must be a string literal.")
        path = str(path_node.this)
    return explode.this.expression, _flatten_bool(kwargs.get("OUTER")), path


def _flatten_subquery(
    input_expr: exp.Expression, *, outer: bool = False, path: str | None = None
) -> exp.Subquery:
    """Build the DuckDB subquery equivalent of Snowflake's ``FLATTEN(input => X)``.

    Snowflake's FLATTEN exposes SEQ/KEY/PATH/INDEX/VALUE/THIS. DuckDB's
    ``json_each()`` exposes key/value/fullkey/...; we map those onto Snowflake's
    names. INDEX is approximated as the array index (``TRY_CAST(key AS BIGINT)``),
    which is NULL for object keys -- matching Snowflake's object behaviour.

    The operand is rewritten first and cast to JSON: a VARIANT column is not JSON to
    DuckDB (``json_each`` rejects it), and this pass runs before the operand's own
    ``col:path`` rewrite, so the cast has to be applied here.
    """
    operand = exp.cast(
        input_expr.transform(_rewrite_snowflake_specifics),
        exp.DataType.build("JSON"),
        dialect=WRITE_DIALECT,
    )
    if path is not None:
        operand = exp.func("json_extract", operand, exp.Literal.string(f"$.{path}"))
    x = operand.sql(dialect=WRITE_DIALECT)
    rows = (
        f"SELECT ROW_NUMBER() OVER () - 1 AS seq, key, fullkey AS path, "
        f"TRY_CAST(key AS BIGINT) AS index, value, {x} AS this "
        f"FROM json_each({x})"
    )
    if outer:
        rows = (
            f"({rows}) UNION ALL "
            "SELECT NULL::BIGINT AS seq, NULL::VARCHAR AS key, NULL::VARCHAR AS path, "
            f"NULL::BIGINT AS index, NULL::JSON AS value, {x} AS this "
            f"WHERE NOT EXISTS (SELECT 1 FROM json_each({x}))"
        )
    select = sqlglot.parse_one(rows, read=WRITE_DIALECT)
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




# Types Snowflake casts a semi-structured value to for its text form.
_TEXT_TYPES = (
    exp.DataType.Type.VARCHAR,
    exp.DataType.Type.CHAR,
    exp.DataType.Type.NCHAR,
    exp.DataType.Type.NVARCHAR,
    exp.DataType.Type.TEXT,
)


def _is_text_type(data_type: exp.Expression | None) -> bool:
    """True for a text type, with or without a length (``VARCHAR``, ``VARCHAR(10)``)."""
    return isinstance(data_type, exp.DataType) and data_type.this in _TEXT_TYPES


def _json_root_extract(operand: exp.Expression) -> exp.Expression:
    """``operand ->> '$'``: the text of a JSON value (a string without its quotes)."""
    return exp.JSONExtractScalar(
        this=operand, expression=exp.JSONPath(expressions=[exp.JSONPathRoot()])
    )


def _rewrite_semi_structured_text_cast(node: exp.Expression) -> exp.Expression:
    """Make ``col:path::VARCHAR`` return the value's text, not DuckDB's JSON serialization.

    ``col:path`` evaluates to a DuckDB JSON value, and ``CAST(<json> AS VARCHAR)`` serializes
    it -- a JSON string keeps its quotes (``'"y"'`` where Snowflake returns ``y``).
    ``->> '$'`` yields the text of a JSON scalar instead, and of a JSON object/array the
    document text, as Snowflake does. An explicit VARIANT is serialized to JSON first for
    the same reason; a bare VARIANT column is left alone, since DuckDB's own VARIANT cast
    already returns a string variant's text (whether a bare column holds one is not known
    until run time).

    The operand is rewritten here rather than serialized: replacing a node stops the walk
    from descending into its children, and a ``col:path`` operand is not yet a JSON value.
    """
    operand = node.this
    if not (
        _is_json_value(operand)
        or _is_variant_value(operand)
        or column_types.is_semi_structured_typed(operand)
    ):
        return node
    inner = operand.transform(_rewrite_snowflake_specifics)
    if _is_variant_value(inner) or column_types.is_variant_typed(inner):
        # ``CAST(<variant> AS JSON)`` is required: DuckDB's ``->>`` on a VARIANT casts it to
        # VARCHAR first, which yields its struct repr (``{'names': [X]}``) and then fails to
        # parse as JSON.
        inner = exp.cast(inner, exp.DataType.build("JSON"), dialect=WRITE_DIALECT)
    extracted = _json_root_extract(inner)
    if isinstance(node, exp.ToChar):
        return extracted  # its DuckDB rendering is already a text cast
    return exp.Cast(this=extracted, to=node.to)



# Snowflake coerces a VARIANT/JSON argument to its JSON *text* in these functions, where DuckDB
# would hand the function a VARIANT (rendered as a struct repr) or a JSON value (rendered with
# quotes). Each entry names the arguments the function treats as text.
_TEXT_ARGUMENTS: dict[type, tuple[str, ...]] = {
    exp.Upper: ("this",),
    exp.Lower: ("this",),
    exp.Initcap: ("this",),
    exp.Reverse: ("this",),
    exp.Length: ("this",),
    exp.Trim: ("this",),
    exp.Substring: ("this",),
    exp.Left: ("this",),
    exp.Right: ("this",),
    exp.Replace: ("this",),
    exp.SplitPart: ("this",),
    exp.Split: ("this",),
    exp.Contains: ("this",),
    exp.Pad: ("this",),
    exp.StrPosition: ("substr",),
    exp.Like: ("this",),
    exp.ILike: ("this",),
    exp.DPipe: ("this", "expression"),
    exp.Concat: ("expressions",),
}


def _select_arms(relation: exp.Expression) -> list[exp.Select]:
    """The SELECTs whose columns a derived relation exposes (both sides of a UNION)."""
    if isinstance(relation, exp.Select):
        return [relation]
    if isinstance(relation, exp.Union):
        arms: list[exp.Select] = []
        pending: list[exp.Expression] = [relation]
        while pending:
            node = pending.pop()
            if isinstance(node, exp.Union):
                pending.extend((node.this, node.expression))
            elif isinstance(node, exp.Select):
                arms.append(node)
        return arms
    return []


def _projection_item(select: exp.Select, name: str) -> exp.Expression | None:
    for item in select.expressions:
        if item.alias_or_name and item.alias_or_name.upper() == name.upper():
            return item
    return None


def _normalize_derived_union(
    node: exp.Column, seen: frozenset[tuple[int, str]] = frozenset()
) -> None:
    """Make a derived UNION's mixed arms comparable as JSON when its column is used as text.

    A UNION of text and JSON is resolved by DuckDB to JSON, and DuckDB then parses the text
    arm (``"abc"`` -> ``Malformed JSON``). Snowflake keeps the text as a value, so the non-JSON
    arms are wrapped in ``TO_JSON``. This is applied only when the column is consumed as text,
    and the cast marker makes it idempotent.
    """
    relation = column_types.derived_relation(node)
    if relation is None:
        return
    key = (id(relation), node.name.upper())
    if key in seen:
        return
    seen = seen | {key}
    for select in _select_arms(relation):
        item = _projection_item(select, node.name)
        if item is None:
            continue
        value = item.this if isinstance(item, exp.Alias) else item
        if isinstance(value, exp.Column) and column_types.derived_relation(value) is not None:
            _normalize_derived_union(value, seen)
            continue
        if _is_semi_structured_value(value):
            continue
        wrapped = exp.cast(
            exp.func("to_json", value), exp.DataType.build("JSON"), dialect=WRITE_DIALECT
        )
        if isinstance(item, exp.Alias):
            item.set("this", wrapped)
        else:
            select.set(
                "expressions",
                [wrapped if expression is item else expression for expression in select.expressions],
            )


def _text_operand(node: exp.Expression | None) -> exp.Expression | None:
    """Present a semi-structured value as the JSON text Snowflake coerces it to."""
    if node is None:
        return node
    if isinstance(node, exp.Column):
        _normalize_derived_union(node)
    if column_types.is_variant_typed(node):
        # ``->>`` on a VARIANT casts to VARCHAR first (its struct repr), so reach real JSON.
        node = exp.cast(node, exp.DataType.build("JSON"), dialect=WRITE_DIALECT)
    if column_types.is_json_typed(node) or _is_json_value(node):
        return _json_root_extract(node)
    return node


def _rewrite_variant_text_function(node: exp.Expression) -> exp.Expression:
    """Coerce a semi-structured argument of a string function to its JSON text.

    ``UPPER(variant_col)`` has to uppercase the JSON text; DuckDB would uppercase (or reverse,
    measure, split, compare) its VARIANT repr instead.
    """
    keys = _TEXT_ARGUMENTS.get(type(node))
    if keys is None:
        return node
    for key in keys:
        value = node.args.get(key)
        if isinstance(value, list):
            rewritten = [_text_operand(item) for item in value]
            if any(new is not old for new, old in zip(rewritten, value)):
                node.set(key, rewritten)
        elif isinstance(value, exp.Expression):
            rewritten = _text_operand(value)
            if rewritten is not value:
                node.set(key, rewritten)
    return node



_TEXT_COLUMN_TYPES = ("VARCHAR", "CHAR", "TEXT", "STRING")
_COMPARISON_NODES = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def _is_semi_structured_value(node: exp.Expression | None) -> bool:
    """True for a value Snowflake stores as JSON/VARIANT (a path, PARSE_JSON, such a column)."""
    return node is not None and (
        _is_json_value(node) or column_types.is_semi_structured_typed(node)
    )


def _is_text_column(declared: str | None) -> bool:
    """True for a DuckDB column type whose Snowflake column holds text."""
    return declared is not None and declared.strip().upper().startswith(_TEXT_COLUMN_TYPES)


def _target_column_type(table: exp.Table, column_name: str) -> str | None:
    """The DuckDB type of ``table.column_name``, when the catalog is in scope."""
    column = exp.column(
        column_name,
        table=table.name,
        db=table.text("db") or None,
        catalog=table.text("catalog") or None,
    )
    return column_types.declared_column_type(column)


def _json_value_of_text(node: exp.Expression) -> exp.Expression:
    """Wrap a text value as JSON, the side Snowflake coerces when comparing with a VARIANT."""
    return exp.func("to_json", node)


def _rewrite_semi_structured_membership(node: exp.Expression) -> exp.Expression:
    """``<text column> IN (<semi-structured>, ...)``: make the text side a JSON string.

    An ``IN`` list compares like ``=``, so the same coercion applies: the text side becomes a
    JSON string and each semi-structured element compares as JSON.
    """
    values = node.expressions or []
    if not any(_is_semi_structured_value(value) for value in values):
        return node
    # ``IN`` is equality-only, so every side compares as JSON: a text column, a string literal
    # and a number all become JSON values, and a mixed list keeps working.
    if not _is_semi_structured_value(node.this):
        node.set("this", _json_value_of_text(node.this))
    node.set(
        "expressions",
        [
            value
            if _is_semi_structured_value(value)
            else _json_value_of_text(value)
            for value in values
        ],
    )
    return node


def _rewrite_semi_structured_comparison(node: exp.Expression) -> exp.Expression:
    """``<text column> = <staged JSON value>``: make the text side a JSON string.

    Snowflake coerces the VARCHAR operand to a VARIANT, so ``u.name = s.$1:name`` matches when
    the JSON holds the same string. DuckDB instead tries to parse ``'ada'`` as JSON and raises
    ``Malformed JSON``. A JSON string and the number it spells stay distinct either way.
    """
    if isinstance(node.parent, exp.Update) and node in (node.parent.expressions or []):
        # ``UPDATE t SET col = ...`` is an EQ *assignment*: the write pass coerces its value to
        # the target column's type, and the column itself must not be rewritten. A WHERE
        # comparison is a separate EQ hanging off the Update's ``where`` argument.
        return node
    left, right = node.this, node.args.get("expression")
    # Equality compares JSON *values*, so wrapping the other side is safe whatever it holds --
    # a text column, a string literal, or a text-valued expression such as ``UPPER(name)``.
    # A range comparison keeps the narrow rule: wrapping a number would compare JSON text.
    equality = isinstance(node, (exp.EQ, exp.NEQ))
    if _is_semi_structured_value(left) and (equality or _is_text_expression(right)):
        node.set("expression", _json_value_of_text(right))
    elif _is_semi_structured_value(right) and (equality or _is_text_expression(left)):
        node.set("this", _json_value_of_text(left))
    return node


def _is_text_expression(node: exp.Expression | None) -> bool:
    """True for a string literal or a text-typed column: the side Snowflake coerces to VARIANT."""
    if isinstance(node, exp.Literal):
        return node.is_string
    return _is_text_column(column_types.declared_column_type(node))


def _insert_target_columns(
    statement: exp.Insert,
) -> tuple[exp.Table | None, list[str]]:
    """The INSERT target table and its target column names in order, when they are known.

    Snowflake allows the column list to be omitted (the table's own order applies) and allows
    an alias before an explicit list (``INSERT INTO t AS x (a, b)``); sqlglot puts the latter
    under the table alias. Both forms need the same target-aware coercion as the plain
    ``INSERT INTO t (a, b)`` form.
    """
    target = statement.this
    if isinstance(target, exp.Schema):
        table = target.this
        columns = [
            item.name
            for item in target.expressions
            if isinstance(item, exp.Expression)
        ]
        return (table if isinstance(table, exp.Table) else None), columns
    if isinstance(target, exp.Table):
        alias = target.args.get("alias")
        columns: list[str] = []
        if isinstance(alias, exp.TableAlias):
            columns = [
                item.name
                for item in (alias.args.get("columns") or [])
                if isinstance(item, exp.Expression)
            ]
        if not columns:
            columns = column_types.declared_table_columns(target) or []
        return target, columns
    return None, []


def _expand_derived_star(select: exp.Select, columns: list[str]) -> bool:
    """Expand ``SELECT *`` over a single derived relation so each target column is addressable.

    The raw staged relation exposes its record as ``$1`` (and CSV/Parquet positional ``$n``
    columns); ``SELECT *`` hides that name from the write coercion. Expanding the projection
    to the relation's own column names keeps the same rows while letting the target types drive
    the coercion.
    """
    if len(select.expressions) != 1 or not isinstance(select.expressions[0], exp.Star):
        return False
    if select.args.get("joins"):
        return False
    from_ = select.args.get("from_")
    source = from_.this if isinstance(from_, exp.From) else None
    if not isinstance(source, exp.Subquery):
        return False
    relation = source.this
    arms = _select_arms(relation)
    if not arms:
        return False
    names = [item.alias_or_name for item in arms[0].expressions]
    if not names or any(not name or name == "*" for name in names):
        return False
    if len(names) != len(columns):
        return False
    alias = source.alias or None
    select.set(
        "expressions",
        [
            exp.column(
                name,
                table=alias,
                quoted=not _SIMPLE_IDENTIFIER_RE.match(name),
            )
            for name in names
        ],
    )
    return True


def _coerce_insert_select(
    select: exp.Expression | None, table: exp.Table, columns: list[str]
) -> None:
    """Coerce each semi-structured select value written into a text target column."""
    if isinstance(select, exp.Union):
        _coerce_insert_select(select.this, table, columns)
        _coerce_insert_select(select.expression, table, columns)
        return
    if not isinstance(select, exp.Select):
        return
    _expand_derived_star(select, columns)
    if len(select.expressions) != len(columns):
        return
    rewritten: list[exp.Expression] = []
    for target, item in zip(columns, select.expressions):
        value = item.this if isinstance(item, exp.Alias) else item
        if _is_text_column(_target_column_type(table, target)) and (
            _is_semi_structured_value(value)
        ):
            converted = _text_operand(value)
            if isinstance(item, exp.Alias):
                item = item.copy()
                item.set("this", converted)
            else:
                item = converted
        rewritten.append(item)
    select.set("expressions", rewritten)


def rewrite_semi_structured_writes(statement: exp.Expression) -> exp.Expression:
    """Coerce a staged JSON value written into a text column to that column's text form.

    ``INSERT INTO t (name) SELECT $1:name FROM @stage``, the same INSERT without a column
    list, and ``UPDATE t SET name = s.$1:name`` would otherwise store DuckDB's JSON
    serialization, quotes and all (``"ada"``), where Snowflake stores the string's text
    (``ada``).
    """
    if isinstance(statement, exp.Insert):
        table, columns = _insert_target_columns(statement)
        if table is not None and columns:
            _coerce_insert_select(statement.args.get("expression"), table, columns)
    elif isinstance(statement, exp.Update) and isinstance(statement.this, exp.Table):
        table = statement.this
        for assignment in statement.expressions:
            if not isinstance(assignment, exp.EQ):
                continue
            column = assignment.this
            value = assignment.args.get("expression")
            if (
                isinstance(column, exp.Column)
                and _is_semi_structured_value(value)
                and _is_text_column(_target_column_type(table, column.name))
            ):
                assignment.set("expression", _text_operand(value))
    return statement

def _rewrite_snowflake_specifics(node: exp.Expression) -> exp.Expression:
    """Rewrite Snowflake nodes that sqlglot can't fully transpile to DuckDB."""
    # ``CURRENT_VERSION()`` -> a Snowflake-style dotted-numeric version literal.
    if isinstance(node, exp.CurrentVersion):
        return exp.Literal.string(EMULATED_SNOWFLAKE_VERSION)

    # ``col:path::VARCHAR`` / ``TO_VARCHAR(col:path)``.
    if isinstance(node, exp.Cast) and _is_text_type(node.to):
        return _rewrite_semi_structured_text_cast(node)
    if isinstance(node, exp.ToChar) and node.args.get("format") is None:
        return _rewrite_semi_structured_text_cast(node)
    if type(node) in _TEXT_ARGUMENTS:
        return _rewrite_variant_text_function(node)
    if isinstance(node, _COMPARISON_NODES):
        return _rewrite_semi_structured_comparison(node)
    if isinstance(node, exp.In):
        return _rewrite_semi_structured_membership(node)

    # ``PARSE_JSON(<variant>)``: DuckDB's ``JSON(variant)`` first renders the VARIANT's
    # internal repr (``{'names': [X]}``) and then fails to parse it. Snowflake parses the
    # value's JSON text, which is exactly what casting the VARIANT to JSON gives.
    if isinstance(node, exp.ParseJSON):
        operand = node.this
        if _is_variant_value(operand) or column_types.is_variant_typed(operand):
            return exp.cast(operand, exp.DataType.build("JSON"), dialect=WRITE_DIALECT)
        return node

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
            input_expr, outer, path = _flatten_arguments(node.this)
            new = node.copy()
            new.set("this", _flatten_subquery(input_expr, outer=outer, path=path))
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
            input_expr, outer, path = _flatten_arguments(node.this)
            sub = _flatten_subquery(input_expr, outer=outer, path=path)
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

    # ``CREATE VOLATILE TABLE`` is Snowflake's (deprecated) synonym for
    # ``CREATE TEMPORARY TABLE``. sqlglot's DuckDB generator drops the VOLATILE marker,
    # which would silently create a permanent table, so it is swapped for the property
    # DuckDB understands and the table keeps its session-scoped lifetime.
    if isinstance(node, exp.Create):
        properties = node.args.get("properties")
        if properties is not None and any(
            isinstance(prop, exp.VolatileProperty) for prop in properties.expressions
        ):
            rewritten = node.copy()
            rewritten_properties = rewritten.args["properties"]
            rewritten_properties.set(
                "expressions",
                [
                    exp.TemporaryProperty() if isinstance(prop, exp.VolatileProperty) else prop
                    for prop in rewritten_properties.expressions
                ],
            )
            return rewritten

    # ``MERGE ... INSERT (t.col, ...)`` -> unqualified target columns. Snowflake accepts the
    # target-table qualifier in the INSERT column list, but DuckDB reads it as a column
    # named after the table (``INSERT (TARGET.ID)`` -> "does not have a column TARGET").
    if isinstance(node, exp.Merge):
        whens = node.args.get("whens")
        for when in whens.expressions if isinstance(whens, exp.Whens) else []:
            then = when.args.get("then") if isinstance(when, exp.When) else None
            if not isinstance(then, exp.Insert):
                continue
            target = then.this
            if isinstance(target, (exp.Tuple, exp.Schema)):
                for column in target.expressions:
                    if isinstance(column, exp.Column):
                        for key in ("table", "db", "catalog"):
                            column.args.pop(key, None)
        return node

    # ``MERGE ... UPDATE SET t.col = ...`` -> unqualified target column. DuckDB rejects
    # the target-table qualifier on the left side of a SET assignment.
    if isinstance(node, exp.Update):
        for eq in node.expressions:
            if isinstance(eq, exp.EQ) and isinstance(eq.this, exp.Column):
                for key in ("table", "db", "catalog"):
                    eq.this.args.pop(key, None)
        return node

    return node
