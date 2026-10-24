"""Snowflake's semi-structured *functions* that DuckDB has no equivalent for.

These are separate from the rewrites in ``semi_structured.py``: they all take a VARIANT/OBJECT
value and are recognised by name (``IS_OBJECT``, ``OBJECT_PICK``, ...), where sqlglot either
leaves the call uninterpreted or maps it onto a DuckDB function with different semantics.

The predicates read DuckDB's ``variant_typeof``, which reports the type *stored* in a VARIANT
(``INT32``, ``DECIMAL(5, 2)``, ``DATE``, ``OBJECT(...)``, ``VARIANT_NULL``, ...), so they match
Snowflake's "contains a value of this type" answers instead of DuckDB's ``typeof``.
"""

from __future__ import annotations

import fnmatch

import sqlglot
from sqlglot import exp

from snowflake_emulator.translator import column_types
from snowflake_emulator.translator.arrays import _is_json_value, _is_variant_value
from snowflake_emulator.translator.constants import WRITE_DIALECT
from snowflake_emulator.translator.errors import TranslationError

# DuckDB ``variant_typeof`` names for the integer kinds Snowflake calls INTEGER.
_INTEGER_TYPES = (
    "INT8", "INT16", "INT32", "INT64", "INT128",
    "UINT8", "UINT16", "UINT32", "UINT64", "UINT128",
)  # fmt: skip

# Each predicate is expressed over the Snowflake type name (see ``_type_name_sql``).
_IS_PREDICATES: dict[str, tuple[str, ...]] = {
    "IS_INTEGER": ("INTEGER",),
    "IS_DECIMAL": ("DECIMAL",),
    "IS_DOUBLE": ("DOUBLE",),
    "IS_REAL": ("DOUBLE",),
    "IS_FLOAT": ("DOUBLE",),
    "IS_BOOLEAN": ("BOOLEAN",),
    "IS_VARCHAR": ("VARCHAR",),
    "IS_CHAR": ("VARCHAR",),
    "IS_BINARY": ("BINARY",),
    "IS_DATE": ("DATE",),
    "IS_DATE_VALUE": ("DATE",),
    "IS_TIME": ("TIME",),
    # The emulator stores TIMESTAMP_LTZ and TIMESTAMP_TZ as one TIMESTAMPTZ, so both
    # predicates answer for either; TIMESTAMP_NTZ is the one without a zone.
    "IS_TIMESTAMP_NTZ": ("TIMESTAMP_NTZ",),
    "IS_TIMESTAMP_LTZ": ("TIMESTAMP_TZ",),
    "IS_TIMESTAMP_TZ": ("TIMESTAMP_TZ",),
    "IS_OBJECT": ("OBJECT",),
    "IS_ARRAY": ("ARRAY",),
}

# Snowflake's TYPEOF name for a DuckDB variant/JSON type. Snowflake's PARSE_JSON keeps a
# decimal exact (docs: 123.45 is NUMBER(5,2)), while DuckDB stores it as a DOUBLE, so DOUBLE is
# reported as DECIMAL unless the value is statically a FLOAT/DOUBLE (an explicit DOUBLE variant).
_TYPEOF_CASES: tuple[tuple[str, str], ...] = (
    (f"t IN ({', '.join(repr(name) for name in _INTEGER_TYPES)})", "INTEGER"),
    # DuckDB's JSON numbers report the SQL width instead (``json_type``).
    (
        "t IN ('BIGINT', 'UBIGINT', 'HUGEINT', 'UHUGEINT', 'INTEGER', 'UINTEGER', "
        "'SMALLINT', 'USMALLINT', 'TINYINT', 'UTINYINT')",
        "INTEGER",
    ),
    ("t = 'DOUBLE'", "#double"),
    ("t = 'FLOAT'", "#double"),
    ("t LIKE 'DECIMAL%'", "DECIMAL"),
    ("t IN ('BOOL_TRUE', 'BOOL_FALSE')", "BOOLEAN"),
    ("t = 'VARCHAR'", "VARCHAR"),
    ("t = 'BLOB'", "BINARY"),
    ("t = 'DATE'", "DATE"),
    # ``TIME%`` would also match TIMESTAMP_*, so the timestamp kinds are excluded.
    ("t LIKE 'TIME%' AND t NOT LIKE 'TIMESTAMP%'", "TIME"),
    ("t LIKE 'TIMESTAMP%TZ'", "TIMESTAMP_TZ"),
    ("t LIKE 'TIMESTAMP%'", "TIMESTAMP_NTZ"),
    ("t LIKE 'OBJECT%'", "OBJECT"),
    ("t LIKE 'ARRAY%'", "ARRAY"),
    ("t = 'VARIANT_NULL'", "NULL_VALUE"),
    # ``json_type`` spells a JSON null just ``NULL``.
    ("t = 'NULL'", "NULL_VALUE"),
)


def _is_float_source(node: exp.Expression) -> bool:
    """True when the value came from a FLOAT/DOUBLE, which Snowflake reports as DOUBLE."""
    inner = node
    if isinstance(inner, exp.ToVariant):
        inner = inner.this
    if (
        isinstance(inner, exp.Cast)
        and isinstance(inner.to, exp.DataType)
        and inner.to.this in (exp.DataType.Type.VARIANT, exp.DataType.Type.JSON)
    ):
        inner = inner.this
    if isinstance(inner, exp.Cast) and isinstance(inner.to, exp.DataType):
        return inner.to.this in (exp.DataType.Type.DOUBLE, exp.DataType.Type.FLOAT)
    declared = column_types.declared_column_type(inner)
    return declared is not None and declared.strip().upper() in ("DOUBLE", "FLOAT")


def _type_name_sql(node: exp.Expression) -> str:
    """The Snowflake type name of a semi-structured value, as one scalar expression.

    A VARIANT exposes its stored type through ``variant_typeof`` (the only way to see a DATE,
    TIME or TIMESTAMP inside one); a DuckDB JSON value (a path result, a staged record) only
    has ``json_type``, whose numbers are doubles, integers or strings.
    """
    double = "'DOUBLE'" if _is_float_source(node) else "'DECIMAL'"
    if _is_variant_value(node) or column_types.is_variant_typed(node):
        source = f"variant_typeof({_sql(node)})"
    else:
        source = f"json_type({_document_sql(node)})"
    cases = " ".join(
        f"WHEN {condition} THEN {double if name == '#double' else repr(name)}"
        for condition, name in _TYPEOF_CASES
    )
    return f"(SELECT CASE {cases} ELSE t END FROM (SELECT {source} AS t))"


def _is_semi_structured(node: exp.Expression | None) -> bool:
    return (
        node is not None
        and (
            _is_json_value(node)
            or _is_variant_value(node)
            or column_types.is_semi_structured_typed(node)
        )
    )


def _rewrite_operand(node: exp.Expression) -> exp.Expression:
    """Rewrite an operand before it is serialized.

    The statement pass is pre-order (a node is visited before its children), so a path such as
    ``v:o`` inside one of these calls has not had its VARIANT -> JSON cast applied yet; entering
    the pass on the operand again puts it in its final form. Imported lazily because
    ``semi_structured`` dispatches to this module.
    """
    from snowflake_emulator.translator.semi_structured import (
        _rewrite_snowflake_specifics,
    )

    return node.transform(_rewrite_snowflake_specifics)


def _sql(node: exp.Expression) -> str:
    return node.sql(dialect=WRITE_DIALECT)


def _document_sql(node: exp.Expression) -> str:
    """The operand as a JSON document; a VARIANT is cast through JSON, a JSON value kept."""
    if _is_json_value(node):
        return _sql(node)
    return f"CAST({_sql(node)} AS JSON)"


def _variant_sql(node: exp.Expression) -> str:
    """The operand as a VARIANT value (Snowflake's predicate argument type)."""
    if _is_variant_value(node) or column_types.is_variant_typed(node):
        return _sql(node)
    return f"CAST({_document_sql(node)} AS VARIANT)"


def rewrite_typeof(node: exp.Typeof) -> exp.Expression:
    """``TYPEOF`` names the type stored *in* the value, as Snowflake documents it.

    ``variant_typeof`` is the only way to see DATE/TIME/TIMESTAMP (DuckDB's JSON casts turn
    them into strings), and it reports a SQL NULL as ``VARIANT_NULL`` just like a JSON null,
    so the SQL NULL case is answered first. Snowflake documents TYPEOF as approximate for
    DECIMAL, and DuckDB's parse normalizes an exponent form; DOUBLE is reported as DECIMAL
    unless the value is statically a FLOAT/DOUBLE.
    """
    operand = node.this
    if isinstance(operand, exp.Null):
        return exp.Literal.string("NULL")
    if not isinstance(operand, exp.Expression):
        return node
    if not _is_semi_structured(operand):
        return node
    operand = _rewrite_operand(operand)
    return sqlglot.parse_one(
        f"CASE WHEN ({_sql(operand)}) IS NULL THEN 'NULL' "
        f"ELSE {_type_name_sql(operand)} END",
        read=WRITE_DIALECT,
    )


def _is_predicate(node: exp.Anonymous) -> exp.Expression | None:
    """``IS_<type>(variant)`` -> a BOOLEAN over the value's stored type."""
    name = (node.name or "").upper()
    wanted = _IS_PREDICATES.get(name)
    if wanted is None:
        return None
    operands = node.expressions or []
    if len(operands) != 1 or not isinstance(operands[0], exp.Expression):
        raise TranslationError(f"{name} takes a single VARIANT argument.")
    operand = _rewrite_operand(operands[0]) if _is_semi_structured(operands[0]) else operands[0]
    names = ", ".join(repr(value) for value in wanted)
    # A NULL argument answers NULL, not FALSE ("without reporting an error").
    return sqlglot.parse_one(
        f"CASE WHEN ({_sql(operand)}) IS NULL THEN NULL "
        f"ELSE {_type_name_sql(operand)} IN ({names}) END",
        read=WRITE_DIALECT,
    )


def _key_list(node: exp.Expression | None) -> tuple[list[str], list[str]] | None:
    """The keys of ``OBJECT_PICK``/``OBJECT_DELETE``: literal keys, or a key-array SQL.

    Returns ``(literal_keys, array_sql)`` where at most one is non-empty, or None when the
    arguments are not something the rewrite can express.
    """
    keys: list[str] = []
    arrays: list[str] = []
    for argument in node.expressions if node is not None else []:
        if isinstance(argument, exp.Literal) and argument.is_string:
            keys.append(str(argument.this))
        elif _is_semi_structured(argument):
            # The documented ``OBJECT_PICK(object, array_of_keys)`` form.
            arrays.append(f"CAST({_document_sql(argument)} AS JSON)")
        else:
            return None
    return keys, arrays


def _key_filter(keys: list[str], arrays: list[str], *, keep: bool) -> str:
    """The ``WHERE`` condition selecting (``keep``) or dropping the given keys."""
    conditions: list[str] = []
    if keys:
        literals = ", ".join(exp.Literal.string(key).sql(dialect=WRITE_DIALECT) for key in keys)
        conditions.append(f"e.key {'IN' if keep else 'NOT IN'} ({literals})")
    for array in arrays:
        conditions.append(
            f"e.key {'IN' if keep else 'NOT IN'} (SELECT json_extract_string(value, '$') "
            f"FROM json_each({array}))"
        )
    if not conditions:
        return "TRUE" if keep else "FALSE"
    joiner = " OR " if keep else " AND "
    return f"({joiner.join(conditions)})"


def _object_rebuild(
    node: exp.Expression, document: str, where: str, *, empty_object: bool = True
) -> exp.Expression:
    """A JSON OBJECT rebuilt from the entries of ``document`` that satisfy ``where``."""
    rebuilt = (
        f"(SELECT json_group_object(e.key, e.value) "
        f"FROM json_each({document}) e WHERE {where})"
    )
    if empty_object:
        # An empty result is an empty OBJECT, not NULL, but a NULL input stays NULL.
        rebuilt = f"COALESCE({rebuilt}, CAST('{{}}' AS JSON))"
    return sqlglot.parse_one(
        f"CASE WHEN ({_sql(node)}) IS NULL THEN NULL ELSE {rebuilt} END",
        read=WRITE_DIALECT,
    )


def _rewrite_object_pick(node: exp.Anonymous) -> exp.Expression | None:
    arguments = node.expressions or []
    if len(arguments) < 2 or not isinstance(arguments[0], exp.Expression):
        raise TranslationError("OBJECT_PICK takes an OBJECT and at least one key.")
    keys = _key_list(exp.Anonymous(expressions=arguments[1:]))
    if keys is None:
        return None
    source = _rewrite_operand(arguments[0])
    document = _document_sql(source)
    return _object_rebuild(source, document, _key_filter(*keys, keep=True))


def _rewrite_object_delete(node: exp.Anonymous) -> exp.Expression | None:
    arguments = node.expressions or []
    if len(arguments) < 2 or not isinstance(arguments[0], exp.Expression):
        raise TranslationError("OBJECT_DELETE takes an OBJECT and at least one key.")
    keys = _key_list(exp.Anonymous(expressions=arguments[1:]))
    if keys is None:
        return None
    source = _rewrite_operand(arguments[0])
    document = _document_sql(source)
    return _object_rebuild(source, document, _key_filter(*keys, keep=False))


def _rewrite_object_insert(node: exp.ObjectInsert) -> exp.Expression:
    """``OBJECT_INSERT(object, key, value [, update])``: insert or update one pair.

    A SQL NULL key or value omits the pair (the object is returned unchanged), a JSON null
    value is inserted, and an existing key is an error unless ``update`` is TRUE.
    """
    source = node.this
    key = node.args.get("key")
    value = node.args.get("value")
    update = node.args.get("update_flag")
    if not isinstance(source, exp.Expression) or not isinstance(key, exp.Expression):
        return node
    source = _rewrite_operand(source)
    if isinstance(value, exp.Expression):
        value = _rewrite_operand(value)
    document = _document_sql(source)
    key_sql = _sql(key)
    if isinstance(value, exp.Expression):
        value_sql = f"CAST({_sql(value)} AS JSON)"
    else:
        value_sql = "CAST(NULL AS JSON)"
    exists = f"list_contains(json_keys({document}), {key_sql})"
    if isinstance(update, exp.Boolean) and update.this:
        duplicate = "FALSE"
    else:
        duplicate = (
            f"(NOT ({key_sql}) IS NULL AND NOT ({value_sql}) IS NULL AND {exists})"
        )
    rebuild = (
        f"(SELECT json_group_object(e.key, e.value) FROM ("
        f"SELECT key, value FROM json_each({document}) "
        f"WHERE key <> {key_sql} "
        f"UNION ALL SELECT {key_sql}, {value_sql}"
        f") e)"
    )
    return sqlglot.parse_one(
        f"CASE WHEN ({_sql(source)}) IS NULL THEN NULL "
        f"WHEN ({key_sql}) IS NULL OR ({value_sql}) IS NULL THEN {document} "
        f"WHEN {duplicate} THEN error('Duplicate key in OBJECT_INSERT; pass TRUE to update') "
        f"ELSE COALESCE({rebuild}, CAST('{{}}' AS JSON)) END",
        read=WRITE_DIALECT,
    )


def _rewrite_get_ignore_case(node: exp.GetIgnoreCase) -> exp.Expression:
    """``GET_IGNORE_CASE(object, field)``: a single field, matched case-insensitively.

    The documented signature takes one *field name* (not a path). The matching key is looked
    up in the object's own keys, then the value is extracted with that key quoted so a space or
    a dot in the name stays part of the key.
    """
    source = node.this
    field = node.args.get("expression")
    if not isinstance(source, exp.Expression) or not isinstance(field, exp.Expression):
        return node
    source = _rewrite_operand(source)
    field = _rewrite_operand(field)
    document = _document_sql(source)
    matched = (
        f"list_extract(list_filter(json_keys({document}), "
        f"k -> lower(k) = lower({_sql(field)})), 1)"
    )
    path = (
        f"'$.\"' || replace({matched}, '\"', '\\\"') || '\"'"
    )
    return sqlglot.parse_one(
        f"CASE WHEN ({_sql(source)}) IS NULL OR ({_sql(field)}) IS NULL THEN NULL "
        f"ELSE json_extract({document}, {path}) END",
        read=WRITE_DIALECT,
    )


def _source_columns(relation: exp.Expression) -> list[str] | None:
    """The column names a FROM source exposes, or None when they cannot be enumerated."""
    if isinstance(relation, exp.Table):
        return column_types.declared_table_columns(relation)
    if isinstance(relation, exp.Subquery):
        select = relation.this
        selects = (
            [select]
            if isinstance(select, exp.Select)
            else [
                arm
                for arm in sqlglot.parse_one(
                    select.sql(dialect=WRITE_DIALECT), read=WRITE_DIALECT
                ).find_all(exp.Select)
            ]
        )
        names: list[str] = []
        for item in selects[0].expressions if selects else []:
            name = item.alias_or_name
            if not name or name == "*":
                return None
            names.append(name)
        return names or None
    return None


def _star_sources(node: exp.StarMap) -> list[exp.Expression]:
    """The FROM/JOIN sources the wildcard covers, from the enclosing SELECT."""
    star = node.this
    wanted = None
    if isinstance(star, exp.Column):
        wanted = star.table.upper() if star.table else None
        star = star.this
    scope = node.parent
    while scope is not None and not isinstance(scope, exp.Select):
        scope = scope.parent
    if scope is None:
        return []
    sources: list[exp.Expression] = []
    from_ = scope.args.get("from_")
    if isinstance(from_, exp.From):
        sources.append(from_.this)
    for join in scope.args.get("joins") or []:
        if isinstance(join, exp.Join):
            sources.append(join.this)
    if wanted is None:
        return sources
    return [
        source
        for source in sources
        if (getattr(source, "alias", None) or getattr(source, "name", "") or "").upper()
        == wanted
    ]


def _matches_name(name: str, pattern: str) -> bool:
    """SQL ``ILIKE`` wildcards (``%`` and ``_``) against a column name."""
    return fnmatch.fnmatchcase(name.lower(), pattern.replace("%", "*").replace("_", "?"))


def _rewrite_star_object(node: exp.StarMap) -> exp.Expression:
    """``OBJECT_CONSTRUCT(*)``: an OBJECT over the columns of the FROM sources in scope.

    The documented ``* ILIKE '<pattern>'`` and ``* EXCLUDE (...)`` filters are applied to the
    column names. A source whose columns the catalog cannot enumerate is a loud error rather
    than a silently empty object.
    """
    star = node.this
    ilike = star.args.get("ilike") if isinstance(star, exp.Star) else None
    excluded = star.args.get("except_") if isinstance(star, exp.Star) else None
    sources = _star_sources(node)
    if not sources:
        raise TranslationError(
            "OBJECT_CONSTRUCT(*) needs a FROM clause whose columns the emulator can name."
        )
    names: list[str] = []
    for source in sources:
        columns = _source_columns(source)
        if columns is None:
            raise TranslationError(
                "OBJECT_CONSTRUCT(*) needs a table or a derived table whose columns the "
                "emulator can name."
            )
        names.extend(columns)
    if isinstance(ilike, exp.Literal) and ilike.is_string:
        pattern = str(ilike.this)
        names = [name for name in names if _matches_name(name, pattern)]
    if isinstance(excluded, exp.Expression):
        excluded = excluded.expressions
    drop = {
        column.name.upper()
        for column in (excluded or [])
        if isinstance(column, exp.Expression) and column.name
    }
    names = [name for name in names if name.upper() not in drop]
    if not names:
        return sqlglot.parse_one("CAST('{}' AS JSON)", read=WRITE_DIALECT)
    return exp.Struct(
        expressions=[
            exp.PropertyEQ(this=exp.Literal.string(name), expression=exp.column(name))
            for name in names
        ]
    )


def rewrite_object_function(node: exp.Expression) -> exp.Expression:
    """Rewrite one semi-structured function call, or return `node` when it is not one."""
    if isinstance(node, exp.Anonymous):
        name = (node.name or "").upper()
        if name in _IS_PREDICATES:
            rewritten = _is_predicate(node)
            if rewritten is not None:
                return rewritten
        if name == "OBJECT_PICK":
            rewritten = _rewrite_object_pick(node)
            if rewritten is not None:
                return rewritten
        if name == "OBJECT_DELETE":
            rewritten = _rewrite_object_delete(node)
            if rewritten is not None:
                return rewritten
        return node
    if isinstance(node, exp.ObjectInsert):
        return _rewrite_object_insert(node)
    if isinstance(node, exp.GetIgnoreCase):
        return _rewrite_get_ignore_case(node)
    if isinstance(node, exp.StarMap):
        return _rewrite_star_object(node)
    return node
