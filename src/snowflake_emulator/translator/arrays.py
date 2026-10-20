"""Snowflake array-function rewrites over DuckDB JSON/VARIANT lists."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Callable

import sqlglot
from sqlglot import exp

from snowflake_emulator.translator.constants import WRITE_DIALECT, _FLATTEN_MARKER
from snowflake_emulator.translator.errors import TranslationError

# ``(table_name_or_None, column_name) -> DuckDB type name``, supplied by the caller that has a
# catalog (see ``snowflake_emulator.catalog``). A bare column has no type in the AST, and an
# array function needs one to tell a real DuckDB LIST from a VARIANT/JSON array.
ColumnTypeResolver = Callable[[str | None, str], str | None]

_column_types: ContextVar[ColumnTypeResolver | None] = ContextVar(
    "_column_types", default=None
)

# Column and table names the *statement itself* defines (select aliases, CTE names, derived
# table aliases). A bare column with one of these names belongs to the query, not to a catalog
# table that happens to share it, so it must not inherit that table's type.
_scoped_names: ContextVar[frozenset[str]] = ContextVar("_scoped_names", default=frozenset())

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


def _is_variant_list(node: exp.Expression) -> bool:
    """True for a ``CAST(... AS VARIANT[])``, the list form these rewrites produce."""
    return (
        isinstance(node, exp.Cast)
        and isinstance(node.to, exp.DataType)
        and node.to.this == exp.DataType.Type.ARRAY
        and any(
            isinstance(inner, exp.DataType) and inner.this == exp.DataType.Type.VARIANT
            for inner in node.to.expressions
        )
    )


def _is_variant_array_value(node: exp.Expression) -> bool:
    """True for expressions the emulator evaluates to DuckDB's VARIANT[] type.

    Covers a VARIANT value, a ``CAST(... AS VARIANT[])``, and an ``ARRAY_SLICE`` over
    either -- which this module turns into a VARIANT[] call, so an enclosing call must
    treat it as an array too.
    """
    if _is_variant_value(node) or _is_variant_list(node):
        return True
    return isinstance(node, exp.ArraySlice) and _is_variant_array_value(node.this)


def _json_list(node: exp.Expression | None) -> exp.Expression | None:
    """Present a JSON array to DuckDB as ``JSON[]``; anything else is unchanged."""
    if node is None or _is_json_list(node) or not _is_json_value(node):
        return node
    return exp.cast(node, exp.DataType.build("JSON[]"), dialect=WRITE_DIALECT)


def _declared_column_type(node: exp.Expression | None) -> str | None:
    """The DuckDB type of a bare column, when the caller supplied a catalog lookup.

    A name the statement defines itself -- a select alias, a CTE, a derived table alias -- is
    not looked up: an unrelated table with the same column name would otherwise decide how the
    column is rewritten, which mis-orders a query-scoped VARIANT value or fails it with a raw
    DuckDB binder error.
    """
    resolver = _column_types.get()
    if resolver is None or not isinstance(node, exp.Column):
        return None
    scoped = _scoped_names.get()
    if node.name.upper() in scoped or (node.table and node.table.upper() in scoped):
        return None
    return resolver(
        node.catalog or None, node.db or None, node.table or None, node.name
    )


def _is_list_type(declared: str) -> bool:
    """True for a DuckDB type that is already a LIST, e.g. ``VARCHAR[]`` or ``LIST(INTEGER)``."""
    return declared.strip().upper().endswith("[]") or declared.upper().startswith("LIST(")


def _is_variant_list_type(declared: str) -> bool:
    """True for a DuckDB type whose elements are VARIANT, which DuckDB mis-orders."""
    return declared.strip().upper().startswith("VARIANT")


def _is_json_list_type(declared: str) -> bool:
    """True for a DuckDB type whose elements are JSON (a ``col:path`` column, or ``JSON[]``)."""
    return declared.strip().upper().startswith("JSON")


def _array_list(node: exp.Expression | None) -> exp.Expression | None:
    """Present a Snowflake array to DuckDB as a LIST.

    A JSON array becomes ``JSON[]`` and a VARIANT array becomes ``VARIANT[]``: both are
    real DuckDB lists, so they bind the list functions, and their elements compare as
    VARIANT values the way Snowflake compares them. An operand whose type is only known at
    run time -- a bare column -- is resolved from the catalog when the caller supplied one:
    a real DuckDB list keeps its own element type (so a CTAS native-list column orders and
    compares natively), a JSON column is cast to ``JSON[]``, and a VARIANT one is cast
    through VARIANT to ``VARIANT[]``. With no catalog (or no match) the blind VARIANT
    round-trip is kept, which is how a bare VARIANT column -- the usual case -- reaches a
    list. An operand already in one of those list forms, a native DuckDB list, or a call
    whose result is already a list, is left alone.
    """
    if node is None or _is_json_list(node) or _is_variant_list(node):
        return node
    if _is_json_value(node):
        return _json_list(node)
    if isinstance(node, exp.Array):
        return node
    declared = _declared_column_type(node)
    if declared is not None:
        if _is_json_list_type(declared):
            return exp.cast(
                node, exp.DataType.build("JSON[]"), dialect=WRITE_DIALECT
            )
        if _is_list_type(declared) and not _is_variant_list_type(declared):
            return node
    variant = exp.DataType.build("VARIANT", dialect=WRITE_DIALECT)
    if _is_variant_value(node):
        return exp.cast(node, exp.DataType.build("VARIANT[]"), dialect=WRITE_DIALECT)
    if isinstance(node, (exp.Column, exp.Placeholder, exp.Parameter)):
        # ``CAST(json AS VARIANT[])`` raises, so go through VARIANT first; a VARIANT or a
        # native list is unaffected by the extra cast.
        return exp.cast(
            exp.cast(node, variant, dialect=WRITE_DIALECT),
            exp.DataType.build("VARIANT[]"),
            dialect=WRITE_DIALECT,
        )
    # Any other expression is a call this module already rewrote to a list result.
    return node


def _json_element(node: exp.Expression | None) -> exp.Expression | None:
    """Encode a searched value so it compares as a VARIANT against ``JSON[]`` elements."""
    if node is None or _is_json_value(node) or _is_variant_value(node):
        return node
    return exp.func("to_json", node)


def _variant_element(node: exp.Expression | None) -> exp.Expression | None:
    """Encode a searched value as a VARIANT so it compares against ``VARIANT[]`` elements.

    The comparison is representation-sensitive: a number derived from ``PARSE_JSON`` is a
    ``UINT64`` VARIANT where ``CAST(2 AS VARIANT)`` is an ``INT32``, and DuckDB's
    ``array_contains`` does not treat those as equal. Round-tripping the value through
    ``VARIANT`` -> ``JSON`` -> ``VARIANT`` lands it on the same representation without needing
    to know its type: a VARIANT or JSON value is normalized, a native number/string/bool gets
    the JSON-derived width, a native text stays text (a bare ``CAST(text AS JSON)`` would try
    to parse it), and a nested list stays a nested list. The value is named once.
    """
    variant = exp.DataType.build("VARIANT", dialect=WRITE_DIALECT)
    json_type = exp.DataType.build("JSON", dialect=WRITE_DIALECT)
    if node is None:
        return node
    as_variant = exp.cast(node, variant, dialect=WRITE_DIALECT)
    as_json = exp.cast(as_variant, json_type, dialect=WRITE_DIALECT)
    return exp.cast(as_json, variant, dialect=WRITE_DIALECT)


def _encoded_element(
    value: exp.Expression | None, array: exp.Expression | None
) -> exp.Expression | None:
    """Encode a searched value the way ``array``'s element type needs it."""
    if _is_json_array_value(array):
        return _json_element(value)
    if _is_variant_array_value(array):
        return _variant_element(value)
    return value


def _is_variant_typed(node: exp.Expression | None) -> bool:
    """True when `node` evaluates to a VARIANT value: a VARIANT column, cast or array."""
    if node is None:
        return False
    if _is_variant_value(node) or _is_variant_array_value(node):
        return True
    declared = _declared_column_type(node)
    return declared is not None and _is_variant_list_type(declared)


def _aggregates_to_variant_list(node: exp.Expression) -> bool:
    """True when `node` is an aggregate that builds a list of VARIANT elements.

    ``ARRAY_AGG`` assembles the list at run time, so it carries no ``VARIANT[]`` cast for
    ``_has_variant_list`` to find -- and an ordering call over an aggregate of a VARIANT column
    therefore slipped through the guard, where DuckDB orders by length. A JSON or native
    argument is still allowed: those lists order the way Snowflake orders them.
    """
    if not isinstance(node, exp.ArrayAgg):
        return False
    argument = node.this
    if isinstance(argument, exp.Distinct):
        argument = argument.this
    return _is_variant_typed(argument)


def _comparable_array_and_value(
    array: exp.Expression | None, value: exp.Expression | None
) -> tuple[exp.Expression | None, exp.Expression | None]:
    """Pick one element space for an array and the value searched in it.

    A VARIANT value compares in VARIANT space: both sides go through the same normalization
    round-trip, where a JSON-derived and a text-built variant land on the same representation.
    The JSON space leaves the variant on one side of the comparison only, and DuckDB then
    treats it as unequal -- ``ARRAY_CONTAINS(<variant 2>, PARSE_JSON('[2,3]'))`` answered FALSE
    although the array holds that number. Anything else keeps the array's own space.
    """
    if _is_variant_typed(value):
        if array is not None and not _is_variant_array_value(array):
            array = _encoded_array_elements(array, "variant")
        return array, _variant_element(value)
    return array, _encoded_element(value, array)


def _has_variant_list(node: exp.Expression) -> bool:
    """True when `node` holds a ``VARIANT[]`` cast -- the array functions' mark for one."""
    return _is_variant_list(node) or any(
        _is_variant_list(cast) for cast in node.find_all(exp.Cast)
    )


# ``json_type`` names a JSON number by its storage type (UBIGINT, BIGINT, DOUBLE, ...), so a
# number is recognised as "anything that is not one of the non-numeric kinds".
_JSON_NON_NUMERIC_TYPES = "('VARCHAR', 'BOOLEAN', 'ARRAY', 'OBJECT', 'NULL')"
_JSON_NUMBER = (
    f"COALESCE(json_type(_sfemu_element) NOT IN {_JSON_NON_NUMERIC_TYPES}, FALSE)"
)

_ORDERING_KEY = (
    "{{'t': CASE WHEN _sfemu_element IS NULL THEN {null_rank} "
    f"WHEN {_JSON_NUMBER} THEN 1 ELSE 2 END, "
    f"'n': CASE WHEN {_JSON_NUMBER} "
    "THEN CAST(_sfemu_element AS DOUBLE) ELSE 0 END, "
    "'s': CASE WHEN _sfemu_element IS NULL THEN '' "
    "ELSE CAST(_sfemu_element AS VARCHAR) END, "
    "'v': _sfemu_element}}"
)

_ORDERING_NAMES = {
    exp.ArrayMax: "ARRAY_MAX",
    exp.ArrayMin: "ARRAY_MIN",
    exp.SortArray: "ARRAY_SORT",
}


def _ordering_flag(node: exp.Expression, name: str) -> bool | None:
    """One of ``ARRAY_SORT``'s boolean arguments, or None when the call did not pass it.

    Only a literal is accepted. Reading ``bool(node.this)`` from an arbitrary expression made
    ``ARRAY_SORT(arr, 1 = 2)`` sort ascending (the left operand is a non-empty node), which
    silently returns the wrong order, and ``ARRAY_SORT(arr, NULL)`` sort descending where
    Snowflake returns NULL for a NULL argument.
    """
    value = node.args.get(name)
    if value is None:
        return None
    if isinstance(value, exp.Boolean):
        return bool(value.this)
    raise TranslationError(
        f"ARRAY_SORT's {name} argument must be a literal TRUE or FALSE in the emulator."
    )


def _rewrite_json_ordering(node: exp.Expression, array: exp.Expression) -> exp.Expression:
    """Order a ``JSON[]`` by a type-aware key, so numbers compare numerically.

    DuckDB compares JSON values as text, so a JSON number array came back in text order
    (``[10,2,33]`` stayed ``[10,2,33]``, and ``ARRAY_MAX([9,100])`` was ``9``). Snowflake orders
    an array the way ``ORDER BY`` orders its flattened elements, so each element carries a key:
    a null rank, then a numeric field for a JSON number, then the element's text. Numbers
    therefore compare numerically, strings lexicographically, and compound values by their
    text. The null rank puts NULL elements last for an ascending sort and first for a
    descending one, which is Snowflake's default, and honours an explicit ``nulls_first``.
    """
    ascending = _ordering_flag(node, "asc")
    ascending = True if ascending is None else ascending
    nulls_first = _ordering_flag(node, "nulls_first")
    nulls_first = (not ascending) if nulls_first is None else nulls_first
    # Rank 0 sorts first and 3 last in ascending order, so the rank that puts NULLs first
    # depends on the direction the sort is applied in.
    null_rank = (0 if nulls_first else 3) if ascending else (3 if nulls_first else 0)
    keyed = (
        f"LIST_TRANSFORM({_sql(array)}, _sfemu_element -> "
        f"{_ORDERING_KEY.format(null_rank=null_rank)})"
    )
    if isinstance(node, exp.SortArray):
        direction = "'ASC'" if ascending else "'DESC'"
        return sqlglot.parse_one(
            f"LIST_TRANSFORM(LIST_SORT({keyed}, {direction}), "
            f"_sfemu_sorted -> _sfemu_sorted.v)",
            read=WRITE_DIALECT,
        )
    aggregate = "LIST_MAX" if isinstance(node, exp.ArrayMax) else "LIST_MIN"
    return sqlglot.parse_one(
        f"LIST_TRANSFORM([{aggregate}({keyed})], _sfemu_pick -> _sfemu_pick.v)[1]",
        read=WRITE_DIALECT,
    )


def _rewrite_ordering_function(node: exp.Expression) -> exp.Expression:
    """``ARRAY_SORT``/``ARRAY_MAX``/``ARRAY_MIN`` over a JSON or native list only.

    DuckDB's ``list_sort``/``list_max``/``list_min`` do not order ``VARIANT[]`` the way its
    own ``<`` operator does -- for strings it goes by length, so DuckDB returns
    ``['apple','other','datafile']`` where Snowflake returns ``['apple','datafile','other']``
    -- and DuckDB raises or crashes outright for a bare VARIANT. A JSON array (``col:path``,
    ``PARSE_JSON('[...]')``) is ordered by a type-aware key so a number compares numerically
    (see ``_rewrite_json_ordering``), and a native list keeps DuckDB's own order. A bare column
    is decided by its declared type when the caller supplied a catalog: a real DuckDB list
    column is left alone, a JSON one takes the JSON path, and only a VARIANT-typed (or
    unresolvable) one -- which DuckDB mis-orders -- is rejected.
    """
    if any(isinstance(node.args.get(name), exp.Null) for name in ("asc", "nulls_first")):
        # "If any of the input arguments is NULL, the function returns NULL."
        return exp.Null()
    array = _rewrite_operand(node.this)
    if _is_json_array_value(array):
        return _rewrite_json_ordering(node, _json_list(array) or array)
    declared = _declared_column_type(array)
    if declared is not None:
        if _is_json_list_type(declared):
            return _rewrite_json_ordering(
                node, exp.cast(array, exp.DataType.build("JSON[]"), dialect=WRITE_DIALECT)
            )
        if _is_list_type(declared) and not _is_variant_list_type(declared):
            # A real DuckDB list -- typically a CTAS ``ARRAY_CONSTRUCT`` column -- keeps its
            # own element type, so DuckDB's order is the one Snowflake gives as well.
            node.set("this", array)
            return node
    if (
        isinstance(array, (exp.Column, exp.Placeholder, exp.Parameter))
        or _is_variant_array_value(array)
        or _has_variant_list(array)
        or _aggregates_to_variant_list(array)
    ):
        name = _ORDERING_NAMES.get(type(node), type(node).__name__)
        raise TranslationError(
            f"{name} over a VARIANT-typed array is not supported by the emulator: DuckDB "
            "orders VARIANT values differently from Snowflake. Write the array as an "
            "ARRAY_CONSTRUCT list or read it through a JSON path (j:path), where a number "
            "array compares numerically."
        )
    node.set("this", array)
    return node


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
    the end, which is what Snowflake does too. A bound is named at least twice by the
    adjustment, which is fine for a literal, a column or a named bind, but not for a
    positional ``?`` -- only one value is sent per placeholder -- so that case goes through a
    one-element list whose lambda parameter is what gets repeated.
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
        # A positional `?` may only appear once in the statement, and the adjustment names the
        # bound twice, so the value is named through a one-element list: the placeholder is
        # written once and the lambda parameter carries the repeated uses.
        return sqlglot.parse_one(
            f"LIST_TRANSFORM([{_sql(bound)}], _sfemu_bound -> "
            f"_sfemu_bound + CAST(_sfemu_bound >= 0 AS INTEGER))[1]",
            read=WRITE_DIALECT,
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
    operand = _array_list(_rewrite_operand(node.this))
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
    array = _array_list(_rewrite_operand(node.this))
    value = _rewrite_operand(node.args.get("expression"))
    # A VARIANT value compares in VARIANT space and a JSON array in JSON space; against a
    # native list the value keeps its type.
    array, value = _comparable_array_and_value(array, value)
    array_sql = _sql(array)
    return sqlglot.parse_one(
        f"COALESCE(ARRAY_CONTAINS({array_sql}, {_sql(value)}), "
        f"NULLIF(ARRAY_LENGTH({array_sql}) <> LIST_COUNT({array_sql}), FALSE))",
        read=WRITE_DIALECT,
    )


def _rewrite_array_size(node: exp.Expression) -> exp.Expression:
    """``ARRAY_SIZE(col:path)``: DuckDB needs a LIST, not the JSON the array is stored as."""
    node.set("this", _array_list(_rewrite_operand(node.this)))
    return node


def _rewrite_array_position(node: exp.Expression) -> exp.Expression:
    """``ARRAY_POSITION`` is 0-based, and NULL when the value is absent.

    Snowflake documents NULL for a value the array does not contain (not 0), and the position
    of the first NULL for a NULL value, which is what DuckDB's ``array_position`` already
    returns -- for a ``JSON[]`` list, where a JSON null element is the match.
    """
    array = _array_list(_rewrite_operand(node.this))
    array, value = _comparable_array_and_value(
        array, _rewrite_operand(node.args.get("expression"))
    )
    return sqlglot.parse_one(
        f"ARRAY_POSITION({_sql(array)}, {_sql(value)}) - 1",
        read=WRITE_DIALECT,
    )


def _rewrite_array_prepend(node: exp.Expression) -> exp.Expression:
    """sqlglot's ``LIST_PREPEND`` does not bind in DuckDB; build it from ``LIST_CONCAT``."""
    array = _array_list(_rewrite_operand(node.args.get("expression")))
    value = _encoded_element(_rewrite_operand(node.this), array)
    return sqlglot.parse_one(
        f"LIST_CONCAT([{_sql(value)}], {_sql(array)})", read=WRITE_DIALECT
    )


def _rewrite_array_unique(node: exp.Expression) -> exp.Expression:
    """DuckDB's ``array_unique`` returns a *count*; Snowflake's returns the distinct array."""
    if len(node.expressions) != 1:
        raise TranslationError("ARRAY_UNIQUE takes a single array argument.")
    array = _array_list(_rewrite_operand(node.expressions[0]))
    return exp.func("list_distinct", array)


def _list_kind(node: exp.Expression | None) -> str:
    """The array form a rewritten operand has: ``json``, ``variant`` or a native ``list``."""
    if _is_json_list(node):
        return "json"
    if _is_variant_list(node):
        return "variant"
    return "native"


def _encoded_array_elements(node: exp.Expression, kind: str) -> exp.Expression:
    """Present a list's elements as `kind`'s list holds them, element by element.

    A native list holds plain values, a ``JSON[]`` holds JSON values and a ``VARIANT[]``
    holds VARIANT values, and DuckDB does not unify them: joining a ``VARCHAR[]`` against a
    ``JSON[]`` casts the text to JSON first, so ``ARRAY_CAT(j:n, ARRAY_CONSTRUCT('z'))``
    raised ``Malformed JSON at byte 0`` instead of appending ``z``. Snowflake coerces the two
    arrays, so one side is re-encoded element-wise here -- ``TO_JSON`` lands a native value
    on the representation a JSON array holds (and leaves a JSON or VARIANT element as its JSON
    form), while ``_variant_element`` lands it on a normalized VARIANT.
    """
    if isinstance(node, exp.Array) and not node.expressions:
        # An empty native list carries no element type, so a lambda over it cannot bind
        # (``Invalid LIST argument during lambda function binding``). Nothing needs encoding.
        return sqlglot.parse_one(
            f"CAST([] AS {'JSON[]' if kind == 'json' else 'VARIANT[]'})",
            read=WRITE_DIALECT,
        )
    if kind == "json":
        return sqlglot.parse_one(
            f"LIST_TRANSFORM({_sql(node)}, _sfemu_element -> "
            f"TO_JSON(_sfemu_element))",
            read=WRITE_DIALECT,
        )
    element = _variant_element(exp.column("_sfemu_element"))
    return sqlglot.parse_one(
        f"CAST(LIST_TRANSFORM({_sql(node)}, _sfemu_element -> {_sql(element)}) "
        f"AS VARIANT[])",
        read=WRITE_DIALECT,
    )


def _json_from_variant_list(node: exp.Expression | None) -> exp.Expression | None:
    """Re-encode a ``VARIANT[]``'s elements as JSON, keeping them a LIST.

    DuckDB raises ``Can't compare values of type VARCHAR and type UBIGINT`` for two VARIANT
    elements of different types (``CAST('2' AS VARIANT) = CAST(2 AS VARIANT)``), while the same
    values as JSON simply compare unequal. A comparison function over a VARIANT array therefore
    compares in JSON space, where a number and the string that spells it are distinct instead
    of an error.
    """
    if node is None or not _is_variant_array_value(node):
        return node
    return sqlglot.parse_one(
        f"CAST(LIST_TRANSFORM({_sql(node)}, "
        f"_sfemu_element -> CAST(_sfemu_element AS JSON)) AS JSON[])",
        read=WRITE_DIALECT,
    )


def _comparable_pair(
    left: exp.Expression | None, right: exp.Expression | None
) -> tuple[exp.Expression | None, exp.Expression | None]:
    """Coerce two array operands onto one element kind that DuckDB can compare."""
    left, right = _coerced_array_pair(left, right)
    if _list_kind(left) == "variant" or _list_kind(right) == "variant":
        left = _json_from_variant_list(left)
        right = _json_from_variant_list(right)
    return left, right


def _coerced_array_pair(
    left: exp.Expression | None, right: exp.Expression | None
) -> tuple[exp.Expression | None, exp.Expression | None]:
    """Force two array operands onto one element kind so DuckDB can compare them.

    A pair of the same kind is left alone -- including two native lists, which keep
    sqlglot's own rendering. A JSON kind wins over a VARIANT one, so a pair that mixes them
    lands on JSON; a native list follows whichever kind the other side has.
    """
    left_kind, right_kind = _list_kind(left), _list_kind(right)
    if left_kind == right_kind:
        return left, right
    target = "json" if "json" in (left_kind, right_kind) else "variant"
    if left_kind != target and left is not None:
        left = _encoded_array_elements(left, target)
    if right_kind != target and right is not None:
        right = _encoded_array_elements(right, target)
    return left, right


def _rewrite_array_set_operation(node: exp.Expression) -> exp.Expression:
    """``ARRAY_EXCEPT``/``ARRAYS_OVERLAP`` over a coerced operand pair.

    A native pair keeps sqlglot's rendering, and with it the multiset semantics it models. A
    JSON or VARIANT pair is first coerced onto one element kind (a native list is re-encoded
    element-wise) and then rendered by sqlglot too: its zip-based multiset subtract binds a
    ``JSON[]`` and a ``VARIANT[]`` -- checked for a repeated null, an empty operand and both
    operands equal -- and gives the same answer as the rank-based form this used to build by
    hand, which was correct but about 240x slower on a 500-element array. ``ARRAYS_OVERLAP``
    uses DuckDB's own ``list_has_any``.
    """
    left = _array_list(_rewrite_operand(node.this))
    right = _array_list(_rewrite_operand(node.args.get("expression")))
    if _list_kind(left) == "native" and _list_kind(right) == "native":
        return node
    left, right = _comparable_pair(left, right)
    if isinstance(node, exp.ArrayOverlaps):
        return exp.func("list_has_any", left, right)
    node.set("this", left)
    node.set("expression", right)
    return node


def _rewrite_array_repeat(node: exp.Expression) -> exp.Expression:
    """``ARRAY_REPEAT`` has no DuckDB function; build it from ``RANGE`` and a transform."""
    if len(node.expressions) != 2:
        raise TranslationError("ARRAY_REPEAT takes a value and a count.")
    value, count = (_rewrite_operand(argument) for argument in node.expressions)
    if _is_json_value(count) or _is_variant_value(count):
        # A count read out of a VARIANT (``j:count``) needs to be a number for RANGE.
        count = exp.cast(count, exp.DataType.build("INTEGER"), dialect=WRITE_DIALECT)
    return sqlglot.parse_one(
        f"LIST_TRANSFORM(RANGE(1, {_sql(count)} + 1), _sfemu_repeat -> {_sql(value)})",
        read=WRITE_DIALECT,
    )


def _array_kind(node: exp.Expression | None) -> str:
    """The element kind of a rewritten array operand: ``json``, ``variant`` or ``native``.

    Unlike ``_list_kind`` this also recognises a rebuilt call -- an ``ARRAY_SLICE`` over a
    VARIANT array, which is an ``array_slice`` expression rather than a ``CAST``.
    """
    if _is_variant_array_value(node):
        return "variant"
    if _is_json_array_value(node):
        return "json"
    return "native"


def _flatten_one_level(previous: str, kind: str, *, guarded: bool) -> str:
    """One ``FLATTEN`` level of `previous`, a SQL expression of list kind `kind`.

    DuckDB cannot flatten a ``JSON[]`` whose elements are JSON arrays, so a JSON or VARIANT
    level casts the elements back to a list; a native list keeps DuckDB's own call.

    `guarded` leaves a level alone when its elements are no longer arrays: casting a scalar
    element to a list would raise, and the legacy depth form means "flatten at most this many
    levels", so a depth past the nesting stops instead of raising. An unguarded level is the
    documented one-argument form, where a non-ARRAY element is an error -- which is what
    DuckDB's own ``FLATTEN`` raises anyway.
    """
    if kind == "variant":
        # A VARIANT[] whose elements are VARIANT arrays -- including an ``ARRAY_SLICE`` over
        # one, which is an ``array_slice`` call rather than a ``CAST``.
        element_type, predicate = "VARIANT[]", "variant_typeof(_sfemu_outer) LIKE 'ARRAY%'"
    elif kind == "json":
        element_type, predicate = "JSON[]", "json_type(_sfemu_outer) = 'ARRAY'"
    else:
        return f"FLATTEN({previous})"
    flattened = (
        f"FLATTEN(LIST_TRANSFORM({previous}, _sfemu_inner -> "
        f"CAST(_sfemu_inner AS {element_type})))"
    )
    if not guarded:
        return flattened
    # ``LENGTH`` rather than ``LIST_COUNT``: the latter skips NULL elements, so a list holding
    # a null compared equal on both sides and was flattened anyway.
    all_arrays = (
        f"LENGTH(LIST_FILTER({previous}, _sfemu_outer -> {predicate})) "
        f"= LENGTH({previous})"
    )
    return f"CASE WHEN {all_arrays} THEN {flattened} ELSE {previous} END"


def _flatten_guarded_reduce(previous: str, kind: str, levels: int) -> str:
    """Flatten up to `levels` levels of a JSON/VARIANT list with one ``LIST_REDUCE``.

    Each level needs a guard ("is every element still an array?"), which names the list three
    times. Applying that per level nested the previous level inside all three, so the SQL text
    *tripled* per level: a depth of 10 produced 108 MB of SQL and took over five minutes to
    translate, which one request could turn into a worker hang. Here the list is named twice --
    as the reduce's initial value and once inside the list of one-per-step copies -- and the
    per-level work is written against the reduce's accumulator, so the text grows by the reduce
    skeleton only and the depth costs one ``FLATTEN`` per step at run time.
    """
    if kind == "variant":
        element_type = "VARIANT[]"
        predicate = "variant_typeof(_sfemu_outer) LIKE 'ARRAY%'"
    else:
        element_type = "JSON[]"
        predicate = "json_type(_sfemu_outer) = 'ARRAY'"
    # ``LENGTH`` rather than ``LIST_COUNT``: the latter skips NULL elements, so a list holding
    # a null compared equal on both sides and was flattened anyway.
    all_arrays = (
        f"LENGTH(LIST_FILTER(acc, _sfemu_outer -> {predicate})) = LENGTH(acc)"
    )
    flattened = (
        f"FLATTEN(LIST_TRANSFORM(acc, _sfemu_inner -> "
        f"CAST(_sfemu_inner AS {element_type})))"
    )
    return (
        f"LIST_REDUCE(LIST_TRANSFORM(RANGE(1, {levels} + 1), _sfemu_step -> {previous}), "
        f"(acc, _sfemu_step) -> CASE WHEN {all_arrays} THEN {flattened} ELSE acc END, "
        f"{previous})"
    )


def _rewrite_array_flatten(node: exp.Expression) -> exp.Expression:
    """``ARRAY_FLATTEN``: the documented one-argument form, plus the legacy constant depth.

    Snowflake's current signature is ``ARRAY_FLATTEN(array)``: it removes one level of nesting,
    reports an error when an element is not an ARRAY, and returns NULL when the array is NULL or
    holds a NULL element. The emulator also accepts the older ``ARRAY_FLATTEN(array, depth)``
    form, which removes up to `depth` levels and stops when the elements are no longer arrays.

    A computed depth is rejected: the documented signature has no depth argument, so there is no
    behaviour to be faithful to, and DuckDB has no recursive list flatten to lower it to.
    """
    array = _array_list(_rewrite_operand(node.expressions[0]))
    kind, sql = _array_kind(array), _sql(array)
    if len(node.expressions) == 1:
        # The documented form: one level, an error for a non-ARRAY element, NULL for a NULL
        # array or a NULL element.
        return sqlglot.parse_one(
            f"CASE WHEN {sql} IS NULL OR LENGTH(LIST_FILTER({sql}, "
            f"_sfemu_outer -> _sfemu_outer IS NULL)) > 0 THEN NULL "
            f"ELSE {_flatten_one_level(sql, kind, guarded=False)} END",
            read=WRITE_DIALECT,
        )
    if len(node.expressions) != 2:
        raise TranslationError(
            "ARRAY_FLATTEN takes an array, and optionally a constant depth."
        )
    depth = node.expressions[1]
    if not (isinstance(depth, exp.Literal) and not depth.is_string and depth.name.isdigit()):
        raise TranslationError(
            "ARRAY_FLATTEN with a computed depth is not supported by the emulator; "
            "Snowflake's documented form is ARRAY_FLATTEN(array), which removes one level of "
            "nesting."
        )
    levels = int(depth.name)
    if not 1 <= levels <= _ARRAY_FLATTEN_MAX_DEPTH:
        raise TranslationError(
            f"ARRAY_FLATTEN depth must be between 1 and {_ARRAY_FLATTEN_MAX_DEPTH}."
        )
    if kind == "native":
        for _ in range(levels):
            sql = _flatten_one_level(sql, kind, guarded=True)
        return sqlglot.parse_one(sql, read=WRITE_DIALECT)
    return sqlglot.parse_one(_flatten_guarded_reduce(sql, kind, levels), read=WRITE_DIALECT)


def _rewrite_array_to_string(node: exp.Expression) -> exp.Expression:
    """``ARRAY_TO_STRING``: unquote JSON elements, and return NULL for a NULL separator.

    sqlglot casts each element with ``CAST(x AS TEXT)``, which keeps a JSON string's quotes, so
    a JSON array is rebuilt from ``e ->> '$'`` -- the value's text, which also renders a nested
    array or object as its JSON text and a number as DuckDB formats it -- with a NULL element
    coalesced to an empty string, as Snowflake documents. DuckDB treats a NULL separator as an
    empty one where Snowflake returns NULL for a NULL argument, so the separator is checked the
    way sqlglot's own rendering checks it.
    """
    array = _array_list(_rewrite_operand(node.this))
    known = {"this", "expression", "null_is_empty", "null_delim_is_null"}
    if set(node.args) - known:
        return node
    separator_node = node.args.get("expression")
    positional = isinstance(separator_node, exp.Placeholder) and separator_node.this is None
    separator = _sql(separator_node, chr(39) + ',' + chr(39))
    if _is_variant_array_value(array):
        # VARIANT elements are already the values themselves, so sqlglot's ``CAST(x AS
        # TEXT)`` rendering returns their text; only the array needs a list form.
        node.set("this", array)
        rendered = f"ARRAY_TO_STRING({_sql(array)}, {separator})"
    elif _is_json_array_value(array):
        rendered = (
            f"ARRAY_TO_STRING(LIST_TRANSFORM({_sql(_json_list(array))}, "
            f"_sfemu_element -> COALESCE(_sfemu_element ->> '$', '')), {separator})"
        )
    else:
        return node
    if positional:
        # A positional `?` may appear once, so the NULL check is dropped rather than repeating
        # it: a NULL separator is then treated as an empty one, where Snowflake returns NULL.
        # Repeating the placeholder instead would make one bind need two values, which broke
        # `ARRAY_TO_STRING(j:n, ?)` that used to work.
        return sqlglot.parse_one(rendered, read=WRITE_DIALECT)
    # The separator is named twice, exactly as sqlglot's own rendering of this call does, so a
    # NULL separator still returns NULL; a literal, a column or a named bind works.
    return sqlglot.parse_one(
        f"CASE WHEN {separator} IS NULL THEN NULL ELSE {rendered} END",
        read=WRITE_DIALECT,
    )


def _rewrite_array_remove(node: exp.Expression) -> exp.Expression:
    """``ARRAY_REMOVE`` drops the value, not every NULL element.

    sqlglot renders ``LIST_FILTER(array, e -> e <> value)``. A comparison against a NULL
    element is NULL, so the filter dropped it: ``ARRAY_REMOVE(['a',null,'b'], 'a')`` returned
    ``['b']`` where Snowflake returns ``[null,'b']`` (the documented example keeps the NULL
    element). ``IS DISTINCT FROM`` keeps every element that is not the value, and the
    documented NULL *value* behaviour -- the whole call is NULL -- is kept as the CASE sqlglot
    emits, since a NULL value means "return NULL", not "remove the NULLs".
    """
    array = _array_list(_rewrite_json_array_functions(node.this))
    value = _rewrite_json_array_functions(node.args.get("expression"))
    if _is_variant_array_value(array) and not _is_variant_typed(value):
        # A VARIANT array with a plain value compares in JSON space, like the other comparison
        # functions (a VARIANT-vs-VARIANT comparison of different types raises).
        array = _json_from_variant_list(array)
    array, value = _comparable_array_and_value(array, value)
    array_sql, value_sql = _sql(array), _sql(value)
    return sqlglot.parse_one(
        f"CASE WHEN {value_sql} IS NULL THEN NULL ELSE "
        f"LIST_FILTER({array_sql}, _sfemu_element -> "
        f"_sfemu_element IS DISTINCT FROM {value_sql}) END",
        read=WRITE_DIALECT,
    )


def _coerce_array_operands(node: exp.Expression) -> exp.Expression:
    """Coerce a two-array call's operands onto one element kind.

    ``ARRAY_CAT`` nests its operands as ``this`` plus ``expressions`` while
    ``ARRAY_INTERSECTION`` keeps both in ``expressions``; each further operand is coerced
    against the first so a chain of mixed operands settles on one kind.
    """
    if isinstance(node, exp.ArrayConcat):
        operands = [node.this, *(node.args.get("expressions") or [])]
    else:
        operands = list(node.args.get("expressions") or [])
    converted = [
        _array_list(_rewrite_operand(operand))
        if isinstance(operand, exp.Expression)
        else operand
        for operand in operands
    ]
    compare = isinstance(node, exp.ArrayIntersect)
    for index in range(1, len(converted)):
        pair = _comparable_pair if compare else _coerced_array_pair
        converted[0], converted[index] = pair(converted[0], converted[index])
    if isinstance(node, exp.ArrayConcat):
        node.set("this", converted[0])
        node.set("expressions", converted[1:])
    else:
        node.set("expressions", converted)
    return node


def _rewrite_lambda_call(node: exp.Expression) -> exp.Expression:
    """``FILTER``/``TRANSFORM``/``REDUCE``: hand the lambda VARIANT elements, as Snowflake does.

    A Snowflake lambda over a semi-structured array receives a VARIANT element, so a body may
    compare it with a plain literal (``FILTER(j:n, x -> x = 'a')``) and Snowflake coerces the
    literal. A DuckDB ``JSON[]`` element is JSON instead, and comparing it with a ``VARCHAR``
    tries to parse the text as JSON, so the call raised ``Malformed JSON at byte 0``. Presenting
    the array as ``VARIANT[]`` of normalized elements makes DuckDB coerce the literal the way
    Snowflake does. A native list keeps its own element type, which is what Snowflake gives a
    structured-array lambda too.
    """
    array = _array_list(_rewrite_operand(node.this))
    if _is_json_array_value(array):
        array = _encoded_array_elements(array, "variant")
    node.set("this", array)
    return node


def _cast_array_operands(node: exp.Expression) -> exp.Expression:
    """Cast the array operand(s) of a call sqlglot otherwise renders faithfully.

    A searched value is encoded with ``TO_JSON``/``CAST(... AS VARIANT)`` when the array it
    is compared against is JSON/VARIANT, so the comparison happens between VARIANT values,
    as Snowflake does. A call that takes two arrays (``ARRAY_CAT``, ``ARRAY_INTERSECTION``)
    also has its operands coerced onto one element kind, so a native list can be combined with
    a JSON or VARIANT one.
    """
    if isinstance(node, (exp.ArrayFilter, exp.Transform, exp.Reduce)):
        return _rewrite_lambda_call(node)
    if isinstance(node, (exp.ArrayConcat, exp.ArrayIntersect)):
        return _coerce_array_operands(node)
    if isinstance(node, exp.ArrayAgg):
        # The argument is a value to aggregate, not an array to index into: only a JSON
        # value is cast, so it aggregates as one element the way Snowflake does.
        node.set("this", _json_list(_rewrite_operand(node.this)))
        return node
    array = None
    if isinstance(node.this, exp.Expression):
        array = _array_list(_rewrite_operand(node.this))
        node.set("this", array)
    if isinstance(node, _ARRAY_VALUE_FUNCTIONS):
        value = node.args.get("expression")
        if isinstance(value, exp.Expression):
            node.set(
                "expression",
                _encoded_element(_rewrite_operand(value), array),
            )
    if node.args.get("expressions"):
        node.set(
            "expressions",
            [
                _array_list(_rewrite_operand(e))
                if isinstance(e, exp.Expression)
                else e
                for e in node.args["expressions"]
            ],
        )
    return node


def _rewrite_json_array_function(node: exp.Expression) -> exp.Expression:
    """Make a Snowflake array function work over a VARIANT/JSON array.

    A Snowflake array the emulator stores as DuckDB JSON (``col:path``, ``$1:path``,
    ``PARSE_JSON('[...]')``) is rejected by DuckDB's list functions, which take a LIST, so
    the array operand is cast to ``JSON[]``; a VARIANT-typed array (``x::VARIANT``,
    ``TO_VARIANT('[...]')``, a bare VARIANT column) is cast to ``VARIANT[]``, a real DuckDB
    list whose elements keep their VARIANT comparison semantics. A searched value is encoded
    with ``TO_JSON``/``CAST(... AS VARIANT)`` so both sides compare as VARIANT values the way
    Snowflake compares them. DuckDB's own ``VARIANT`` -> ``JSON[]`` cast is no substitute:
    for a VARIANT built from text it returns unquoted element text (``ada`` for ``"ada"``),
    so the search would compare unequal against ``TO_JSON(...)``. The functions whose sqlglot
    mapping is wrong for *any* operand (indexing, ``SLICE``, ``ARRAY_UNIQUE``,
    ``ARRAY_REPEAT``, ``ARRAY_FLATTEN``) are rebuilt.

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
    if isinstance(node, (exp.ArrayMax, exp.ArrayMin, exp.SortArray)):
        return _rewrite_ordering_function(node)
    name = _array_call_name(node)
    if name == "ARRAY_UNIQUE":
        return _rewrite_array_unique(node)
    if name == "ARRAY_REPEAT":
        return _rewrite_array_repeat(node)
    if name == _FLATTEN_MARKER:
        return _rewrite_array_flatten(node)
    if isinstance(node, exp.ArrayRemove):
        return _rewrite_array_remove(node)
    if isinstance(node, exp.ArrayContains):
        return _rewrite_array_contains(node)
    if isinstance(node, exp.ArraySize):
        return _rewrite_array_size(node)
    if isinstance(node, _JSON_ARRAY_FUNCTIONS):
        return _cast_array_operands(node)
    # Anything else -- including an uninterpreted call such as a Snowflake function sqlglot
    # does not know -- is left for DuckDB, whose error names it.
    return node


def _rewrite_operand(node: exp.Expression | None) -> exp.Expression | None:
    """Rewrite an operand with the resolver still in scope.

    A rebuild needs its operands in their final form before it serializes them, so it re-enters
    the pass on that subtree. The re-entry must carry the column-type resolver: entering with
    the default ``None`` made a nested call treat a bare native-list column as an unknown
    VARIANT, so ``ARRAY_SORT(ARRAY_UNIQUE(n))`` was rejected while ``ARRAY_SORT(n)`` worked.
    """
    if node is None:
        return None
    return _rewrite_json_array_functions(node, _column_types.get())


def _rewrite_json_array_functions(
    statement: exp.Expression, column_types: ColumnTypeResolver | None = None
) -> exp.Expression:
    """Apply ``_rewrite_json_array_function`` to every array call in `statement`.

    A separate pass over a snapshot, because ``transform`` visits a parent before its
    children: rebuilding a call from its operands' SQL requires those operands to have
    been rewritten first, so an operand that is itself a rewritten call is handled
    recursively (``_rewrite_json_array_function``). A node detached by an ancestor's
    replacement is skipped.

    `column_types` answers a bare column's DuckDB type for the call sites that turn on one --
    ``ARRAY_SORT``/``ARRAY_MAX``/``ARRAY_MIN`` and the list form an operand is cast to. It is
    published for the duration of the pass (the rewrites are otherwise pure AST transforms)
    and restored afterwards, so a nested call from a rebuild reuses the same resolver.
    """
    # Snowflake documents SLICE as an alias of ARRAY_SLICE, but sqlglot leaves it as an
    # uninterpreted call that DuckDB rejects. Nested calls are converted by the loop too:
    # a converted call holds the live child nodes, so an inner SLICE is still in the tree.
    for alias in list(statement.find_all(exp.Anonymous)):
        if alias.name.upper() == "SLICE" and len(alias.expressions) == 3:
            array, start, end = alias.expressions
            alias.replace(exp.ArraySlice(this=array, start=start, end=end, zero_based=True))

    root = statement
    scoped = _scoped_names.get() | {
        name.upper()
        for name in (
            *[alias.alias for alias in statement.find_all(exp.Alias) if alias.alias],
            *[cte.alias for cte in statement.find_all(exp.CTE) if cte.alias],
            *[
                table.alias
                for table in statement.find_all(exp.Table)
                if table.alias
            ],
        )
        if name
    }
    token = _column_types.set(column_types)
    scoped_token = _scoped_names.set(frozenset(scoped))
    try:
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
    finally:
        _scoped_names.reset(scoped_token)
        _column_types.reset(token)
    return root
