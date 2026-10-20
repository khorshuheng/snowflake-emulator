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
    """The DuckDB type of a bare column, when the caller supplied a catalog lookup."""
    resolver = _column_types.get()
    if resolver is None or not isinstance(node, exp.Column):
        return None
    return resolver(node.table or None, node.name)


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


def _has_variant_list(node: exp.Expression) -> bool:
    """True when `node` holds a ``VARIANT[]`` cast -- the array functions' mark for one."""
    return _is_variant_list(node) or any(
        _is_variant_list(cast) for cast in node.find_all(exp.Cast)
    )


def _rewrite_ordering_function(node: exp.Expression) -> exp.Expression:
    """``ARRAY_SORT``/``ARRAY_MAX``/``ARRAY_MIN`` over a JSON or native list only.

    DuckDB's ``list_sort``/``list_max``/``list_min`` do not order ``VARIANT[]`` the way its
    own ``<`` operator does -- for strings it goes by length, so DuckDB returns
    ``['apple','other','datafile']`` where Snowflake returns ``['apple','datafile','other']``
    -- and DuckDB raises or crashes outright for a bare VARIANT. A JSON array (``col:path``,
    ``PARSE_JSON('[...]')``) sorts lexicographically -- right for strings, wrong for numbers,
    which come back in text order (``[10,2,33]`` stays ``[10,2,33]``) -- and a native list
    keeps DuckDB's own order. A bare column is decided by its declared type when the caller
    supplied a catalog: a real DuckDB list column is left alone, a JSON one is cast, and only
    a VARIANT-typed (or unresolvable) one -- which DuckDB mis-orders -- is rejected.
    """
    array = _rewrite_json_array_functions(node.this)
    if _is_json_array_value(array):
        node.set("this", _json_list(array))
        return node
    declared = _declared_column_type(array)
    if declared is not None:
        if _is_json_list_type(declared):
            # A JSON column sorts lexicographically, like a ``col:path`` operand.
            node.set("this", exp.cast(array, exp.DataType.build("JSON[]"), dialect=WRITE_DIALECT))
            return node
        if _is_list_type(declared) and not _is_variant_list_type(declared):
            # A real DuckDB list -- typically a CTAS ``ARRAY_CONSTRUCT`` column -- keeps its
            # own element type, so DuckDB's order is the one Snowflake gives as well.
            node.set("this", array)
            return node
    if (
        isinstance(array, (exp.Column, exp.Placeholder, exp.Parameter))
        or _is_variant_array_value(array)
        or _has_variant_list(array)
    ):
        name = type(node).__name__
        name = {
            exp.ArrayMax: "ARRAY_MAX",
            exp.ArrayMin: "ARRAY_MIN",
            exp.SortArray: "ARRAY_SORT",
        }.get(type(node), name)
        raise TranslationError(
            f"{name} over a VARIANT-typed array is not supported by the emulator: DuckDB "
            "orders VARIANT values differently from Snowflake. Write the array as an "
            "ARRAY_CONSTRUCT list (a col:path JSON array sorts lexicographically, which is "
            "wrong for numbers)."
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
    operand = _array_list(_rewrite_json_array_functions(node.this))
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
    array = _array_list(_rewrite_json_array_functions(node.this))
    value = _rewrite_json_array_functions(node.args.get("expression"))
    # A VARIANT/JSON element is encoded so both sides compare as VARIANT values; against a
    # native list the value keeps its type.
    value = _encoded_element(value, array)
    array_sql = _sql(array)
    return sqlglot.parse_one(
        f"COALESCE(ARRAY_CONTAINS({array_sql}, {_sql(value)}), "
        f"NULLIF(ARRAY_LENGTH({array_sql}) <> LIST_COUNT({array_sql}), FALSE))",
        read=WRITE_DIALECT,
    )


def _rewrite_array_size(node: exp.Expression) -> exp.Expression:
    """``ARRAY_SIZE(col:path)``: DuckDB needs a LIST, not the JSON the array is stored as."""
    node.set("this", _array_list(_rewrite_json_array_functions(node.this)))
    return node


def _rewrite_array_position(node: exp.Expression) -> exp.Expression:
    """``ARRAY_POSITION`` is 0-based, and NULL when the value is absent.

    Snowflake documents NULL for a value the array does not contain (not 0), and the position
    of the first NULL for a NULL value, which is what DuckDB's ``array_position`` already
    returns -- for a ``JSON[]`` list, where a JSON null element is the match.
    """
    array = _array_list(_rewrite_json_array_functions(node.this))
    value = _encoded_element(
        _rewrite_json_array_functions(node.args.get("expression")), array
    )
    return sqlglot.parse_one(
        f"ARRAY_POSITION({_sql(array)}, {_sql(value)}) - 1",
        read=WRITE_DIALECT,
    )


def _rewrite_array_prepend(node: exp.Expression) -> exp.Expression:
    """sqlglot's ``LIST_PREPEND`` does not bind in DuckDB; build it from ``LIST_CONCAT``."""
    array = _array_list(_rewrite_json_array_functions(node.args.get("expression")))
    value = _encoded_element(_rewrite_json_array_functions(node.this), array)
    return sqlglot.parse_one(
        f"LIST_CONCAT([{_sql(value)}], {_sql(array)})", read=WRITE_DIALECT
    )


def _rewrite_array_unique(node: exp.Expression) -> exp.Expression:
    """DuckDB's ``array_unique`` returns a *count*; Snowflake's returns the distinct array."""
    if len(node.expressions) != 1:
        raise TranslationError("ARRAY_UNIQUE takes a single array argument.")
    array = _array_list(_rewrite_json_array_functions(node.expressions[0]))
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


def _multiset_except_sql(left: str, right: str) -> str:
    """``ARRAY_EXCEPT`` as Snowflake's multiset subtract: a value keeps ``N - M`` copies.

    Snowflake removes as many copies of a value as the excluded array holds, not every copy
    (``['a','b','b','b','c']`` except ``['b']`` keeps two ``b``s), so a plain
    ``LIST_FILTER(... NOT LIST_CONTAINS(...))`` is wrong. A copy is kept when its occurrence
    index (its rank among the equal elements of the source, counted from the front) is at
    most ``N - M``; the rank is the length of the same-valued prefix, which is what the inner
    ``LIST_FILTER`` over ``ARRAY_SLICE`` measures. Each operand is therefore named several
    times, which a bound ``?`` operand does not survive -- the same limitation the other
    rebuilt calls have.

    The comparison is ``IS NOT DISTINCT FROM``, not ``=``, because the function is NULL-safe
    and a JSON null element of a ``JSON[]`` *is* an SQL NULL to DuckDB: with ``=`` every
    comparison against it is NULL, so the copy was never counted on either side and the
    multiset arithmetic was wrong for a repeated null (``['a',null,null]`` except ``[null]``
    kept none of them, where Snowflake keeps one).
    """
    rank = (
        f"LENGTH(LIST_FILTER(ARRAY_SLICE({left}, 1, _sfemu_index), "
        f"_sfemu_seen -> _sfemu_seen IS NOT DISTINCT FROM {left}[_sfemu_index]))"
    )
    budget = (
        f"LENGTH(LIST_FILTER({left}, _sfemu_all -> "
        f"_sfemu_all IS NOT DISTINCT FROM {left}[_sfemu_index])) "
        f"- LENGTH(LIST_FILTER({right}, _sfemu_excluded -> "
        f"_sfemu_excluded IS NOT DISTINCT FROM {left}[_sfemu_index]))"
    )
    return (
        f"LIST_TRANSFORM(LIST_FILTER(RANGE(1, LENGTH({left}) + 1), "
        f"_sfemu_index -> {rank} <= {budget}), "
        f"_sfemu_index -> {left}[_sfemu_index])"
    )


def _rewrite_array_set_operation(node: exp.Expression) -> exp.Expression:
    """``ARRAY_EXCEPT``/``ARRAYS_OVERLAP``: sqlglot's rendering does not bind a JSON[]/VARIANT[].

    A native pair keeps sqlglot's rendering (and with it the multiset semantics it models);
    a JSON or VARIANT pair is rebuilt, where DuckDB has no equivalent for the ``&&`` operator
    or the zip-based except -- ``_multiset_except_sql`` reproduces the multiset subtract. A
    *mixed* pair -- one JSON/VARIANT operand and one native list -- is coerced element-wise
    onto one kind first (see ``_coerced_array_pair``), because DuckDB compares a JSON or
    VARIANT element against an ``INT`` or ``VARCHAR`` element as unequal.
    """
    left = _array_list(_rewrite_json_array_functions(node.this))
    right = _array_list(_rewrite_json_array_functions(node.args.get("expression")))
    left_kind, right_kind = _list_kind(left), _list_kind(right)
    if left_kind == right_kind == "native":
        return node
    left, right = _coerced_array_pair(left, right)
    if isinstance(node, exp.ArrayOverlaps):
        return exp.func("list_has_any", left, right)
    return sqlglot.parse_one(
        _multiset_except_sql(_sql(left), _sql(right)), read=WRITE_DIALECT
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
    array = _array_list(_rewrite_json_array_functions(array))
    sql = _sql(array)
    for _ in range(levels):
        if _is_variant_array_value(array):
            # A VARIANT[] whose elements are VARIANT arrays -- including an ``ARRAY_SLICE``
            # over one, which is an ``array_slice`` call rather than a ``CAST`` -- casts each
            # element back to VARIANT[] a level at a time. A level whose elements are no
            # longer arrays is left alone, where Snowflake simply stops flattening.
            all_arrays = (
                f"LIST_COUNT(LIST_FILTER({sql}, _sfemu_outer -> "
                f"variant_typeof(_sfemu_outer) LIKE 'ARRAY%')) = LIST_COUNT({sql})"
            )
            flattened = (
                f"FLATTEN(LIST_TRANSFORM({sql}, "
                f"_sfemu_inner -> CAST(_sfemu_inner AS VARIANT[])))"
            )
            sql = f"CASE WHEN {all_arrays} THEN {flattened} ELSE {sql} END"
            continue
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
    array = _array_list(_rewrite_json_array_functions(node.this))
    known = {"this", "expression", "null_is_empty", "null_delim_is_null"}
    if set(node.args) - known:
        return node
    if _is_variant_array_value(array):
        # VARIANT elements are already the values themselves, so sqlglot's ``CAST(x AS
        # TEXT)`` rendering returns their text; only the array needs a list form.
        node.set("this", array)
        return node
    if not _is_json_array_value(array):
        return node
    return sqlglot.parse_one(
        f"ARRAY_TO_STRING(LIST_TRANSFORM({_sql(_json_list(array))}, "
        f"_sfemu_element -> COALESCE(_sfemu_element ->> '$', '')), "
        f"{_sql(node.args.get('expression'), chr(39) + ',' + chr(39))})",
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
        _array_list(_rewrite_json_array_functions(operand))
        if isinstance(operand, exp.Expression)
        else operand
        for operand in operands
    ]
    for index in range(1, len(converted)):
        converted[0], converted[index] = _coerced_array_pair(
            converted[0], converted[index]
        )
    if isinstance(node, exp.ArrayConcat):
        node.set("this", converted[0])
        node.set("expressions", converted[1:])
    else:
        node.set("expressions", converted)
    return node


def _cast_array_operands(node: exp.Expression) -> exp.Expression:
    """Cast the array operand(s) of a call sqlglot otherwise renders faithfully.

    A searched value is encoded with ``TO_JSON``/``CAST(... AS VARIANT)`` when the array it
    is compared against is JSON/VARIANT, so the comparison happens between VARIANT values,
    as Snowflake does. A call that takes two arrays (``ARRAY_CAT``, ``ARRAY_INTERSECTION``)
    also has its operands coerced onto one element kind, so a native list can be combined with
    a JSON or VARIANT one.
    """
    if isinstance(node, (exp.ArrayConcat, exp.ArrayIntersect)):
        return _coerce_array_operands(node)
    if isinstance(node, exp.ArrayAgg):
        # The argument is a value to aggregate, not an array to index into: only a JSON
        # value is cast, so it aggregates as one element the way Snowflake does.
        node.set("this", _json_list(_rewrite_json_array_functions(node.this)))
        return node
    array = None
    if isinstance(node.this, exp.Expression):
        array = _array_list(_rewrite_json_array_functions(node.this))
        node.set("this", array)
    if isinstance(node, _ARRAY_VALUE_FUNCTIONS):
        value = node.args.get("expression")
        if isinstance(value, exp.Expression):
            node.set(
                "expression",
                _encoded_element(_rewrite_json_array_functions(value), array),
            )
    if node.args.get("expressions"):
        node.set(
            "expressions",
            [
                _array_list(_rewrite_json_array_functions(e))
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
    if isinstance(node, exp.ArrayContains):
        return _rewrite_array_contains(node)
    if isinstance(node, exp.ArraySize):
        return _rewrite_array_size(node)
    if isinstance(node, _JSON_ARRAY_FUNCTIONS):
        return _cast_array_operands(node)
    # Anything else -- including an uninterpreted call such as a Snowflake function sqlglot
    # does not know -- is left for DuckDB, whose error names it.
    return node


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
    token = _column_types.set(column_types)
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
        _column_types.reset(token)
    return root
