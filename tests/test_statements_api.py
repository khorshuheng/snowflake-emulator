"""End-to-end tests for the Snowflake SQL API v2 emulation."""

from __future__ import annotations

import datetime

import gzip
import json
import os


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_simple_select(client):
    resp = client.post("/api/v2/statements", json={"statement": "SELECT 1 AS x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == [[1]]
    assert body["resultSetMetaData"]["rowType"][0]["name"] == "X"
    assert body["resultSetMetaData"]["rowType"][0]["type"] == "fixed"


def test_current_version_returns_numeric_version(client):
    resp = client.post("/api/v2/statements", json={"statement": "SELECT CURRENT_VERSION()"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == [["8.23.1"]]
    assert body["resultSetMetaData"]["rowType"][0]["type"] == "text"


def test_describe_table_returns_snowflake_types(client):
    stmt = (
        "CREATE TABLE dt (a INT, b VARCHAR, c DOUBLE, d BOOLEAN, e DATE, "
        "f TIMESTAMP, g DECIMAL(10,2), h VARCHAR[])"
    )
    resp = client.post("/api/v2/statements", json={"statement": stmt})
    assert resp.status_code == 200

    resp = client.post("/api/v2/statements", json={"statement": "DESCRIBE TABLE dt"})
    assert resp.status_code == 200
    body = resp.json()
    types = {row[0]: row[1] for row in body["data"]}
    assert types == {
        "A": "NUMBER(38,0)",
        "B": "VARCHAR",
        "C": "FLOAT",
        "D": "BOOLEAN",
        "E": "DATE",
        "F": "TIMESTAMP_NTZ(6)",
        "G": "NUMBER(10,2)",
        "H": "ARRAY",
    }


def test_multi_statement_ddl_dml_query(client):
    stmt = (
        "CREATE TABLE t (a INT, b VARCHAR); "
        "INSERT INTO t VALUES (1, 'x'), (2, 'y'); "
        "SELECT * FROM t ORDER BY a"
    )
    # Chaining statements requires opting in: Snowflake's MULTI_STATEMENT_COUNT defaults to 1.
    resp = client.post(
        "/api/v2/statements",
        json={"statement": stmt, "parameters": {"MULTI_STATEMENT_COUNT": "0"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == [[1, "x"], [2, "y"]]


def test_snowflake_qualify_and_ilike(client):
    stmt = (
        "CREATE TABLE u (name VARCHAR, grp INT, val INT); "
        "INSERT INTO u VALUES ('Foo', 1, 10), ('Bar', 1, 20); "
        "SELECT name, grp, val FROM u WHERE name ILIKE '%oo%' "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY grp ORDER BY val DESC) = 1"
    )
    resp = client.post(
        "/api/v2/statements",
        json={"statement": stmt, "parameters": {"MULTI_STATEMENT_COUNT": "0"}},
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == [["Foo", 1, 10]]


def test_variant_and_array_types(client):
    stmt = (
        "SELECT OBJECT_CONSTRUCT('a', 1, 'b', 'x') AS v, "
        "ARRAY_CONSTRUCT(1, 2, 3) AS arr"
    )
    resp = client.post("/api/v2/statements", json={"statement": stmt})
    assert resp.status_code == 200
    body = resp.json()
    row_type = body["resultSetMetaData"]["rowType"]
    assert row_type[0]["type"] == "object"
    assert row_type[1]["type"] == "array"
    assert body["data"] == [[{"a": 1, "b": "x"}, [1, 2, 3]]]


def test_array_functions_over_variant_arrays(client):
    """A VARIANT array (`col:path`) must work with ARRAY_CONTAINS/ARRAY_SIZE.

    The emulator stores a semi-structured array as DuckDB JSON, while DuckDB's list
    functions take a LIST, so `ARRAY_CONTAINS(v, col:path)` used to fail with
    `No function matches ... 'array_length(JSON)'`.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE holds (j VARIANT, x VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO holds SELECT PARSE_JSON('{\"names\": [\"ada\", \"bob\"]}'), "
            "TO_VARIANT('ada')"
        ).status_code
        == 200
    )

    for statement, expected in (
        ("SELECT ARRAY_CONTAINS('ada', j:names) FROM holds", [[True]]),
        ("SELECT ARRAY_CONTAINS('zoe', j:names) FROM holds", [[False]]),
        ("SELECT ARRAY_CONTAINS('ada'::VARIANT, j:names) FROM holds", [[True]]),
        ("SELECT ARRAY_CONTAINS(TO_VARIANT('ada'), j:names) FROM holds", [[True]]),
        ("SELECT ARRAY_CONTAINS(x, j:names) FROM holds", [[True]]),
        ("SELECT ARRAY_CONTAINS('ada', PARSE_JSON('[\"ada\"]'))", [[True]]),
        ("SELECT ARRAY_CONTAINS('ada', j:missing) FROM holds", [[None]]),
        ("SELECT ARRAY_SIZE(j:names) FROM holds", [[2]]),
        ("SELECT ARRAY_SIZE(j:missing) FROM holds", [[None]]),
        # Native DuckDB lists must keep working untouched.
        ("SELECT ARRAY_CONTAINS('ada', ARRAY_CONSTRUCT('ada', 'bob'))", [[True]]),
        ("SELECT ARRAY_SIZE(ARRAY_CONSTRUCT(1, 2, 3))", [[3]]),
    ):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        assert resp.json()["data"] == expected, statement


def test_array_contains_with_a_bound_value(client):
    """A bound value must be one parameter: sqlglot's own rendering names it twice."""
    assert (
        client.post(
            "/api/v2/statements",
            json={"statement": "CREATE TABLE bound (j VARIANT)"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v2/statements",
            json={
                "statement": "INSERT INTO bound SELECT "
                "PARSE_JSON('{\"tags\": [\"ada\", \"bob\"]}')"
            },
        ).status_code
        == 200
    )

    for value, expected in (("ada", [[True]]), ("zoe", [[False]])):
        resp = client.post(
            "/api/v2/statements",
            json={
                "statement": "SELECT ARRAY_CONTAINS(?, j:tags) FROM bound",
                "bindings": {"1": {"type": "TEXT", "value": value}},
            },
        )
        assert resp.status_code == 200, resp.json()
        assert resp.json()["data"] == expected

    filtered = client.post(
        "/api/v2/statements",
        json={
            "statement": "SELECT ARRAY_SIZE(j:tags) AS n FROM bound "
            "WHERE ARRAY_CONTAINS(?, j:tags)",
            "bindings": {"1": {"type": "TEXT", "value": "bob"}},
        },
    )
    assert filtered.status_code == 200, filtered.json()
    assert filtered.json()["data"] == [[2]]


def test_array_slice_over_variant_arrays(client):
    """ARRAY_SLICE is 0-based and inclusive, with negatives counted from the end.

    sqlglot's own rendering moved only the start bound: a JSON array became a slice of the
    JSON *text* (`ARRAY_SLICE(j:names, 0, 1)` returned `'['`) and a native list came back
    one element short.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE slices (j VARIANT, idx VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO slices SELECT "
            "PARSE_JSON('{\"names\": [\"a\",\"b\",\"c\",\"d\",\"e\"], \"idx\": 1}'), "
            "TO_VARIANT(1)"
        ).status_code
        == 200
    )

    for statement, expected in (
        ("SELECT ARRAY_SLICE(j:names, 0, 4) FROM slices", [[['a', 'b', 'c', 'd', 'e']]]),
        ("SELECT ARRAY_SLICE(j:names, 1, 3) FROM slices", [[['b', 'c', 'd']]]),
        ("SELECT ARRAY_SLICE(j:names, -3, -1) FROM slices", [[['c', 'd', 'e']]]),
        ("SELECT ARRAY_SLICE(j:names, 0, 0) FROM slices", [[['a']]]),
        ("SELECT ARRAY_SLICE(j:names, -1, -1) FROM slices", [[['e']]]),
        # Out-of-range bounds clamp, and an empty slice is an empty array, not an error.
        ("SELECT ARRAY_SLICE(j:names, 0, 99) FROM slices", [[['a', 'b', 'c', 'd', 'e']]]),
        ("SELECT ARRAY_SLICE(j:names, -100, 100) FROM slices", [[['a', 'b', 'c', 'd', 'e']]]),
        ("SELECT ARRAY_SLICE(j:names, 10, 20) FROM slices", [[[]]]),
        ("SELECT ARRAY_SLICE(j:names, 3, 1) FROM slices", [[[]]]),
        ("SELECT ARRAY_SLICE(j:missing, 0, 1) FROM slices", [[None]]),
        # A nested call, and a bound that is itself a VARIANT value.
        ("SELECT ARRAY_CONTAINS('c', ARRAY_SLICE(j:names, 1, 3)) FROM slices", [[True]]),
        ("SELECT ARRAY_SIZE(ARRAY_SLICE(j:names, 1, 3)) FROM slices", [[3]]),
        ("SELECT ARRAY_SLICE(j:names, j:idx, 3) FROM slices", [[['b', 'c', 'd']]]),
    ):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        assert resp.json()["data"] == expected, statement


def test_slice_alias_matches_array_slice(client, tmp_path):
    """`SLICE` is Snowflake's alias of `ARRAY_SLICE`; sqlglot left it as an unknown call."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE alias_t (j VARIANT)").status_code == 200
    assert (
        post("INSERT INTO alias_t SELECT PARSE_JSON('{\"n\": [\"a\",\"b\",\"c\",\"d\"]}')")
    ).status_code == 200

    json_path = tmp_path / "names.json"
    json_path.write_text(json.dumps({"names": ["a", "b", "c"]}) + "\n")
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    for statement, expected in (
        ("SELECT SLICE(j:n, 0, 1) FROM alias_t", [[['a', 'b']]]),
        ("SELECT SLICE(j:n, -2, -1) FROM alias_t", [[['c', 'd']]]),
        ("SELECT slice(j:n, 1, 2) FROM alias_t", [[['b', 'c']]]),
        ("SELECT SLICE(ARRAY_CONSTRUCT('a','b','c'), 0, 1)", [[["a", "b"]]]),
        ("SELECT ARRAY_SIZE(SLICE(j:n, 0, 1)) FROM alias_t", [[2]]),
        ("SELECT ARRAY_CONTAINS('a', SLICE(j:n, 0, 1)) FROM alias_t", [[True]]),
        ("SELECT SLICE(SLICE(j:n, 0, 2), 1, 1) FROM alias_t", [[['b']]]),
    ):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        assert resp.json()["data"] == expected, statement

    staged = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT SLICE($1:names, 1, 2) FROM @~/staged/names.json"},
        headers=headers,
    )
    assert staged.status_code == 200, staged.json()
    assert staged.json()["data"] == [[['b', 'c']]]


def test_array_slice_requires_both_bounds(client):
    """A missing bound is invalid Snowflake; DuckDB would otherwise return NULL."""
    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT ARRAY_SLICE(ARRAY_CONSTRUCT('a','b'), 0)"},
    )
    assert resp.status_code == 422
    assert "requires both a start and an end index" in resp.json()["detail"]["message"]


def test_array_slice_over_native_lists(client):
    """A native list slices correctly too (it used to come back one element short)."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE native (names VARCHAR[])").status_code == 200
    assert post("INSERT INTO native SELECT ARRAY_CONSTRUCT('a','b','c')").status_code == 200

    for statement, expected in (
        ("SELECT ARRAY_SLICE(ARRAY_CONSTRUCT('a','b','c'), 0, 1)", [[["a", "b"]]]),
        ("SELECT ARRAY_SLICE(ARRAY_CONSTRUCT('a','b','c'), 1, 2)", [[["b", "c"]]]),
        ("SELECT ARRAY_SLICE(names, 0, 2) FROM native", [[["a", "b", "c"]]]),
        ("SELECT ARRAY_SLICE(names, -2, -1) FROM native", [[["b", "c"]]]),
        ("SELECT ARRAY_SLICE(names, 0, 9) FROM native", [[["a", "b", "c"]]]),
        ("SELECT ARRAY_SLICE(names, 5, 9) FROM native", [[[]]]),
    ):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        assert resp.json()["data"] == expected, statement


def test_other_array_functions_over_variant_arrays(client):
    """The rest of Snowflake's array functions over a VARIANT array, and their native controls."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE arr_t (j VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO arr_t SELECT PARSE_JSON('{\"n\": [\"a\",\"b\",\"a\"], "
            "\"nums\": [3,1,2], \"deep\": [[1,2],[3]], \"withnull\": [\"a\",null], "
            "\"flat\": [1,2,3]}')"
        ).status_code
        == 200
    )

    for statement, expected in (
        # Order is unspecified where DuckDB uses a hash set, so sort before comparing.
        ("SELECT ARRAY_SORT(ARRAY_UNIQUE(j:n)) FROM arr_t", [[['a', 'b']]]),
        ("SELECT ARRAY_REMOVE_AT(j:n, 0) FROM arr_t", [[['b', 'a']]]),
        ("SELECT ARRAY_POSITION('b', j:n) FROM arr_t", [[1]]),
        ("SELECT ARRAY_POSITION('zz', j:n) FROM arr_t", [[None]]),
        ("SELECT ARRAY_PREPEND('z', j:n) FROM arr_t", [[['z', 'a', 'b', 'a']]]),
        ("SELECT ARRAY_REPEAT('x', 3) FROM arr_t", [[['x', 'x', 'x']]]),
        ("SELECT ARRAY_FLATTEN(j:deep, 1) FROM arr_t", [[[1, 2, 3]]]),
        ("SELECT ARRAY_FLATTEN(j:deep, 2) FROM arr_t", [[[1, 2, 3]]]),
        # A depth past the nesting leaves the array as it is rather than raising.
        ("SELECT ARRAY_FLATTEN(j:flat, 2) FROM arr_t", [[[1, 2, 3]]]),
        ("SELECT ARRAY_APPEND(j:n, 'z') FROM arr_t", [[['a', 'b', 'a', 'z']]]),
        ("SELECT ARRAY_COMPACT(j:withnull) FROM arr_t", [[['a']]]),
        ("SELECT ARRAY_SORT(ARRAY_DISTINCT(j:n)) FROM arr_t", [[['a', 'b']]]),
        ("SELECT ARRAY_EXCEPT(j:n, j:n) FROM arr_t", [[[]]]),
        ("SELECT ARRAYS_OVERLAP(j:n, j:n) FROM arr_t", [[True]]),
        ("SELECT ARRAY_SORT(j:n) FROM arr_t", [[['a', 'a', 'b']]]),
        ("SELECT ARRAY_MAX(j:nums) FROM arr_t", [[3]]),
        ("SELECT ARRAY_MIN(j:nums) FROM arr_t", [[1]]),
        # JSON string elements are unquoted, unlike sqlglot's own CAST(x AS TEXT) rendering.
        ("SELECT ARRAY_TO_STRING(j:n, ',') FROM arr_t", [["a,b,a"]]),
        ("SELECT ARRAY_REMOVE(j:n, 'a') FROM arr_t", [[['b']]]),
        ("SELECT ARRAY_INSERT(j:n, 0, 'z') FROM arr_t", [[['z', 'a', 'b', 'a']]]),
        ("SELECT ARRAY_AGG(j:n) FROM arr_t", [[[['a', 'b', 'a']]]]),
        ("SELECT ARRAY_SIZE(ARRAY_CAT(j:n, j:n)) FROM arr_t", [[6]]),
        # A lambda over a semi-structured array receives a VARIANT element, so a plain literal
        # compares the way Snowflake compares it. ``TO_JSON`` returns VARCHAR (JSON text), so
        # comparing against ``TO_JSON('a')`` is comparing against '"a"' and matches nothing.
        ("SELECT FILTER(j:n, x -> x = 'a') FROM arr_t", [[['a', 'a']]]),
        ("SELECT FILTER(j:n, x -> x = TO_JSON('a')) FROM arr_t", [[[]]]),
        ("SELECT ARRAY_SIZE(TRANSFORM(j:n, x -> TO_VARCHAR(x))) FROM arr_t", [[3]]),
        ("SELECT FILTER(j:nums, x -> x > 1) FROM arr_t", [[[3, 2]]]),
        (
            "SELECT REDUCE(j:nums, 0, (acc, x) -> CAST(acc AS INT) + CAST(x AS INT)) "
            "FROM arr_t",
            [[6]],
        ),
        # Native lists keep their own rendering, including the two that were wrong before.
        ("SELECT ARRAY_SORT(ARRAY_UNIQUE(ARRAY_CONSTRUCT('a','a','b')))", [[["a", "b"]]]),
        ("SELECT ARRAY_PREPEND('z', ARRAY_CONSTRUCT('a','b'))", [[["z", "a", "b"]]]),
        ("SELECT ARRAY_REPEAT('x', 2)", [[["x", "x"]]]),
    ):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        assert resp.json()["data"] == expected, statement

    # Snowflake's ARRAY_INTERSECTION is a multiset (bag) operation: a value appears
    # min(N, M) times, where N and M are its counts in each array. ['a','b','a'] against
    # itself keeps both 'a's, which is what sqlglot's multiset model produces.
    intersected = post("SELECT ARRAY_INTERSECTION(j:n, j:n) FROM arr_t")
    assert intersected.status_code == 200, intersected.json()
    assert intersected.json()["data"] == [[['a', 'b', 'a']]]
    assert post(
        "SELECT ARRAY_INTERSECTION(PARSE_JSON('[\"b\",\"b\",\"b\"]'), "
        "PARSE_JSON('[\"b\",\"b\"]'))"
    ).json()["data"] == [[['b', 'b']]]


def test_array_flatten_requires_a_literal_depth(client):
    """The legacy two-argument form needs a constant depth; a computed one names the 1-arg form."""
    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT ARRAY_FLATTEN(ARRAY_CONSTRUCT(1), j) FROM arr_t"},
    )
    assert resp.status_code == 422
    assert "ARRAY_FLATTEN" in resp.json()["detail"]["message"]
    assert "ARRAY_FLATTEN(array)" in resp.json()["detail"]["message"]


def test_array_ordering_compares_json_numbers_numerically(client):
    """A JSON number array orders numerically, as Snowflake orders it by value.

    DuckDB compares JSON values as text, so ``[10,2,33]`` stayed in that order and
    ``ARRAY_MAX([9,100])`` was ``9``. The JSON path orders by a type-aware key instead: numbers
    numerically, strings lexicographically, and NULL elements last by default (first with an
    explicit ``nulls_first``, or with a descending sort).
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE ordj (v VARIANT)").status_code == 200
    assert (
        post(
            'INSERT INTO ordj SELECT PARSE_JSON(\'{"nums":[10,2,33],"big":[9,100],'
            '"strs":["b","aa","ab"],"withnull":[3,null,1],"bools":[true,false]}\')'
        ).status_code
        == 200
    )

    assert data("SELECT ARRAY_SORT(v:nums) FROM ordj") == [[[2, 10, 33]]]
    assert data("SELECT ARRAY_MAX(v:big) FROM ordj") == [[100]]
    assert data("SELECT ARRAY_MIN(v:big) FROM ordj") == [[9]]
    # A string array keeps lexicographic order, and booleans order false before true.
    assert data("SELECT ARRAY_SORT(v:strs) FROM ordj") == [[['aa', 'ab', 'b']]]
    assert data("SELECT ARRAY_SORT(v:bools) FROM ordj") == [[[False, True]]]
    # A literal JSON array takes the same path.
    assert data("SELECT ARRAY_SORT(PARSE_JSON('[10,2,33]'))") == [[[2, 10, 33]]]

    # Direction and NULL placement: last by default when ascending.
    assert data("SELECT ARRAY_SORT(v:nums, FALSE) FROM ordj") == [[[33, 10, 2]]]
    assert data("SELECT ARRAY_SORT(v:withnull) FROM ordj") == [[[1, 3, None]]]
    assert data("SELECT ARRAY_SORT(v:withnull, TRUE, TRUE) FROM ordj") == [[[None, 1, 3]]]

    # A native list keeps DuckDB's own order.
    assert data("SELECT ARRAY_SORT(ARRAY_CONSTRUCT(10,2,33))") == [[[2, 10, 33]]]
    assert data("SELECT ARRAY_SORT(ARRAY_CONSTRUCT(10,2,33), FALSE)") == [[[33, 10, 2]]]


def test_array_remove_keeps_null_elements(client):
    """``ARRAY_REMOVE`` drops the value, not every NULL element; a NULL value returns NULL.

    Snowflake's documented example keeps the NULL element (the function is NULL-safe about
    elements), while ``value_of_elements_to_remove`` being NULL makes the whole call NULL.
    sqlglot renders ``e <> value``, which is NULL for a NULL element and so dropped it.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE rem (v VARIANT)").status_code == 200
    assert post(
        'INSERT INTO rem SELECT PARSE_JSON(\'{"n":["a",null,"b","a"],'
        '"dup":["a",null,null]}\')'
    ).status_code == 200

    # A NULL element survives the removal of a non-NULL value, in a JSON array and a native one.
    assert data("SELECT ARRAY_REMOVE(v:n, 'a') FROM rem") == [[[None, 'b']]]
    assert data("SELECT ARRAY_REMOVE(ARRAY_CONSTRUCT('a', NULL, 'b'), 'a')") == [[[None, 'b']]]
    assert data("SELECT ARRAY_REMOVE(ARRAY_CONSTRUCT(1, 5, NULL), 5)") == [[[1, None]]]
    # Every equal element goes, and the documented cast-to-VARIANT form works.
    assert data("SELECT ARRAY_REMOVE(ARRAY_CONSTRUCT(5, 5), 5)") == [[[]]]
    assert data(
        "SELECT ARRAY_REMOVE(ARRAY_CONSTRUCT('a','b','a','c','d','a'), 'a'::VARIANT)"
    ) == [[['b', 'c', 'd']]]
    # A NULL value returns NULL, as documented -- it does not mean "remove the NULLs".
    assert data("SELECT ARRAY_REMOVE(v:n, NULL) FROM rem") == [[None]]
    assert data("SELECT ARRAY_REMOVE(v:dup, NULL) FROM rem") == [[None]]


def test_array_comparisons_span_variant_value_types(client):
    """A comparison over VARIANT values answers instead of raising, and matches across spaces.

    DuckDB raises for ``CAST('2' AS VARIANT) = CAST(2 AS VARIANT)``, so INTERSECTION/EXCEPT/
    OVERLAP over a VARIANT array compared in JSON space, where the two are simply unequal. A
    VARIANT *value* searched in a JSON array compares in VARIANT space, where both sides go
    through the same normalization (that pair used to answer FALSE for a number the array
    holds).
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE cx (n VARIANT, s VARIANT, two VARIANT)").status_code == 200
    assert post(
        'INSERT INTO cx SELECT PARSE_JSON(\'[3,1,2]\'), '
        'PARSE_JSON(\'["2","3"]\'), PARSE_JSON(\'2\')'
    ).status_code == 200

    # Cross-type pairs over a VARIANT array answer instead of raising.
    assert data("SELECT ARRAY_INTERSECTION(n, ARRAY_CONSTRUCT('2')) FROM cx") == [[[]]]
    assert data("SELECT ARRAY_EXCEPT(n, ARRAY_CONSTRUCT('2')) FROM cx") == [[[3, 1, 2]]]
    assert data("SELECT ARRAYS_OVERLAP(n, ARRAY_CONSTRUCT('2')) FROM cx") == [[False]]
    assert data("SELECT ARRAY_INTERSECTION(n, s) FROM cx") == [[[]]]
    assert data("SELECT ARRAYS_OVERLAP(s, n) FROM cx") == [[False]]

    # Same-type comparisons are unchanged.
    assert data("SELECT ARRAY_INTERSECTION(n, ARRAY_CONSTRUCT(2)) FROM cx") == [[[2]]]
    assert data("SELECT ARRAY_EXCEPT(s, ARRAY_CONSTRUCT('2')) FROM cx") == [[['3']]]

    # A VARIANT value finds its number in a JSON array.
    assert data("SELECT ARRAY_CONTAINS(two, PARSE_JSON('[2,3]')) FROM cx") == [[True]]
    assert data("SELECT ARRAY_POSITION(two, PARSE_JSON('[2,3]')) FROM cx") == [[0]]
    assert data("SELECT ARRAY_REMOVE(PARSE_JSON('[2,3]'), two) FROM cx") == [[[3]]]
    # A number is still not the string that spells it.
    assert data("SELECT ARRAY_CONTAINS('2'::VARIANT, PARSE_JSON('[2,3]')) FROM cx") == [[False]]


def test_array_to_string_null_and_argument_shapes(client):
    """``ARRAY_TO_STRING`` over a JSON array: NULL separator, NULL element, nested values.

    Snowflake documents two arguments, a NULL element becoming an empty string, and NULL for a
    NULL argument. sqlglot drops a third argument silently, so it is rejected before parsing;
    DuckDB treats a NULL separator as an empty one, so the separator is checked the way
    sqlglot's own rendering checks it.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE ats (j VARIANT)").status_code == 200
    assert (
        post(
            'INSERT INTO ats SELECT PARSE_JSON(\'{"plain":["a","b"],'
            '"withnull":["a",null,"b"],'
            '"nested":[["x","y"],{"a":1}],"nums":[1,-1.2e-3]}\')'

        ).status_code
        == 200
    )

    assert data("SELECT ARRAY_TO_STRING(j:plain, ',') FROM ats") == [["a,b"]]
    assert data("SELECT ARRAY_TO_STRING(j:withnull, ',') FROM ats") == [["a,,b"]]
    assert data("SELECT ARRAY_TO_STRING(j:nested, ' ') FROM ats") == [[
        '["x","y"] {"a":1}'
    ]]
    assert data("SELECT ARRAY_TO_STRING(j:nums, ', ') FROM ats") == [["1, -0.0012"]]

    # A NULL argument returns NULL, as Snowflake documents (DuckDB would use an empty one).
    assert data("SELECT ARRAY_TO_STRING(j:plain, NULL) FROM ats") == [[None]]
    # A NULL element is an empty string, not a skipped one.
    assert data("SELECT ARRAY_TO_STRING(j:withnull, '|') FROM ats") == [["a||b"]]
    # A separator expression that contains a comma is still one argument.
    assert data("SELECT ARRAY_TO_STRING(j:plain, IFF(1=1, ',', ';')) FROM ats") == [["a,b"]]

    # A positional separator keeps working: the NULL check is dropped rather than repeating
    # the placeholder, so one bind value is still enough.
    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "SELECT ARRAY_TO_STRING(j:plain, ?) FROM ats",
            "bindings": {"1": {"type": "TEXT", "value": ","}},
        },
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["data"] == [["a,b"]]

    # A third argument is not Snowflake SQL; say so instead of dropping it.
    for statement in (
        "SELECT ARRAY_TO_STRING(j:withnull, ',', 'X') FROM ats",
        "SELECT ARRAY_TO_STRING(j:withnull, ',', 'X', 'Y') FROM ats",
    ):
        resp = post(statement)
        assert resp.status_code == 422, (statement, resp.json())
        assert "ARRAY_TO_STRING takes an array and a separator" in resp.json()["detail"]["message"]


def test_array_flatten_one_argument_form(client):
    """Snowflake's documented ``ARRAY_FLATTEN(array)``: one level, strict about elements.

    The docs give no depth argument: the function removes one level of nesting, reports an
    error when an element is not an ARRAY, and returns NULL for a NULL array or a NULL element.
    The two-argument form is the emulator's legacy extension and stops at the nesting instead.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE flat_src (j VARIANT, v VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO flat_src SELECT PARSE_JSON('{\"deep\":[[1,2],[3]],"
            "\"deeper\":[[[1,2]],[[3]]],\"scalar\":[1,2,3],"
            "\"withnull\":[[1],null]}'), "
            "ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(1,2), ARRAY_CONSTRUCT(3))"
        ).status_code
        == 200
    )

    # One level, over a native list, a JSON path and a VARIANT column.
    assert data("SELECT ARRAY_FLATTEN([[1,2],[3]])") == [[[1, 2, 3]]]
    assert data("SELECT ARRAY_FLATTEN(j:deep) FROM flat_src") == [[[1, 2, 3]]]
    assert data("SELECT ARRAY_FLATTEN(j:deeper) FROM flat_src") == [[[[1, 2], [3]]]]
    assert data("SELECT ARRAY_FLATTEN(v) FROM flat_src") == [[[1, 2, 3]]]

    # The legacy depth form still stops at the nesting.
    assert data("SELECT ARRAY_FLATTEN(j:deeper, 2) FROM flat_src") == [[[1, 2, 3]]]
    assert data("SELECT ARRAY_FLATTEN(j:scalar, 2) FROM flat_src") == [[[1, 2, 3]]]

    # A non-ARRAY element is an error, not a silent no-op.
    resp = post("SELECT ARRAY_FLATTEN(j:scalar) FROM flat_src")
    assert resp.status_code == 422, resp.json()
    assert "ARRAY" in resp.json()["detail"]["message"]

    # A NULL array, and an array holding a NULL element, return NULL.
    assert data("SELECT ARRAY_FLATTEN(NULL)") == [[None]]
    assert data("SELECT ARRAY_FLATTEN(j:withnull) FROM flat_src") == [[None]]

    # A depth outside 1..10 stays a loud error.
    for depth in (0, 11):
        resp = post(f"SELECT ARRAY_FLATTEN(j:deep, {depth}) FROM flat_src")
        assert resp.status_code == 422, resp.json()
        assert "between 1 and 10" in resp.json()["detail"]["message"]


def test_flatten_over_a_variant_column(client):
    """`json_each` needs JSON: a VARIANT column (and `col:path`) is cast, as `col:path` is."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE flat_v (j VARIANT)").status_code == 200
    assert (
        post("INSERT INTO flat_v SELECT PARSE_JSON('{\"n\": [\"a\",\"b\"]}')").status_code
        == 200
    )

    for statement in (
        "SELECT f.value AS v FROM flat_v t, LATERAL FLATTEN(input => t.j:n) AS f",
        "SELECT f.value AS v FROM flat_v t, TABLE(FLATTEN(input => t.j:n)) AS f",
    ):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        assert {row[0] for row in resp.json()["data"]} == {'a', 'b'}


def test_variant_cells_report_as_variant(client):
    """A DuckDB VARIANT cell is typed `variant`, and the SQL API inlines its value."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE vk (s VARIANT, n VARIANT)").status_code == 200
    assert (
        post("INSERT INTO vk SELECT TO_VARIANT('y'), TO_VARIANT(42)").status_code == 200
    )

    selected = post("SELECT s, n FROM vk").json()
    assert [col["type"] for col in selected["resultSetMetaData"]["rowType"]] == [
        "variant",
        "variant",
    ]
    assert selected["data"] == [["y", 42]]

    described = post("DESCRIBE TABLE vk").json()["data"]
    assert [(row[0], row[1]) for row in described] == [("S", "VARIANT"), ("N", "VARIANT")]


def test_semi_structured_text_cast_returns_the_value(client):
    """`col:path::VARCHAR` returns the string itself, not DuckDB's JSON serialization (`'"y"'`)."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE txt (j VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO txt SELECT "
            "PARSE_JSON('{\"s\":\"y\",\"n\":1,\"o\":{\"k\":1},\"a\":[1,2],\"nl\":null}')"
        ).status_code
        == 200
    )

    for statement, expected in (
        ("SELECT j:s::VARCHAR FROM txt", [["y"]]),
        ("SELECT TO_VARCHAR(j:s) FROM txt", [["y"]]),
        ("SELECT CAST(j:s AS VARCHAR) FROM txt", [["y"]]),
        ("SELECT j:s::VARCHAR(1) FROM txt", [["y"]]),
        ("SELECT j:s::CHAR(3) FROM txt", [["y"]]),
        ("SELECT CAST(PARSE_JSON('\"y\"') AS VARCHAR)", [["y"]]),
        # A number is its text, a path that matches nothing is NULL, and a JSON null is the
        # word `null` (TO_VARCHAR: "A JSON null value is converted to a string containing the
        # word "null""; a missing value is converted to SQL NULL).
        ("SELECT j:n::VARCHAR FROM txt", [["1"]]),
        ("SELECT j:nl::VARCHAR FROM txt", [["null"]]),
        ("SELECT j:missing::VARCHAR FROM txt", [[None]]),
        # An object or array keeps its document text.
        ("SELECT j:o::VARCHAR FROM txt", [["{\"k\":1}"]]),
        ("SELECT j:a::VARCHAR FROM txt", [["[1,2]"]]),
        # The cast is usable in an expression.
        ("SELECT j:s::VARCHAR = 'y' FROM txt", [[True]]),
        ("SELECT LENGTH(j:s::VARCHAR) FROM txt", [[1]]),
    ):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        assert resp.json()["data"] == expected, statement


def test_to_json_of_a_variant_returns_the_json_representation(client):
    """`TO_JSON` returns the value's JSON document, not a string of DuckDB's VARIANT form.

    DuckDB's ``to_json`` casts a VARIANT to VARCHAR first, so a numeric variant became the
    JSON string ``"42"`` and an object became a quoted internal repr (``"{'k': v}"``).
    Snowflake documents TO_JSON as the JSON *representation* of the value: ``42``, and the
    document text for an object.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE tj (v VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO tj SELECT "
            "PARSE_JSON('{\"k\":\"a\\\"b\",\"tags\":[\"a\\\"b\",1]}')"
        ).status_code
        == 200
    )

    document = '{"k":"a\\"b","tags":["a\\"b",1]}'
    for statement, expected in (
        # A VARIANT column keeps its document, quotes inside strings and all.
        ("SELECT TO_JSON(v) FROM tj", [[document]]),
        # A string variant stays a JSON string; a number and a boolean do not.
        ("SELECT TO_JSON(TO_VARIANT('a\"b'))", [['"a\\"b"']]),
        (
            "SELECT TO_JSON(TO_VARIANT(42)), TO_JSON(TO_VARIANT(true))",
            [["42", "true"]],
        ),
        # A struct-backed VARIANT is the document, not DuckDB's repr in a JSON string.
        (
            "SELECT TO_JSON(TO_VARIANT(OBJECT_CONSTRUCT('k', 'a\"b')))",
            [['{"k":"a\\"b"}']],
        ),
        # A JSON value already holds its document text, and NULL stays NULL.
        ("SELECT TO_JSON(PARSE_JSON('{\"a\":1}'))", [['{"a":1}']]),
        ("SELECT TO_JSON(NULL)", [[None]]),
        # The result is VARCHAR (JSON text), so it compares as text.
        ("SELECT TYPEOF(TO_JSON(TO_VARIANT(42)))", [["VARCHAR"]]),
        ("SELECT TO_JSON(TO_VARIANT(42)) = '42'", [[True]]),
    ):
        assert data(statement) == expected, statement


def test_parse_json_of_a_string_variant_parses_its_text(client):
    """`PARSE_JSON(variant)` parses the value's text, as Snowflake's coercion does.

    A VARIANT is coercible to VARCHAR, and that conversion preserves a stored string as is
    (no JSON quotes). Parsing the value's *document* instead made a string variant that holds
    JSON text stay a string: ``PARSE_JSON(TO_VARIANT('{"a":1}'))`` answered ``{"a":1}``
    rather than the object Snowflake parses.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    # A string variant holding a document is reparsed into that document.
    assert data("SELECT PARSE_JSON(TO_VARIANT('{\"a\":1}')):a::VARCHAR") == [["1"]]
    assert data("SELECT PARSE_JSON(TO_VARIANT('[1,2]'))::VARCHAR") == [["[1,2]"]]
    # A number variant is parsed from its digits, so the result is the number.
    assert data("SELECT PARSE_JSON(TO_VARIANT('42')) = 42") == [[True]]
    assert data("SELECT PARSE_JSON(TO_VARIANT('42')) = '42'") == [[False]]
    # A JSON null variant parses back to a JSON null, which is not an SQL NULL.
    assert data("SELECT PARSE_JSON(TO_VARIANT('null')) IS NULL") == [[False]]
    assert data("SELECT TO_VARCHAR(PARSE_JSON(TO_VARIANT('null')))") == [["null"]]
    # A string variant that is not valid JSON fails the parse, as in Snowflake.
    resp = post("SELECT PARSE_JSON(TO_VARIANT('abc'))")
    assert resp.status_code == 422
    assert "Malformed JSON" in resp.json()["detail"]["message"]


def test_flatten_value_is_a_variant_when_used_as_text(client):
    """FLATTEN's VALUE/THIS are VARIANT, so a string element's JSON quotes are dropped.

    The relation is rewritten to a ``json_each``-shaped subquery, which has no catalog entry,
    so without typing it ``VALUE::VARCHAR`` kept DuckDB's JSON serialization (``"a\\"b"``)
    and ``LENGTH``/``UPPER`` measured and transformed those quotes -- 6 and ``"A\\"B"`` where
    Snowflake answers ``a"b``, 3 and ``A"B``.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE flat_txt (j VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO flat_txt SELECT "
            "PARSE_JSON('{\"tags\":[\"a\\\"b\",1]}')"
        ).status_code
        == 200
    )

    qualified = (
        "SELECT f.value::VARCHAR, LENGTH(f.value), UPPER(f.value) "
        "FROM flat_txt t, LATERAL FLATTEN(input => t.j:tags) f ORDER BY f.index"
    )
    assert data(qualified) == [["a\"b", 3, 'A"B'], ["1", 1, "1"]]
    # A FLATTEN's columns are also visible without a qualifier.
    assert data(
        "SELECT value::VARCHAR, LENGTH(value) "
        "FROM LATERAL FLATTEN(input => PARSE_JSON('[\"a\\\"b\"]'))"
    ) == [["a\"b", 3]]
    # TABLE(FLATTEN(...)) is the same relation, and THIS keeps an object's document text.
    assert data(
        "SELECT f.value::VARCHAR, f.this::VARCHAR FROM "
        "TABLE(FLATTEN(input => PARSE_JSON('{\"k\":\"a\\\"b\"}'))) f"
    ) == [["a\"b", '{"k":"a\\"b"}']]


def test_variant_value_used_as_text_in_nested_expressions(client):
    """A VARIANT column is text even when wrapped in PARSE_JSON/REGEXP_SUBSTR.

    ``PARSE_JSON(variant_col)`` used to render ``JSON(variant)``, which parses DuckDB's VARIANT
    struct repr and raised ``Malformed JSON``; ``REGEXP_SUBSTR`` saw that repr too, so a pattern
    looking for the JSON document's double quotes answered NULL. A qualified column must work
    the same way as an unqualified one.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE vtext (ident VARIANT)").status_code == 200
    assert (
        post("INSERT INTO vtext SELECT PARSE_JSON('{\"names\":[\"N300\"]}')").status_code
        == 200
    )

    for statement, expected in (
        ("SELECT TO_VARCHAR(PARSE_JSON(ident):names[0]) FROM vtext", [["N300"]]),
        ("SELECT TO_VARCHAR(PARSE_JSON(v.ident):names[0]) FROM vtext v", [["N300"]]),
        # PARSE_JSON of a variant is the same document, not a reparse of its repr.
        ("SELECT TO_VARCHAR(PARSE_JSON(ident)) FROM vtext", [['{"names":["N300"]}']]),
        (
            "SELECT REGEXP_SUBSTR(ident, '\"([^\"]*)\"', 1, 1, 'e') FROM vtext",
            [["names"]],
        ),
        (
            "SELECT REGEXP_SUBSTR(v.ident, '\"([^\"]*)\"', 1, 1, 'e') FROM vtext v",
            [["names"]],
        ),
        (
            "SELECT REGEXP_SUBSTR(PARSE_JSON(ident), '\"([^\"]*)\"', 1, 1, 'e') "
            "FROM vtext",
            [["names"]],
        ),
    ):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        assert resp.json()["data"] == expected, statement


def test_semi_structured_text_cast_over_staged_json(client, tmp_path):
    json_path = tmp_path / "doc.json"
    json_path.write_text(json.dumps({"s": "y", "o": {"k": 1}}) + "\n")
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    selected = _select(
        client,
        headers,
        "SELECT $1:s::VARCHAR AS s, TO_VARCHAR($1:o) AS o FROM @~/staged/doc.json",
    )
    assert selected["data"] == [["y", '{"k":1}']]


def test_uninterpreted_function_is_left_to_duckdb(client):
    """A function sqlglot does not know must pass through, not crash the array rewrite."""
    resp = client.post(
        "/api/v2/statements", json={"statement": "SELECT NOT_A_REAL_FUNCTION(1) AS x"}
    )
    assert resp.status_code == 422
    assert "NOT_A_REAL_FUNCTION" in resp.json()["detail"]["message"]


def test_array_functions_over_a_bare_variant_array(client):
    """A bare VARIANT column (and ``x::VARIANT``) is presented as ``VARIANT[]``, not left raw.

    DuckDB rejects a VARIANT where its list functions want a LIST (``Binder Error: length
    can only be used on arrays or lists``), so the operand is cast through VARIANT to
    ``VARIANT[]`` -- a real list whose elements keep Snowflake's VARIANT comparison
    semantics (``'1'`` and ``1`` are different values). A searched value is round-tripped
    ``VARIANT`` -> ``JSON`` -> ``VARIANT`` so a VARIANT number (an ``INT32``) matches the
    ``UINT64`` a ``PARSE_JSON`` array element has.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert (
        post("CREATE TABLE bare (names VARIANT, nums VARIANT, deep VARIANT, two VARIANT)")
        .status_code
        == 200
    )
    assert (
        post(
            "INSERT INTO bare SELECT PARSE_JSON('[\"ada\",\"bob\"]'), "
            "PARSE_JSON('[3,1,2]'), PARSE_JSON('[[1,2],[3]]'), TO_VARIANT(2)"
        ).status_code
        == 200
    )

    for statement, expected in (
        ("SELECT ARRAY_CONTAINS('ada', names) FROM bare", [[True]]),
        ("SELECT ARRAY_CONTAINS(CAST('ada' AS VARIANT), names) FROM bare", [[True]]),
        ("SELECT ARRAY_CONTAINS('zoe', names) FROM bare", [[False]]),
        ("SELECT ARRAY_CONTAINS('ada', names::VARIANT) FROM bare", [[True]]),
        ("SELECT ARRAY_CONTAINS('ada', TO_VARIANT(names)) FROM bare", [[True]]),
        ("SELECT ARRAY_CONTAINS(2, nums) FROM bare", [[True]]),
        # A string is not the number it spells, as in Snowflake.
        ("SELECT ARRAY_CONTAINS('2', nums) FROM bare", [[False]]),
        # A VARIANT-typed value keeps its numeric width; normalizing it matches the array.
        ("SELECT ARRAY_CONTAINS(two, nums) FROM bare", [[True]]),
        ("SELECT ARRAY_CONTAINS(2::VARIANT, nums) FROM bare", [[True]]),
        ("SELECT ARRAY_POSITION(two, nums) FROM bare", [[2]]),
        ("SELECT ARRAY_REMOVE(nums, two) FROM bare", [[[3, 1]]]),
        ("SELECT ARRAY_CONTAINS(TO_VARIANT(ARRAY_CONSTRUCT(1,2)), deep) FROM bare", [[True]]),
        ("SELECT ARRAY_SIZE(names) FROM bare", [[2]]),
        ("SELECT ARRAY_POSITION('bob', names) FROM bare", [[1]]),
        ("SELECT ARRAY_POSITION('zoe', names) FROM bare", [[None]]),
        ("SELECT ARRAY_SLICE(names, 0, 0) FROM bare", [[['ada']]]),
        ("SELECT ARRAY_CONTAINS('ada', ARRAY_SLICE(names, 0, 0)) FROM bare", [[True]]),
        # Flatten follows an ARRAY_SLICE result, which is a call rather than a CAST.
        ("SELECT ARRAY_FLATTEN(ARRAY_SLICE(deep, 0, 0), 1) FROM bare", [[[1, 2]]]),
        ("SELECT ARRAY_EXCEPT(names, names) FROM bare", [[[]]]),
        ("SELECT ARRAYS_OVERLAP(names, names) FROM bare", [[True]]),
        ("SELECT ARRAY_TO_STRING(names, '|') FROM bare", [['ada|bob']]),
        ("SELECT ARRAY_SIZE(ARRAY_AGG(names)) FROM bare", [[1]]),
    ):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        assert resp.json()["data"] == expected, statement


def test_array_slice_accepts_a_bound_index(client):
    """A ``?``/``:name`` slice index works: the value is named once and adjusted through it.

    The 0-based to 1-based adjustment has to name the bound twice, which a positional ``?``
    cannot do (one value per placeholder), so the bound is written once inside a one-element
    list and the repeated reference is the lambda parameter.
    """
    def post(statement, bindings=None):
        body = {"statement": statement}
        if bindings:
            body["bindings"] = bindings
        return client.post("/api/v2/statements", json=body)

    def data(statement, bindings=None):
        resp = post(statement, bindings)
        assert resp.status_code == 200, (statement, bindings, resp.json())
        return resp.json()["data"]

    def fixed(value):
        return {"type": "FIXED", "value": str(value)}

    assert post("CREATE TABLE sl (arr VARIANT)").status_code == 200
    assert post("INSERT INTO sl SELECT PARSE_JSON('[10,20,30,40]')").status_code == 200

    # Both bounds positional, over a native list and over a JSON path alike.
    assert data(
        "SELECT ARRAY_SLICE(ARRAY_CONSTRUCT(1,2,3), ?, ?)", {"1": fixed(0), "2": fixed(1)}
    ) == [[[1, 2]]]
    assert data(
        "SELECT ARRAY_SLICE(ARRAY_CONSTRUCT(1,2,3), ?, ?)", {"1": fixed(-2), "2": fixed(-1)}
    ) == [[[2, 3]]]
    assert data(
        "SELECT ARRAY_SLICE(arr, ?, ?) FROM sl", {"1": fixed(1), "2": fixed(2)}
    ) == [[[20, 30]]]
    assert data("SELECT ARRAY_SLICE(j, ?, ?) FROM (SELECT PARSE_JSON('[1,2,3]') AS j)", {
        "1": fixed(0),
        "2": fixed(0),
    }) == [[[1]]]

    # A literal on one side and a bind on the other.
    assert data("SELECT ARRAY_SLICE(arr, ?, 2) FROM sl", {"1": fixed(1)}) == [[[20, 30]]]
    assert data("SELECT ARRAY_SLICE(arr, 0, ?) FROM sl", {"1": fixed(1)}) == [[[10, 20]]]

    # A named bind repeats safely, and a literal out-of-range bound still clamps.
    assert data(
        "SELECT ARRAY_SLICE(ARRAY_CONSTRUCT(1,2,3), :lo, :hi)",
        {"lo": fixed(1), "hi": fixed(2)},
    ) == [[[2, 3]]]
    assert data("SELECT ARRAY_SLICE(ARRAY_CONSTRUCT(1,2,3), 5, 9)") == [[[]]]


def test_lambda_bodies_over_a_json_array(client):
    """A lambda over a semi-structured array receives a VARIANT element, as in Snowflake.

    The array is presented as ``VARIANT[]`` so a body may compare the element with a plain
    literal (``x -> x = 'a'``) or do arithmetic on it, which DuckDB coerces the way Snowflake
    does. A ``JSON`` element would instead try to parse the literal as JSON and raise
    ``Malformed JSON at byte 0``. A native list keeps its own element type.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE lam (j VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO lam SELECT PARSE_JSON('{\"n\":[\"a\",\"b\",\"a\"],"
            "\"nums\":[3,1,2]}')"
        ).status_code
        == 200
    )
    assert post("CREATE TABLE lam_bare (names VARIANT)").status_code == 200
    assert (
        post("INSERT INTO lam_bare SELECT PARSE_JSON('[\"ada\",\"bob\"]')").status_code
        == 200
    )

    # A string literal compares against the element without an encoding helper.
    assert data("SELECT FILTER(j:n, x -> x = 'a') FROM lam") == [[['a', 'a']]]
    assert data("SELECT FILTER(j:n, x -> x IN ('a')) FROM lam") == [[['a', 'a']]]
    assert data("SELECT ARRAY_SIZE(FILTER(j:n, x -> x = 'a')) FROM lam") == [[2]]
    assert data("SELECT FILTER(names, x -> x = 'ada') FROM lam_bare") == [[['ada']]]

    # A number compares and computes the same way, and a nested lambda sees VARIANT[] again.
    assert data("SELECT FILTER(j:nums, x -> x > 1) FROM lam") == [[[3, 2]]]
    assert data("SELECT FILTER(j:nums, x -> x = 2) FROM lam") == [[[2]]]
    assert data("SELECT TRANSFORM(j:nums, x -> x * 2) FROM lam") == [[[6, 2, 4]]]
    assert data("SELECT FILTER(FILTER(j:n, x -> x = 'a'), y -> y = 'a') FROM lam") == [
        [['a', 'a']]
    ]
    assert data("SELECT REDUCE(j:nums, 0, (acc, x) -> acc + CAST(x AS INT)) FROM lam") == [[6]]

    # A native list keeps its own element type.
    assert data("SELECT FILTER(ARRAY_CONSTRUCT('a','b'), x -> x = 'a')") == [[['a']]]
    assert data("SELECT TRANSFORM(ARRAY_CONSTRUCT(1,2), x -> x + 1)") == [[[2, 3]]]
    assert data("SELECT REDUCE(ARRAY_CONSTRUCT(1,2,3), 0, (a, x) -> a + x)") == [[6]]


def test_documented_array_semantics(client):
    """The four semantics that were inherited from sqlglot, checked against Snowflake's docs.

    Verified on docs.snowflake.com: ``ARRAY_INTERSECTION`` and ``ARRAY_EXCEPT`` compare with
    multi-set (bag) semantics, ``ARRAY_POSITION`` returns NULL for a value the array does not
    contain and the position of the first NULL for a NULL value, and ``ARRAY_AGG`` omits NULL
    values. The multiset intersection and the NULL omission were already what sqlglot models;
    the 0-based ``ARRAY_POSITION`` fallback of 0 and ARRAY_EXCEPT's set-semantics JSON rebuild
    were not.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE json_arr (j VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO json_arr SELECT PARSE_JSON('{\"n\":[\"a\",\"b\",\"b\",\"b\",\"c\"]}')"
        ).status_code
        == 200
    )
    assert post("CREATE TABLE json_null_arr (j VARIANT)").status_code == 200
    assert (
        post("INSERT INTO json_null_arr SELECT PARSE_JSON('{\"n\":[\"a\",null,\"b\"]}')")
        .status_code
        == 200
    )
    assert post("CREATE TABLE null_arr (x INT)").status_code == 200
    assert post("INSERT INTO null_arr VALUES (1), (NULL), (2)").status_code == 200
    assert post("CREATE TABLE variant_arr (n VARIANT)").status_code == 200
    assert (
        post("INSERT INTO variant_arr SELECT ARRAY_CONSTRUCT('a','b','b','b','c')").status_code
        == 200
    )

    # ARRAY_EXCEPT removes M copies of a value, leaving N - M: ['a','b','b','b','c'] minus
    # ['b'] keeps two 'b's, for a JSON array and a VARIANT one alike.
    assert data("SELECT ARRAY_EXCEPT(j:n, PARSE_JSON('[\"b\"]')) FROM json_arr") == [
        [['a', 'b', 'b', 'c']]
    ]
    assert data(
        "SELECT ARRAY_EXCEPT(PARSE_JSON('[\"a\",\"b\",\"b\",\"b\",\"c\"]'), "
        "PARSE_JSON('[\"b\"]'))"
    ) == [[['a', 'b', 'b', 'c']]]
    assert data(
        "SELECT ARRAY_EXCEPT(n, TO_VARIANT(ARRAY_CONSTRUCT('b'))) FROM variant_arr"
    ) == [[['a', 'b', 'b', 'c']]]
    assert data(
        "SELECT ARRAY_EXCEPT(ARRAY_CONSTRUCT('a','b','b','b','c'), ARRAY_CONSTRUCT('b'))"
    ) == [[['a', 'b', 'b', 'c']]]
    assert data("SELECT ARRAY_EXCEPT(j:n, PARSE_JSON('[\"b\",\"b\"]')) FROM json_arr") == [
        [['a', 'b', 'c']]
    ]
    assert data("SELECT ARRAY_EXCEPT(n, n) FROM variant_arr") == [[[]]]

    # The multiset arithmetic is NULL-safe: a JSON null is an SQL NULL to DuckDB, so the
    # `=` comparison counted no copy of it on either side and a repeated null was dropped
    # entirely. Two nulls minus one null keeps one.
    assert data(
        "SELECT ARRAY_EXCEPT(PARSE_JSON('[\"a\",null,null]'), PARSE_JSON('[null]'))"
    ) == [[['a', None]]]
    assert data(
        "SELECT ARRAY_EXCEPT(TO_VARIANT(ARRAY_CONSTRUCT('a', NULL)), "
        "TO_VARIANT(ARRAY_CONSTRUCT(NULL)))"
    ) == [[['a']]]
    assert data(
        "SELECT ARRAY_INTERSECTION(PARSE_JSON('[\"a\",null]'), PARSE_JSON('[null]'))"
    ) == [[[None]]]

    # ARRAY_POSITION answers NULL, not 0, for a value the array does not hold...
    assert data("SELECT ARRAY_POSITION('zz', j:n) FROM json_arr") == [[None]]
    assert data("SELECT ARRAY_POSITION('zz', ARRAY_CONSTRUCT('a'))") == [[None]]
    # ...and the position of the first NULL for a NULL value, in a JSON array and a native
    # list alike.
    assert data("SELECT ARRAY_POSITION(NULL, j:n) FROM json_null_arr") == [[1]]
    assert data("SELECT ARRAY_POSITION(NULL, ARRAY_CONSTRUCT('a', NULL, 'b'))") == [[1]]

    # ARRAY_AGG omits NULL values.
    assert data("SELECT ARRAY_AGG(x) FROM null_arr") == [[[1, 2]]]


def test_array_functions_coerce_a_mixed_operand_pair(client):
    """A native list and a JSON/VARIANT array are coerced element-wise, as Snowflake does.

    DuckDB does not unify the element types: joining a ``VARCHAR[]`` against a ``JSON[]``
    casts the text to JSON (``Malformed JSON at byte 0``), and comparing a JSON element
    against a plain value is silently unequal. Snowflake coerces the two arrays, so the
    native side is re-encoded with ``TO_JSON``/``_variant_element`` per element, which keeps
    a number distinct from the string that spells it.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE mix (nums VARIANT, j VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO mix SELECT PARSE_JSON('[3,1,2]'), "
            "PARSE_JSON('{\"nums\":[3,1,2],\"s\":[\"1\",\"2\"],"
            "\"nulls\":[\"a\",null,null],\"names\":[\"a\",\"b\",\"a\"]}')"
        ).status_code
        == 200
    )

    # A shared value is found, and a value in the array that the native list does not hold is
    # left behind -- in either operand order.
    assert data("SELECT ARRAYS_OVERLAP(nums, ARRAY_CONSTRUCT(2)) FROM mix") == [[True]]
    assert data("SELECT ARRAYS_OVERLAP(nums, ARRAY_CONSTRUCT(9)) FROM mix") == [[False]]
    assert data("SELECT ARRAY_INTERSECTION(nums, ARRAY_CONSTRUCT(2)) FROM mix") == [[[2]]]
    assert data("SELECT ARRAY_INTERSECTION(j:names, ARRAY_CONSTRUCT('a')) FROM mix") == [
        [['a']]
    ]
    # Snowflake leaves the order of ARRAY_EXCEPT's result unspecified, so compare as a
    # multiset rather than pinning one.
    assert sorted(data("SELECT ARRAY_EXCEPT(j:names, ARRAY_CONSTRUCT('a')) FROM mix")[0][0]) == [
        'a', 'b'
    ]
    assert data("SELECT ARRAY_EXCEPT(ARRAY_CONSTRUCT(2,3), nums) FROM mix") == [[[]]]
    assert data("SELECT ARRAY_CAT(j:nums, ARRAY_CONSTRUCT(4)) FROM mix") == [[[3, 1, 2, 4]]]
    assert data("SELECT ARRAY_CAT(ARRAY_CONSTRUCT(4), nums) FROM mix") == [[[4, 3, 1, 2]]]

    # The encoding keeps a number apart from the string that spells it. A bare VARIANT column
    # is compared as a VARIANT array, where DuckDB cannot compare a string element against a
    # number element at all, so the cross-type case is exercised on the JSON path.
    assert data("SELECT ARRAY_INTERSECTION(j:nums, ARRAY_CONSTRUCT('2')) FROM mix") == [[[]]]
    assert data("SELECT ARRAY_INTERSECTION(j:s, ARRAY_CONSTRUCT('2')) FROM mix") == [
        [['2']]
    ]
    assert data("SELECT ARRAY_INTERSECTION(j:s, ARRAY_CONSTRUCT(2)) FROM mix") == [[[]]]

    # A NULL in the native list matches the array's null element and is subtracted once.
    assert data("SELECT ARRAY_EXCEPT(j:nulls, ARRAY_CONSTRUCT(NULL)) FROM mix") == [
        [['a', None]]
    ]
    assert data("SELECT ARRAY_INTERSECTION(j:nulls, ARRAY_CONSTRUCT(NULL)) FROM mix") == [
        [[None]]
    ]

    # Same-kind pairs are unaffected, native pairs included.
    assert post("SELECT ARRAYS_OVERLAP(nums, nums) FROM mix").json()["data"] == [[True]]
    assert post("SELECT ARRAY_EXCEPT(nums, nums) FROM mix").json()["data"] == [[[]]]
    assert post(
        "SELECT ARRAY_EXCEPT(ARRAY_CONSTRUCT('a','b'), ARRAY_CONSTRUCT('b'))"
    ).json()["data"] == [[['a']]]


def test_array_flatten_deep_depth_is_cheap(client):
    """A depth of 10 must not expand into an exponential expression.

    The guarded form used to name the previous level three times per level, so the SQL text
    tripled with each one: depth 10 produced 108 MB of SQL and took over five minutes to
    translate. ``LIST_REDUCE`` keeps the text one size regardless of depth.
    """
    from snowflake_emulator.translator import transpile_to_duckdb
    from snowflake_emulator.translator.constants import READ_DIALECT
    from snowflake_emulator.translator.parsing import preprocess_array_calls

    sizes = []
    for depth in (1, 10):
        duckdb_sql = transpile_to_duckdb(
            sqlparse_one(preprocess_array_calls(f"SELECT ARRAY_FLATTEN(j:deep, {depth})"))
        )
        sizes.append(len(duckdb_sql))
    assert max(sizes) < 2000, sizes

    assert client.post(
        "/api/v2/statements", json={"statement": "CREATE TABLE deep_t (j VARIANT)"}
    ).status_code == 200
    assert client.post(
        "/api/v2/statements",
        json={
            "statement": 'INSERT INTO deep_t SELECT PARSE_JSON(\'{"deep":[[1,2],[3]]}\')'
        },
    ).status_code == 200
    resp = client.post(
        "/api/v2/statements", json={"statement": "SELECT ARRAY_FLATTEN(j:deep, 10) FROM deep_t"}
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["data"] == [[[1, 2, 3]]]


def sqlparse_one(sql):
    """Parse Snowflake SQL the way the service does (preprocessing included)."""
    import sqlglot
    from snowflake_emulator.translator.constants import READ_DIALECT

    return sqlglot.parse_one(sql, read=READ_DIALECT)


def test_array_ordering_rejects_a_non_literal_direction(client):
    """``ARRAY_SORT``'s direction arguments are literals; anything else is not guessed.

    Reading the AST's first child made ``ARRAY_SORT(arr, 1 = 2)`` sort ascending (wrong) and
    ``ARRAY_SORT(arr, NULL)`` sort descending where Snowflake returns NULL.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE dir (v VARIANT)").status_code == 200
    assert post(
        'INSERT INTO dir SELECT PARSE_JSON(\'{"nums":[10,2,33]}\')'
    ).status_code == 200

    for statement in (
        "SELECT ARRAY_SORT(v:nums, 1 = 2) FROM dir",
        "SELECT ARRAY_SORT(v:nums, TRUE = FALSE) FROM dir",
        "SELECT ARRAY_SORT(v:nums, TRUE, 1 = 2) FROM dir",
    ):
        resp = post(statement)
        assert resp.status_code == 422, (statement, resp.json())
        assert "literal TRUE or FALSE" in resp.json()["detail"]["message"]

    # A NULL argument makes the call NULL, as Snowflake documents.
    assert post("SELECT ARRAY_SORT(v:nums, NULL) FROM dir").json()["data"] == [[None]]
    # Literals keep working.
    assert post("SELECT ARRAY_SORT(v:nums, FALSE) FROM dir").json()["data"] == [[[33, 10, 2]]]


def test_array_column_type_resolution_is_schema_aware(client):
    """A qualified column resolves within its own schema, and only when the type is unique.

    The resolver matched on the table name alone and skipped the type-disagreement check for a
    qualified name, so ``SELECT ARRAY_SORT(arr.n) FROM s_variant.arr`` picked the other
    schema's ``VARCHAR[]`` and reached DuckDB's VARIANT ``list_sort`` -- an out-of-memory
    error. A name the query itself defines (an alias) is not looked up either.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE SCHEMA s_native").status_code == 200
    assert post(
        "CREATE TABLE s_native.arr AS SELECT ARRAY_CONSTRUCT('b','a') AS n"
    ).status_code == 200
    assert post("CREATE SCHEMA s_variant").status_code == 200
    assert post("CREATE TABLE s_variant.arr (n VARIANT)").status_code == 200
    assert post(
        'INSERT INTO s_variant.arr SELECT PARSE_JSON(\'["b","aa","ab"]\')'
    ).status_code == 200

    assert post("SELECT ARRAY_SORT(s_native.arr.n) FROM s_native.arr").json()["data"] == [
        [['a', 'b']]
    ]
    for statement in (
        "SELECT ARRAY_SORT(s_variant.arr.n) FROM s_variant.arr",
        "SELECT ARRAY_SORT(arr.n) FROM s_variant.arr",
        "SELECT ARRAY_SORT(s_variant.arr.n) FROM s_native.arr",
    ):
        resp = post(statement)
        assert resp.status_code == 422, (statement, resp.json())
        assert "VARIANT-typed array" in resp.json()["detail"]["message"]

    # A derived table's alias has no catalog meaning: resolving it to a real table of the same
    # name reached DuckDB with the wrong type (LIST_SORT over a VARIANT -> an OOM error).
    assert post("CREATE TABLE nlist_src AS SELECT ARRAY_CONSTRUCT('x') AS n").status_code == 200
    assert post("CREATE TABLE vlist_src (n VARIANT)").status_code == 200
    assert post("INSERT INTO vlist_src SELECT PARSE_JSON('[\"b\",\"aa\"]')").status_code == 200
    resp = post(
        "SELECT ARRAY_SORT(nlist_src.n) FROM (SELECT n FROM vlist_src) AS nlist_src"
    )
    assert resp.status_code == 422, resp.json()
    assert "VARIANT-typed array" in resp.json()["detail"]["message"]
    assert post("SELECT ARRAY_SORT(nlist_src.n) FROM nlist_src").json()["data"] == [
        [['x']]
    ]

    # A query-scoped alias that happens to match a catalog column name is not resolved either.
    assert post("CREATE TABLE nums_tbl AS SELECT ARRAY_CONSTRUCT(1) AS nums").status_code == 200
    resp = post(
        'SELECT ARRAY_SORT(nums) FROM (SELECT PARSE_JSON(\'[9,100]\') AS nums) x'
    )
    assert resp.status_code == 422, resp.json()
    assert "VARIANT-typed array" in resp.json()["detail"]["message"]


def test_array_rewrites_keep_the_resolver_through_nesting(client):
    """A nested array call must not lose the bare column's type.

    Re-entering the rewrite pass on an operand used to reset the resolver to None, so every
    wrapping call treated a real ``VARCHAR[]``/``INTEGER[]`` column as an unknown VARIANT and
    ``ARRAY_SORT(ARRAY_UNIQUE(n))`` was rejected while ``ARRAY_SORT(n)`` worked.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post(
        "CREATE TABLE nested (n VARCHAR[], nums INTEGER[])"
    ).status_code == 200
    assert post(
        "INSERT INTO nested SELECT ARRAY_CONSTRUCT('b','a'), ARRAY_CONSTRUCT(30,4,100)"
    ).status_code == 200

    for statement, expected in (
        ("SELECT ARRAY_SORT(ARRAY_UNIQUE(n)) FROM nested", [[['a', 'b']]]),
        ("SELECT ARRAY_SORT(ARRAY_DISTINCT(n)) FROM nested", [[['a', 'b']]]),
        ("SELECT ARRAY_MAX(ARRAY_SLICE(nums, 0, 1)) FROM nested", [[30]]),
        ("SELECT ARRAY_SORT(ARRAY_CAT(n, ARRAY_CONSTRUCT('z'))) FROM nested", [[['a', 'b', 'z']]]),
        ("SELECT ARRAY_SORT(ARRAY_APPEND(n, 'z')) FROM nested", [[['a', 'b', 'z']]]),
    ):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        assert resp.json()["data"] == expected, statement


def test_array_except_scales(client):
    """``ARRAY_EXCEPT`` over 1000 elements finishes promptly.

    The hand-built rank-based rewrite recomputed a prefix count per element (quadratic, ~240x
    slower than sqlglot's own multiset rendering at 500 elements), which made a single row of
    a large array take seconds.
    """
    import json as json_module
    import time

    payload = json_module.dumps({"a": list(range(1000)), "b": [0]})
    assert client.post(
        "/api/v2/statements", json={"statement": "CREATE TABLE big_arr (v VARIANT)"}
    ).status_code == 200
    assert client.post(
        "/api/v2/statements",
        json={"statement": "INSERT INTO big_arr SELECT PARSE_JSON(\'" + payload + "\')"},
    ).status_code == 200
    started = time.time()
    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT ARRAY_SIZE(ARRAY_EXCEPT(v:a, v:b)) FROM big_arr"},
    )
    elapsed = time.time() - started
    assert resp.status_code == 200, resp.json()
    assert resp.json()["data"] == [[999]]
    assert elapsed < 5, elapsed


def test_create_function_named_like_a_builtin(client):
    """A UDF may shadow a built-in name; its signature is not a call.

    The argument-count guard counted commas in the token stream, so the three-parameter
    signature of ``CREATE FUNCTION array_to_string`` was rejected as a bad built-in call.
    """
    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "CREATE FUNCTION array_to_string(a VARCHAR, b VARCHAR, c VARCHAR) "
            "RETURNS VARCHAR AS 'a || b || c'"
        },
    )
    assert resp.status_code == 200, resp.json()


def test_malformed_call_reports_the_parse_error(client):
    """An unterminated argument list must not be reported as a bad ARRAY_TO_STRING call."""
    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT ARRAY_TO_STRING(v:n, ',', 'X'"},
    )
    assert resp.status_code == 422
    message = resp.json()["detail"]["message"]
    assert "null-replacement" not in message
    assert "parse" in message.lower() or "expect" in message.lower()


def test_array_except_with_an_empty_native_list(client):
    """An empty ``ARRAY_CONSTRUCT()`` is a typed empty list, not an unbindable one."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    resp = post('SELECT ARRAY_EXCEPT(PARSE_JSON(\'["a","b"]\'), ARRAY_CONSTRUCT())')
    assert resp.status_code == 200, resp.json()
    assert resp.json()["data"] == [[['a', 'b']]]
    assert post(
        'SELECT ARRAYS_OVERLAP(PARSE_JSON(\'["a"]\'), ARRAY_CONSTRUCT())'
    ).json()["data"] == [[False]]
    assert post('SELECT ARRAY_CAT(PARSE_JSON(\'[1]\'), ARRAY_CONSTRUCT())').json()["data"] == [
        [[1]]
    ]


def test_array_ordering_over_a_variant_array_is_rejected(client):
    """DuckDB's list_sort/list_max/list_min order VARIANT by length, not by the ``<`` operator.

    ``['aa','ab','b']`` comes back ``['b','aa','ab']`` where Snowflake sorts it
    lexicographically, so an ordering function over a VARIANT array is a loud error rather
    than a silently wrong order. A JSON path (a ``JSON[]`` list) sorts lexicographically,
    which matches Snowflake for strings but not for numbers.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE ord (names VARIANT, j VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO ord SELECT PARSE_JSON('[\"aa\",\"ab\",\"b\"]'), "
            "PARSE_JSON('{\"names\":[\"aa\",\"ab\",\"b\"]}')"
        ).status_code
        == 200
    )

    for statement in (
        "SELECT ARRAY_SORT(names) FROM ord",
        "SELECT ARRAY_SORT(names::VARIANT) FROM ord",
        "SELECT ARRAY_SORT(ARRAY_UNIQUE(names)) FROM ord",
        "SELECT ARRAY_MAX(names) FROM ord",
        "SELECT ARRAY_MIN(names) FROM ord",
    ):
        resp = post(statement)
        assert resp.status_code == 422, (statement, resp.json())
        assert "VARIANT" in resp.json()["detail"]["message"]

    # An aggregate of a VARIANT column builds its list at run time, so it carries no VARIANT[]
    # cast for the guard to find; it is rejected the same way, which DuckDB would otherwise
    # order by length (['b','aa','ab'] staying ['b','aa','ab']).
    assert post("CREATE TABLE aggs (g INT, names VARIANT, n INTEGER)").status_code == 200
    assert (
        post(
            "INSERT INTO aggs VALUES (1, PARSE_JSON('\"b\"'), 30), "
            "(1, PARSE_JSON('\"aa\"'), 4), (1, PARSE_JSON('\"ab\"'), 100)"
        ).status_code
        == 200
    )
    for statement in (
        "SELECT ARRAY_SORT(ARRAY_AGG(names)) FROM aggs GROUP BY g",
        "SELECT ARRAY_MAX(ARRAY_AGG(names)) FROM aggs GROUP BY g",
        "SELECT ARRAY_MIN(ARRAY_AGG(names)) FROM aggs GROUP BY g",
    ):
        resp = post(statement)
        assert resp.status_code == 422, (statement, resp.json())
        assert "VARIANT" in resp.json()["detail"]["message"]
    # An aggregate of a native column, or of a VARIANT cast to VARCHAR, still orders.
    assert post("SELECT ARRAY_SORT(ARRAY_AGG(n)) FROM aggs GROUP BY g").json()["data"] == [
        [[4, 30, 100]]
    ]
    assert post(
        "SELECT ARRAY_SORT(ARRAY_AGG(CAST(names AS VARCHAR))) FROM aggs GROUP BY g"
    ).json()["data"] == [[['aa', 'ab', 'b']]]

    # The JSON path (stored as JSON[], which DuckDB sorts lexicographically) still works.
    sorted_json = post("SELECT ARRAY_SORT(j:names) FROM ord")
    assert sorted_json.status_code == 200, sorted_json.json()
    assert sorted_json.json()["data"] == [[['aa', 'ab', 'b']]]


def test_array_ordering_over_a_native_list_column(client):
    """A CTAS ``ARRAY_CONSTRUCT`` column is a real DuckDB list, so it orders natively.

    The guard above has to reject a VARIANT-typed array without rejecting a bare column that
    DuckDB already stores as ``VARCHAR[]``/``INTEGER[]``. The column type is resolved from the
    catalog, so a native list keeps DuckDB's own order -- including the numeric order a JSON
    or VARIANT array cannot give -- while a VARIANT column is still a loud error.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert (
        post(
            "CREATE TABLE nlist AS SELECT ARRAY_CONSTRUCT('b','a') AS names, "
            "ARRAY_CONSTRUCT(30,4,100) AS nums"
        ).status_code
        == 200
    )
    assert data("SELECT DISTINCT typeof(names), typeof(nums) FROM nlist") == [
        ['VARCHAR[]', 'INTEGER[]']
    ]

    assert data("SELECT ARRAY_SORT(names) FROM nlist") == [[['a', 'b']]]
    assert data("SELECT ARRAY_SORT(nlist.names) FROM nlist") == [[['a', 'b']]]
    assert data("SELECT ARRAY_MAX(names) FROM nlist") == [['b']]
    assert data("SELECT ARRAY_MIN(names) FROM nlist") == [['a']]
    # Numeric order: a JSON/VARIANT array would sort these as text (100, 30, 4).
    assert data("SELECT ARRAY_SORT(nums) FROM nlist") == [[[4, 30, 100]]]
    assert data("SELECT ARRAY_MAX(nums) FROM nlist") == [[100]]

    # The rest of the array family reads the same native column, encoded against its own
    # element type rather than through a VARIANT round-trip.
    assert data("SELECT ARRAY_CONTAINS('a', names) FROM nlist") == [[True]]
    assert data("SELECT ARRAY_SIZE(names) FROM nlist") == [[2]]
    assert data("SELECT ARRAY_POSITION('b', names) FROM nlist") == [[0]]
    assert data("SELECT ARRAY_EXCEPT(names, ARRAY_CONSTRUCT('a')) FROM nlist") == [[['b']]]
    assert data("SELECT ARRAY_INTERSECTION(nums, ARRAY_CONSTRUCT(4)) FROM nlist") == [[[4]]]

    # A JSON-typed column keeps the JSON path's semantics.
    assert post("CREATE TABLE jcol AS SELECT PARSE_JSON('[\"aa\",\"ab\"]') AS j").status_code == 200
    assert data("SELECT ARRAY_SORT(j) FROM jcol") == [[['aa', 'ab']]]
    assert data("SELECT ARRAY_SIZE(j) FROM jcol") == [[2]]

    # A VARIANT column is still rejected, named or not.
    assert post("CREATE TABLE vcol (v VARIANT)").status_code == 200
    assert post("INSERT INTO vcol SELECT PARSE_JSON('[3,1,2]')").status_code == 200
    for statement in ("SELECT ARRAY_SORT(v) FROM vcol", "SELECT ARRAY_MIN(vcol.v) FROM vcol"):
        resp = post(statement)
        assert resp.status_code == 422, (statement, resp.json())
        assert "VARIANT" in resp.json()["detail"]["message"]


def test_array_functions_over_staged_json(client, tmp_path):
    """The same fix covers a staged file's `$1:path` array."""
    json_path = tmp_path / "names.json"
    json_path.write_text(json.dumps({"names": ["ada", "bob"]}) + "\n")
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "SELECT ARRAY_CONTAINS('bob', $1:names) AS hit, "
            "ARRAY_SIZE($1:names) AS n, "
            "ARRAY_SLICE($1:names, 1, 1) AS second FROM @~/staged/names.json"
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["data"] == [[True, 2, ['bob']]]


def test_snowflake_date_functions(client):
    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT DATEADD(day, 1, '2024-01-01'::date) AS d"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"][0][0].startswith("2024-01-02")


def test_get_statement_result_by_handle(client):
    resp = client.post("/api/v2/statements", json={"statement": "SELECT 42 AS answer"})
    handle = resp.json()["statementHandle"]

    fetched = client.get(f"/api/v2/statements/{handle}")
    assert fetched.status_code == 200
    assert fetched.json()["data"] == [[42]]


def test_unknown_statement_handle_returns_404(client):
    resp = client.get("/api/v2/statements/does-not-exist")
    assert resp.status_code == 404


def test_invalid_sql_returns_422(client):
    resp = client.post("/api/v2/statements", json={"statement": "SELEKT * FROM"})
    assert resp.status_code == 422


def test_session_token_persists_use_database(client):
    token_resp = client.post("/session/token")
    token = token_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    use_resp = client.post(
        "/api/v2/statements", json={"statement": "USE DATABASE MYDB"}, headers=headers
    )
    assert use_resp.status_code == 200

    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "CREATE TABLE t2 (a INT); INSERT INTO t2 VALUES (99); SELECT * FROM t2",
            "parameters": {"MULTI_STATEMENT_COUNT": "0"},
        },
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == [[99]]


def test_database_and_schema_isolated_across_sessions(client):
    # Session A creates a table in its default namespace.
    token_a = client.post("/session/token").json()["token"]
    client.post(
        "/api/v2/statements",
        json={"statement": "CREATE TABLE isolated_t (a INT)"},
        headers={"Authorization": f"Bearer {token_a}"},
    )

    # A brand new anonymous session (no token) still gets the shared default
    # database/schema, so the table should be visible there too.
    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT COUNT(*) FROM isolated_t"},
    )
    assert resp.status_code == 200


def test_login_request_issues_token(client):
    resp = client.post("/session/v1/login-request")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["token"]


# -- Staging (PUT / GET / LIST / REMOVE / COPY INTO @stage) -------------------


def _stage_headers(client):
    token = client.post("/session/token").json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _write_csv(path, rows=((1, "x"), (2, "y"))):
    with open(path, "w") as fh:
        fh.write("a,b\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")


def _write_json(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_stage_put_list_copy_into_csv(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    headers = _stage_headers(client)

    put = client.post(
        "/api/v2/statements",
        json={"statement": f"PUT file://{csv_path} @~/staged AUTO_COMPRESS = FALSE"},
        headers=headers,
    )
    assert put.status_code == 200
    assert put.json()["data"] == [
        ["data.csv", "staged/data.csv", 12, 12, "NONE", "NONE", "UPLOADED", ""]
    ]

    listed = client.post(
        "/api/v2/statements", json={"statement": "LIST @~/staged"}, headers=headers
    ).json()
    assert [(row[0], row[1]) for row in listed["data"]] == [("staged/data.csv", 12)]

    client.post(
        "/api/v2/statements",
        json={"statement": "CREATE TABLE staged_t (a INT, b VARCHAR)"},
        headers=headers,
    )
    copied = client.post(
        "/api/v2/statements",
        json={
            "statement": "COPY INTO staged_t FROM @~/staged "
        "FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1)"
        },
        headers=headers,
    )
    assert copied.status_code == 200
    copied_rows = copied.json()["data"]
    assert len(copied_rows) == 1
    assert copied_rows[0][1] == "LOADED"
    assert copied_rows[0][3] == 2

    selected = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT * FROM staged_t ORDER BY a"},
        headers=headers,
    ).json()
    assert selected["data"] == [[1, "x"], [2, "y"]]


def test_stage_put_json_and_copy_into(client, tmp_path):
    json_path = tmp_path / "data.json"
    _write_json(json_path, [{"a": 3, "b": "z"}])
    headers = _stage_headers(client)

    put = client.post(
        "/api/v2/statements",
        json={"statement": f"PUT file://{json_path} @~/staged"},
        headers=headers,
    )
    assert put.status_code == 200

    client.post(
        "/api/v2/statements",
        json={"statement": "CREATE TABLE staged_j (a INT, b VARCHAR)"},
        headers=headers,
    )
    copied = client.post(
        "/api/v2/statements",
        json={
            "statement": "COPY INTO staged_j FROM @~/staged FILE_FORMAT = (TYPE = JSON)"
        },
        headers=headers,
    )
    assert copied.status_code == 200
    copied_rows = copied.json()["data"]
    assert len(copied_rows) == 1
    assert copied_rows[0][1] == "LOADED"
    assert copied_rows[0][3] == 1

    selected = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT * FROM staged_j"},
        headers=headers,
    ).json()
    assert selected["data"] == [[3, "z"]]


def test_stage_remove_and_get(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    headers = _stage_headers(client)

    client.post(
        "/api/v2/statements",
        json={"statement": f"PUT file://{csv_path} @~/staged AUTO_COMPRESS = FALSE"},
        headers=headers,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    got = client.post(
        "/api/v2/statements",
        json={"statement": f"GET @~/staged file://{out_dir}"},
        headers=headers,
    )
    assert got.status_code == 200
    assert got.json()["data"] == [["staged/data.csv", 12, "DOWNLOADED", ""]]
    assert (out_dir / "data.csv").read_text() == csv_path.read_text()

    removed = client.post(
        "/api/v2/statements",
        json={"statement": "REMOVE @~/staged"},
        headers=headers,
    )
    assert removed.json()["data"] == [["staged/data.csv", "removed", ""]]

    listed = client.post(
        "/api/v2/statements", json={"statement": "LIST @~/staged"}, headers=headers
    ).json()
    assert listed["data"] == []


def test_stage_ddl_create_show_drop(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    headers = _stage_headers(client)

    created = client.post(
        "/api/v2/statements",
        json={"statement": "CREATE STAGE my_stage"},
        headers=headers,
    )
    assert created.status_code == 200

    put = client.post(
        "/api/v2/statements",
        json={"statement": f"PUT file://{csv_path} @my_stage"},
        headers=headers,
    )
    assert put.status_code == 200

    shown = client.post(
        "/api/v2/statements", json={"statement": "SHOW STAGES"}, headers=headers
    ).json()
    assert [row[0] for row in shown["data"]] == ["my_stage"]

    dropped = client.post(
        "/api/v2/statements",
        json={"statement": "DROP STAGE my_stage"},
        headers=headers,
    )
    assert dropped.status_code == 200

    shown = client.post(
        "/api/v2/statements", json={"statement": "SHOW STAGES"}, headers=headers
    ).json()
    assert shown["data"] == []


def test_stage_put_without_overwrite_errors(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    headers = _stage_headers(client)
    sql = f"PUT file://{csv_path} @~/staged"

    assert client.post("/api/v2/statements", json={"statement": sql}, headers=headers).status_code == 200
    resp = client.post("/api/v2/statements", json={"statement": sql}, headers=headers)
    assert resp.status_code == 422
    assert "already exists" in resp.json()["detail"]["message"]

    ok = client.post(
        "/api/v2/statements",
        json={"statement": f"{sql} OVERWRITE=TRUE"},
        headers=headers,
    )
    assert ok.status_code == 200


# -- Querying staged files directly (SELECT ... FROM @stage/file) -------------


def _select(client, headers, statement):
    """Run a statement and return its parsed body, asserting it succeeded."""
    resp = client.post(
        "/api/v2/statements", json={"statement": statement}, headers=headers
    )
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _stage(client, headers, path, target):
    staged = client.post(
        "/api/v2/statements",
        # Parsing is what these tests exercise; AUTO_COMPRESS would rename the file.
        json={"statement": f"PUT file://{path} {target} AUTO_COMPRESS = FALSE"},
        headers=headers,
    )
    assert staged.status_code == 200, staged.json()


def test_select_from_staged_json_file(client, tmp_path):
    json_path = tmp_path / "data.json"
    _write_json(json_path, [{"col": 1, "a": "x"}, {"col": 2, "a": "y"}])
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    # A JSON record is exposed as a single `$1` value, as in Snowflake.
    selected = _select(client, headers, "SELECT * FROM @~/staged/data.json")
    assert selected["resultSetMetaData"]["rowType"][0]["name"] == "$1"
    assert selected["data"] == [[{"col": 1, "a": "x"}], [{"col": 2, "a": "y"}]]

    # `$1:path` extracts a field through the JSON path syntax.
    extracted = _select(
        client, headers, "SELECT $1:col AS c FROM @~/staged/data.json ORDER BY c"
    )
    assert extracted["data"] == [[1], [2]]


def test_select_from_staged_json_file_errors_on_missing_column(client, tmp_path):
    json_path = tmp_path / "data.json"
    _write_json(json_path, [{"col": 1}])
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT $2 FROM @~/staged/data.json"},
        headers=headers,
    )
    assert resp.status_code == 422
    assert "$2 does not exist" in resp.json()["detail"]["message"]


def test_select_from_staged_csv_file(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,x\n2,y\n")
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")

    # Without SKIP_HEADER the header line is data, as Snowflake's default CSV format has
    # SKIP_HEADER = 0.
    default = _select(client, headers, "SELECT $1 FROM @~/staged/data.csv ORDER BY 1")
    assert [row[0] for row in default["data"]] == ["1", "2", "a"]

    skipped = _select(
        client,
        headers,
        "SELECT $1, $2 FROM @~/staged/data.csv "
        "(FILE_FORMAT => (TYPE = CSV, SKIP_HEADER = 1)) ORDER BY $1",
    )
    assert skipped["data"] == [[1, "x"], [2, "y"]]
    assert [col["name"] for col in skipped["resultSetMetaData"]["rowType"]] == ["$1", "$2"]


def test_select_from_staged_csv_field_delimiter(client, tmp_path):
    csv_path = tmp_path / "data.psv"
    csv_path.write_text("1|x\n2|y\n")
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")

    selected = _select(
        client,
        headers,
        "SELECT $1, $2 FROM @~/staged/data.psv "
        "(FILE_FORMAT => (TYPE = CSV, FIELD_DELIMITER = '|')) ORDER BY $1",
    )
    assert selected["data"] == [[1, "x"], [2, "y"]]


def test_select_from_staged_csv_file_uses_snowflake_defaults(client, tmp_path):
    """Snowflake's CSV defaults: no quoting, empty fields and `\\N` read as NULL."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b,c\n1,,\n2,,\\N\n")
    quoted_path = tmp_path / "quoted.csv"
    quoted_path.write_text('x,"y"\n1,2\n')
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")
    _stage(client, headers, quoted_path, "@~/staged")

    default = _select(client, headers, "SELECT $1, $2, $3 FROM @~/staged/data.csv")
    assert default["data"] == [["a", "b", "c"], ["1", None, None], ["2", None, None]]

    # Snowflake only honours quoting when FIELD_OPTIONALLY_ENCLOSED_BY asks for it.
    unquoted = _select(client, headers, "SELECT $2 FROM @~/staged/quoted.csv")
    assert [row[0] for row in unquoted["data"]] == ['"y"', "2"]
    quoted = _select(
        client,
        headers,
        "SELECT $2 FROM @~/staged/quoted.csv "
        "(FILE_FORMAT => (TYPE = 'CSV', FIELD_OPTIONALLY_ENCLOSED_BY = '\"'))",
    )
    assert [row[0] for row in quoted["data"]] == ["y", "2"]

    kept_empty = _select(
        client,
        headers,
        "SELECT $2 FROM @~/staged/data.csv "
        "(FILE_FORMAT => (TYPE = CSV, EMPTY_FIELD_AS_NULL = FALSE))",
    )
    assert [row[0] for row in kept_empty["data"]] == ["b", "", ""]


def test_select_from_staged_csv_with_named_file_format(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,x\n")
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")

    _select(client, headers, "CREATE FILE FORMAT ff TYPE = CSV SKIP_HEADER = 1")
    selected = _select(
        client, headers, "SELECT $1, $2 FROM @~/staged/data.csv (FILE_FORMAT => 'ff')"
    )
    assert selected["data"] == [[1, "x"]]


def test_select_from_staged_parquet_file(client, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"n": [1, 2], "s": ["x", "y"]}), parquet_path)
    headers = _stage_headers(client)
    _stage(client, headers, parquet_path, "@~/staged")

    # Parquet keeps its own column names; `$n` addresses the nth column positionally.
    selected = _select(client, headers, "SELECT * FROM @~/staged/data.parquet ORDER BY n")
    assert [col["name"] for col in selected["resultSetMetaData"]["rowType"]] == ["n", "s"]
    assert selected["data"] == [[1, "x"], [2, "y"]]

    positional = _select(
        client, headers, "SELECT $1, $2 FROM @~/staged/data.parquet ORDER BY $1"
    )
    assert positional["data"] == [[1, "x"], [2, "y"]]


def test_select_from_stage_directory_prefix_and_pattern(client, tmp_path):
    _write_json(tmp_path / "part1.json", [{"col": 1}])
    _write_json(tmp_path / "part2.json", [{"col": 2}])
    headers = _stage_headers(client)
    for path in ("part1.json", "part2.json"):
        _stage(client, headers, tmp_path / path, "@~/staged")

    # A directory prefix reads every file under it.
    prefixed = _select(
        client, headers, "SELECT $1:col AS c FROM @~/staged/ ORDER BY c"
    )
    assert prefixed["data"] == [[1], [2]]

    patterned = _select(
        client,
        headers,
        "SELECT $1:col AS c FROM @~/staged (PATTERN => '.*part2.*')",
    )
    assert patterned["data"] == [[2]]


def test_select_from_staged_json_strip_outer_array(client, tmp_path):
    json_path = tmp_path / "array.json"
    json_path.write_text(json.dumps([{"col": 1}, {"col": 2}], indent=2))
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    # STRIP_OUTER_ARRAY = TRUE reads the file as one document and explodes the array.
    selected = _select(
        client,
        headers,
        "SELECT $1:col AS c FROM @~/staged/array.json "
        "(FILE_FORMAT => (TYPE = JSON, STRIP_OUTER_ARRAY = TRUE)) ORDER BY c",
    )
    assert selected["data"] == [[1], [2]]


def test_hash_agg_and_ratio_to_report(client):
    """``HASH_AGG`` keeps its documented contract and ``RATIO_TO_REPORT`` divides by the sum.

    DuckDB has neither function. ``HASH_AGG`` is a ``SUM(HASH(...))`` folded into a signed
    64-bit value: order-independent, duplicates count, NULLs count, never NULL, and empty
    input hashes to 0. ``RATIO_TO_REPORT`` is the row's value over its partition's sum, with
    an ORDER BY inside OVER ignored and a zero partition sum yielding NULL.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE hagg (g INT, x INT)").status_code == 200
    assert post("INSERT INTO hagg VALUES (1, 10), (1, 20), (1, NULL)").status_code == 200

    hashed = data("SELECT HASH_AGG(x) FROM hagg")[0][0]
    assert isinstance(hashed, int)
    # Deterministic and order-independent.
    assert data("SELECT HASH_AGG(x) FROM (SELECT x FROM hagg ORDER BY x DESC)")[0][0] == hashed
    # Empty input hashes to 0, never NULL; a NULL-only group still hashes.
    assert data("SELECT HASH_AGG(x) FROM hagg WHERE FALSE") == [[0]]
    assert data("SELECT HASH_AGG(x) FROM hagg WHERE x IS NULL")[0][0] != 0
    # Duplicates and multiple arguments are part of the fingerprint.
    assert data("SELECT HASH_AGG(g, x) FROM hagg")[0][0] != hashed
    assert post("SELECT HASH_AGG(*) FROM hagg").status_code == 422
    resp = post("SELECT HASH_AGG(x) OVER (PARTITION BY g) FROM hagg")
    assert resp.status_code == 422
    assert "window function" in resp.json()["detail"]["message"]

    assert post("CREATE TABLE ratio (g INT, v INT)").status_code == 200
    assert post("INSERT INTO ratio VALUES (1, 1), (1, 3), (2, 5), (2, 5)").status_code == 200
    assert data(
        "SELECT g, v, RATIO_TO_REPORT(v) OVER (PARTITION BY g) AS r "
        "FROM ratio ORDER BY g, v"
    ) == [[1, 1, 0.25], [1, 3, 0.75], [2, 5, 0.5], [2, 5, 0.5]]
    # ORDER BY inside OVER does not change the calculation.
    assert data(
        "SELECT g, v, RATIO_TO_REPORT(v) OVER (PARTITION BY g ORDER BY v) AS r "
        "FROM ratio ORDER BY g, v"
    ) == [[1, 1, 0.25], [1, 3, 0.75], [2, 5, 0.5], [2, 5, 0.5]]
    # Without PARTITION BY the sum spans the whole result.
    whole = data("SELECT RATIO_TO_REPORT(v) OVER () FROM ratio ORDER BY v")
    assert abs(whole[0][0] - 1 / 14) < 1e-12

    # A partition that sums to zero yields NULL instead of a division error.
    assert post("CREATE TABLE zero_sum (g INT, v INT)").status_code == 200
    assert post("INSERT INTO zero_sum VALUES (1, 0), (1, 0)").status_code == 200
    assert data("SELECT RATIO_TO_REPORT(v) OVER (PARTITION BY g) FROM zero_sum") == [
        [None],
        [None],
    ]


def test_staged_json_writes_coerce_to_the_target_columns(client, tmp_path):
    """A staged JSON value written into a text column is its text, not DuckDB's JSON serialization.

    Three paths are covered: ``INSERT ... SELECT $1:c`` and ``UPDATE ... SET c = s.$1:c`` (which
    stored ``"ada"``, quotes and all), a comparison between a VARCHAR column and a staged value
    (which raised ``Malformed JSON ... Input: "ada"``, where Snowflake coerces the text side to a
    VARIANT), and a ``COPY INTO ... FROM (SELECT ...)`` projection (which bypassed the translator
    and so disagreed with the same expression in a plain SELECT).
    """
    data = tmp_path / "rows.json"
    data.write_text('{"id": 1, "name": "ada"}\n{"id": 2, "name": "bob"}\n')
    headers = _stage_headers(client)
    _stage(client, headers, data, "@~/staged")

    def run(statement):
        resp = client.post(
            "/api/v2/statements", json={"statement": statement}, headers=headers
        )
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    # A plain SELECT already rendered the string's text; the write paths must agree with it.
    assert run("SELECT $1:name::VARCHAR AS n FROM @~/staged/rows.json ORDER BY $1:id") == [
        ["ada"],
        ["bob"],
    ]

    assert run("CREATE TABLE stage_t (id INT, name VARCHAR)") == [
        ["Statement executed successfully."]
    ]
    run("INSERT INTO stage_t (id, name) SELECT $1:id, $1:name FROM @~/staged/rows.json")
    assert run("SELECT id, name, LENGTH(name) FROM stage_t ORDER BY id") == [
        [1, "ada", 3],
        [2, "bob", 3],
    ]
    # The column list may be omitted (the table's own order applies) or preceded by an alias.
    assert run("CREATE TABLE stage_nolist (id INT, name VARCHAR)") == [
        ["Statement executed successfully."]
    ]
    run("INSERT INTO stage_nolist SELECT $1:id, $1:name FROM @~/staged/rows.json")
    assert run("SELECT id, name, LENGTH(name) FROM stage_nolist ORDER BY id") == [
        [1, "ada", 3],
        [2, "bob", 3],
    ]
    assert run("CREATE TABLE stage_alias (id INT, name VARCHAR)") == [
        ["Statement executed successfully."]
    ]
    run("INSERT INTO stage_alias AS x (id, name) SELECT $1:id, $1:name FROM @~/staged/rows.json")
    assert run("SELECT id, name, LENGTH(name) FROM stage_alias ORDER BY id") == [
        [1, "ada", 3],
        [2, "bob", 3],
    ]

    # A number staged into a text column becomes its text, as Snowflake's VARIANT coercion does.
    assert run("CREATE TABLE stage_num (n VARCHAR)") == [["Statement executed successfully."]]
    run("INSERT INTO stage_num (n) SELECT $1:id FROM @~/staged/rows.json")
    assert run("SELECT n, LENGTH(n) FROM stage_num ORDER BY n") == [["1", 1], ["2", 1]]

    # UPDATE reports Snowflake's two columns: rows updated, and multi-joined rows.
    assert run("UPDATE stage_t SET name = s.$1:name FROM @~/staged/rows.json s "
               "WHERE stage_t.id = s.$1:id") == [[2, 0]]
    assert run("SELECT name FROM stage_t ORDER BY id") == [["ada"], ["bob"]]
    assert run(
        "SELECT COUNT(*) FROM stage_t t, @~/staged/rows.json s WHERE t.name = s.$1:name"
    ) == [[2]]
    # The WHERE comparison coerces the VARCHAR side just like the SET value: comparing it
    # with the staged JSON value used to raise ``Malformed JSON``.
    assert run("UPDATE stage_t SET name = 'x'") == [[2, 0]]
    assert run("UPDATE stage_t SET name = s.$1:name FROM @~/staged/rows.json s "
               "WHERE stage_t.name = s.$1:name") == [[0, 0]]
    assert run("UPDATE stage_t SET name = s.$1:name FROM @~/staged/rows.json s "
               "WHERE stage_t.id = s.$1:id") == [[2, 0]]
    assert run("SELECT name FROM stage_t ORDER BY id") == [["ada"], ["bob"]]

    # A derived relation over the stage is still a staged JSON value: its projected
    # ``$1:name`` column is JSON, so the SET value and comparison are coerced too.
    assert run("CREATE TABLE stage_derived (id INT, name VARCHAR)") == [
        ["Statement executed successfully."]
    ]
    run("INSERT INTO stage_derived SELECT $1:id, $1:name FROM @~/staged/rows.json")
    assert run("UPDATE stage_derived SET name = s.name FROM "
               "(SELECT $1:id AS id, $1:name AS name FROM @~/staged/rows.json) s "
               "WHERE stage_derived.id = s.id") == [[2, 0]]
    assert run("SELECT id, name, LENGTH(name) FROM stage_derived ORDER BY id") == [
        [1, "ada", 3],
        [2, "bob", 3],
    ]

    # The projection form now goes through the same rewrites as a plain SELECT.
    assert run("CREATE TABLE stage_p (name VARCHAR)") == [["Statement executed successfully."]]
    run("COPY INTO stage_p FROM (SELECT $1:name::VARCHAR FROM @~/staged/rows.json)")
    assert run("SELECT name FROM stage_p ORDER BY name") == [["ada"], ["bob"]]

    # A JSON value written into a VARIANT column is untouched: still a VARIANT holding the string
    # (TYPEOF names the type *stored* in the VARIANT, which is VARCHAR for that string).
    assert run("CREATE TABLE stage_v (v VARIANT)") == [["Statement executed successfully."]]
    run("INSERT INTO stage_v (v) SELECT $1:name FROM @~/staged/rows.json")
    assert run("SELECT typeof(v), v::VARCHAR FROM stage_v ORDER BY 1 LIMIT 1") == [
        ["VARCHAR", "ada"]
    ]


def test_semi_structured_comparisons_coerce_the_text_side(client):
    """Comparing text with a staged/JSON value coerces the text side, as Snowflake does.

    Snowflake coerces the VARCHAR operand to a VARIANT, so a text column, a string literal and a
    text-valued expression all compare as JSON values. DuckDB instead parses the text as JSON and
    raises -- or, for ``LIKE``, silently answers no rows because it compares the serialization.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE cmp_t (name VARCHAR, num INT)").status_code == 200
    assert post("INSERT INTO cmp_t VALUES ('ada', 9)").status_code == 200
    staged = '''(SELECT PARSE_JSON('{"name":"ada","num":9}') AS j) s'''

    for predicate in (
        "cmp_t.name = s.j:name",
        "s.j:name = cmp_t.name",
        "'ada' = s.j:name",
        "LOWER(cmp_t.name) = s.j:name",
        "UPPER(cmp_t.name) = UPPER(s.j:name)",
    ):
        assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE {predicate}") == [[1]], predicate

    # Inequality, LIKE/ILIKE (which used to answer no rows) and IN with a mixed list.
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE cmp_t.name <> s.j:name") == [[0]]
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE s.j:name LIKE 'a%'") == [[1]]
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE s.j:name ILIKE 'A%'") == [[1]]
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE cmp_t.name IN (s.j:name)") == [[1]]
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE 'ada' IN (s.j:name, 'zzz')") == [
        [1]
    ]
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE cmp_t.num IN (s.j:num, 3)") == [[1]]
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE cmp_t.name IN ('zzz')") == [[0]]
    # A number still compares as a number, and ordering keeps numeric semantics.
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE cmp_t.num = s.j:num") == [[1]]
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE cmp_t.num < s.j:num") == [[0]]
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE 8 < s.j:num") == [[1]]
    assert data(f"SELECT COUNT(*) FROM cmp_t, {staged} WHERE 10 < s.j:num") == [[0]]


def test_copy_json_array_document_is_one_record_without_strip(client, tmp_path):
    """Both JSON readers agree: without STRIP_OUTER_ARRAY an array document is one record.

    ``read_json_auto`` auto-detects an outer array and explodes it, so the direct COPY INTO
    loaded ``[{...},{...}]`` as one row per element while the projected reader loaded it as a
    single record. The direct reader now forces the newline-delimited format, so an array
    document is that one record's value (Snowflake's model) and a pretty-printed document is
    rejected by the same check on both paths.
    """
    ndjson = tmp_path / "rows.json"
    ndjson.write_text('{"a": 1}\n{"a": 2}\n')
    single_line_array = tmp_path / "array.json"
    single_line_array.write_text('[{"a": 1}, {"a": 2}]\n')
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps([{"a": 1}, {"a": 2}], indent=2))
    headers = _stage_headers(client)
    _stage(client, headers, ndjson, "@~/staged")
    _stage(client, headers, single_line_array, "@~/staged")
    _stage(client, headers, pretty, "@~/staged")

    def copy(statement):
        return client.post(
            "/api/v2/statements", json={"statement": statement}, headers=headers
        )

    def count(table):
        resp = client.post(
            "/api/v2/statements",
            json={"statement": f"SELECT COUNT(*) FROM {table}"},
            headers=headers,
        )
        assert resp.status_code == 200, resp.json()
        return resp.json()["data"][0][0]

    for form, source in (
        ("direct", "COPY INTO {table} FROM @~/staged/{name} FILE_FORMAT = (TYPE = JSON)"),
        (
            "projected",
            "COPY INTO {table} FROM (SELECT $1 FROM @~/staged/{name}) "
            "FILE_FORMAT = (TYPE = JSON)",
        ),
    ):
        ok = copy("CREATE OR REPLACE TABLE arr_t (v VARIANT)")
        assert ok.status_code == 200, ok.json()
        assert copy(source.format(table="arr_t", name="rows.json")).status_code == 200
        assert count("arr_t") == 2, form

        ok = copy("CREATE OR REPLACE TABLE arr_t (v VARIANT)")
        assert ok.status_code == 200, ok.json()
        assert copy(source.format(table="arr_t", name="array.json")).status_code == 200
        # One record, holding the array -- not one row per element.
        assert count("arr_t") == 1, form

        ok = copy("CREATE OR REPLACE TABLE arr_t (v VARIANT)")
        assert ok.status_code == 200, ok.json()
        rejected = copy(source.format(table="arr_t", name="pretty.json"))
        assert rejected.status_code == 422, (form, rejected.json())
        assert "newline-delimited" in rejected.json()["detail"]["message"], form

    # STRIP_OUTER_ARRAY stays available on the projected form, where it explodes the array.
    assert copy(
        "CREATE OR REPLACE TABLE arr_t (v VARIANT)"
    ).status_code == 200
    stripped = copy(
        "COPY INTO arr_t FROM (SELECT $1 FROM @~/staged/array.json) "
        "FILE_FORMAT = (TYPE = JSON, STRIP_OUTER_ARRAY = TRUE)"
    )
    assert stripped.status_code == 200, stripped.json()
    assert count("arr_t") == 2


def test_select_from_staged_file_rejects_non_newline_json(client, tmp_path):
    json_path = tmp_path / "pretty.json"
    json_path.write_text(json.dumps({"col": 1}, indent=2))
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT $1 FROM @~/staged/pretty.json"},
        headers=headers,
    )
    assert resp.status_code == 422
    assert "not newline-delimited JSON" in resp.json()["detail"]["message"]


def test_select_from_stage_rejects_mixed_formats(client, tmp_path):
    _write_json(tmp_path / "data.json", [{"col": 1}])
    (tmp_path / "data.csv").write_text("1,x\n")
    headers = _stage_headers(client)
    for path in ("data.json", "data.csv"):
        _stage(client, headers, tmp_path / path, "@~/staged")

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT $1 FROM @~/staged"},
        headers=headers,
    )
    assert resp.status_code == 422
    assert "mixes file formats" in resp.json()["detail"]["message"]


def test_select_from_stage_rejects_unsupported_and_missing(client, tmp_path):
    json_path = tmp_path / "data.json"
    _write_json(json_path, [{"col": 1}])
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    unsupported = client.post(
        "/api/v2/statements",
        json={
            "statement": "SELECT $1 FROM @~/staged/data.json (FILE_FORMAT => (TYPE = AVRO))"
        },
        headers=headers,
    )
    assert unsupported.status_code == 422
    assert "AVRO" in unsupported.json()["detail"]["message"]

    missing = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT $1 FROM @~/staged/nope.json"},
        headers=headers,
    )
    assert missing.status_code == 422
    assert "No files found" in missing.json()["detail"]["message"]


def test_select_from_multiple_stages_is_rejected(client, tmp_path):
    json_path = tmp_path / "data.json"
    _write_json(json_path, [{"col": 1}])
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT a.$1 FROM @~/staged/data.json a, @~/staged/data.json b"},
        headers=headers,
    )
    assert resp.status_code == 422
    assert "only one staged file location" in resp.json()["detail"]["message"]


def test_stage_options_accept_spaces_around_equals(client, tmp_path):
    """`PATTERN = '...'` (Snowflake's spelling) must not be dropped for a no-op pattern."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "people.parquet"
    pq.write_table(pa.table({"n": [1, 2]}), parquet_path)
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    headers = _stage_headers(client)
    _stage(client, headers, parquet_path, "@~/spaced")
    _stage(client, headers, csv_path, "@~/spaced")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    got = _select(
        client, headers, f"GET @~/spaced file://{out_dir} PATTERN = '.*[.]parquet'"
    )
    assert [row[0] for row in got["data"]] == ["spaced/people.parquet"]
    assert sorted(os.listdir(out_dir)) == ["people.parquet"]

    listed = _select(client, headers, "LIST @~/spaced PATTERN = '.*[.]csv'")
    assert [row[0] for row in listed["data"]] == ["spaced/data.csv"]

    # A PUT flag written with spaces still overwrites (it used to become `PARQUET`-style junk).
    again = _select(
        client, headers, f"PUT file://{csv_path} @~/spaced OVERWRITE = TRUE"
    )
    assert again["data"][0][6] == "UPLOADED"


def test_copy_into_parquet_file(client, tmp_path):
    """Staged Parquet loads into a table, with or without an explicit TYPE."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "people.parquet"
    pq.write_table(pa.table({"name": ["ada", "bob"], "age": [36, 41]}), parquet_path)
    headers = _stage_headers(client)
    _stage(client, headers, parquet_path, "@~/staged")

    _select(client, headers, "CREATE TABLE pq_inferred (name VARCHAR, age INT)")
    _select(client, headers, "CREATE TABLE pq_typed (name VARCHAR, age INT)")

    # An exact file reference is enough: the extension selects the reader.
    inferred = _select(client, headers, "COPY INTO pq_inferred FROM @~/staged/people.parquet")
    assert inferred["data"][0][3] == 2
    typed = _select(
        client,
        headers,
        "COPY INTO pq_typed FROM @~/staged/people.parquet FILE_FORMAT = (TYPE = PARQUET)",
    )
    assert typed["data"][0][3] == 2

    for table in ("pq_inferred", "pq_typed"):
        rows = _select(client, headers, f"SELECT * FROM {table} ORDER BY age")
        assert rows["data"] == [["ada", 36], ["bob", 41]]


def test_copy_into_from_exact_file_reference(client, tmp_path):
    """`COPY INTO t FROM @stage/dir/file.csv` names one file, not a directory prefix."""
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")

    _select(client, headers, "CREATE TABLE from_file (a INT, b VARCHAR)")
    copied = _select(
        client,
        headers,
        "COPY INTO from_file FROM @~/staged/data.csv "
        "FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1)",
    )
    assert copied["data"][0][3] == 2
    rows = _select(client, headers, "SELECT * FROM from_file ORDER BY a")
    assert rows["data"] == [[1, "x"], [2, "y"]]


def test_copy_into_pattern_filters_staged_files(client, tmp_path):
    """`PATTERN` limits which staged files a COPY INTO loads (Snowflake regex semantics)."""
    for name in ("part1.csv", "part2.csv"):
        (tmp_path / name).write_text(f"1,{name}\n")
    _write_json(tmp_path / "other.json", [{"a": 9}])
    headers = _stage_headers(client)
    for name in ("part1.csv", "part2.csv", "other.json"):
        _stage(client, headers, tmp_path / name, "@~/staged")

    _select(client, headers, "CREATE TABLE patterned (a INT, b VARCHAR)")
    first = _select(
        client,
        headers,
        "COPY INTO patterned FROM @~/staged PATTERN = '.*part1[.]csv' "
        "FILE_FORMAT = (TYPE = CSV)",
    )
    assert first["data"][0][3] == 1
    both = _select(
        client,
        headers,
        "COPY INTO patterned FROM @~/staged PATTERN = '.*[.]csv' "
        "FILE_FORMAT = (TYPE = CSV)",
    )
    assert len(both["data"]) == 2
    assert sum(row[3] for row in both["data"]) == 2
    rows = _select(client, headers, "SELECT b FROM patterned ORDER BY b")
    assert [row[0] for row in rows["data"]] == ["part1.csv", "part1.csv", "part2.csv"]

    # A pattern that matches nothing is an error, not a silent no-op.
    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "COPY INTO patterned FROM @~/staged PATTERN = '.*nope.*' "
            "FILE_FORMAT = (TYPE = CSV)"
        },
        headers=headers,
    )
    assert resp.status_code == 422
    assert "No files found" in resp.json()["detail"]["message"]


def test_copy_into_stage_source_is_not_rewritten_as_a_query(client, tmp_path):
    """COPY INTO names its source with a stage reference; that must keep working."""
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")

    _select(client, headers, "CREATE TABLE copied (a INT, b VARCHAR)")
    copied = _select(
        client,
        headers,
        "COPY INTO copied FROM @~/staged "
        "FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1)",
    )
    assert copied["data"][0][3] == 2


def test_copy_into_stage_subquery_applies_projection(client, tmp_path):
    """`COPY INTO t FROM (SELECT ... FROM @stage)` transforms each record on load."""
    json_path = tmp_path / "data.json"
    _write_json(json_path, [{"name": "a", "age": 3}, {"name": "b", "age": 4}])
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    _select(client, headers, "CREATE TABLE projected (n VARCHAR, a INT)")
    copied = _select(
        client,
        headers,
        "COPY INTO projected FROM (SELECT $1:name AS n, $1:age AS a "
        "FROM @~/staged/data.json) FILE_FORMAT = (TYPE = JSON)",
    )
    assert copied["data"][0][3] == 2

    # The projection, not a 1:1 column copy, is what landed in the table.
    stored = _select(client, headers, "SELECT * FROM projected ORDER BY a")
    assert stored["data"] == [["\"a\"", 3], ["\"b\"", 4]]


def test_copy_into_stage_subquery_honours_strip_outer_array(client, tmp_path):
    """A projected copy of an outer-array JSON file explodes the array, as the straight
    copy does, instead of reading the whole array as one NULL record."""
    json_path = tmp_path / "array.json"
    json_path.write_text(json.dumps([{"a": 1}, {"a": 2}]) + "\n")
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    _select(client, headers, "CREATE TABLE projected (a INT)")
    copied = _select(
        client,
        headers,
        "COPY INTO projected FROM (SELECT $1:a AS a FROM @~/staged/array.json) "
        "FILE_FORMAT = (TYPE = JSON, STRIP_OUTER_ARRAY = TRUE)",
    )
    assert copied["data"][0][3] == 2
    assert _select(client, headers, "SELECT * FROM projected ORDER BY a")["data"] == [
        [1],
        [2],
    ]


def test_copy_into_stage_subquery_rejects_csv_projection(client, tmp_path):
    """The projection is JSON-only; a CSV source must not silently drop it."""
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")

    _select(client, headers, "CREATE TABLE projected (a INT, b VARCHAR)")
    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "COPY INTO projected FROM (SELECT $1 FROM @~/staged/data.csv) "
            "FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1)"
        },
        headers=headers,
    )
    assert resp.status_code == 422
    assert "projection only for staged JSON" in resp.json()["detail"]["message"]


def test_copy_into_stage_subquery_rejects_dropped_clause(client, tmp_path):
    """A WHERE (or any non-projection clause) must be rejected, not silently ignored."""
    json_path = tmp_path / "data.json"
    _write_json(json_path, [{"a": 1, "b": 10}, {"a": 2, "b": 20}])
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    _select(client, headers, "CREATE TABLE projected (a INT)")
    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "COPY INTO projected FROM "
            "(SELECT $1:a AS a FROM @~/staged/data.json WHERE $1:b = 10) "
            "FILE_FORMAT = (TYPE = JSON)"
        },
        headers=headers,
    )
    assert resp.status_code == 422
    assert "unsupported clause" in resp.json()["detail"]["message"]


def test_correlated_subquery_limit_is_rejected(client):
    """Snowflake allows LIMIT/FETCH only in an uncorrelated scalar subquery.

    DuckDB executes a correlated subquery with LIMIT (and returns rows), so without a check
    the emulator would report results Snowflake rejects.
    """
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE ct (id INT, g INT)").status_code == 200
    assert post("INSERT INTO ct VALUES (1, 1), (2, 1), (3, 2)").status_code == 200

    for statement in (
        "SELECT id FROM ct c WHERE id = "
        "(SELECT id FROM ct WHERE g = c.g ORDER BY id LIMIT 1)",
        "SELECT id FROM ct c WHERE EXISTS "
        "(SELECT 1 FROM ct WHERE g = c.g LIMIT 1)",
        "SELECT id FROM ct c WHERE id IN "
        "(SELECT id FROM ct WHERE g = c.g FETCH FIRST 1 ROWS ONLY)",
    ):
        resp = post(statement)
        assert resp.status_code == 422, (statement, resp.json())
        detail = resp.json()["detail"]
        assert detail["code"] == "002031", statement
        assert detail["sqlState"] == "42601", statement
        assert "Unsupported subquery type cannot be evaluated" in detail["message"], statement

    # An uncorrelated scalar subquery may still use LIMIT.
    assert post(
        "SELECT id FROM ct WHERE id = (SELECT id FROM ct ORDER BY id DESC LIMIT 1)"
    ).json()["data"] == [[3]]

    # An unqualified outer column is still a correlation; the local source does not hold it.
    assert post("CREATE TABLE outr (id INT, g INT)").status_code == 200
    assert post("CREATE TABLE innr (x INT)").status_code == 200
    assert post("INSERT INTO outr VALUES (1, 10), (2, 20)").status_code == 200
    assert post("INSERT INTO innr VALUES (10), (20)").status_code == 200
    unqualified = post(
        "SELECT id, (SELECT max(x) FROM innr WHERE x = g LIMIT 1) AS m "
        "FROM outr ORDER BY id"
    )
    assert unqualified.status_code == 422, unqualified.json()
    detail = unqualified.json()["detail"]
    assert detail["code"] == "002031"
    assert detail["sqlState"] == "42601"


def test_derived_alias_types_do_not_leak_across_scopes(client):
    """A JSON projection under one alias must not type a text relation using the same alias."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    resp = post(
        "SELECT s.v::VARCHAR AS x FROM (SELECT 'abc' AS v) s "
        "UNION ALL "
        "SELECT s.v::VARCHAR AS x FROM (SELECT PARSE_JSON('\"xyz\"') AS v) s"
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["data"] == [["abc"], ["xyz"]]

    resp = post(
        "SELECT s.v::VARCHAR AS x FROM (SELECT 'abc' AS v) s "
        "WHERE EXISTS (SELECT 1 FROM (SELECT PARSE_JSON('\"x\"') AS v) s)"
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["data"] == [["abc"]]


def test_derived_stage_writes_cover_nested_union_and_bare_positional(client, tmp_path):
    """Derived JSON types survive wrappers, UNION arms, ``$1`` and ``SELECT *``."""
    json_path = tmp_path / "nested.json"
    json_path.write_text('{"id": 1, "name": "N300"}\n')
    string_path = tmp_path / "string.json"
    string_path.write_text('"N300"\n')
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/nested")
    _stage(client, headers, string_path, "@~/scalar")

    def run(statement):
        resp = client.post(
            "/api/v2/statements", json={"statement": statement}, headers=headers
        )
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert run("CREATE TABLE nested_t (id INT, name VARCHAR)") == [
        ["Statement executed successfully."]
    ]
    run(
        "INSERT INTO nested_t SELECT s.id, s.name "
        "FROM (SELECT q.id, q.name "
        "FROM (SELECT $1:id AS id, $1:name AS name FROM @~/nested/nested.json) q) s"
    )
    assert run("SELECT id, name, LENGTH(name) FROM nested_t") == [[1, "N300", 4]]

    # The raw staged relation's ``$1`` (and ``SELECT *`` over it) is JSON too.
    for table, projection in (("bare_t", "$1"), ("star_t", "*")):
        assert run(f"CREATE TABLE {table} (v VARCHAR)") == [
            ["Statement executed successfully."]
        ]
        run(f"INSERT INTO {table} SELECT {projection} FROM @~/scalar/string.json")
        assert run(f"SELECT v, LENGTH(v) FROM {table}") == [["N300", 4]]

    # A UNION of text and JSON is coerced to JSON values, then to the target's text.
    assert run("CREATE TABLE union_t (name VARCHAR)") == [
        ["Statement executed successfully."]
    ]
    run(
        "INSERT INTO union_t SELECT s.name FROM "
        "(SELECT 'abc' AS name UNION ALL SELECT PARSE_JSON('\"xyz\"') AS name) s"
    )
    assert run("SELECT name, LENGTH(name) FROM union_t ORDER BY name") == [
        ["abc", 3],
        ["xyz", 3],
    ]


def test_insert_select_from_staged_file(client, tmp_path):
    json_path = tmp_path / "data.json"
    _write_json(json_path, [{"col": 7}])
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    _select(client, headers, "CREATE TABLE loaded (j VARIANT)")
    inserted = _select(
        client, headers, "INSERT INTO loaded SELECT $1 FROM @~/staged/data.json"
    )
    assert inserted["data"] == [[1]]

    stored = _select(client, headers, "SELECT j:col AS c FROM loaded")
    assert stored["data"] == [[7]]


# -- Stage path containment, extremes and option strictness -------------------


def _session(headers):
    """The session behind a bearer token, for inspecting its stage directory."""
    import snowflake_emulator.sessions as sessions_module

    token = headers["Authorization"].split()[1]
    return sessions_module.get_session_manager().get_or_create(token)


def _stage_base(client, headers, ref="@my_stage"):
    import snowflake_emulator.stages as stages_module

    _select(client, headers, f"CREATE STAGE {ref.lstrip('@')}")
    manager = stages_module.get_stage_manager()
    return manager.stage_base_dir(_session(headers), stages_module.parse_stage_ref(ref))


def test_stage_reference_cannot_escape_its_stage(client, tmp_path):
    """A stage is a namespace: `..` must not read, list or delete outside it."""
    headers = _stage_headers(client)
    base = _stage_base(client, headers)
    os.makedirs(base, exist_ok=True)
    decoy = os.path.join(os.path.dirname(base), "decoy.csv")
    with open(decoy, "w") as handle:
        handle.write("secret,value\n")

    for statement in (
        "SELECT $1, $2 FROM @my_stage/../decoy.csv (FILE_FORMAT => (TYPE = CSV))",
        "LIST @my_stage/..",
        "REMOVE @my_stage/../decoy.csv",
        "REMOVE @my_stage/.. PATTERN = '.*'",
        "SELECT $1 FROM @../decoy.csv (FILE_FORMAT => (TYPE = CSV))",
    ):
        resp = client.post(
            "/api/v2/statements", json={"statement": statement}, headers=headers
        )
        assert resp.status_code == 422, (statement, resp.json())
        message = resp.json()["detail"]["message"]
        assert "escapes its stage location" in message or "Invalid stage reference" in message
    assert os.path.exists(decoy), "a REMOVE escaped the stage and deleted a sibling file"

    # Legitimate nested paths keep working.
    nested = tmp_path / "nested.csv"
    nested.write_text("1,x\n")
    _stage(client, headers, nested, "@my_stage/deep/dir")
    assert _select(
        client, headers, "SELECT $1, $2 FROM @my_stage/deep/dir/nested.csv"
    )["data"] == [[1, "x"]]


def test_list_and_remove_accept_an_exact_file(client, tmp_path):
    """`LIST`/`REMOVE @stage/dir/file.csv` name one file, not an empty directory walk."""
    headers = _stage_headers(client)
    _write_csv(tmp_path / "keep.csv")
    _write_csv(tmp_path / "drop.csv")
    _stage(client, headers, tmp_path / "keep.csv", "@~/staged")
    _stage(client, headers, tmp_path / "drop.csv", "@~/staged")

    listed = _select(client, headers, "LIST @~/staged/keep.csv")
    assert [row[0] for row in listed["data"]] == ["staged/keep.csv"]

    removed = _select(client, headers, "REMOVE @~/staged/drop.csv")
    assert [row[0] for row in removed["data"]] == ["staged/drop.csv"]
    remaining = _select(client, headers, "LIST @~/staged")
    assert [row[0] for row in remaining["data"]] == ["staged/keep.csv"]

    # GET names one file too (the same listing helper backs all three commands).
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    got = _select(client, headers, f"GET @~/staged/keep.csv file://{out_dir}")
    assert [row[0] for row in got["data"]] == ["staged/keep.csv"]
    assert (out_dir / "keep.csv").exists()


def test_select_from_staged_json_large_record(client, tmp_path):
    """Snowflake allows a semi-structured value up to 16MB, so a 2MB record must read."""
    json_path = tmp_path / "big.json"
    json_path.write_text(json.dumps({"s": "y" * 2_000_000}) + "\n")
    csv_path = tmp_path / "big.csv"
    csv_path.write_text("z" * 3_000_000 + ",tail\n")
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")
    _stage(client, headers, csv_path, "@~/staged")

    selected = _select(client, headers, "SELECT LENGTH($1:s) AS n FROM @~/staged/big.json")
    # The string's own length: a semi-structured string is coerced to its text without the
    # JSON quotes (Snowflake: "a string stored in VARIANT is preserved as is"), where this
    # used to count the serialized `"..."`.
    assert selected["data"] == [[2_000_000]]
    wide = _select(client, headers, "SELECT LENGTH($1) AS n FROM @~/staged/big.csv")
    assert wide["data"] == [[3_000_000]]


def test_select_from_staged_json_rejects_a_corrupt_record(client, tmp_path):
    """A malformed record must not become a NULL row (the guard only samples a prefix)."""
    json_path = tmp_path / "partial.json"
    valid = "\n".join(json.dumps({"i": i}) for i in range(1, 6))
    json_path.write_text(valid + '\n{"i": 6\n' + json.dumps({"i": 7}) + "\n")
    headers = _stage_headers(client)
    _stage(client, headers, json_path, "@~/staged")

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT $1:i AS i FROM @~/staged/partial.json"},
        headers=headers,
    )
    assert resp.status_code == 422
    assert "Malformed JSON" in resp.json()["detail"]["message"]

    # Blank lines are still skipped rather than rejected.
    blank = tmp_path / "blank.json"
    blank.write_text('{"i": 1}\n\n{"i": 2}\n')
    _stage(client, headers, blank, "@~/staged")
    rows = _select(client, headers, "SELECT $1:i AS i FROM @~/staged/blank.json ORDER BY i")
    assert rows["data"] == [[1], [2]]


def test_select_from_corrupt_gzip_errors(client, tmp_path):
    """DuckDB's CSV reader returns zero rows for a truncated gzip; that must not be silent."""
    corrupt = tmp_path / "bad.csv.gz"
    corrupt.write_bytes(b"\x1f\x8b\x08\x00truncated")
    headers = _stage_headers(client)
    _stage(client, headers, corrupt, "@~/gz")

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT $1 FROM @~/gz/bad.csv.gz (FILE_FORMAT => (TYPE = CSV))"},
        headers=headers,
    )
    assert resp.status_code == 422
    assert "decompress" in resp.json()["detail"]["message"]

    # A valid gzip stream still reads.
    valid = tmp_path / "ok.csv.gz"
    with gzip.open(valid, "wb") as handle:
        handle.write(b"1,x\n2,y\n")
    _stage(client, headers, valid, "@~/gz")
    assert _select(client, headers, "SELECT $1, $2 FROM @~/gz/ok.csv.gz")["data"] == [
        [1, "x"],
        [2, "y"],
    ]


def test_select_from_ragged_stage_location_widens(client, tmp_path):
    """Files of different widths must not silently drop the extra fields."""
    (tmp_path / "a.csv").write_text("1,x\n")
    (tmp_path / "b.csv").write_text("2,y,z\n")
    headers = _stage_headers(client)
    for name in ("a.csv", "b.csv"):
        _stage(client, headers, tmp_path / name, "@~/staged")

    selected = _select(client, headers, "SELECT * FROM @~/staged/ ORDER BY $1")
    assert [col["name"] for col in selected["resultSetMetaData"]["rowType"]] == ["$1", "$2", "$3"]
    assert selected["data"] == [[1, "x", None], [2, "y", "z"]]


def test_copy_into_compressed_csv_with_explicit_type(client, tmp_path):
    """`.csv.gz` is a CSV file: an explicit TYPE must not skip it."""
    gz = tmp_path / "data.csv.gz"
    with gzip.open(gz, "wb") as handle:
        handle.write(b"1,x\n2,y\n")
    headers = _stage_headers(client)
    _stage(client, headers, gz, "@~/staged")

    _select(client, headers, "CREATE TABLE gz_t (a INT, b VARCHAR)")
    copied = _select(
        client,
        headers,
        "COPY INTO gz_t FROM @~/staged/data.csv.gz FILE_FORMAT = (TYPE = CSV)",
    )
    assert copied["data"][0][3] == 2


def test_select_from_stage_rejects_unsupported_file_format_options(client, tmp_path):
    """An option the emulator cannot translate must fail, not silently change results."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("1, x \n")
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")

    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "SELECT $1 FROM @~/staged/data.csv "
            "(FILE_FORMAT => (TYPE = CSV, TRIM_SPACE = TRUE))"
        },
        headers=headers,
    )
    assert resp.status_code == 422
    assert "TRIM_SPACE" in resp.json()["detail"]["message"]

    # A default value is what DuckDB already does, and a full FILE_FORMAT clause still runs.
    accepted = _select(
        client,
        headers,
        "SELECT $1, $2 FROM @~/staged/data.csv (FILE_FORMAT => (TYPE = CSV, "
        "TRIM_SPACE = FALSE, SKIP_HEADER = 0, FIELD_DELIMITER = ',', "
        "EMPTY_FIELD_AS_NULL = TRUE, COMPRESSION = AUTO))",
    )
    assert accepted["data"] == [[1, " x "]]


def test_copy_into_returns_snowflake_load_report(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("1,x\n2,y\n")
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")
    _select(client, headers, "CREATE TABLE report_t (a INT, b VARCHAR)")

    copied = _select(
        client,
        headers,
        "COPY INTO report_t FROM @~/staged FILE_FORMAT = (TYPE = CSV)",
    )
    names = [col["name"] for col in copied["resultSetMetaData"]["rowType"]]
    assert names == [
        "file",
        "status",
        "rows_parsed",
        "rows_loaded",
        "error_limit",
        "errors_seen",
        "first_error",
        "first_error_line",
        "first_error_character",
        "first_error_column_name",
    ]
    assert len(copied["data"]) == 1
    row = copied["data"][0]
    assert row[0] == "staged/data.csv"
    assert row[1] == "LOADED"
    assert row[2] == 2
    assert row[3] == 2
    assert row[4] == 0
    assert row[5] == 0


def test_copy_into_skip_header_is_a_count(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\njunk,junk\n1,x\n2,y\n")
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")
    _select(client, headers, "CREATE TABLE skip_t (a INT, b VARCHAR)")

    _select(
        client,
        headers,
        "COPY INTO skip_t FROM @~/staged "
        "FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 2)",
    )
    assert _select(client, headers, "SELECT * FROM skip_t ORDER BY a")["data"] == [
        [1, "x"],
        [2, "y"],
    ]


def test_copy_into_multiple_files_reports_one_row_per_file(client, tmp_path):
    for name in ("part1.csv", "part2.csv"):
        (tmp_path / name).write_text("1,x\n")
    headers = _stage_headers(client)
    for name in ("part1.csv", "part2.csv"):
        _stage(client, headers, tmp_path / name, "@~/staged")
    _select(client, headers, "CREATE TABLE multi_copy (a INT, b VARCHAR)")

    copied = _select(
        client,
        headers,
        "COPY INTO multi_copy FROM @~/staged PATTERN = '.*[.]csv' "
        "FILE_FORMAT = (TYPE = CSV)",
    )
    assert sorted(row[0] for row in copied["data"]) == [
        "staged/part1.csv",
        "staged/part2.csv",
    ]
    assert sum(row[3] for row in copied["data"]) == 2


def test_copy_into_rejects_options_it_cannot_honor(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("1,x\n")
    headers = _stage_headers(client)
    _stage(client, headers, csv_path, "@~/staged")
    _select(client, headers, "CREATE TABLE reject_t (a INT, b VARCHAR)")

    trimmed = client.post(
        "/api/v2/statements",
        json={
            "statement": "COPY INTO reject_t FROM @~/staged "
            "FILE_FORMAT = (TYPE = CSV, TRIM_SPACE = TRUE)"
        },
        headers=headers,
    )
    assert trimmed.status_code == 422
    assert "TRIM_SPACE" in trimmed.json()["detail"]["message"]

    on_error = client.post(
        "/api/v2/statements",
        json={
            "statement": "COPY INTO reject_t FROM @~/staged "
            "FILE_FORMAT = (TYPE = CSV) ON_ERROR = 'CONTINUE'"
        },
        headers=headers,
    )
    assert on_error.status_code == 422
    assert "ON_ERROR" in on_error.json()["detail"]["message"]

    json_path = tmp_path / "data.json"
    json_path.write_text('{"a": 1}\n')
    _stage(client, headers, json_path, "@~/staged")
    strip = client.post(
        "/api/v2/statements",
        json={
            "statement": "COPY INTO reject_t FROM @~/staged/data.json "
            "FILE_FORMAT = (TYPE = JSON, STRIP_OUTER_ARRAY = TRUE)"
        },
        headers=headers,
    )
    assert strip.status_code == 422
    assert "STRIP_OUTER_ARRAY" in strip.json()["detail"]["message"]


def test_request_parameters(client):
    """The SQL API ``parameters`` object: multi-statement count, timezone, output formats.

    Snowflake's ``MULTI_STATEMENT_COUNT`` defaults to 1, so a chained body needs 0 or the exact
    count; ``TIMEZONE`` applies to the request; the four output formats decide the rowset's
    text, and an unknown value is a request error rather than a silent default.
    """
    def post(statement, parameters=None):
        body = {"statement": statement}
        if parameters is not None:
            body["parameters"] = parameters
        return client.post("/api/v2/statements", json=body)

    chained = "SELECT 1 AS a; SELECT 2 AS b"

    # Default 1: two statements are rejected unless the request opts in.
    rejected = post(chained)
    assert rejected.status_code == 422, rejected.json()
    assert "Multi-statement count 1" in rejected.json()["detail"]["message"]
    assert post(chained, {"MULTI_STATEMENT_COUNT": "0"}).json()["data"] == [[2]]
    assert post(chained, {"MULTI_STATEMENT_COUNT": "2"}).json()["data"] == [[2]]
    assert post(chained, {"MULTI_STATEMENT_COUNT": 2}).json()["data"] == [[2]]
    assert post(chained, {"MULTI_STATEMENT_COUNT": "1"}).status_code == 422
    assert post(chained, {"MULTI_STATEMENT_COUNT": "3"}).status_code == 422
    assert post("SELECT 1", {"MULTI_STATEMENT_COUNT": "1"}).json()["data"] == [[1]]
    negative = post(chained, {"MULTI_STATEMENT_COUNT": "-1"})
    assert negative.status_code == 422
    assert "negative" in negative.json()["detail"]["message"]
    assert post(chained, {"MULTI_STATEMENT_COUNT": "many"}).status_code == 422

    # TIMEZONE applies to the request: a zone without DST shifts the timestamp by its offset.
    utc = post("SELECT CURRENT_TIMESTAMP() AS ts", {"TIMEZONE": "UTC"}).json()["data"][0][0]
    shifted = post(
        "SELECT CURRENT_TIMESTAMP() AS ts", {"TIMEZONE": "Asia/Tokyo"}
    ).json()["data"][0][0]
    delta = datetime.datetime.fromisoformat(shifted) - datetime.datetime.fromisoformat(utc)
    assert abs(delta - datetime.timedelta(hours=9)) < datetime.timedelta(minutes=1)
    for unknown in ("Mars/Olympus", "+05:00"):
        invalid_zone = post("SELECT 1", {"TIMEZONE": unknown})
        assert invalid_zone.status_code == 422, unknown
        assert "Invalid TIMEZONE" in invalid_zone.json()["detail"]["message"]

    # Output formats decide the rowset text; the defaults are Snowflake's.
    assert post(
        "SELECT CAST('2024-03-04' AS DATE) AS d", {"DATE_OUTPUT_FORMAT": "DD.MM.YYYY"}
    ).json()["data"] == [["04.03.2024"]]
    assert post("SELECT CAST('2024-03-04' AS DATE) AS d").json()["data"] == [["2024-03-04"]]
    assert post(
        "SELECT CAST('2024-03-04 05:06:07.891234' AS TIMESTAMP_NTZ) AS ts",
        {"TIMESTAMP_OUTPUT_FORMAT": "YYYY/MM/DD HH24:MI:SS.FF6"},
    ).json()["data"] == [["2024/03/04 05:06:07.891234"]]
    assert post(
        "SELECT CAST('2024-03-04 05:06:07.891234' AS TIMESTAMP_NTZ) AS ts"
    ).json()["data"] == [["2024-03-04 05:06:07.891"]]
    assert post(
        "SELECT CAST('13:05:06' AS TIME) AS t", {"TIME_OUTPUT_FORMAT": "HH12:MI AM"}
    ).json()["data"] == [["01:05 PM"]]
    assert post("SELECT CAST('13:05:06' AS TIME) AS t").json()["data"] == [["13:05:06"]]

    # BINARY defaults to HEX and can be asked for BASE64.
    assert post("SELECT 'abc'::BINARY AS b").json()["data"] == [["616263"]]
    assert post(
        "SELECT 'abc'::BINARY AS b", {"BINARY_OUTPUT_FORMAT": "BASE64"}
    ).json()["data"] == [["YWJj"]]
    for bad in (
        {"DATE_OUTPUT_FORMAT": "YYYY-QQ"},
        {"BINARY_OUTPUT_FORMAT": "OCTAL"},
    ):
        resp = post("SELECT CURRENT_DATE()", bad)
        assert resp.status_code == 422, (bad, resp.json())

    # The formats travel with the result, so a later GET renders it the same way.
    submitted = post(
        "SELECT CAST('2024-03-04' AS DATE) AS d", {"DATE_OUTPUT_FORMAT": "DD.MM.YYYY"}
    )
    handle = submitted.json()["statementHandle"]
    fetched = client.get(f"/api/v2/statements/{handle}")
    assert fetched.json()["data"] == [["04.03.2024"]]


def test_async_statement_returns_202_and_is_pollable(client):
    submitted = client.post(
        "/api/v2/statements?async=true", json={"statement": "SELECT 1 AS x"}
    )
    assert submitted.status_code == 202
    body = submitted.json()
    assert "resultSetMetaData" not in body
    handle = body["statementHandle"]

    polled = client.get(f"/api/v2/statements/{handle}")
    assert polled.status_code == 200
    polled_body = polled.json()
    assert polled_body["data"] == [[1]]
    assert polled_body["resultSetMetaData"]["partitionInfo"] == [
        {"rowCount": 1, "uncompressedSize": 0}
    ]


def test_get_statement_supports_partition_polling(client):
    submitted = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT 1 AS x UNION ALL SELECT 2 AS x"},
    )
    handle = submitted.json()["statementHandle"]

    first = client.get(f"/api/v2/statements/{handle}?partition=0").json()
    assert first["data"] == [[1], [2]]
    assert first["resultSetMetaData"]["partitionInfo"] == [
        {"rowCount": 2, "uncompressedSize": 0}
    ]

    # The emulator has one materialized partition; later partitions are empty but
    # still report the same metadata.
    second = client.get(f"/api/v2/statements/{handle}?partition=1").json()
    assert second["data"] == []
    assert second["resultSetMetaData"]["numRows"] == 0
    assert second["resultSetMetaData"]["rowType"][0]["name"] == "X"


def test_get_and_get_path_read_a_variant_column(client):
    """`GET`/`GET_PATH` read the VARIANT's document; `GET` used to parse its internal repr."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE gv (v VARIANT)").status_code == 200
    assert (
        post("""INSERT INTO gv SELECT PARSE_JSON('{"s":"x","arr":[1,2]}')""").status_code
        == 200
    )
    assert data("SELECT GET(v,'s'), GET_PATH(v,'s'), GET_PATH(v,'arr[1]') FROM gv") == [
        ["x", "x", 2]
    ]
    # A JSON operand reads the same way.
    assert data("SELECT GET(PARSE_JSON('{\"a\":1}'),'a')") == [[1]]


def test_typeof_names_the_type_stored_in_the_variant(client):
    """`TYPEOF` answers Snowflake's names for the stored value, not DuckDB's JSON/VARIANT."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE tv (v VARIANT)").status_code == 200
    assert (
        post(
            "INSERT INTO tv SELECT "
            """PARSE_JSON('{"s":"x","i":1,"dec":1.5,"b":true,"arr":[1,2],"""
            """"o":{"k":1},"nl":null}')"""
        ).status_code
        == 200
    )

    assert data("SELECT TYPEOF(v) FROM tv") == [["OBJECT"]]
    # A JSON null is NULL_VALUE; a value that is not there (SQL NULL) is NULL.
    assert data(
        "SELECT TYPEOF(v:s), TYPEOF(v:i), TYPEOF(v:dec), TYPEOF(v:b), TYPEOF(v:arr), "
        "TYPEOF(v:o), TYPEOF(v:nl), TYPEOF(v:missing), TYPEOF(NULL) FROM tv"
    ) == [
        [
            "VARCHAR", "INTEGER", "DECIMAL", "BOOLEAN", "ARRAY", "OBJECT",
            "NULL_VALUE", "NULL", "NULL",
        ]
    ]


def test_flatten_output_columns_follow_snowflake(client):
    """SEQ counts the input record, KEY is NULL for arrays, PATH is `[i]`/`key` (docs table)."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE fv (v VARIANT)").status_code == 200
    assert (
        post(
            """INSERT INTO fv SELECT PARSE_JSON('{"arr":[1,2],"o":{"k":"v"}}')"""
        ).status_code
        == 200
    )

    # An array element: KEY NULL, PATH `[i]`, INDEX i, and one SEQ per input record.
    assert data(
        "SELECT f.seq, f.key, f.path, f.index, f.value::VARCHAR "
        "FROM fv t, LATERAL FLATTEN(input => t.v:arr) f ORDER BY f.index"
    ) == [[1, None, "[0]", 0, "1"], [1, None, "[1]", 1, "2"]]
    # An object member: the key is both KEY and PATH, and INDEX is NULL.
    assert data(
        "SELECT f.seq, f.key, f.path, f.index "
        "FROM fv t, LATERAL FLATTEN(input => t.v:o) f"
    ) == [[1, "k", "k", None]]
    # The PATH argument prefixes the path (`b[0]` in the docs, `o.k` for an object).
    assert data(
        "SELECT f.path FROM fv t, LATERAL FLATTEN(input => t.v, path => 'o') f"
    ) == [["o.k"]]
    # A zero-row OUTER expansion: SEQ 1 and an empty PATH, as the docs table shows.
    assert data(
        "SELECT seq, key, path, index, value FROM "
        "TABLE(FLATTEN(input => PARSE_JSON('[]'), outer => TRUE))"
    ) == [[1, None, "", None, None]]


def test_parse_json_rejects_duplicate_keys_and_maps_empty_to_null(client):
    """`PARSE_JSON` is strict about duplicates and returns NULL for an empty document."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert data("SELECT PARSE_JSON(''), PARSE_JSON('   ')") == [[None, None]]
    # A string that is only known at run time can be empty too.
    assert data("SELECT PARSE_JSON(x) FROM (SELECT '' x)") == [[None]]

    resp = post("""SELECT PARSE_JSON('{"a":1,"a":2}')""")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert (detail["sqlState"], detail["code"]) == ("22000", "100069")
    assert 'duplicate object attribute "a"' in detail["message"]

    # The 'd' parameter allows duplicates (last key wins); TRY_PARSE_JSON answers NULL.
    assert data("""SELECT PARSE_JSON('{"a":1,"a":2}', 'd')""") == [[{"a": 2}]]
    assert data("""SELECT TRY_PARSE_JSON('{"a":1,"a":2}')""") == [[None]]


def test_object_construct_omits_sql_null_pairs(client):
    """An OBJECT omits a pair whose key or value is SQL NULL; a JSON null keeps its key."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert data("SELECT OBJECT_CONSTRUCT('a', NULL, 'b', 2)") == [[{"b": 2}]]
    # A JSON null is not omitted, and is not the string 'null'.
    assert data("SELECT OBJECT_CONSTRUCT('a', PARSE_JSON('null'), 'b', 1)") == [
        [{"a": None, "b": 1}]
    ]
    # A SQL NULL key omits the pair too; dropping every pair is an empty object.
    assert data("SELECT OBJECT_CONSTRUCT(NULL, 1)") == [[{}]]
    # A run-time NULL value still keeps its key: a DuckDB STRUCT cannot drop a field.
    assert data("SELECT OBJECT_CONSTRUCT('a', x, 'b', 2) FROM (SELECT NULL x)") == [
        [{"a": None, "b": 2}]
    ]


def test_pad_with_an_empty_pad_string_inserts_no_padding(client):
    """`LPAD`/`RPAD` with an empty pad truncate but never pad, as the docs state."""
    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT LPAD('abc', 5, ''), RPAD('abc', 5, ''), "
        "LPAD('abc', 2, ''), RPAD('abc', 2, '')"},
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["data"] == [["abc", "abc", "ab", "ab"]]


def test_dateadd_and_date_trunc_keep_a_date(client):
    """A DATE input stays a DATE for a day-or-larger part, and is a TIMESTAMP below it."""
    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "SELECT DATEADD('day',2,'2024-01-30'::DATE) AS d1, "
            "DATEADD('hour',2,'2024-01-30'::DATE) AS d2, "
            "DATE_TRUNC('week','2024-01-03'::DATE) AS d3, "
            "DATE_TRUNC('hour','2024-01-03'::DATE) AS d4"
        },
    )
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert [col["type"] for col in body["resultSetMetaData"]["rowType"]] == [
        "date",
        "timestamp_ntz",
        "date",
        "timestamp_ntz",
    ]
    assert body["data"] == [
        ["2024-02-01", "2024-01-30 02:00:00.000", "2024-01-01", "2024-01-03 00:00:00.000"]
    ]

    # A DATE column is typed from the catalog, and a TIMESTAMP column keeps its own type.
    assert client.post(
        "/api/v2/statements",
        json={"statement": "CREATE TABLE dts (d DATE, t TIMESTAMP_NTZ)"},
    ).status_code == 200
    assert client.post(
        "/api/v2/statements",
        json={"statement": "INSERT INTO dts SELECT '2024-01-31', '2024-01-31 10:20:30'"},
    ).status_code == 200
    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "SELECT DATEADD(day,1,d) AS a, DATEADD(hour,1,d) AS b, "
            "DATE_TRUNC(day,d) AS c, DATE_TRUNC(hour,d) AS e, DATE_TRUNC(day,t) AS f FROM dts"
        },
    )
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert [col["type"] for col in body["resultSetMetaData"]["rowType"]] == [
        "date",
        "timestamp_ntz",
        "date",
        "timestamp_ntz",
        "timestamp_ntz",
    ]
    assert body["data"] == [
        [
            "2024-02-01",
            "2024-01-31 01:00:00.000",
            "2024-01-31",
            "2024-01-31 00:00:00.000",
            "2024-01-31 00:00:00.000",
        ]
    ]


def test_is_type_predicates_read_the_stored_type(client):
    """`IS_<type>` answers from the type stored in the VARIANT, not DuckDB's storage type."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert data(
        "SELECT IS_OBJECT(PARSE_JSON('{}')), IS_ARRAY(PARSE_JSON('[]')), "
        "IS_VARCHAR(PARSE_JSON('\"x\"')), IS_BOOLEAN(PARSE_JSON('true'))"
    ) == [[True, True, True, True]]
    # Snowflake stores a JSON decimal exactly, so it is DECIMAL rather than DOUBLE.
    assert data(
        "SELECT IS_INTEGER(PARSE_JSON('1')), IS_DECIMAL(PARSE_JSON('1.5')), "
        "IS_DOUBLE(PARSE_JSON('1.5'))"
    ) == [[True, True, False]]
    # A variant built from a typed value keeps that type.
    assert data(
        "SELECT IS_DATE(TO_VARIANT(TO_DATE('2024-02-24'))), "
        "IS_TIME(TO_VARIANT(TO_TIME('10:00:00'))), "
        "IS_TIMESTAMP_NTZ(TO_VARIANT(TO_TIMESTAMP('2024-02-24 10:00:00'))), "
        "IS_DOUBLE(TO_VARIANT(1.5::DOUBLE)), IS_BINARY(TO_VARIANT('ab'::BINARY))"
    ) == [[True, True, True, True, True]]
    # A SQL NULL answers NULL; a JSON null is a value of type NULL_VALUE.
    assert data(
        "SELECT IS_INTEGER(NULL), IS_INTEGER(PARSE_JSON('null')), "
        "IS_VARCHAR(PARSE_JSON('1'))"
    ) == [[None, False, False]]
    assert data(
        "SELECT TYPEOF(TO_VARIANT(TO_DATE('2024-02-24'))), TYPEOF(TO_VARIANT(1.5::DOUBLE))"
    ) == [["DATE", "DOUBLE"]]


def test_object_pick_delete_insert(client):
    """`OBJECT_PICK`/`OBJECT_DELETE`/`OBJECT_INSERT` select, drop and insert key-value pairs."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert data(
        """SELECT OBJECT_PICK(PARSE_JSON('{"a":1,"b":2,"c":3}'), 'a', 'c')"""
    ) == [[{"a": 1, "c": 3}]]
    # A key that is not present is ignored, and every key missing is an empty OBJECT.
    assert data("""SELECT OBJECT_PICK(PARSE_JSON('{"a":1}'), 'zz')""") == [[{}]]
    # The documented array-of-keys form.
    assert data(
        """SELECT OBJECT_PICK(PARSE_JSON('{"a":1,"b":2}'), PARSE_JSON('["a"]'))"""
    ) == [[{"a": 1}]]
    assert data(
        """SELECT OBJECT_DELETE(PARSE_JSON('{"a":1,"b":2}'), 'a')"""
    ) == [[{"b": 2}]]
    assert data("""SELECT OBJECT_DELETE(PARSE_JSON('{"a":1}'), 'a')""") == [[{}]]

    assert data("""SELECT OBJECT_INSERT(PARSE_JSON('{"a":1}'), 'b', 2)""") == [
        [{"a": 1, "b": 2}]
    ]
    assert data("""SELECT OBJECT_INSERT(PARSE_JSON('{"a":1}'), 'a', 2, TRUE)""") == [
        [{"a": 2}]
    ]
    # An existing key without updateFlag is an error.
    resp = post("""SELECT OBJECT_INSERT(PARSE_JSON('{"a":1}'), 'a', 2)""")
    assert resp.status_code == 422
    assert "Duplicate key" in resp.json()["detail"]["message"]
    # A SQL NULL value omits the pair; a JSON null value is inserted.
    assert data("""SELECT OBJECT_INSERT(PARSE_JSON('{"a":1}'), 'b', NULL)""") == [[{"a": 1}]]
    assert data(
        """SELECT OBJECT_INSERT(PARSE_JSON('{"a":1}'), 'b', PARSE_JSON('null'))"""
    ) == [[{"a": 1, "b": None}]]
    # The functions also accept an OBJECT (a struct-backed value).
    assert data("SELECT OBJECT_PICK(OBJECT_CONSTRUCT('a', 1, 'b', 2), 'a')") == [[{"a": 1}]]


def test_get_ignore_case_matches_a_field_name(client):
    """`GET_IGNORE_CASE` matches one field name without regard to case."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert data(
        """SELECT GET_IGNORE_CASE(PARSE_JSON('{"Aa":1,"b b":2}'), 'aa'),
                  GET_IGNORE_CASE(PARSE_JSON('{"Aa":1}'), 'zz')"""
    ) == [[1, None]]
    # A field name with a space is one key, not a path.
    assert data("""SELECT GET_IGNORE_CASE(PARSE_JSON('{"b b":2}'), 'B B')""") == [[2]]
    assert data("SELECT GET_IGNORE_CASE(NULL, 'a')") == [[None]]


def test_object_construct_star_uses_the_source_columns(client):
    """`OBJECT_CONSTRUCT(*)` takes the columns in scope, with the ILIKE/EXCLUDE filters."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE star_t (province VARCHAR, created_date DATE)").status_code == 200
    assert (
        post("INSERT INTO star_t SELECT 'Alberta', '2024-01-19'::DATE").status_code == 200
    )
    both = {"PROVINCE": "Alberta", "CREATED_DATE": "2024-01-19"}
    assert data("SELECT OBJECT_CONSTRUCT(*) FROM star_t") == [[both]]
    assert data("SELECT OBJECT_CONSTRUCT(t.*) FROM star_t t") == [[both]]
    assert data("SELECT OBJECT_CONSTRUCT(* ILIKE 'prov%') FROM star_t") == [
        [{"PROVINCE": "Alberta"}]
    ]
    assert data("SELECT OBJECT_CONSTRUCT(* EXCLUDE created_date) FROM star_t") == [
        [{"PROVINCE": "Alberta"}]
    ]


def test_array_size_of_a_non_array_variant_is_null(client):
    """A VARIANT that is not an array answers NULL; a real array keeps its size."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert data("""SELECT ARRAY_SIZE(PARSE_JSON('1')), ARRAY_SIZE(PARSE_JSON('{"a":1}'))""") == [
        [None, None]
    ]
    assert data("SELECT ARRAY_SIZE(PARSE_JSON('[1,2,3]')), ARRAY_SIZE(ARRAY_CONSTRUCT(1,2))") == [
        [3, 2]
    ]
    assert data("SELECT ARRAY_SIZE(NULL)") == [[None]]


def test_negative_array_index_answers_null(client):
    """The documented path grammar allows only non-negative indices; DuckDB reads -1."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert data(
        """SELECT v:arr[-1], v:arr[0] FROM (SELECT PARSE_JSON('{"arr":[10,20]}') v)"""
    ) == [[None, 10]]


def test_numeric_format_models(client):
    """`TO_CHAR`/`TO_VARCHAR` apply Snowflake's numeric format model (default sign blank)."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    # The documented examples from the SQL format models page.
    assert data("SELECT TO_VARCHAR(1.2, '999,999.999'), TO_VARCHAR(123.456, '999,999.999')") == [
        ["      1.2  ", "    123.456"]
    ]
    assert data(
        "SELECT TO_VARCHAR(1.2, 'S000,000.000'), TO_VARCHAR(-123456.789, 'S000,000.000')"
    ) == [["+000,001.200", "-123,456.789"]]
    assert data(
        "SELECT TO_VARCHAR(1.2, 'FM999,999.999'), "
        "TO_VARCHAR(-123456.789, 'FM999,999.999')"
    ) == [["1.2", "-123,456.789"]]
    assert data("SELECT TO_CHAR(1234, '9d999EE'), TO_CHAR(12, '\">\"99\"<\"')") == [
        [" 1.234E3", "> 12<"]
    ]
    assert data("SELECT TO_CHAR(-123.45, '999.99MI'), TO_VARCHAR(0.25, 'TM9%')") == [
        ["123.45-", "25%"]
    ]
    # A DATE/TIME format model keeps sqlglot's own translation.
    assert data("SELECT TO_CHAR('2024-01-02'::DATE, 'YYYY-MM-DD')") == [["2024-01-02"]]


def test_numeric_format_model_parsing(client):
    """`TO_NUMBER(text, model)` parses with the model; the default scale is 0 (docs)."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    # Default precision 38 / scale 0 rounds the parsed value, as documented for TO_NUMBER.
    assert data("SELECT TO_NUMBER('1,234.50', '9,999.99')") == [[1235]]
    assert data("SELECT TO_NUMBER('1,234.50', '9,999.99', 8, 2)") == [[1234.5]]
    assert data("SELECT TO_NUMBER('012,345.67', '000,000.00', 8, 2)") == [[12345.67]]
    assert data("SELECT TO_NUMBER('-1,234.50', '9,999.99', 8, 2)") == [[-1234.5]]
    # TRY_TO_NUMBER answers NULL; TO_NUMBER raises a data error.
    assert data("SELECT TRY_TO_NUMBER('bad', '999')") == [[None]]
    resp = post("SELECT TO_NUMBER('bad', '999')")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert (detail["sqlState"], detail["code"]) == ("22000", "100038")
    assert "Invalid number format" in detail["message"]
    # A semi-structured operand is converted to its text first (an unquoted string variant).
    assert data(
        "SELECT TO_NUMBER(j:s, '99,999.99', 8, 2), TO_CHAR(j:n, '999,999') FROM "
        "(SELECT PARSE_JSON('{\"s\":\"12,345.67\",\"n\":1234}') j)"
    ) == [[12345.67, "   1,234"]]
    # A model the emulator cannot honour is refused at translation time.
    resp = post("SELECT TO_CHAR(1.5, 'V99')")
    assert resp.status_code == 422
    assert "not supported" in resp.json()["detail"]["message"]


def test_binary_format_models(client):
    """`TO_CHAR`/`TO_VARCHAR` over BINARY use Snowflake's binary format models."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    # HEX is the default; BASE64 and UTF-8 are the other documented models.
    assert data("SELECT TO_CHAR('abc'::BINARY, 'HEX')") == [["616263"]]
    assert data("SELECT TO_VARCHAR('abc'::BINARY, 'BASE64')") == [["YWJj"]]
    assert data("SELECT TO_CHAR('abc'::BINARY, 'UTF-8')") == [["abc"]]
    assert data("SELECT TO_VARCHAR('abc'::BINARY)") == [["616263"]]
    # A text cast of BINARY uses the same default.
    assert data("SELECT CAST('abc'::BINARY AS VARCHAR)") == [["616263"]]
    resp = post("SELECT TO_CHAR('abc'::BINARY, 'NOPE')")
    assert resp.status_code == 422
    assert "Binary format model" in resp.json()["detail"]["message"]


def test_text_column_length_in_row_type(client):
    """`rowType.length` carries the declared text length, which DuckDB does not keep."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def lengths(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return [col["length"] for col in resp.json()["resultSetMetaData"]["rowType"]]

    assert post("CREATE TABLE len_t (s VARCHAR(5), c CHAR(3), d VARCHAR)").status_code == 200
    assert lengths("SELECT s, c, d FROM len_t") == [5, 3, 16777216]
    assert lengths(
        "SELECT CAST('x' AS VARCHAR(7)) AS a, 'lit' AS b, CAST('a' AS VARCHAR) AS c"
    ) == [7, 3, 16777216]
    # A non-text column reports no length.
    assert lengths("SELECT CAST(1.25 AS NUMBER(10,2)) AS n") == [None]


def test_mixed_alter_table_actions(client):
    """`ALTER TABLE ... ADD CONSTRAINT ..., ADD COLUMN ...` runs each action."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE mixed_alt (id INT, v INT)").status_code == 200
    assert data("ALTER TABLE mixed_alt ADD CONSTRAINT pk PRIMARY KEY (id), ADD COLUMN w INT") == [
        ["Statement executed successfully."]
    ]
    assert [row[0] for row in data("DESCRIBE TABLE mixed_alt")] == ["ID", "V", "W"]
    # The constraint is recorded but not enforced, as in Snowflake.
    assert data("INSERT INTO mixed_alt (id, v, w) VALUES (1, 1, 1), (1, 2, 3)") == [[2]]
    assert [row[6] for row in data("SHOW PRIMARY KEYS IN TABLE mixed_alt")] == ["PK"]
    assert data("ALTER TABLE mixed_alt ADD COLUMN x INT, DROP COLUMN v") == [
        ["Statement executed successfully."]
    ]
    assert [row[0] for row in data("DESCRIBE TABLE mixed_alt")] == ["ID", "W", "X"]


def test_sample_forms(client):
    """`SAMPLE` keeps Snowflake's row vs percentage semantics (and REPEATABLE)."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE TABLE sample_src (a INT)").status_code == 200
    assert data("INSERT INTO sample_src VALUES (1), (2), (3)") == [[3]]
    assert len(data("SELECT * FROM sample_src SAMPLE (2 ROWS)")) == 2
    assert len(data("SELECT * FROM sample_src SAMPLE (10 ROWS)")) == 3  # n > rows: all rows
    assert len(data("SELECT * FROM sample_src SAMPLE (100)")) == 3
    assert data("SELECT * FROM sample_src SAMPLE (0)") == []
    # A percentage sample is probabilistic: 1% of three rows can legitimately be empty.
    assert data("SELECT * FROM sample_src SAMPLE (1)") in ([], [[1]], [[2]], [[3]])
    first = sorted(row[0] for row in data("SELECT * FROM sample_src SAMPLE (50) REPEATABLE (7)"))
    second = sorted(row[0] for row in data("SELECT * FROM sample_src SAMPLE (50) REPEATABLE (7)"))
    assert first == second


def test_stage_metadata_backs_show_and_information_schema(client):
    """CREATE STAGE records the URL, owner, comment and FILE_FORMAT; DROP removes them."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post(
        "CREATE STAGE meta_st FILE_FORMAT = (TYPE = CSV FIELD_DELIMITER = '|') "
        "COMMENT = 'pipe stage'"
    ).status_code == 200
    assert data("SHOW STAGES LIKE 'META_ST'") == [
        [
            "meta_st",
            "EMULATOR_DB",
            "PUBLIC",
            "s3://sfemu/EMULATOR_DB.PUBLIC.meta_st",
        ]
    ]
    view = data(
        "SELECT STAGE_NAME, STAGE_URL, STAGE_TYPE, STAGE_OWNER, COMMENT "
        "FROM INFORMATION_SCHEMA.STAGES WHERE lower(STAGE_NAME) = 'meta_st'"
    )
    assert view == [
        [
            "meta_st",
            "s3://sfemu/EMULATOR_DB.PUBLIC.meta_st",
            "Internal Named",
            "ACCOUNTADMIN",
            "pipe stage",
        ]
    ]
    # An external stage is rejected rather than silently created as an internal one.
    resp = post("CREATE STAGE meta_ext URL = 's3://bucket/prefix'")
    assert resp.status_code == 422
    assert "External stages are not supported" in resp.json()["detail"]["message"]
    assert data("DROP STAGE meta_st") == [["Stage meta_st successfully dropped."]]
    assert data(
        "SELECT STAGE_NAME FROM INFORMATION_SCHEMA.STAGES WHERE lower(STAGE_NAME) = 'meta_st'"
    ) == []


def test_file_formats_are_persisted_and_visible(client):
    """Named file formats live in the catalog side table, so SHOW and the view agree."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post(
        "CREATE FILE FORMAT pipes TYPE = CSV FIELD_DELIMITER = '|' SKIP_HEADER = 1"
    ).status_code == 200
    assert post("CREATE FILE FORMAT jsf TYPE = JSON").status_code == 200
    assert [row[0] for row in data("SHOW FILE FORMATS")] == ["JSF", "PIPES"]
    view = data(
        "SELECT FILE_FORMAT_NAME, FILE_FORMAT_TYPE, FIELD_DELIMITER, SKIP_HEADER "
        "FROM INFORMATION_SCHEMA.FILE_FORMATS ORDER BY 1"
    )
    assert view == [["JSF", "JSON", None, None], ["PIPES", "CSV", "|", 1]]
    assert data("DROP FILE FORMAT jsf") == [["File format JSF successfully dropped."]]
    assert [row[0] for row in data("SHOW FILE FORMATS")] == ["PIPES"]


def test_stage_file_format_is_the_default_for_reads(client, tmp_path):
    """A stage's own FILE_FORMAT parses reads that do not name one; explicit wins."""
    def post(statement, headers=None):
        return client.post("/api/v2/statements", json={"statement": statement}, headers=headers)

    headers = _stage_headers(client)
    pipes = tmp_path / "pipes.csv"
    pipes.write_text("1|alpha\n2|beta\n")
    commas = tmp_path / "commas.csv"
    commas.write_text("3,gamma\n")

    assert post("CREATE TABLE sf_read (a INT, b VARCHAR)").status_code == 200
    assert post(
        "CREATE STAGE pipes_st FILE_FORMAT = (TYPE = CSV FIELD_DELIMITER = '|')"
    ).status_code == 200
    assert post(
        f"PUT file://{pipes} @pipes_st AUTO_COMPRESS = FALSE", headers
    ).status_code == 200
    # COPY INTO uses the stage's format (no FILE_FORMAT clause).
    assert post("COPY INTO sf_read FROM @pipes_st").status_code == 200
    assert post("SELECT * FROM sf_read ORDER BY a").json()["data"] == [
        [1, "alpha"],
        [2, "beta"],
    ]
    # The staged-file query path uses it too.
    assert post("SELECT $1, $2 FROM @pipes_st/pipes.csv").json()["data"] == [
        [1, "alpha"],
        [2, "beta"],
    ]
    # An explicit clause still wins over the stage default.
    assert post(f"PUT file://{commas} @pipes_st AUTO_COMPRESS = FALSE", headers).status_code == 200
    assert post(
        "COPY INTO sf_read FROM @pipes_st/commas.csv FILE_FORMAT = (TYPE = CSV)"
    ).status_code == 200
    assert post("SELECT * FROM sf_read WHERE a = 3").json()["data"] == [[3, "gamma"]]


def test_put_auto_compresses_by_default(client, tmp_path):
    """PUT gzips by default (AUTO_COMPRESS), skips compressed sources, and COPY reads gzip."""
    def post(statement, headers=None):
        return client.post("/api/v2/statements", json={"statement": statement}, headers=headers)

    headers = _stage_headers(client)
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    assert post("CREATE TABLE gz_t (a INT, b VARCHAR)").status_code == 200

    staged = post(f"PUT file://{csv_path} @~/gz", headers).json()["data"]
    # source_size is the local file's size, target_size the gzipped one.
    assert staged == [
        ["data.csv.gz", "gz/data.csv.gz", 12, staged[0][3], "NONE", "GZIP", "UPLOADED", ""]
    ]
    assert staged[0][3] > 12
    assert [row[0] for row in post("LIST @~/gz", headers).json()["data"]] == ["gz/data.csv.gz"]
    assert (
        post(
            "COPY INTO gz_t FROM @~/gz FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1)",
            headers,
        ).status_code
        == 200
    )
    assert post("SELECT * FROM gz_t ORDER BY a").json()["data"] == [[1, "x"], [2, "y"]]

    # AUTO_COMPRESS = FALSE keeps the name, and an already-compressed source is left alone.
    plain = post(
        f"PUT file://{csv_path} @~/gz AUTO_COMPRESS = FALSE", headers
    ).json()["data"]
    assert plain[0][0] == "data.csv"
    listed = {row[0] for row in post("LIST @~/gz", headers).json()["data"]}
    assert {"gz/data.csv", "gz/data.csv.gz"} <= listed


def test_put_rejects_a_directory_and_divergent_paths(client, tmp_path):
    """A bare directory and a glob across subdirectories are rejected, as Snowflake does."""
    def post(statement, headers=None):
        return client.post("/api/v2/statements", json={"statement": statement}, headers=headers)

    headers = _stage_headers(client)
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    (tmp_path / "d1" / "a.csv").write_text("1,x\n")
    (tmp_path / "d2" / "b.csv").write_text("2,y\n")
    assert post(f"PUT 'file://{tmp_path}' @~/bad", headers).status_code == 422
    divergent = post(f"PUT 'file://{tmp_path}/*/*.csv' @~/bad", headers)
    assert divergent.status_code == 422
    message = divergent.json()["detail"]["message"]
    assert "divergent directory paths" in message
    # A wildcard within one directory is fine.
    assert post(f"PUT 'file://{tmp_path}/d1/*.csv' @~/ok", headers).status_code == 200


def test_alter_stage_updates_the_recorded_format_and_comment(client):
    """`ALTER STAGE ... SET FILE_FORMAT/COMMENT` rewrites the stage's metadata."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE STAGE alter_st COMMENT = 'before'").status_code == 200
    assert data("ALTER STAGE alter_st SET FILE_FORMAT = (TYPE = CSV FIELD_DELIMITER = ';')") == [
        ["Stage ALTER_ST successfully altered."]
    ]
    assert data(
        "SELECT COMMENT FROM INFORMATION_SCHEMA.STAGES WHERE lower(STAGE_NAME) = 'alter_st'"
    ) == [["before"]]
    assert data(
        "ALTER STAGE alter_st SET FILE_FORMAT = (TYPE = CSV), COMMENT = 'after'"
    ) == [["Stage ALTER_ST successfully altered."]]
    assert data(
        "SELECT COMMENT FROM INFORMATION_SCHEMA.STAGES WHERE lower(STAGE_NAME) = 'alter_st'"
    ) == [["after"]]
    resp = post("ALTER STAGE missing_st SET COMMENT = 'x'")
    assert resp.status_code == 422
    assert "does not exist" in resp.json()["detail"]["message"]


def test_create_stage_respects_existence(client):
    """Plain CREATE errors on an existing stage, IF NOT EXISTS keeps it, OR REPLACE replaces."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE STAGE exist_st COMMENT = 'a'").status_code == 200
    resp = post("CREATE STAGE exist_st COMMENT = 'b'")
    assert resp.status_code == 422
    assert "already exists" in resp.json()["detail"]["message"]
    assert data("CREATE STAGE IF NOT EXISTS exist_st COMMENT = 'c'") == [
        ["Stage exist_st successfully created."]
    ]
    assert data(
        "SELECT COMMENT FROM INFORMATION_SCHEMA.STAGES WHERE lower(STAGE_NAME) = 'exist_st'"
    ) == [["a"]]
    assert data("CREATE OR REPLACE STAGE exist_st COMMENT = 'd'") == [
        ["Stage exist_st successfully created."]
    ]
    assert data(
        "SELECT COMMENT FROM INFORMATION_SCHEMA.STAGES WHERE lower(STAGE_NAME) = 'exist_st'"
    ) == [["d"]]


def test_stage_names_are_case_sensitive_like_their_directory(client):
    """A stage's metadata identity is exact, so a case variant is a different stage."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE STAGE case_st").status_code == 200
    assert post("CREATE STAGE CASE_ST").status_code == 200
    listed = {row[0] for row in data("SHOW STAGES")}
    assert {"case_st", "CASE_ST"} <= listed
    assert data("DROP STAGE case_st") == [["Stage case_st successfully dropped."]]
    remaining = {row[0] for row in data("SHOW STAGES")}
    assert "CASE_ST" in remaining and "case_st" not in remaining


def test_alter_stage_rejects_unmodelled_set_forms(client):
    """An ALTER STAGE SET the emulator cannot apply is an error, not a silent success."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE STAGE alter_set").status_code == 200
    for statement in (
        "ALTER STAGE alter_set SET URL = 's3://b/x'",
        "ALTER STAGE alter_set SET CREDENTIALS = (AWS_KEY_ID = 'k')",
        "ALTER STAGE alter_set SET DIRECTORY = (ENABLE = TRUE)",
    ):
        resp = post(statement)
        assert resp.status_code == 422, (statement, resp.json())
        assert "Unsupported ALTER STAGE SET form" in resp.json()["detail"]["message"]


def test_alter_stage_format_keeps_a_parenthesis_in_a_value(client, tmp_path):
    """The FILE_FORMAT parentheses are scanned, so a ')' delimiter survives the ALTER."""
    def post(statement, headers=None):
        return client.post("/api/v2/statements", json={"statement": statement}, headers=headers)

    headers = _stage_headers(client)
    paren = tmp_path / "paren.csv"
    paren.write_text("1)alpha\n2)beta\n")
    assert post("CREATE STAGE paren_st FILE_FORMAT = (TYPE = CSV)").status_code == 200
    assert post("ALTER STAGE paren_st SET FILE_FORMAT = (TYPE = CSV, FIELD_DELIMITER = ')')").status_code == 200
    assert post(f"PUT file://{paren} @paren_st AUTO_COMPRESS = FALSE", headers).status_code == 200
    assert post("CREATE TABLE paren_t (a INT, b VARCHAR)").status_code == 200
    assert post("COPY INTO paren_t FROM @paren_st").status_code == 200
    assert post("SELECT * FROM paren_t ORDER BY a").json()["data"] == [[1, "alpha"], [2, "beta"]]


def test_create_stage_requires_an_existing_database_and_schema(client):
    """A stage in a missing database/schema is an error, not a silent filesystem directory."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    resp = post("CREATE STAGE no_such_db.public.s")
    assert resp.status_code == 422
    assert "Database 'NO_SUCH_DB' does not exist" in resp.json()["detail"]["message"]
    resp = post("CREATE STAGE EMULATOR_DB.no_such_schema.s")
    assert resp.status_code == 422
    assert "Schema 'NO_SUCH_SCHEMA' does not exist" in resp.json()["detail"]["message"]


def test_stage_ddl_quoted_names_with_dots_and_spaces(client):
    """A quoted stage name keeps its dots/spaces through CREATE, ALTER and DROP."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post('CREATE STAGE "dot.name" COMMENT = \'a\'').status_code == 200
    assert post('CREATE STAGE "space name" COMMENT = \'b\'').status_code == 200
    assert data("ALTER STAGE \"dot.name\" SET COMMENT = 'x'") == [
        ["Stage DOT.NAME successfully altered."]
    ]
    assert data("ALTER STAGE \"space name\" SET COMMENT = 'y'") == [
        ["Stage SPACE NAME successfully altered."]
    ]
    assert data(
        "SELECT STAGE_NAME, COMMENT FROM INFORMATION_SCHEMA.STAGES "
        "WHERE STAGE_NAME IN ('dot.name', 'space name') ORDER BY STAGE_NAME"
    ) == [["dot.name", "x"], ["space name", "y"]]
    assert data('DROP STAGE "dot.name"') == [["Stage dot.name successfully dropped."]]


def test_create_stage_rejects_an_empty_url(client):
    resp = client.post(
        "/api/v2/statements", json={"statement": "CREATE STAGE empty_url_st URL = ''"}
    )
    assert resp.status_code == 422
    assert "URL must not be empty" in resp.json()["detail"]["message"]


def test_put_honours_source_compression(client, tmp_path):
    """SOURCE_COMPRESSION classifies a source whose name does not say it is compressed."""
    import gzip

    headers = _stage_headers(client)
    plain = tmp_path / "plain.csv"
    with gzip.open(plain, "wb") as fh:
        fh.write(b"1,x\n")
    put = client.post(
        "/api/v2/statements",
        json={
            "statement": f"PUT 'file://{plain}' @~/sc SOURCE_COMPRESSION = GZIP AUTO_COMPRESS = FALSE"
        },
        headers=headers,
    )
    assert put.status_code == 200, put.json()
    row = put.json()["data"][0]
    assert row[0] == "plain.csv"
    assert row[4] == "GZIP" and row[5] == "GZIP"


def test_stage_metadata_side_table_is_recreated_on_demand(client):
    """A missing side table is recreated by the next write instead of losing metadata."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("DROP TABLE EMULATOR_DB.SF_INFORMATION_SCHEMA._sf_stages_ext").status_code == 200
    assert post("CREATE STAGE healed_st COMMENT = 'h'").status_code == 200
    assert data(
        "SELECT STAGE_NAME, COMMENT FROM INFORMATION_SCHEMA.STAGES WHERE STAGE_NAME = 'healed_st'"
    ) == [["healed_st", "h"]]


def test_alter_stage_preserves_created_and_owner(client):
    """ALTER STAGE updates the row in place, keeping CREATED and the original owner."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE STAGE kept_st COMMENT = 'c1'").status_code == 200
    before = data(
        "SELECT STAGE_OWNER, CREATED FROM INFORMATION_SCHEMA.STAGES WHERE STAGE_NAME = 'kept_st'"
    )[0]
    assert data("ALTER STAGE kept_st SET COMMENT = 'c2'") == [
        ["Stage KEPT_ST successfully altered."]
    ]
    after = data(
        "SELECT STAGE_OWNER, CREATED, COMMENT FROM INFORMATION_SCHEMA.STAGES "
        "WHERE STAGE_NAME = 'kept_st'"
    )[0]
    assert after[0] == before[0]
    assert after[1] == before[1]
    assert after[2] == "c2"


def test_file_format_ddl_honours_its_namespace_and_quoted_case(client):
    """`CREATE FILE FORMAT db.schema.name` lands there, and a quoted name keeps its case."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    def data(statement):
        resp = post(statement)
        assert resp.status_code == 200, (statement, resp.json())
        return resp.json()["data"]

    assert post("CREATE DATABASE ffdb").status_code == 200
    assert post("CREATE FILE FORMAT ffdb.public.ff_here TYPE = CSV").status_code == 200
    # An unquoted name is upper-cased, as Snowflake does.
    assert data("SELECT FILE_FORMAT_NAME FROM ffdb.INFORMATION_SCHEMA.FILE_FORMATS") == [
        ["FF_HERE"]
    ]
    assert data("SHOW FILE FORMATS LIKE 'FF_HERE'") == []
    assert post('CREATE FILE FORMAT "MixedCaseFF" TYPE = JSON').status_code == 200
    assert [row[0] for row in data("SHOW FILE FORMATS LIKE 'MIXEDCASEFF'")] == ["MixedCaseFF"]
