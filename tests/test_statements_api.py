"""End-to-end tests for the Snowflake SQL API v2 emulation."""

from __future__ import annotations

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
    resp = client.post("/api/v2/statements", json={"statement": stmt})
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
    resp = client.post("/api/v2/statements", json={"statement": stmt})
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


def test_array_slice_rejects_a_bound_index(client):
    """A `?` index cannot work (the sign adjustment names it twice); say so clearly."""
    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": "SELECT ARRAY_SLICE(ARRAY_CONSTRUCT('a','b','c'), ?, 2)",
            "bindings": {"1": {"type": "FIXED", "value": "1"}},
        },
    )
    assert resp.status_code == 422
    assert "ARRAY_SLICE with a '?' index" in resp.json()["detail"]["message"]


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
        ("SELECT ARRAY_POSITION('zz', j:n) FROM arr_t", [[0]]),
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
        # A lambda compares against a VARIANT element, so the literal is encoded.
        ("SELECT FILTER(j:n, x -> x = TO_JSON('a')) FROM arr_t", [[['a', 'a']]]),
        ("SELECT ARRAY_SIZE(TRANSFORM(j:n, x -> TO_VARCHAR(x))) FROM arr_t", [[3]]),
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

    # sqlglot models ARRAY_INTERSECTION as a multiset operation, so compare as a set here.
    intersected = post("SELECT ARRAY_INTERSECTION(j:n, j:n) FROM arr_t")
    assert intersected.status_code == 200, intersected.json()
    assert set(intersected.json()["data"][0][0]) == {'a', 'b'}


def test_array_flatten_requires_a_literal_depth(client):
    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT ARRAY_FLATTEN(ARRAY_CONSTRUCT(1), j) FROM arr_t"},
    )
    assert resp.status_code == 422
    assert "ARRAY_FLATTEN" in resp.json()["detail"]["message"]


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
        # A number is its text, and a JSON null or missing path is NULL as in Snowflake.
        ("SELECT j:n::VARCHAR FROM txt", [["1"]]),
        ("SELECT j:nl::VARCHAR FROM txt", [[None]]),
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


def test_array_functions_over_a_bare_variant_column_still_error(client):
    """A bare VARIANT column has no static array type, so this stays a loud error."""
    def post(statement):
        return client.post("/api/v2/statements", json={"statement": statement})

    assert post("CREATE TABLE bare (names VARIANT)").status_code == 200
    assert post("INSERT INTO bare SELECT PARSE_JSON('[\"ada\"]')").status_code == 200
    resp = post("SELECT ARRAY_CONTAINS('ada', names) FROM bare")
    assert resp.status_code == 422


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
        json={"statement": "CREATE TABLE t2 (a INT); INSERT INTO t2 VALUES (99); SELECT * FROM t2"},
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
        json={"statement": f"PUT file://{csv_path} @~/staged"},
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
            "statement": "COPY INTO staged_t FROM @~/staged FILE_FORMAT = (TYPE = CSV)"
        },
        headers=headers,
    )
    assert copied.status_code == 200
    assert copied.json()["data"] == [[2]]

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
    assert copied.json()["data"] == [[1]]

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
        json={"statement": f"PUT file://{csv_path} @~/staged"},
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
        json={"statement": f"PUT file://{path} {target}"},
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
    assert inferred["data"] == [[2]]
    typed = _select(
        client,
        headers,
        "COPY INTO pq_typed FROM @~/staged/people.parquet FILE_FORMAT = (TYPE = PARQUET)",
    )
    assert typed["data"] == [[2]]

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
        "COPY INTO from_file FROM @~/staged/data.csv FILE_FORMAT = (TYPE = CSV)",
    )
    assert copied["data"] == [[2]]
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
    assert first["data"] == [[1]]
    both = _select(
        client,
        headers,
        "COPY INTO patterned FROM @~/staged PATTERN = '.*[.]csv' "
        "FILE_FORMAT = (TYPE = CSV)",
    )
    assert both["data"] == [[2]]
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
        "COPY INTO copied FROM @~/staged FILE_FORMAT = (TYPE = CSV)",
    )
    assert copied["data"] == [[2]]


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
    assert copied["data"] == [[2]]

    # The projection, not a 1:1 column copy, is what landed in the table.
    stored = _select(client, headers, "SELECT * FROM projected ORDER BY a")
    assert stored["data"] == [["\"a\"", 3], ["\"b\"", 4]]


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
    assert selected["data"] == [[2_000_002]]
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
    assert copied["data"] == [[2]]


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
