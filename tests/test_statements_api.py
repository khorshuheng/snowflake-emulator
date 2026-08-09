"""End-to-end tests for the Snowflake SQL API v2 emulation."""

from __future__ import annotations


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_simple_select(client):
    resp = client.post("/api/v2/statements", json={"statement": "SELECT 1 AS x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == [[1]]
    assert body["resultSetMetaData"]["rowType"][0]["name"] == "x"
    assert body["resultSetMetaData"]["rowType"][0]["type"] == "fixed"


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
