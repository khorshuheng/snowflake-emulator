"""End-to-end tests for the Snowflake SQL API v2 emulation."""

from __future__ import annotations

import json


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
        "F": "TIMESTAMP_NTZ",
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
