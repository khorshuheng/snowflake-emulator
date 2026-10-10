"""SQL feature ports: constructs Snowflake supports that DuckDB does not."""

from __future__ import annotations


def _post(client, statement, headers=None):
    return client.post("/api/v2/statements", json={"statement": statement}, headers=headers)


def _auth(client):
    token = client.post("/session/token").json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_create_table_clone_copies_rows(client):
    assert _post(client, "CREATE TABLE src (a INT, b VARCHAR)").status_code == 200
    assert _post(client, "INSERT INTO src VALUES (1, 'x'), (2, 'y')").status_code == 200
    assert _post(client, "CREATE TABLE dst CLONE src").status_code == 200
    assert _post(client, "SELECT * FROM dst ORDER BY a").json()["data"] == [[1, "x"], [2, "y"]]


def test_clone_is_independent_of_source(client):
    _post(client, "CREATE TABLE src2 (a INT)")
    _post(client, "INSERT INTO src2 VALUES (1)")
    _post(client, "CREATE TABLE dst2 CLONE src2")
    _post(client, "INSERT INTO src2 VALUES (2)")
    assert _post(client, "SELECT COUNT(*) AS n FROM dst2").json()["data"] == [[1]]


def test_create_or_replace_table_clone(client):
    _post(client, "CREATE TABLE osrc (a INT)")
    _post(client, "INSERT INTO osrc VALUES (7)")
    _post(client, "CREATE TABLE odst (a INT)")
    assert _post(client, "CREATE OR REPLACE TABLE odst CLONE osrc").status_code == 200
    assert _post(client, "SELECT * FROM odst").json()["data"] == [[7]]


def test_clone_time_travel_is_rejected(client):
    _post(client, "CREATE TABLE tsrc (a INT)")
    resp = _post(
        client, "CREATE TABLE tdst CLONE tsrc AT (TIMESTAMP => '2020-01-01'::timestamp)"
    )
    assert resp.status_code == 422
    assert "time travel" in resp.json()["detail"]["message"].lower()


def test_create_database_clone_is_rejected(client):
    _post(client, "CREATE TABLE dsrc (a INT)")
    resp = _post(client, "CREATE DATABASE dclone CLONE EMULATOR_DB")
    assert resp.status_code == 422
    assert "CLONE" in resp.json()["detail"]["message"]


def test_create_schema_clone_is_rejected(client):
    resp = _post(client, "CREATE SCHEMA sclone CLONE PUBLIC")
    assert resp.status_code == 422
    assert "CLONE" in resp.json()["detail"]["message"]


def test_result_scan_last_query_id(client):
    headers = _auth(client)
    _post(client, "CREATE TABLE rs (a INT)", headers)
    _post(client, "INSERT INTO rs VALUES (1), (2)", headers)
    _post(client, "SELECT * FROM rs ORDER BY a", headers)
    resp = _post(client, "SELECT * FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))", headers)
    assert resp.status_code == 200
    assert resp.json()["data"] == [[1], [2]]


def test_result_scan_explicit_handle(client):
    handle = _post(client, "SELECT 41 AS x, 'y' AS s").json()["statementHandle"]
    resp = _post(client, f"SELECT * FROM TABLE(RESULT_SCAN('{handle}'))")
    assert resp.status_code == 200
    assert resp.json()["data"] == [[41, "y"]]


def test_result_scan_unknown_id_reports_02000(client):
    resp = _post(client, "SELECT * FROM TABLE(RESULT_SCAN('nope'))")
    assert resp.status_code == 422
    assert resp.json()["detail"]["sqlState"] == "02000"


def test_last_query_id_returns_previous_handle(client):
    headers = _auth(client)
    handle = _post(client, "SELECT 1 AS n", headers).json()["statementHandle"]
    resp = _post(client, "SELECT LAST_QUERY_ID() AS qid", headers)
    assert resp.json()["data"] == [[handle]]


def test_last_query_id_without_prior_query_reports_02000(client):
    headers = _auth(client)
    resp = _post(client, "SELECT LAST_QUERY_ID() AS qid", headers)
    assert resp.status_code == 422
    assert resp.json()["detail"]["sqlState"] == "02000"
