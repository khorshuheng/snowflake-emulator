"""SQL feature ports: constructs Snowflake supports that DuckDB does not."""

from __future__ import annotations


def _post(client, statement):
    return client.post("/api/v2/statements", json={"statement": statement})


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
