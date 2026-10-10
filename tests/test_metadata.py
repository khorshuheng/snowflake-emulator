"""Snowflake metadata / SHOW surface tests."""

from __future__ import annotations


def _post(client, statement, headers=None):
    return client.post("/api/v2/statements", json={"statement": statement}, headers=headers)


def test_show_columns_in_table_has_snowflake_shape(client):
    _post(client, "CREATE TABLE sc (id INT, name VARCHAR)")
    resp = _post(client, "SHOW COLUMNS IN TABLE sc")
    assert resp.status_code == 200
    body = resp.json()
    names = [col["name"] for col in body["resultSetMetaData"]["rowType"]]
    assert names[:4] == ["table_name", "schema_name", "column_name", "data_type"]
    assert {row[2]: row[3] for row in body["data"]} == {"ID": "NUMBER(38,0)", "NAME": "VARCHAR"}


def test_show_columns_in_schema_lists_all_tables(client):
    _post(client, "CREATE TABLE a1 (x INT)")
    _post(client, "CREATE TABLE a2 (y INT)")
    resp = _post(client, "SHOW COLUMNS IN SCHEMA PUBLIC")
    assert {"A1", "A2"} <= {row[0] for row in resp.json()["data"]}


def test_show_columns_fully_qualified(client):
    _post(client, "CREATE TABLE EMULATOR_DB.PUBLIC.q (z INT)")
    resp = _post(client, "SHOW COLUMNS IN TABLE EMULATOR_DB.PUBLIC.q")
    assert [row[2] for row in resp.json()["data"]] == ["Z"]


def test_show_columns_with_quoted_identifier_does_not_break_sql(client):
    resp = _post(client, "SHOW COLUMNS IN TABLE \"we'ird\"")
    assert resp.status_code == 200
    assert resp.json()["data"] == []
