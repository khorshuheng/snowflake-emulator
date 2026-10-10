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


def test_show_primary_keys(client):
    _post(client, "CREATE TABLE pk (id INT PRIMARY KEY, name VARCHAR)")
    rows = _post(client, "SHOW PRIMARY KEYS IN TABLE pk").json()["data"]
    assert [(row[4], row[5]) for row in rows] == [("ID", 1)]


def test_show_unique_keys(client):
    _post(client, "CREATE TABLE uk (email VARCHAR UNIQUE)")
    rows = _post(client, "SHOW UNIQUE KEYS IN TABLE uk").json()["data"]
    assert [row[4] for row in rows] == ["EMAIL"]


def test_show_foreign_keys(client):
    _post(client, "CREATE TABLE parent (id INT PRIMARY KEY)")
    _post(client, "CREATE TABLE child (pid INT, FOREIGN KEY (pid) REFERENCES parent(id))")
    resp = _post(client, "SHOW FOREIGN KEYS IN TABLE child")
    assert resp.status_code == 200
    rows = resp.json()["data"]
    assert any(row[8] == "PID" and row[7] == "CHILD" for row in rows)


def test_show_imported_keys_composite(client):
    _post(client, "CREATE TABLE par2 (id1 INT, id2 INT, PRIMARY KEY (id1, id2))")
    _post(client, "CREATE TABLE chi2 (px INT, py INT, FOREIGN KEY (px, py) REFERENCES par2(id1, id2))")
    rows = _post(client, "SHOW IMPORTED KEYS IN TABLE chi2").json()["data"]
    assert [(row[4], row[8], row[9]) for row in rows] == [("ID1", "PX", 1), ("ID2", "PY", 2)]
