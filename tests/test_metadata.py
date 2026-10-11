"""Snowflake metadata / SHOW surface tests."""

from __future__ import annotations


def _post(client, statement, headers=None):
    return client.post("/api/v2/statements", json={"statement": statement}, headers=headers)


def _auth(client):
    token = client.post("/session/token").json()["token"]
    return {"Authorization": f"Bearer {token}"}


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


def test_show_databases_lists_catalogs(client):
    rows = _post(client, "SHOW DATABASES").json()["data"]
    assert "EMULATOR_DB" in {row[1] for row in rows}


def test_show_warehouses_returns_session_warehouse(client):
    rows = _post(client, "SHOW WAREHOUSES").json()["data"]
    assert rows[0][0] == "EMULATOR_WH"


def test_show_sequences_lists_created_sequence(client):
    _post(client, "CREATE SEQUENCE sq1")
    rows = _post(client, "SHOW SEQUENCES").json()["data"]
    assert "SQ1" in {row[1] for row in rows}


def test_show_parameters_returns_session_parameters(client):
    rows = _post(client, "SHOW PARAMETERS").json()["data"]
    assert {row[0] for row in rows} == {"AUTOCOMMIT", "TIMEZONE", "CLIENT_PREFETCH_THREADS"}


def test_show_users_procedures_functions_return_empty(client):
    assert _post(client, "SHOW USERS").json()["data"] == []
    assert _post(client, "SHOW PROCEDURES").json()["data"] == []
    assert _post(client, "SHOW FUNCTIONS").json()["data"] == []


def test_show_databases_excludes_duckdb_internals(client):
    rows = _post(client, "SHOW DATABASES").json()["data"]
    assert {row[1].lower() for row in rows}.isdisjoint({"system", "temp", "memory"})


def test_show_sequences_scoped_to_current_schema(client):
    _post(client, "CREATE SCHEMA other_s")
    _post(client, "CREATE SEQUENCE other_s.seq_x")
    rows = _post(client, "SHOW SEQUENCES").json()["data"]
    assert "SEQ_X" not in {row[1] for row in rows}


def test_varchar_length_preserved_in_describe_and_show_columns(client):
    _post(client, "CREATE TABLE vc (name VARCHAR(255), n INT, amount NUMBER(10,2))")
    describe = {row[0]: row[1] for row in _post(client, "DESCRIBE TABLE vc").json()["data"]}
    assert describe == {"NAME": "VARCHAR(255)", "N": "NUMBER(38,0)", "AMOUNT": "NUMBER(10,2)"}
    show = {row[2]: row[3] for row in _post(client, "SHOW COLUMNS IN TABLE vc").json()["data"]}
    assert show["NAME"] == "VARCHAR(255)"


def test_information_schema_columns_view(client):
    _post(client, "CREATE TABLE isc (name VARCHAR(50), n INT)")
    resp = _post(
        client,
        "SELECT COLUMN_NAME, DATA_TYPE FROM EMULATOR_DB.INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_NAME = 'ISC' ORDER BY ORDINAL_POSITION",
    )
    assert resp.json()["data"] == [["NAME", "VARCHAR(50)"], ["N", "NUMBER(38,0)"]]


def test_create_or_replace_refreshes_declared_length(client):
    _post(client, "CREATE OR REPLACE TABLE rl (name VARCHAR(10))")
    _post(client, "CREATE OR REPLACE TABLE rl (name VARCHAR(300))")
    describe = {row[0]: row[1] for row in _post(client, "DESCRIBE TABLE rl").json()["data"]}
    assert describe["NAME"] == "VARCHAR(300)"


def test_temporary_table_metadata_is_visible_to_its_session(client):
    """DESCRIBE, SHOW COLUMNS and INFORMATION_SCHEMA cover the session's temp tables."""
    headers = _auth(client)
    _post(client, "CREATE TEMPORARY TABLE tt_meta (a INT, name VARCHAR(7))", headers)
    describe = {row[0]: row[1] for row in _post(client, "DESCRIBE TABLE tt_meta", headers).json()["data"]}
    assert describe == {"A": "NUMBER(38,0)", "NAME": "VARCHAR"}
    show = {row[2]: row[3] for row in _post(client, "SHOW COLUMNS IN TABLE tt_meta", headers).json()["data"]}
    assert show == describe
    columns = _post(
        client,
        "SELECT TABLE_CATALOG, TABLE_SCHEMA, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_NAME = 'TT_META' ORDER BY ORDINAL_POSITION",
        headers,
    ).json()["data"]
    assert columns == [["EMULATOR_DB", "PUBLIC", "A"], ["EMULATOR_DB", "PUBLIC", "NAME"]]
    tables = _post(
        client,
        "SELECT TABLE_CATALOG, TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'TT_META'",
        headers,
    ).json()["data"]
    assert tables == [["EMULATOR_DB", "PUBLIC"]]


def test_temporary_table_metadata_is_hidden_from_other_sessions(client):
    owner = _auth(client)
    other = _auth(client)
    _post(client, "CREATE TEMPORARY TABLE tt_secret (a INT)", owner)
    assert _post(client, "DESCRIBE TABLE tt_secret", owner).status_code == 200
    assert _post(client, "DESCRIBE TABLE tt_secret", other).json()["data"] == []
    assert _post(client, "SHOW COLUMNS IN TABLE tt_secret", other).json()["data"] == []
    assert (
        _post(
            client,
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'TT_SECRET'",
            other,
        ).json()["data"]
        == []
    )


def test_temporary_table_shadows_a_permanent_one(client):
    """A temp table wins name resolution, so its metadata replaces the permanent one's."""
    headers = _auth(client)
    _post(client, "CREATE TABLE sh_tbl (b VARCHAR(7))", headers)
    _post(client, "CREATE TEMPORARY TABLE sh_tbl (b NUMBER(5,2))", headers)
    describe = _post(client, "DESCRIBE TABLE sh_tbl", headers).json()["data"]
    assert [row[0] for row in describe] == ["B"]
    assert describe[0][1] == "NUMBER(5,2)"
    assert [row[1] for row in _post(client, "SHOW TABLES", headers).json()["data"]].count("SH_TBL") == 1
    # The permanent table keeps its own declared type, which another session still sees.
    permanent = {row[0]: row[1] for row in _post(client, "DESCRIBE TABLE sh_tbl", _auth(client)).json()["data"]}
    assert permanent == {"B": "VARCHAR(7)"}


def test_show_tables_like_and_starts_with_filter(client):
    _post(client, "CREATE TABLE alpha_one (a INT)")
    _post(client, "CREATE TABLE beta_two (a INT)")
    names = lambda resp: [row[1] for row in resp.json()["data"]]
    assert names(_post(client, "SHOW TABLES LIKE 'alpha%'")) == ["ALPHA_ONE"]
    assert names(_post(client, "SHOW TABLES LIKE 'ALPHA_ONE'")) == ["ALPHA_ONE"]
    assert names(_post(client, "SHOW TABLES LIKE 'zzz%'")) == []
    assert names(_post(client, "SHOW TABLES STARTS WITH 'BET'")) == ["BETA_TWO"]
    assert names(_post(client, "SHOW TABLES LIMIT 1")) == ["ALPHA_ONE"]
    # The IN-scoped form takes LIKE before IN, as Snowflake's grammar does.
    _post(client, "CREATE SCHEMA like_schema")
    _post(client, "CREATE TABLE like_schema.alpha_scoped (a INT)")
    assert names(_post(client, "SHOW TABLES LIKE 'alpha%' IN SCHEMA EMULATOR_DB.LIKE_SCHEMA")) == [
        "ALPHA_SCOPED"
    ]
    assert _post(client, "SHOW TABLES LIKE 'zzz%' IN SCHEMA EMULATOR_DB.LIKE_SCHEMA").json()["data"] == []


def test_show_tables_in_database_spans_schemas(client):
    _post(client, "CREATE SCHEMA span_s")
    _post(client, "CREATE TABLE span_s.spanned (a INT)")
    rows = _post(client, "SHOW TABLES IN DATABASE EMULATOR_DB").json()["data"]
    assert "SPANNED" in {row[1] for row in rows}


def test_show_views_and_objects_respect_scope_and_shadowing(client):
    headers = _auth(client)
    _post(client, "CREATE SCHEMA v_schema", headers)
    _post(client, "CREATE VIEW v_schema.v1 AS SELECT 1 AS c", headers)
    _post(client, "CREATE TABLE v_schema.t1 (a INT)", headers)
    objects = lambda: [
        (row[1], row[2])
        for row in _post(client, "SHOW OBJECTS IN SCHEMA EMULATOR_DB.V_SCHEMA", headers).json()["data"]
    ]
    assert sorted(objects()) == [("T1", "TABLE"), ("V1", "VIEW")]
    assert "V1" in {
        row[1] for row in _post(client, "SHOW VIEWS IN DATABASE EMULATOR_DB", headers).json()["data"]
    }
    _post(client, "USE SCHEMA v_schema", headers)
    assert [row[1] for row in _post(client, "SHOW VIEWS LIKE 'V%'", headers).json()["data"]] == ["V1"]
    assert _post(client, "SHOW VIEWS LIKE 'zzz%'", headers).json()["data"] == []
    # A temporary table of the same name shadows the view: the object is still listed,
    # once, as the session's temporary table.
    _post(client, "CREATE TEMPORARY TABLE v1 (x INT)", headers)
    assert objects() == [("T1", "TABLE"), ("V1", "TABLE")]
    assert [row[1] for row in _post(client, "SHOW TABLES", headers).json()["data"]] == ["T1", "V1"]
    assert _post(client, "SHOW VIEWS LIKE 'V%'", headers).json()["data"] == []
