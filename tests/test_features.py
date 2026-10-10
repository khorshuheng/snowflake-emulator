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


def test_set_and_use_string_variable(client):
    headers = _auth(client)
    _post(client, "SET my_var = 'hello'", headers)
    assert _post(client, "SELECT $my_var AS v", headers).json()["data"] == [["hello"]]


def test_set_and_use_numeric_variable(client):
    headers = _auth(client)
    _post(client, "SET n = 41", headers)
    assert _post(client, "SELECT $n + 1 AS v", headers).json()["data"] == [[42]]


def test_show_variables_lists_session_variables(client):
    headers = _auth(client)
    _post(client, "SET a = 'x'", headers)
    resp = _post(client, "SHOW VARIABLES", headers)
    assert resp.status_code == 200
    assert {row[0] for row in resp.json()["data"]} == {"A"}


def test_variables_are_session_scoped(client):
    _post(client, "SET scoped = 'yes'", _auth(client))
    other = _post(client, "SHOW VARIABLES", _auth(client))
    assert other.json()["data"] == []


def test_set_expression_variable_expands_expression_not_string(client):
    headers = _auth(client)
    _post(client, "SET x = 1 + 1", headers)
    assert _post(client, "SELECT $x AS v", headers).json()["data"] == [[2]]


def _write_csv(path, text):
    path.write_text(text)


def test_named_file_format_used_by_copy_into(client, tmp_path):
    csv = tmp_path / "data.csv"
    _write_csv(csv, "a,b\n1,x\n2,y\n")
    headers = _auth(client)
    _post(client, "CREATE FILE FORMAT ff TYPE = CSV SKIP_HEADER = 1", headers)
    _post(client, f"PUT file://{csv} @~/ffstage", headers)
    _post(client, "CREATE TABLE fft (a INT, b VARCHAR)", headers)
    resp = _post(
        client, "COPY INTO fft FROM @~/ffstage FILE_FORMAT = (FORMAT_NAME = 'ff')", headers
    )
    assert resp.status_code == 200
    assert _post(client, "SELECT * FROM fft ORDER BY a", headers).json()["data"] == [[1, "x"], [2, "y"]]


def test_show_file_formats(client):
    headers = _auth(client)
    _post(client, "CREATE FILE FORMAT sf TYPE = JSON", headers)
    resp = _post(client, "SHOW FILE FORMATS", headers)
    assert {row[0] for row in resp.json()["data"]} == {"SF"}


def test_drop_file_format(client):
    headers = _auth(client)
    _post(client, "CREATE FILE FORMAT df TYPE = CSV", headers)
    _post(client, "DROP FILE FORMAT df", headers)
    assert _post(client, "SHOW FILE FORMATS", headers).json()["data"] == []


def test_copy_into_missing_named_format_errors(client, tmp_path):
    csv = tmp_path / "d.csv"
    _write_csv(csv, "a\n1\n")
    headers = _auth(client)
    _post(client, f"PUT file://{csv} @~/mf", headers)
    _post(client, "CREATE TABLE mft (a INT)", headers)
    resp = _post(
        client, "COPY INTO mft FROM @~/mf FILE_FORMAT = (FORMAT_NAME = 'nope')", headers
    )
    assert resp.status_code == 422
    assert "does not exist" in resp.json()["detail"]["message"]
