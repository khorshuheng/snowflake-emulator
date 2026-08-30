"""Regression tests for Snowflake-compatibility fixes.

Covers: ``USE ... IDENTIFIER('...')``, ``ALTER SESSION`` / transaction-control
no-ops, the per-catalog ``INFORMATION_SCHEMA``, and the ``seq.NEXTVAL`` /
``OBJECT`` / ``ARRAY`` / ``MERGE`` translation rewrites.
"""

from __future__ import annotations


def _auth_headers(client) -> dict:
    token = client.post("/session/token").json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _post(client, statement, headers=None):
    return client.post("/api/v2/statements", json={"statement": statement}, headers=headers)


def test_use_database_and_schema_identifier_fn(client):
    headers = _auth_headers(client)

    assert _post(client, "USE DATABASE IDENTIFIER('ALT_DB')", headers).status_code == 200
    assert _post(client, "USE SCHEMA IDENTIFIER('PUBLIC')", headers).status_code == 200

    # A table created afterwards must land in ALT_DB, proving IDENTIFIER() resolved.
    assert _post(client, "CREATE TABLE idf_t (a INT)", headers).status_code == 200
    resp = _post(
        client,
        "SELECT COUNT(*) AS n FROM ALT_DB.INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = 'PUBLIC' AND TABLE_NAME = 'IDF_T'",
        headers,
    )
    assert resp.json()["data"] == [[1]]


def test_unquoted_identifiers_uppercased_quoted_preserved(client):
    # Unquoted table/column/alias names become uppercase (Snowflake semantics).
    assert _post(client, "CREATE TABLE case_t (mixed_col INT)").status_code == 200
    resp = _post(client, "SELECT mixed_col AS my_alias FROM case_t")
    assert resp.status_code == 200
    names = [c["name"] for c in resp.json()["resultSetMetaData"]["rowType"]]
    assert names == ["MY_ALIAS"]

    # Quoted identifiers preserve their exact case.
    assert _post(client, 'CREATE TABLE "QuotedTbl" ("QuotedCol" INT)').status_code == 200
    resp = _post(client, 'SELECT "QuotedCol" AS "MyAlias" FROM "QuotedTbl"')
    names = [c["name"] for c in resp.json()["resultSetMetaData"]["rowType"]]
    assert names == ["MyAlias"]


def test_use_unquoted_names_are_uppercased(client):
    headers = _auth_headers(client)
    assert _post(client, "USE DATABASE lower_db", headers).status_code == 200
    assert _post(client, "USE SCHEMA lower_schema", headers).status_code == 200
    resp = _post(
        client, "SELECT CURRENT_DATABASE() AS db, CURRENT_SCHEMA() AS sch", headers
    )
    assert resp.json()["data"] == [["LOWER_DB", "LOWER_SCHEMA"]]


def test_alter_session_is_noop(client):
    for stmt in (
        "ALTER SESSION SET QUERY_TAG = 'x'",
        "ALTER SESSION UNSET QUERY_TAG",
    ):
        assert _post(client, stmt).status_code == 200


def test_transaction_control_is_noop(client):
    for stmt in ("BEGIN", "COMMIT", "ROLLBACK"):
        assert _post(client, stmt).status_code == 200


def test_information_schema_schemata_and_tables(client):
    assert _post(client, "CREATE TABLE is_t (a INT)").status_code == 200

    resp = _post(
        client,
        "SELECT COUNT(*) AS n FROM EMULATOR_DB.INFORMATION_SCHEMA.SCHEMATA "
        "WHERE SCHEMA_NAME = 'PUBLIC'",
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == [[1]]

    resp = _post(
        client,
        "SELECT CREATED, LAST_ALTERED, TABLE_NAME FROM EMULATOR_DB.INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = 'PUBLIC' AND TABLE_NAME = 'IS_T'",
    )
    assert resp.status_code == 200
    rows = resp.json()["data"]
    assert len(rows) == 1
    assert rows[0][2] == "IS_T"  # CREATED/LAST_ALTERED are synthesized (NULL), name matches


def test_information_schema_string_literal_not_rewritten(client):
    # The INFORMATION_SCHEMA rewrite must not touch string literals/comments.
    resp = _post(client, "SELECT 'INFORMATION_SCHEMA' AS lit")
    assert resp.status_code == 200
    assert resp.json()["data"] == [["INFORMATION_SCHEMA"]]


def test_information_schema_tables_lists_views(client):
    assert _post(client, "CREATE TABLE base_tbl (a INT)").status_code == 200
    assert _post(client, "CREATE VIEW some_view AS SELECT 1 AS x").status_code == 200

    resp = _post(
        client,
        "SELECT TABLE_NAME, TABLE_TYPE FROM EMULATOR_DB.INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = 'PUBLIC' AND TABLE_NAME IN ('BASE_TBL', 'SOME_VIEW') "
        "ORDER BY TABLE_NAME",
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == [
        ["BASE_TBL", "BASE TABLE"],
        ["SOME_VIEW", "VIEW"],
    ]


def test_seq_nextval_in_select_and_default(client):
    assert _post(client, "CREATE SEQUENCE compat_seq").status_code == 200

    resp = _post(client, "SELECT compat_seq.NEXTVAL AS n")
    assert resp.status_code == 200
    assert resp.json()["data"] == [[1]]

    # ``DEFAULT <seq>.NEXTVAL`` used to be left untranslated and rejected by DuckDB.
    assert (
        _post(
            client,
            "CREATE TABLE seqdef (id NUMBER DEFAULT compat_seq.NEXTVAL, sku VARCHAR)",
        ).status_code
        == 200
    )


def test_bare_semi_structured_column_types(client):
    # Bare OBJECT / ARRAY map to DuckDB JSON; VARIANT is already accepted.
    resp = _post(client, "CREATE TABLE semi_t (o OBJECT, a ARRAY, v VARIANT)")
    assert resp.status_code == 200

    resp = _post(
        client,
        "INSERT INTO semi_t SELECT "
        "OBJECT_CONSTRUCT('k', 'v'), ARRAY_CONSTRUCT(1, 2), PARSE_JSON('{\"x\": 1}')",
    )
    assert resp.status_code == 200

    resp = _post(client, "SELECT COUNT(*) AS n FROM semi_t")
    assert resp.json()["data"] == [[1]]


def test_lateral_flatten_object(client):
    assert _post(client, "CREATE TABLE flat_t (attrs OBJECT)").status_code == 200
    assert (
        _post(client, "INSERT INTO flat_t SELECT OBJECT_CONSTRUCT('a', 1, 'b', 2)").status_code
        == 200
    )

    resp = _post(
        client,
        "SELECT f.key AS k, f.value AS v, f.seq AS s "
        "FROM flat_t t, LATERAL FLATTEN(input => t.attrs) f",
    )
    assert resp.status_code == 200
    rows = resp.json()["data"]
    assert len(rows) == 2
    assert {r[0] for r in rows} == {"a", "b"}
    assert {r[1] for r in rows} == {"1", "2"}


def test_table_flatten_form(client):
    assert _post(client, "CREATE TABLE flat_arr (arr VARIANT)").status_code == 200
    assert _post(client, "INSERT INTO flat_arr SELECT ARRAY_CONSTRUCT(10, 20)").status_code == 200
    resp = _post(
        client,
        "SELECT f.value FROM flat_arr t, TABLE(FLATTEN(input => t.arr)) f",
    )
    assert resp.status_code == 200
    assert {r[0] for r in resp.json()["data"]} == {"10", "20"}


def test_merge_qualified_update_set(client):
    stmt = (
        "CREATE TABLE m_t (id INT, v INT); "
        "INSERT INTO m_t VALUES (1, 10); "
        "MERGE INTO m_t t USING (SELECT 1 AS id, 99 AS v) s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET t.v = s.v "
        "WHEN NOT MATCHED THEN INSERT (id, v) VALUES (s.id, s.v)"
    )
    assert _post(client, stmt).status_code == 200

    resp = _post(client, "SELECT v FROM m_t WHERE id = 1")
    assert resp.json()["data"] == [[99]]


def test_javascript_udf_rejected(client):
    # Non-SQL UDFs have no runtime in DuckDB; they must fail loudly, not silently
    # create a function whose body is the JS source text.
    resp = _post(
        client,
        "CREATE FUNCTION js_double(x NUMBER) RETURNS NUMBER LANGUAGE JAVASCRIPT "
        "AS $$ return X * 2; $$",
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "JAVASCRIPT" in detail["message"]
    assert "not supported" in detail["message"]


def test_internal_error_returns_snowflake_shaped_422(client):
    # An invalid identifier raises ValueError inside the executor; it must surface
    # as a Snowflake-shaped 422 rather than a raw 500 (which made clients hang).
    resp = _post(client, 'USE DATABASE "bad-name"')
    assert resp.status_code == 422
    assert resp.json()["detail"]["sqlState"] == "42000"
    assert "Invalid identifier" in resp.json()["detail"]["message"]
