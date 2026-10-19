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


def test_show_terse_schemas_snowflake_shape(client):
    resp = _post(client, "SHOW TERSE SCHEMAS IN DATABASE EMULATOR_DB")
    assert resp.status_code == 200
    body = resp.json()
    names = [c["name"] for c in body["resultSetMetaData"]["rowType"]]
    assert "name" in names  # dbt reads row["name"] from this result
    idx = {n: i for i, n in enumerate(names)}
    assert any(r[idx["name"]] == "PUBLIC" and r[idx["kind"]] == "SCHEMA" for r in body["data"])


def test_show_objects_snowflake_shape(client):
    assert _post(client, "CREATE TABLE obj_t (a INT)").status_code == 200
    resp = _post(client, "SHOW OBJECTS IN EMULATOR_DB.PUBLIC")
    assert resp.status_code == 200
    body = resp.json()
    names = [c["name"] for c in body["resultSetMetaData"]["rowType"]]
    for required in ("database_name", "schema_name", "name", "kind", "is_dynamic", "is_iceberg"):
        assert required in names  # columns dbt selects from SHOW OBJECTS
    idx = {n: i for i, n in enumerate(names)}
    assert any(
        r[idx["name"]] == "OBJ_T" and r[idx["kind"]] == "TABLE" for r in body["data"]
    )


def test_show_user_functions_returns_empty_shape(client):
    resp = _post(client, "SHOW USER FUNCTIONS IN EMULATOR_DB.PUBLIC")
    assert resp.status_code == 200
    body = resp.json()
    names = [c["name"] for c in body["resultSetMetaData"]["rowType"]]
    assert "name" in names and "catalog_name" in names and "is_builtin" in names
    assert body["data"] == []


def test_describe_table_snowflake_column_names(client):
    assert _post(client, "CREATE TABLE desc_s (id INT, name VARCHAR)").status_code == 200
    resp = _post(client, "DESCRIBE TABLE desc_s")
    assert resp.status_code == 200
    body = resp.json()
    names = [c["name"] for c in body["resultSetMetaData"]["rowType"]]
    assert names[:2] == ["name", "type"]  # dbt reads row["name"]/row["type"]
    rows = body["data"]
    types = {r[0]: r[1] for r in rows}
    assert set(types) == {"ID", "NAME"}
    assert types["ID"] == "NUMBER(38,0)"
    assert types["NAME"] == "VARCHAR"


def test_insert_overwrite_clears_then_inserts(client):
    stmt = (
        "CREATE TABLE iow (id INT, v INT); "
        "INSERT INTO iow VALUES (1, 10); "
        "INSERT OVERWRITE INTO iow (id, v) VALUES (2, 20)"
    )
    assert _post(client, stmt).status_code == 200
    resp = _post(client, "SELECT id, v FROM iow ORDER BY id")
    assert resp.json()["data"] == [[2, 20]]


def test_create_database(client):
    resp = _post(client, "CREATE DATABASE sales_db")
    assert resp.status_code == 200
    assert resp.json()["data"] == [["Database SALES_DB successfully created."]]
    # A new database ships with a default PUBLIC schema, visible via INFORMATION_SCHEMA.
    resp = _post(
        client,
        "SELECT COUNT(*) AS n FROM SALES_DB.INFORMATION_SCHEMA.SCHEMATA "
        "WHERE SCHEMA_NAME = 'PUBLIC'",
    )
    assert resp.json()["data"] == [[1]]


def test_create_database_if_not_exists_and_duplicate(client):
    assert _post(client, "CREATE DATABASE dup_db").status_code == 200
    assert _post(client, "CREATE DATABASE IF NOT EXISTS dup_db").status_code == 200
    resp = _post(client, "CREATE DATABASE dup_db")
    assert resp.status_code == 422
    assert "already exists" in resp.json()["detail"]["message"]


def test_create_or_replace_database(client):
    assert _post(client, "CREATE DATABASE repl_db").status_code == 200
    assert _post(client, "CREATE TABLE REPL_DB.PUBLIC.t1 (a INT)").status_code == 200
    assert _post(client, "CREATE OR REPLACE DATABASE repl_db").status_code == 200
    # Replacing the database drops its previous contents.
    resp = _post(
        client,
        "SELECT COUNT(*) AS n FROM REPL_DB.INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = 'PUBLIC' AND TABLE_NAME = 'T1'",
    )
    assert resp.json()["data"] == [[0]]


def test_ddl_returns_status_shape(client):
    # DuckDB reports DDL as a bare Count/Success column; Snowflake returns a status.
    assert _post(client, "CREATE TABLE ddl_t (a INT)").status_code == 200
    resp = _post(client, "CREATE SCHEMA ddl_s")
    assert [c["name"] for c in resp.json()["resultSetMetaData"]["rowType"]] == ["status"]
    resp = _post(client, "CREATE SEQUENCE ddl_sq")
    assert [c["name"] for c in resp.json()["resultSetMetaData"]["rowType"]] == ["status"]
    resp = _post(client, "DROP TABLE ddl_t")
    assert [c["name"] for c in resp.json()["resultSetMetaData"]["rowType"]] == ["status"]


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


def test_json_path_extraction_on_bound_variant_operand(client):
    # A VARIANT value bound from a Python dict is stored by DuckDB as a Python-repr
    # string (``{'a': 1}``) that is not valid JSON, and ``->`` used to raise when it
    # tried to parse it. The operand must be cast back to JSON first.
    headers = _auth_headers(client)

    def wire(statement, bindings=None):
        body = {"sqlText": statement}
        if bindings is not None:
            body["bindings"] = bindings
        return client.post("/queries/v1/query-request", json=body, headers=headers)

    assert wire("CREATE OR REPLACE TABLE v_t (id INT, data VARIANT)").json()["success"]
    assert wire(
        "INSERT INTO v_t VALUES (?, ?)",
        {
            "1": {"type": "FIXED", "value": 1},
            "2": {"type": "VARIANT", "value": {"a": 1, "b": {"c": 2}}},
        },
    ).json()["success"]

    resp = wire(
        "SELECT data:a AS a, data:b.c AS c, data:a::int AS n FROM v_t"
    ).json()
    assert resp["success"], resp
    import base64

    import pyarrow as pa

    table = pa.ipc.open_stream(
        pa.BufferReader(base64.b64decode(resp["data"]["rowsetBase64"]))
    ).read_all()
    assert [str(value) for value in table.column(0).to_pylist()] == ["1"]
    assert [str(value) for value in table.column(1).to_pylist()] == ["2"]
    assert table.column(2).to_pylist() == [1]


def test_timestamp_precision_translation_and_metadata(client):
    # Snowflake's TIMESTAMP_TZ(p)/TIMESTAMP_LTZ(p) precision parameter must not make
    # the translated DuckDB DDL/CAST invalid (DuckDB rejects TIMESTAMPTZ(p)), and the
    # reported type/scale must reflect the precision DuckDB actually stores.
    headers = _auth_headers(client)
    assert _post(
        client,
        "CREATE TABLE ts_p (a TIMESTAMP(3), b TIMESTAMP(0), c TIMESTAMP(9), "
        "d TIMESTAMP_LTZ(3), e TIMESTAMP_TZ(6))",
        headers,
    ).status_code == 200

    resp = _post(client, "DESCRIBE TABLE ts_p", headers)
    types = {row[0]: row[1] for row in resp.json()["data"]}
    assert types == {
        "A": "TIMESTAMP_NTZ(3)",
        "B": "TIMESTAMP_NTZ(0)",
        "C": "TIMESTAMP_NTZ(9)",
        # DuckDB has no LTZ type; LTZ maps to TIMESTAMP_TZ at microsecond precision.
        "D": "TIMESTAMP_TZ(6)",
        "E": "TIMESTAMP_TZ(6)",
    }

    # LTZ/TZ casts must execute rather than raise "does not take any type parameters".
    resp = _post(
        client,
        "SELECT CAST('2020-01-01 12:00:00.123456789' AS TIMESTAMP(3)) AS a, "
        "CAST('2020-01-01 12:00:00.123456789' AS TIMESTAMP_LTZ(3)) AS b, "
        "CAST('2020-01-01 12:00:00.123456789' AS TIMESTAMP_TZ(3)) AS c",
        headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"][0][0] == "2020-01-01T12:00:00.123000"

    # The connector wire metadata advertises the real granularity, not a fixed 6.
    wire = client.post(
        "/queries/v1/query-request",
        json={"sqlText": "SELECT a, b, c FROM ts_p"},
        headers=headers,
    ).json()
    scales = {col["name"]: col["scale"] for col in wire["data"]["rowtype"]}
    assert scales == {"A": 3, "B": 0, "C": 9}


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
    assert {r[1] for r in rows} == {1, 2}


def test_table_flatten_form(client):
    assert _post(client, "CREATE TABLE flat_arr (arr VARIANT)").status_code == 200
    assert _post(client, "INSERT INTO flat_arr SELECT ARRAY_CONSTRUCT(10, 20)").status_code == 200
    resp = _post(
        client,
        "SELECT f.value FROM flat_arr t, TABLE(FLATTEN(input => t.arr)) f",
    )
    assert resp.status_code == 200
    assert {r[0] for r in resp.json()["data"]} == {10, 20}


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


def test_backslash_escapes_stay_literal_in_string_literals(client):
    """sqlglot >= 30.21 decodes backslash escapes inside Snowflake string literals.

    An unknown escape lost its backslash (``\\d`` -> ``d``) and a numeric escape was
    decoded (``\\x41`` -> ``A``), silently breaking regex calls and stored paths. The
    emulator pins the Snowflake tokenizer back to the pre-30.21 behaviour.
    """
    resp = _post(client, r"SELECT REGEXP_REPLACE('a1b', '\d', 'x') AS v")
    assert resp.status_code == 200
    assert resp.json()["data"] == [["axb"]]

    resp = _post(client, r"SELECT '1' RLIKE '\d' AS v")
    assert resp.json()["data"] == [[True]]

    # Numeric escapes are not decoded either.
    resp = _post(client, r"SELECT '\x41' AS v")
    assert resp.json()["data"] == [[r"\x41"]]

    # A backslash in stored text survives a round trip.
    assert _post(client, "CREATE TABLE esc_t (s VARCHAR)").status_code == 200
    assert _post(client, r"INSERT INTO esc_t VALUES ('C:\Users\me')").status_code == 200
    resp = _post(client, "SELECT s FROM esc_t")
    assert resp.json()["data"] == [["C:\\Users\\me"]]


def test_constraints_are_recorded_but_not_enforced(client):
    """Snowflake records PK/UNIQUE/FK/CHECK but only enforces NOT NULL."""
    assert _post(client, "CREATE TABLE parent_nf (id INT PRIMARY KEY)").status_code == 200
    assert (
        _post(
            client,
            "CREATE TABLE child_nf ("
            "id INT PRIMARY KEY, "
            "u INT UNIQUE, "
            "c INT CHECK (c > 0), "
            "pid INT REFERENCES parent_nf(id))",
        ).status_code
        == 200
    )

    # Duplicate primary key and unique value, a failing CHECK and an orphan FK are all
    # accepted, matching Snowflake's non-enforcement.
    assert (
        _post(client, "INSERT INTO child_nf VALUES (1, 1, 1, 99)").status_code == 200
    )
    assert (
        _post(client, "INSERT INTO child_nf VALUES (1, 1, -1, 99)").status_code == 200
    )
    assert _post(client, "SELECT COUNT(*) AS n FROM child_nf").json()["data"] == [[2]]

    rows = _post(client, "SHOW PRIMARY KEYS IN TABLE child_nf").json()["data"]
    assert [(row[4], row[5]) for row in rows] == [("ID", 1)]
    rows = _post(client, "SHOW UNIQUE KEYS IN TABLE child_nf").json()["data"]
    assert [row[4] for row in rows] == ["U"]
    rows = _post(client, "SHOW FOREIGN KEYS IN TABLE child_nf").json()["data"]
    assert any(row[4] == "ID" and row[8] == "PID" for row in rows)

    # NOT NULL is the one constraint Snowflake does enforce.
    assert _post(client, "CREATE TABLE nn_nf (v INT NOT NULL)").status_code == 200
    assert _post(client, "INSERT INTO nn_nf VALUES (NULL)").status_code == 422


def test_transactions_commit_and_rollback(client):
    assert _post(client, "CREATE TABLE tx_t (a INT)").status_code == 200
    assert _post(client, "INSERT INTO tx_t VALUES (1)").status_code == 200

    assert _post(client, "BEGIN").status_code == 200
    assert _post(client, "INSERT INTO tx_t VALUES (2)").status_code == 200
    assert _post(client, "ROLLBACK").status_code == 200
    assert _post(client, "SELECT * FROM tx_t ORDER BY a").json()["data"] == [[1]]

    assert _post(client, "BEGIN").status_code == 200
    assert _post(client, "INSERT INTO tx_t VALUES (3)").status_code == 200
    assert _post(client, "COMMIT").status_code == 200
    assert _post(client, "SELECT * FROM tx_t ORDER BY a").json()["data"] == [[1], [3]]

    # A multi-statement request shares the same transaction.
    resp = _post(
        client,
        "BEGIN; INSERT INTO tx_t VALUES (4); ROLLBACK; "
        "SELECT * FROM tx_t ORDER BY a",
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == [[1], [3]]


def test_commit_or_rollback_outside_transaction_is_a_noop(client):
    for statement in ("COMMIT", "ROLLBACK"):
        assert _post(client, statement).status_code == 200
    assert _post(client, "SELECT 1 AS x").json()["data"] == [[1]]
