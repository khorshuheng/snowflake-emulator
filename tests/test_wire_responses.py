"""Wire-protocol response fidelity: statement type ids, session echo, describeOnly
and the monitoring endpoint."""

from __future__ import annotations

from snowflake_emulator import statement_types as st


def _wire(client, sql, **extra):
    token = client.post("/session/token").json()["token"]
    return client.post(
        "/queries/v1/query-request",
        json={"sqlText": sql, **extra},
        headers={"Authorization": f"Bearer {token}"},
    )


def test_statement_type_id_select(client):
    assert _wire(client, "SELECT 1 AS x").json()["data"]["statementTypeId"] == st.SELECT


def test_statement_type_id_insert(client):
    _wire(client, "CREATE TABLE sti (a INT)")
    assert _wire(client, "INSERT INTO sti VALUES (1)").json()["data"]["statementTypeId"] == st.INSERT


def test_statement_type_id_ddl(client):
    assert _wire(client, "CREATE TABLE std (a INT)").json()["data"]["statementTypeId"] == st.DDL


def test_final_names_returned_on_every_response(client):
    data = _wire(client, "SELECT 1").json()["data"]
    assert data["finalDatabaseName"] == "EMULATOR_DB"
    assert data["finalSchemaName"] == "PUBLIC"


def test_describe_only_select_returns_metadata_without_rows(client):
    body = _wire(client, "SELECT 1 AS n, 'x' AS s", describeOnly=True).json()
    assert body["success"] is True
    assert body["data"]["rowset"] == []
    assert [c["name"] for c in body["data"]["rowtype"]] == ["N", "S"]
    assert body["data"]["statementTypeId"] == st.SELECT


def test_describe_only_reports_bind_count(client):
    body = _wire(client, "SELECT ? AS n", describeOnly=True).json()
    assert body["data"]["numberOfBinds"] == 1
    assert [c["name"] for c in body["data"]["rowtype"]] == ["N"]


def test_describe_only_insert_does_not_execute(client):
    _wire(client, "CREATE TABLE dsc (a INT)")
    body = _wire(client, "INSERT INTO dsc VALUES (1)", describeOnly=True).json()
    assert body["data"]["arrayBindSupported"] is True
    assert body["data"]["rowset"] == []
    count = client.post(
        "/api/v2/statements", json={"statement": "SELECT COUNT(*) AS n FROM dsc"}
    ).json()
    assert count["data"] == [[0]]  # the described INSERT must not have run


def test_monitoring_endpoint_returns_empty_for_unknown_query(client):
    token = client.post("/session/token").json()["token"]
    resp = client.get(
        "/monitoring/queries/does-not-exist", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"data": {"queries": []}, "success": True}


def test_describe_only_merge_does_not_execute(client):
    """Describing a MERGE must not run the write."""
    _wire(client, "CREATE TABLE mrg (id INT, v INT)")
    _wire(client, "INSERT INTO mrg VALUES (1, 10)")
    _wire(
        client,
        "MERGE INTO mrg t USING (SELECT 1 AS id, 99 AS v) s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET t.v = s.v",
        describeOnly=True,
    )
    row = client.post(
        "/api/v2/statements", json={"statement": "SELECT v FROM mrg WHERE id = 1"}
    ).json()
    assert row["data"] == [[10]]


def test_describe_only_use_does_not_change_session(client):
    token = client.post("/session/token").json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    def wire(sql, **extra):
        return client.post(
            "/queries/v1/query-request", json={"sqlText": sql, **extra}, headers=headers
        )

    wire("USE DATABASE EMULATOR_DB")
    wire("USE DATABASE desc_alt", describeOnly=True)
    body = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT CURRENT_DATABASE() AS d"},
        headers=headers,
    ).json()
    assert body["data"] == [["EMULATOR_DB"]]  # the described USE must not have applied


def test_describe_only_named_parameter(client):
    token = client.post("/session/token").json()["token"]
    body = client.post(
        "/queries/v1/query-request",
        json={"sqlText": "SELECT $a AS x", "describeOnly": True},
        headers={"Authorization": f"Bearer {token}"},
    ).json()
    assert body["success"] is True, body
    assert [col["name"] for col in body["data"]["rowtype"]] == ["X"]
    assert body["data"]["numberOfBinds"] == 1


def test_describe_only_merge_reports_only_its_actions(client):
    _wire(client, "CREATE TABLE mrg_desc (id INT, v INT)")
    body = _wire(
        client,
        "MERGE INTO mrg_desc t USING (SELECT 1 AS id) s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET v = 1",
        describeOnly=True,
    ).json()
    assert [col["name"] for col in body["data"]["rowtype"]] == ["number of rows updated"]
