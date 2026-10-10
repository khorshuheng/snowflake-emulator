"""Error-classification tests.

A failed statement must answer with the Snowflake ``errno``/``sqlState`` pair that
matches the underlying DuckDB failure, not a single generic code. Clients branch on
this: dbt's ``retry_on_database_errors`` and migration tooling's "does it exist"
checks both key off it.
"""

from __future__ import annotations


def _post(client, statement):
    return client.post("/api/v2/statements", json={"statement": statement})


def test_missing_table_reports_catalog_error(client):
    resp = _post(client, "SELECT * FROM table_that_does_not_exist")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sqlState"] == "42S02"
    assert detail["code"] == "002003"  # six-digit, zero-padded like Snowflake


def test_unknown_column_reports_binder_error(client):
    resp = _post(client, "SELECT unknown_column FROM (SELECT 1 AS a)")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sqlState"] == "02000"
    assert detail["code"] == "002043"


def test_unparseable_sql_reports_compilation_error(client):
    resp = _post(client, "SELEKT * FROM nowhere")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sqlState"] == "42000"
    assert detail["code"] == "001003"


def test_conversion_failure_reports_data_error(client):
    resp = _post(client, "SELECT CAST('abc' AS INT)")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sqlState"] == "22000"
    assert detail["code"] == "100038"


def test_invalid_identifier_still_reports_42000(client):
    resp = _post(client, 'USE DATABASE "bad-name"')
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["sqlState"] == "42000"
    assert "Invalid identifier" in detail["message"]


def test_wire_protocol_reports_classified_catalog_error(client):
    token = client.post("/session/token").json()["token"]
    resp = client.post(
        "/queries/v1/query-request",
        json={"sqlText": "SELECT * FROM no_such_table"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["code"] == "002003"
    assert body["data"]["errorCode"] == "002003"
    assert body["data"]["sqlState"] == "42S02"


def test_wire_protocol_reports_classified_binder_error(client):
    token = client.post("/session/token").json()["token"]
    resp = client.post(
        "/queries/v1/query-request",
        json={"sqlText": "SELECT unknown_column FROM (SELECT 1 AS a)"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.json()["code"] == "002043"
    assert resp.json()["data"]["sqlState"] == "02000"
