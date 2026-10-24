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


def test_parse_error_has_no_ansi_escapes_and_no_stray_hint(client):
    detail = _post(client, "SELEKT * FROM nowhere").json()["detail"]
    assert "\x1b" not in detail["message"]
    assert "LIKE before IN" not in detail["message"]


def test_show_like_after_in_names_the_snowflake_clause_order(client):
    """``SHOW ... IN ... LIKE`` is MySQL's order; Snowflake wants LIKE first, so say so."""
    resp = _post(client, "SHOW TABLES IN SCHEMA EMULATOR_DB.PUBLIC LIKE 'a%'")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert (detail["sqlState"], detail["code"]) == ("42000", "001003")
    assert "LIKE before IN" in detail["message"]
    assert "SHOW TABLES LIKE '<pattern>' IN SCHEMA <database>.<schema>." in detail["message"]


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


def test_division_by_zero_reports_snowflake_division_error(client):
    """Snowflake raises ``100051 (22012)`` for ``1/0``; DuckDB answers NULL."""
    resp = _post(client, "SELECT 1/0")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert (detail["sqlState"], detail["code"]) == ("22012", "100051")
    assert "Division by zero" in detail["message"]


def test_division_by_zero_guards_are_not_errors(client):
    """The rewritten division keeps NULL dividends, NULL divisors and DIV0 answering."""
    resp = _post(
        client,
        "SELECT DIV0(1,0), DIV0(NULL,0), DIV0NULL(NULL,0), 1/NULL, 4/2",
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == [[0.0, None, 0.0, None, 2.0]]
