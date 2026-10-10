"""Server operations: isolation mode, configurable no-ops and CLI flags."""

from __future__ import annotations

import pytest

from snowflake_emulator.settings import settings


@pytest.fixture()
def restore_settings():
    """Snapshot and restore the settings mutated by a test."""
    saved = {
        "isolation": settings.persistence.isolation,
        "noop_regexes": list(settings.statements.noop_regexes),
        "host": settings.server.host,
        "port": settings.server.port,
        "database_path": settings.persistence.database_path,
        "stage_root": settings.staging.stage_root,
    }
    yield
    settings.persistence.isolation = saved["isolation"]
    settings.statements.noop_regexes = saved["noop_regexes"]
    settings.server.host = saved["host"]
    settings.server.port = saved["port"]
    settings.persistence.database_path = saved["database_path"]
    settings.staging.stage_root = saved["stage_root"]


def _auth(client):
    return {"Authorization": f"Bearer {client.post('/session/token').json()['token']}"}


def _post(client, statement, headers=None):
    return client.post("/api/v2/statements", json={"statement": statement}, headers=headers)


def test_configured_noop_regex_makes_statement_succeed(client, restore_settings):
    settings.statements.noop_regexes = [r"^ALTER ACCOUNT\b"]
    # Without the no-op this would be a DuckDB parse error.
    resp = _post(client, "ALTER ACCOUNT SET STATEMENT_TIMEOUT_IN_SECONDS = 60")
    assert resp.status_code == 200
    assert resp.json()["data"] == [["Statement executed successfully."]]


def test_non_matching_statement_is_not_skipped(client, restore_settings):
    settings.statements.noop_regexes = [r"^ALTER ACCOUNT\b"]
    assert _post(client, "SELEKT nope").status_code == 422


def test_isolated_mode_gives_each_session_a_private_database(client, restore_settings):
    settings.persistence.isolation = "isolated"
    first = _auth(client)
    second = _auth(client)

    assert _post(client, "CREATE TABLE iso_t (a INT)", first).status_code == 200
    assert _post(client, "SELECT COUNT(*) AS n FROM iso_t", first).status_code == 200
    # The other session must not see the first session's table.
    other = _post(client, "SELECT COUNT(*) AS n FROM iso_t", second)
    assert other.status_code == 422
    assert other.json()["detail"]["sqlState"] == "42S02"


def test_shared_mode_shares_tables(client, restore_settings):
    settings.persistence.isolation = "shared"
    first = _auth(client)
    second = _auth(client)

    assert _post(client, "CREATE TABLE shared_t (a INT)", first).status_code == 200
    assert _post(client, "SELECT COUNT(*) AS n FROM shared_t", second).status_code == 200


def test_cli_parser_applies_overrides(restore_settings):
    from snowflake_emulator.main import apply_args, build_parser

    args = build_parser().parse_args(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
            "--db-path",
            "/tmp/x.duckdb",
            "--stage-root",
            "/tmp/stages",
            "--isolation",
            "isolated",
            "--no-reload",
        ]
    )
    apply_args(args)

    assert settings.server.host == "0.0.0.0"
    assert settings.server.port == 9999
    assert settings.persistence.database_path == "/tmp/x.duckdb"
    assert settings.staging.stage_root == "/tmp/stages"
    assert settings.persistence.isolation == "isolated"
    assert args.reload is False


def test_cli_defaults_leave_settings_untouched(restore_settings):
    from snowflake_emulator.main import apply_args, build_parser

    before = (settings.server.port, settings.persistence.database_path)
    apply_args(build_parser().parse_args([]))
    assert (settings.server.port, settings.persistence.database_path) == before


def test_closing_session_releases_isolated_database(client, restore_settings):
    import snowflake_emulator.database as database_module

    settings.persistence.isolation = "isolated"
    token = client.post("/session/token").json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    _post(client, "SELECT 1", headers)
    assert token in database_module._isolated_managers

    client.post("/session", params={"delete": "true"}, headers=headers)
    assert token not in database_module._isolated_managers
