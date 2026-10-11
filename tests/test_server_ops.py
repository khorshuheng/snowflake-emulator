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
        "max_sessions": settings.sessions.max_sessions,
    }
    yield
    settings.persistence.isolation = saved["isolation"]
    settings.statements.noop_regexes = saved["noop_regexes"]
    settings.server.host = saved["host"]
    settings.server.port = saved["port"]
    settings.persistence.database_path = saved["database_path"]
    settings.staging.stage_root = saved["stage_root"]
    settings.sessions.max_sessions = saved["max_sessions"]


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


def test_temporary_table_survives_in_isolated_mode(client, restore_settings):
    settings.persistence.isolation = "isolated"
    headers = _auth(client)

    assert _post(client, "CREATE TEMPORARY TABLE iso_tmp (a INT)", headers).status_code == 200
    assert _post(client, "INSERT INTO iso_tmp VALUES (3)", headers).status_code == 200
    assert _post(client, "SELECT a FROM iso_tmp", headers).json()["data"] == [[3]]


def test_tokenless_requests_share_one_session(client):
    """A request without a token must not retain a session (and connection) of its own."""
    import snowflake_emulator.sessions as sessions_module

    for _ in range(5):
        assert _post(client, "SELECT 1").status_code == 200
    manager = sessions_module.get_session_manager()
    assert manager._sessions == {}
    assert len(manager._sessions) == 0
    # ...and a tokenless client still gets session state: its temp table survives.
    assert _post(client, "CREATE TEMPORARY TABLE anon_keep (a INT)").status_code == 200
    assert _post(client, "INSERT INTO anon_keep VALUES (1)").status_code == 200
    assert _post(client, "SELECT a FROM anon_keep").json()["data"] == [[1]]


def test_sessions_are_capped_and_eviction_releases_the_connection(client, restore_settings):
    import snowflake_emulator.database as database_module
    import snowflake_emulator.sessions as sessions_module

    settings.sessions.max_sessions = 2
    sessions_module._manager = None
    oldest = client.post("/session/token").json()["token"]
    hdr = {"Authorization": f"Bearer {oldest}"}
    _post(client, "CREATE TEMPORARY TABLE capped (a INT)", hdr)
    assert _post(client, "SELECT * FROM capped", hdr).status_code == 200
    middle = client.post("/session/token").json()["token"]
    newest = client.post("/session/token").json()["token"]  # evicts `oldest`
    assert set(sessions_module.get_session_manager()._sessions) == {middle, newest}
    assert oldest not in database_module.get_manager(None)._session_cursors
    # The evicted token's next statement starts a fresh session, without its temp table.
    assert _post(client, "SELECT * FROM capped", hdr).status_code == 422


def test_session_close_waits_for_an_in_flight_statement(client):
    """A logout must not pull the connection out from under a running statement."""
    import threading
    import time

    from snowflake_emulator.database import get_manager
    from snowflake_emulator.sessions import get_session_manager

    token = client.post("/session/token").json()["token"]
    manager = get_manager(None)
    _post(client, "SELECT 1", {"Authorization": f"Bearer {token}"})
    closed = threading.Event()

    def closer():
        get_session_manager().close(token)
        closed.set()

    thread = threading.Thread(target=closer)
    with manager.session_scope(token):  # stand in for a statement holding the session
        thread.start()
        time.sleep(0.2)
        assert not closed.is_set()
    thread.join(timeout=5)
    assert closed.is_set()


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
        ]
    )
    apply_args(args)

    assert settings.server.host == "0.0.0.0"
    assert settings.server.port == 9999
    assert settings.persistence.database_path == "/tmp/x.duckdb"
    assert settings.staging.stage_root == "/tmp/stages"
    assert settings.persistence.isolation == "isolated"


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
