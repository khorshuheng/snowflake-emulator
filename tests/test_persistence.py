"""File-backed persistence: data survives a manager restart."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.settings import settings


@pytest.fixture()
def persistent_db(tmp_path):
    """Point persistence at a scratch file and restore the settings afterwards."""
    saved_path = settings.persistence.database_path
    saved_default = settings.snowflake.default_database
    saved_isolation = settings.persistence.isolation
    settings.persistence.database_path = str(tmp_path / "emu.duckdb")
    yield settings.persistence.database_path
    settings.persistence.database_path = saved_path
    settings.snowflake.default_database = saved_default
    settings.persistence.isolation = saved_isolation


def _storage_dir(path: str) -> Path:
    """The dedicated directory holding one DuckDB file per Snowflake database."""
    return Path(f"{path}.d")


def _post(client, statement):
    return client.post("/api/v2/statements", json={"statement": statement})


def test_non_default_database_survives_reopen(persistent_db):
    path = persistent_db
    manager = DuckDBManager(path)
    manager.ensure_namespace("MY_DB", "PUBLIC")
    manager.connection.execute('CREATE TABLE "MY_DB"."PUBLIC".t (x INTEGER)')
    manager.connection.execute('INSERT INTO "MY_DB"."PUBLIC".t VALUES (42)')
    manager.connection.close()

    assert (_storage_dir(path) / "MY_DB.duckdb").exists()
    reopened = DuckDBManager(path)
    assert reopened.has_database("MY_DB")
    assert reopened.connection.execute('SELECT * FROM "MY_DB"."PUBLIC".t').fetchall() == [(42,)]
    reopened.connection.close()


def test_default_database_survives_reopen(persistent_db):
    path = persistent_db
    default = settings.default_database
    manager = DuckDBManager(path)
    manager.ensure_namespace(default, "PUBLIC")
    manager.connection.execute(f'CREATE TABLE "{default}"."PUBLIC".d (x INTEGER)')
    manager.connection.execute(f'INSERT INTO "{default}"."PUBLIC".d VALUES (7)')
    manager.connection.close()

    assert (_storage_dir(path) / f"{default}.duckdb").exists()
    reopened = DuckDBManager(path)
    assert reopened.connection.execute(f'SELECT * FROM "{default}"."PUBLIC".d').fetchall() == [(7,)]
    reopened.connection.close()


def test_changing_default_database_does_not_relabel(persistent_db):
    """A database keeps its identity when `snowflake.default_database` changes."""
    path = persistent_db
    settings.snowflake.default_database = "MY_DB"
    manager = DuckDBManager(path)
    manager.ensure_namespace("MY_DB", "PUBLIC")
    manager.connection.execute('CREATE TABLE "MY_DB"."PUBLIC".t (x INTEGER)')
    manager.connection.execute('INSERT INTO "MY_DB"."PUBLIC".t VALUES (1)')
    manager.connection.close()

    settings.snowflake.default_database = "EMULATOR_DB"
    reopened = DuckDBManager(path)
    assert reopened.has_database("MY_DB")
    assert reopened.connection.execute('SELECT * FROM "MY_DB"."PUBLIC".t').fetchall() == [(1,)]
    reopened.connection.close()


def test_create_or_replace_database_clears_persisted_contents(persistent_db):
    path = persistent_db
    manager = DuckDBManager(path)
    manager.ensure_namespace("MY_DB", "PUBLIC")
    manager.connection.execute('CREATE TABLE "MY_DB"."PUBLIC".t (x INTEGER)')
    manager.connection.execute('INSERT INTO "MY_DB"."PUBLIC".t VALUES (1)')

    manager.replace_database("MY_DB")
    with pytest.raises(duckdb.CatalogException):
        manager.connection.execute('SELECT * FROM "MY_DB"."PUBLIC".t')
    manager.connection.close()

    reopened = DuckDBManager(path)
    with pytest.raises(duckdb.CatalogException):
        reopened.connection.execute('SELECT * FROM "MY_DB"."PUBLIC".t')
    reopened.connection.close()


def test_memory_mode_writes_no_files(tmp_path):
    manager = DuckDBManager(":memory:")
    manager.ensure_namespace("MY_DB", "PUBLIC")
    manager.connection.execute('CREATE TABLE "MY_DB"."PUBLIC".t (x INTEGER)')
    assert list(tmp_path.iterdir()) == []
    manager.connection.close()


def test_foreign_matching_file_outside_storage_dir_is_not_attached(persistent_db):
    """Only the dedicated directory is scanned; parent-directory files are ignored."""
    path = persistent_db
    foreign = Path(path).with_name("emu__PAYROLL.duckdb")
    connection = duckdb.connect(str(foreign))
    connection.execute("CREATE TABLE secrets (ssn VARCHAR)")
    connection.close()

    manager = DuckDBManager(path)
    assert not manager.has_database("PAYROLL")
    assert "PAYROLL" not in {
        row[0]
        for row in manager.connection.execute(
            "SELECT database_name FROM duckdb_databases()"
        ).fetchall()
    }
    manager.connection.close()


def test_unreadable_database_file_is_skipped(persistent_db):
    """One bad file must not make the whole manager unusable."""
    path = persistent_db
    storage = _storage_dir(path)
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "NOTES.duckdb").write_text("not a database file")

    with pytest.warns(UserWarning, match="NOTES"):
        manager = DuckDBManager(path)

    assert "NOTES" not in {
        row[0]
        for row in manager.connection.execute(
            "SELECT database_name FROM duckdb_databases()"
        ).fetchall()
    }
    manager.connection.close()


def test_isolated_mode_warns_when_database_path_is_set(persistent_db, monkeypatch):
    import snowflake_emulator.database as database_module

    monkeypatch.setattr(database_module, "_isolated_persistence_warned", False)
    settings.persistence.isolation = "isolated"
    try:
        with pytest.warns(UserWarning, match="isolated"):
            manager = database_module.get_manager(session=SimpleNamespace(token="t1"))
        assert manager._persistent is False
    finally:
        settings.persistence.isolation = "shared"
        database_module._isolated_managers.clear()


def test_persisted_databases_are_visible_after_restart(client, persistent_db):
    """A new manager re-attaches prior database files, so SHOW DATABASES lists them."""
    import snowflake_emulator.database as database_module

    assert _post(client, "CREATE DATABASE MY_DB").status_code == 200
    assert _post(client, "CREATE TABLE MY_DB.PUBLIC.t (x INT)").status_code == 200
    assert _post(client, "INSERT INTO MY_DB.PUBLIC.t VALUES (5)").status_code == 200

    # Simulate a process restart: drop the shared manager, closing its file handles.
    database_module._manager.connection.close()
    database_module._manager = None

    names = {row[1] for row in _post(client, "SHOW DATABASES").json()["data"]}
    assert "MY_DB" in names
    assert _post(client, "SELECT * FROM MY_DB.PUBLIC.t").json()["data"] == [[5]]
