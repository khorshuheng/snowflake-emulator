"""Tests for TOML/env-based configuration (settings.py)."""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    """Point `Path.home()` at a scratch directory and reload the settings module so
    `~/.sfemu/config.toml` is created/read from there instead of the real home dir.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    # Ensure no leftover env overrides from other tests bleed into this one.
    for key in list(os.environ):
        if key.startswith("SNOWFLAKE_EMULATOR_"):
            monkeypatch.delenv(key, raising=False)

    import snowflake_emulator.settings as settings_module

    importlib.reload(settings_module)
    yield settings_module, tmp_path
    importlib.reload(settings_module)


def test_config_file_created_with_defaults(isolated_home):
    settings_module, tmp_path = isolated_home
    config_path = tmp_path / ".sfemu" / "config.toml"

    assert config_path.exists()
    assert settings_module.settings.host == "localhost"
    assert settings_module.settings.port == 8000
    assert settings_module.settings.default_database == "EMULATOR_DB"
    assert settings_module.settings.default_schema == "PUBLIC"
    assert settings_module.settings.database_path == ":memory:"
    assert settings_module.settings.max_cached_statements == 500
    assert settings_module.settings.stage_root == ""


def test_config_file_values_are_applied(isolated_home):
    settings_module, tmp_path = isolated_home
    config_path = tmp_path / ".sfemu" / "config.toml"
    config_path.write_text(
        """
[server]
host = "0.0.0.0"
port = 9999

[snowflake]
default_database = "MY_DB"
default_schema = "MY_SCHEMA"
default_warehouse = "EMULATOR_WH"
default_role = "ACCOUNTADMIN"

[persistence]
database_path = "/tmp/sfemu-test.duckdb"

[statements]
max_cached_statements = 42

[staging]
stage_root = "/tmp/sfemu-stages"
"""
    )

    settings_module.Settings.model_config["toml_file"] = str(config_path)
    reloaded = settings_module.Settings()

    assert reloaded.host == "0.0.0.0"
    assert reloaded.port == 9999
    assert reloaded.default_database == "MY_DB"
    assert reloaded.default_schema == "MY_SCHEMA"
    assert reloaded.database_path == "/tmp/sfemu-test.duckdb"
    assert reloaded.max_cached_statements == 42
    assert reloaded.stage_root == "/tmp/sfemu-stages"


def test_env_var_overrides_toml_file(isolated_home, monkeypatch):
    settings_module, _ = isolated_home
    monkeypatch.setenv("SNOWFLAKE_EMULATOR_SERVER__PORT", "7777")

    reloaded = settings_module.Settings()

    assert reloaded.port == 7777
    # Untouched settings should still come from the TOML file/defaults.
    assert reloaded.host == "localhost"
