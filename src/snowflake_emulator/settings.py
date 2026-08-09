"""Application configuration.

Configuration is loaded (in increasing priority order) from:

1. Field defaults defined below.
2. A TOML config file at ``~/.sfemu/config.toml`` (created with defaults on first run).
3. Environment variables prefixed with ``SNOWFLAKE_EMULATOR_`` (use ``__`` to reach
   nested fields, e.g. ``SNOWFLAKE_EMULATOR_SERVER__PORT=9000``).
4. Keyword arguments passed directly to ``Settings()``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_DIR = Path.home() / ".sfemu"
CONFIG_PATH = CONFIG_DIR / "config.toml"

_DEFAULT_CONFIG_TOML = """\
# Snowflake Emulator configuration.

[server]
# Host/port the FastAPI server binds to.
host = "localhost"
port = 8000

[snowflake]
# Default Snowflake-style identifiers used when a request/session omits them.
default_database = "EMULATOR_DB"
default_schema = "PUBLIC"
default_warehouse = "EMULATOR_WH"
default_role = "ACCOUNTADMIN"

[persistence]
# Path to a DuckDB database file for durable storage across restarts.
# Use ":memory:" (the default) for an ephemeral, in-process database.
database_path = ":memory:"

[statements]
# Maximum number of cached statement results kept in memory.
max_cached_statements = 500
"""


def ensure_config_file(path: Path = CONFIG_PATH) -> Path:
    """Create the config file (and its parent directory) with defaults if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(_DEFAULT_CONFIG_TOML)
    return path


class ServerSettings(BaseModel):
    host: str = "localhost"
    port: int = 8000


class SnowflakeDefaults(BaseModel):
    default_database: str = "EMULATOR_DB"
    default_schema: str = "PUBLIC"
    default_warehouse: str = "EMULATOR_WH"
    default_role: str = "ACCOUNTADMIN"


class PersistenceSettings(BaseModel):
    # Path to the DuckDB database file. Use ":memory:" for an ephemeral, in-process database.
    database_path: str = ":memory:"


class StatementSettings(BaseModel):
    max_cached_statements: int = 500


class Settings(BaseSettings):
    """Runtime configuration for the emulator, overridable via TOML file and/or env vars."""

    model_config = SettingsConfigDict(
        env_prefix="SNOWFLAKE_EMULATOR_",
        env_nested_delimiter="__",
        toml_file=str(ensure_config_file()),
        extra="ignore",
    )

    server: ServerSettings = ServerSettings()
    snowflake: SnowflakeDefaults = SnowflakeDefaults()
    persistence: PersistenceSettings = PersistenceSettings()
    statements: StatementSettings = StatementSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority (highest first): init kwargs > env vars > TOML file > field defaults.
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )

    # -- Convenience flat accessors, kept for backwards compatibility with existing
    # -- call sites (`settings.database_path`, `settings.default_database`, ...).
    @property
    def database_path(self) -> str:
        return self.persistence.database_path

    @property
    def default_database(self) -> str:
        return self.snowflake.default_database

    @property
    def default_schema(self) -> str:
        return self.snowflake.default_schema

    @property
    def default_warehouse(self) -> str:
        return self.snowflake.default_warehouse

    @property
    def default_role(self) -> str:
        return self.snowflake.default_role

    @property
    def max_cached_statements(self) -> int:
        return self.statements.max_cached_statements

    @property
    def host(self) -> str:
        return self.server.host

    @property
    def port(self) -> int:
        return self.server.port


settings = Settings()
