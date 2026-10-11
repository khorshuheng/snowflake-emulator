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
# Anchor path for durable storage across restarts. When set to a file path, a
# directory "<database_path>.d" is created next to it holding one DuckDB file per
# Snowflake database, e.g. emulator.duckdb.d/MY_DB.duckdb. Use ":memory:" (the
# default) for an ephemeral, in-process database.
database_path = ":memory:"
# Isolation mode. "shared" (default) keeps one database for all sessions;
# "isolated" gives each session its own private in-memory database.
isolation = "shared"

[statements]
# Maximum number of cached statement results kept in memory.
max_cached_statements = 500
# Regexes (matched case-insensitively against the whole statement) whose statements
# succeed without being executed, e.g. session parameters the emulator does not model.
noop_regexes = []

[sessions]
# Maximum number of live bearer-token sessions. Above this the least recently used
# session is evicted, releasing its connection and any temporary tables it held, as a
# real Snowflake session does when it expires. 0 or less keeps every session.
max_sessions = 1000

[staging]
# Root directory under which staged files (PUT/GET/COPY INTO @stage) are stored.
# Leave empty to use an auto-created temporary directory.
stage_root = ""
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
    # Anchor path for durable storage. A file path creates a `<path>.d` directory
    # holding one DuckDB file per database; ":memory:" is ephemeral.
    database_path: str = ":memory:"
    # "shared" (one database for all sessions) or "isolated" (one per session).
    isolation: str = "shared"


class StatementSettings(BaseModel):
    max_cached_statements: int = 500
    # Statements matching any of these regexes succeed without being executed.
    noop_regexes: list[str] = []


class SessionSettings(BaseModel):
    # Live bearer-token sessions kept before the least recently used one is evicted;
    # 0 or less keeps every session (unbounded).
    max_sessions: int = 1000


class StagingSettings(BaseModel):
    # Root directory for staged files; empty means an auto-created temp directory.
    stage_root: str = ""


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
    sessions: SessionSettings = SessionSettings()
    staging: StagingSettings = StagingSettings()

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
    def max_sessions(self) -> int:
        return self.sessions.max_sessions

    @property
    def isolation(self) -> str:
        return self.persistence.isolation

    @property
    def noop_regexes(self) -> list[str]:
        return self.statements.noop_regexes

    @property
    def stage_root(self) -> str:
        return self.staging.stage_root

    @property
    def host(self) -> str:
        return self.server.host

    @property
    def port(self) -> int:
        return self.server.port


settings = Settings()
