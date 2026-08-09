"""Shared pytest fixtures: a TestClient with fully reset in-memory state per test."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Reset the module-level singletons (DuckDB connection, sessions, result cache)
    before each test so tests don't leak state into one another.
    """
    import snowflake_emulator.database as database_module
    import snowflake_emulator.sessions as sessions_module
    import snowflake_emulator.statement_store as statement_store_module
    from snowflake_emulator.main import app

    database_module._manager = None
    sessions_module._manager = None
    statement_store_module._store = None

    with TestClient(app) as test_client:
        yield test_client
