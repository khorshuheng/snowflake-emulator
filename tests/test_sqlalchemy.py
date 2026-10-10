"""Real-client tests using SQLAlchemy via the ``snowflake-sqlalchemy`` dialect.

Like ``test_connector_integration.py``, this drives a real uvicorn server with an
official client rather than FastAPI's TestClient, so it exercises the parts of the
wire protocol that SQLAlchemy's dialect relies on (version probe, DESCRIBE,
per-connection session state).
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn

import snowflake_emulator.database as database_module
import snowflake_emulator.file_formats as file_formats_module
import snowflake_emulator.sessions as sessions_module
import snowflake_emulator.stages as stages_module
import snowflake_emulator.statement_store as statement_store_module
from snowflake_emulator.main import app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def emulator_port():
    database_module._manager = None
    database_module._isolated_managers.clear()
    sessions_module._manager = None
    stages_module._manager = None
    statement_store_module._store = None
    file_formats_module._manager = None

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "emulator server failed to start"

    yield port

    server.should_exit = True
    thread.join(timeout=5)


def _engine(port: int):
    from sqlalchemy import create_engine

    url = "snowflake://test_user:test_password@test_account/EMULATOR_DB/PUBLIC"
    return create_engine(
        url,
        connect_args={
            "host": "127.0.0.1",
            "port": port,
            "protocol": "http",
            "disable_ocsp_checks": True,
        },
    )


def test_sqlalchemy_select_and_dml(emulator_port):
    from sqlalchemy import text

    engine = _engine(emulator_port)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1 AS x")).scalar() == 1
        conn.execute(text("CREATE OR REPLACE TABLE sa_t (a INT, b VARCHAR)"))
        conn.execute(text("INSERT INTO sa_t VALUES (1, 'x'), (2, 'y')"))
        assert conn.execute(text("SELECT * FROM sa_t ORDER BY a")).fetchall() == [
            (1, "x"),
            (2, "y"),
        ]


def test_sqlalchemy_tokenized_identifier(emulator_port):
    """The dialect quotes identifiers; a quoted mixed-case name must round-trip."""
    from sqlalchemy import text

    engine = _engine(emulator_port)
    with engine.connect() as conn:
        conn.execute(text('CREATE OR REPLACE TABLE "MixedCase" ("Col" INT)'))
        conn.execute(text('INSERT INTO "MixedCase" VALUES (7)'))
        assert conn.execute(text('SELECT "Col" FROM "MixedCase"')).scalar() == 7


def test_sqlalchemy_reflect_columns(emulator_port):
    from sqlalchemy import inspect, text

    engine = _engine(emulator_port)
    with engine.connect() as conn:
        conn.execute(text("CREATE OR REPLACE TABLE sa_meta (id INT, name VARCHAR)"))
    columns = inspect(engine).get_columns("sa_meta", schema="PUBLIC")
    assert {column["name"].upper() for column in columns} == {"ID", "NAME"}
