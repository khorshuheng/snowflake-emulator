"""Integration tests using the real `snowflake-connector-python` driver.

Unlike `tests/test_statements_api.py` (which talks to the SQL API v2 surface via
FastAPI's `TestClient`), these tests spin up an actual `uvicorn` server and drive it
with the official Snowflake Python connector, exercising the private wire-protocol
endpoints implemented in `snowflake_emulator.routers.auth` / `.queries`
(`/session/v1/login-request`, `/queries/v1/query-request`, `/session?delete=true`).
"""

from __future__ import annotations

import datetime
import json
import socket
import threading
import time

import pytest
import snowflake.connector
import uvicorn
from snowflake.connector.constants import FIELD_NAME_TO_ID as FIELD_ID_TO_NAME_LOOKUP

import snowflake_emulator.database as database_module
import snowflake_emulator.sessions as sessions_module
import snowflake_emulator.statement_store as statement_store_module
from snowflake_emulator.main import app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def emulator_port():
    """Run the real emulator app on a background thread, with fresh in-memory state."""
    database_module._manager = None
    sessions_module._manager = None
    statement_store_module._store = None

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "emulator server failed to start"

    yield port

    server.should_exit = True
    thread.join(timeout=5)


def _connect(port: int, **kwargs):
    return snowflake.connector.connect(
        user="test_user",
        password="test_password",
        account="test_account",
        host="127.0.0.1",
        port=port,
        protocol="http",
        disable_ocsp_checks=True,
        login_timeout=10,
        network_timeout=10,
        **kwargs,
    )


@pytest.fixture()
def connection(emulator_port):
    con = _connect(emulator_port)
    yield con
    con.close()


def test_connect_and_select_literal(connection):
    cur = connection.cursor()
    cur.execute("SELECT 1 AS x")
    assert cur.fetchall() == [(1,)]
    # Note: unlike real Snowflake, the emulator does not uppercase unquoted
    # identifiers (DuckDB preserves case as written).
    assert cur.description[0].name == "x"


def test_ddl_dml_and_typed_roundtrip(connection):
    cur = connection.cursor()
    cur.execute(
        "CREATE OR REPLACE TABLE t (a INT, b VARCHAR, c DOUBLE, d BOOLEAN, e DATE, f TIMESTAMP)"
    )
    cur.execute(
        "INSERT INTO t VALUES (1, 'hello', 3.14, TRUE, '2024-01-01', '2024-01-01 10:20:30.123456')"
    )
    cur.execute("SELECT * FROM t")
    rows = cur.fetchall()
    assert rows == [
        (
            1,
            "hello",
            3.14,
            True,
            datetime.date(2024, 1, 1),
            datetime.datetime(2024, 1, 1, 10, 20, 30, 123456),
        )
    ]


def test_snowflake_qualify_and_ilike(connection):
    cur = connection.cursor()
    cur.execute("CREATE OR REPLACE TABLE u (name VARCHAR, grp INT, val INT)")
    cur.execute("INSERT INTO u VALUES ('Foo', 1, 10), ('Bar', 1, 20)")
    cur.execute(
        "SELECT name, grp, val FROM u WHERE name ILIKE '%oo%' "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY grp ORDER BY val DESC) = 1"
    )
    assert cur.fetchall() == [("Foo", 1, 10)]


def test_variant_object_and_array_types(connection):
    """VARIANT/OBJECT/ARRAY columns: the connector performs no client-side parsing
    for these types, so `fetchall()` should return the raw JSON text verbatim."""
    cur = connection.cursor()
    cur.execute(
        "SELECT OBJECT_CONSTRUCT('a', 1, 'b', 'x') AS v, "
        "ARRAY_CONSTRUCT(1, 2, 3) AS arr, "
        "PARSE_JSON('{\"a\": 1}') AS j"
    )
    row = cur.fetchone()
    assert json.loads(row[0]) == {"a": 1, "b": "x"}
    assert json.loads(row[1]) == [1, 2, 3]
    assert json.loads(row[2]) == {"a": 1}

    assert cur.description[0].type_code == FIELD_ID_TO_NAME_LOOKUP["OBJECT"]
    assert cur.description[1].type_code == FIELD_ID_TO_NAME_LOOKUP["ARRAY"]
    assert cur.description[2].type_code == FIELD_ID_TO_NAME_LOOKUP["VARIANT"]


def test_variant_and_array_column_round_trip(connection):
    """A table with native VARIANT/ARRAY columns should also round-trip correctly.

    Note: bare `ARRAY` (Snowflake's typeless array) has no DuckDB DDL equivalent, so
    we declare the column as `VARCHAR[]` (an explicitly-typed DuckDB array) here.
    """
    cur = connection.cursor()
    cur.execute("CREATE OR REPLACE TABLE semi_types (id INT, payload VARIANT, tags VARCHAR[])")
    cur.execute(
        "INSERT INTO semi_types SELECT 1, OBJECT_CONSTRUCT('k', 'v'), ARRAY_CONSTRUCT('x', 'y')"
    )
    cur.execute("SELECT id, payload, tags FROM semi_types")
    row = cur.fetchone()
    assert row[0] == 1
    assert json.loads(row[1]) == {"k": "v"}
    assert json.loads(row[2]) == ["x", "y"]


def test_qmark_bind_params(emulator_port):
    con = _connect(emulator_port, paramstyle="qmark")
    try:
        cur = con.cursor()
        cur.execute("CREATE OR REPLACE TABLE bound (a INT, b VARCHAR)")
        cur.execute("INSERT INTO bound VALUES (?, ?)", (42, "world"))
        cur.execute("SELECT * FROM bound WHERE a = ?", (42,))
        assert cur.fetchall() == [(42, "world")]
    finally:
        con.close()


def test_dict_cursor(connection):
    connection.cursor().execute("CREATE OR REPLACE TABLE dictt (a INT, b VARCHAR)")
    connection.cursor().execute("INSERT INTO dictt VALUES (1, 'x')")
    dict_cursor = connection.cursor(snowflake.connector.DictCursor)
    dict_cursor.execute("SELECT * FROM dictt")
    assert dict_cursor.fetchall() == [{"a": 1, "b": "x"}]


def test_use_database_updates_session(connection):
    cur = connection.cursor()
    cur.execute("USE DATABASE ALT_DB")
    cur.execute("CREATE TABLE t3 (a INT)")
    cur.execute("INSERT INTO t3 VALUES (5)")
    cur.execute("SELECT * FROM t3")
    assert cur.fetchall() == [(5,)]


def test_invalid_sql_raises_programming_error(connection):
    cur = connection.cursor()
    with pytest.raises(snowflake.connector.errors.ProgrammingError):
        cur.execute("SELEKT * FROM nowhere")
