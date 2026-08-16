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
import snowflake_emulator.stages as stages_module
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
    stages_module._manager = None
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


# -- Staging (PUT / GET / LIST / REMOVE / COPY INTO @stage) ------------------


def _write_csv(path, header=("a", "b"), rows=((1, "x"), (2, "y"))):
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")


def _write_json(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_copy_into_file_format_options(connection, tmp_path):
    """FILE_FORMAT options (SKIP_HEADER, FIELD_DELIMITER) are honored by COPY INTO."""
    cur = connection.cursor()

    headerless = tmp_path / "no_header.csv"
    headerless.write_text("1,x\n2,y\n")
    cur.execute(f"PUT file://{headerless} @~/nh")
    cur.execute("CREATE OR REPLACE TABLE th (a INT, b VARCHAR)")
    cur.execute("COPY INTO th FROM @~/nh FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1)")
    assert cur.fetchall() == [(1,)]
    cur.execute("SELECT * FROM th ORDER BY a")
    assert cur.fetchall() == [(2, "y")]

    tsv = tmp_path / "data.tsv"
    tsv.write_text("a\tb\n5\tq\n")
    cur.execute(f"PUT file://{tsv} @~/t")
    cur.execute("CREATE OR REPLACE TABLE tt (a INT, b VARCHAR)")
    cur.execute(
        "COPY INTO tt FROM @~/t FILE_FORMAT = (TYPE = CSV, FIELD_DELIMITER = '\\t')"
    )
    assert cur.fetchall() == [(1,)]
    cur.execute("SELECT * FROM tt")
    assert cur.fetchall() == [(5, "q")]


def test_put_csv_then_copy_into(connection, tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    cur = connection.cursor()
    cur.execute(f"PUT file://{csv_path} @~/staged")
    row = cur.fetchone()
    assert row[0] == "data.csv"  # source
    assert row[1] == "data.csv"  # target
    assert row[6] == "UPLOADED"

    cur.execute("CREATE OR REPLACE TABLE staged_csv (a INT, b VARCHAR)")
    cur.execute("COPY INTO staged_csv FROM @~/staged FILE_FORMAT = (TYPE = CSV)")
    assert cur.fetchall() == [(2,)]

    cur.execute("SELECT * FROM staged_csv ORDER BY a")
    assert cur.fetchall() == [(1, "x"), (2, "y")]


def test_put_json_then_copy_into(connection, tmp_path):
    json_path = tmp_path / "data.json"
    _write_json(json_path, [{"a": 3, "b": "z"}, {"a": 4, "b": "w"}])

    cur = connection.cursor()
    cur.execute(f"PUT file://{json_path} @~/staged")
    assert cur.fetchone()[6] == "UPLOADED"

    cur.execute("CREATE OR REPLACE TABLE staged_json (a INT, b VARCHAR)")
    cur.execute("COPY INTO staged_json FROM @~/staged FILE_FORMAT = (TYPE = JSON)")
    assert cur.fetchall() == [(2,)]

    cur.execute("SELECT * FROM staged_json ORDER BY a")
    assert cur.fetchall() == [(3, "z"), (4, "w")]


def test_put_list_remove(connection, tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    cur = connection.cursor()
    cur.execute(f"PUT file://{csv_path} @~/staged")
    cur.execute("LIST @~/staged")
    rows = cur.fetchall()
    assert [(r[0], r[1]) for r in rows] == [("staged/data.csv", 12)]
    assert cur.description[0].name == "name"

    cur.execute("REMOVE @~/staged")
    assert cur.fetchall() == [("staged/data.csv", "removed", "")]

    cur.execute("LIST @~/staged")
    assert cur.fetchall() == []


def test_put_glob_and_list_pattern(connection, tmp_path):
    _write_csv(tmp_path / "part1.csv")
    _write_csv(tmp_path / "part2.csv")
    _write_json(tmp_path / "meta.json", [{"a": 9, "b": "q"}])

    cur = connection.cursor()
    cur.execute(f"PUT file://{tmp_path}/*.csv @~/staged")
    assert cur.rowcount == 2

    cur.execute("LIST @~/staged")
    assert {r[0] for r in cur.fetchall()} == {
        "staged/part1.csv",
        "staged/part2.csv",
    }

    cur.execute("LIST @~/staged PATTERN='.*part1.*'")
    assert [r[0] for r in cur.fetchall()] == ["staged/part1.csv"]


def test_put_overwrite_default_and_force(connection, tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    cur = connection.cursor()
    cur.execute(f"PUT file://{csv_path} @~/staged")
    # Without OVERWRITE, re-putting an existing file fails (like real Snowflake).
    with pytest.raises(snowflake.connector.errors.ProgrammingError):
        cur.execute(f"PUT file://{csv_path} @~/staged")
    # With OVERWRITE=TRUE it succeeds.
    cur.execute(f"PUT file://{csv_path} @~/staged OVERWRITE=TRUE")
    assert cur.fetchone()[6] == "UPLOADED"


def test_get_from_stage(connection, tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cur = connection.cursor()
    cur.execute(f"PUT file://{csv_path} @~/staged")
    cur.execute(f"GET @~/staged file://{out_dir}")
    # The connector reports the downloaded file name (stage path stripped).
    assert cur.fetchall() == [("data.csv", 12, "DOWNLOADED", "")]

    downloaded = out_dir / "data.csv"
    assert downloaded.read_text() == csv_path.read_text()


def test_named_stage_roundtrip(connection, tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    cur = connection.cursor()
    cur.execute("CREATE STAGE my_stage")
    cur.execute(f"PUT file://{csv_path} @my_stage")
    assert cur.fetchone()[6] == "UPLOADED"

    cur.execute("SHOW STAGES")
    stages = cur.fetchall()
    assert any(row[0] == "my_stage" for row in stages)

    cur.execute("CREATE OR REPLACE TABLE named_t (a INT, b VARCHAR)")
    cur.execute("COPY INTO named_t FROM @my_stage FILE_FORMAT = (TYPE = CSV)")
    assert cur.fetchall() == [(2,)]
    cur.execute("SELECT * FROM named_t ORDER BY a")
    assert cur.fetchall() == [(1, "x"), (2, "y")]

    cur.execute("DROP STAGE my_stage")
    cur.execute("SHOW STAGES")
    assert not any(row[0] == "my_stage" for row in cur.fetchall())
