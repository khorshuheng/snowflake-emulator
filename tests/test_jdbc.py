"""Real-client JDBC smoke test, gated on a local ``snowflake-jdbc`` jar.

Set ``SNOWFLAKE_JDBC_JAR`` to the path of a ``snowflake-jdbc-*.jar`` to run it;
otherwise the test is skipped. It uses Java's single-file source execution, so no
build step is needed. The driver also probes cloud metadata endpoints
(169.254.169.254), which fails harmlessly on a local machine - only stdout is
asserted on.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import textwrap
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

_JDBC_JAR = os.environ.get("SNOWFLAKE_JDBC_JAR")
_JAVA = shutil.which("java")

_JAVA_SOURCE = textwrap.dedent(
    """
    import java.sql.*;

    public class JdbcSmoke {
        public static void main(String[] args) throws Exception {
            try (Connection c = DriverManager.getConnection(args[0], "test_user", "test_password");
                 Statement s = c.createStatement()) {
                try (ResultSet rs = s.executeQuery("SELECT 1 AS x")) {
                    rs.next();
                    System.out.println("select=" + rs.getInt(1));
                }
                s.execute("CREATE OR REPLACE TABLE jdbc_t (a INT, b VARCHAR)");
                s.execute("INSERT INTO jdbc_t VALUES (1, 'x'), (2, 'y')");
                try (ResultSet rs = s.executeQuery("SELECT COUNT(*) FROM jdbc_t")) {
                    rs.next();
                    System.out.println("count=" + rs.getInt(1));
                }
            }
        }
    }
    """
)


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


@pytest.mark.skipif(
    not (_JDBC_JAR and _JAVA),
    reason="set SNOWFLAKE_JDBC_JAR (and install java) to run the JDBC integration test",
)
def test_jdbc_select_and_dml(emulator_port, tmp_path):
    source = tmp_path / "JdbcSmoke.java"
    source.write_text(_JAVA_SOURCE)
    url = (
        f"jdbc:snowflake://127.0.0.1:{emulator_port}/?ssl=off"
        "&account=test_account&db=EMULATOR_DB&schema=PUBLIC"
    )
    result = subprocess.run(
        [
            _JAVA,
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "-cp",
            _JDBC_JAR,
            str(source),
            url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "select=1" in result.stdout
    assert "count=2" in result.stdout
