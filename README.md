# Snowflake Emulator

A local emulation of the [Snowflake SQL API v2](https://docs.snowflake.com/en/developer-guide/sql-api/index),
built with **FastAPI**, **sqlglot**, and **DuckDB**.

Point tools/clients that speak the Snowflake SQL API at this service during local
development or automated testing — no real Snowflake account required.

## How it works

1. A client `POST`s Snowflake SQL to `/api/v2/statements`, same as the real SQL API.
2. [`sqlglot`](https://github.com/tobymao/sqlglot) parses the statement(s) with the
   `snowflake` read dialect and transpiles them to the `duckdb` write dialect.
3. The translated SQL is executed against an embedded [DuckDB](https://duckdb.org/)
   database.
4. Results are returned in a Snowflake SQL API-shaped JSON payload
   (`resultSetMetaData`, `data`, `statementHandle`, ...).

Snowflake `database.schema` namespacing is mapped onto DuckDB's catalog/schema model:
each Snowflake "database" becomes an attached DuckDB catalog, and each Snowflake
"schema" becomes a schema within that catalog.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)

## Getting started

```bash
uv sync
uv run snowflake-emulator          # starts the dev server on http://localhost:8000
# or, equivalently:
uv run uvicorn snowflake_emulator.main:app --reload
```

## Configuration

On first run, a config file is created (if missing) at `~/.sfemu/config.toml` with
defaults:

```toml
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

[staging]
# Root directory under which staged files (PUT/GET/COPY INTO @stage) are stored.
# Leave empty to use an auto-created temporary directory.
stage_root = ""
```

Edit this file to change the bind address/port, default database/schema, or to
persist data to disk (set `persistence.database_path` to a file path, e.g.
`"/var/lib/sfemu/emulator.duckdb"`, instead of `":memory:"`). Set
`staging.stage_root` to a directory to keep staged files there instead of an
auto-created temp directory.

Settings can also be overridden with environment variables prefixed with
`SNOWFLAKE_EMULATOR_`, using `__` to reach nested fields — these take priority
over the TOML file (e.g. `SNOWFLAKE_EMULATOR_SERVER__PORT=9000`,
`SNOWFLAKE_EMULATOR_PERSISTENCE__DATABASE_PATH=/tmp/sfemu.duckdb`).

## Usage

Execute a statement:

```bash
curl -X POST http://localhost:8000/api/v2/statements \
  -H 'Content-Type: application/json' \
  -d '{"statement": "SELECT 1 AS x"}'
```

Multiple semicolon-separated statements are supported; only the final statement's
result set is returned (matching Snowflake's own behavior):

```bash
curl -X POST http://localhost:8000/api/v2/statements \
  -H 'Content-Type: application/json' \
  -d '{"statement": "CREATE TABLE t (a INT, b VARCHAR); INSERT INTO t VALUES (1, '\''x'\''); SELECT * FROM t"}'
```

Obtain a bearer token to keep session state (current database/schema/warehouse/role)
across requests:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/session/token | jq -r .token)

curl -X POST http://localhost:8000/api/v2/statements \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"statement": "USE DATABASE MY_DB"}'
```

Fetch a previously executed statement's result:

```bash
curl http://localhost:8000/api/v2/statements/<statementHandle>
```

Interactive API docs are available at `http://localhost:8000/docs` while the server
is running.

## Supported SQL

Anything `sqlglot` can transpile from the `snowflake` dialect to `duckdb` works,
including `QUALIFY`, `ILIKE`, `DATEADD`/`DATEDIFF`, semi-structured functions like
`OBJECT_CONSTRUCT`/`PARSE_JSON`, and `MERGE`. Session-scoped `USE DATABASE/SCHEMA/
WAREHOUSE/ROLE` statements are intercepted and applied to the emulator's in-memory
session state rather than being sent to DuckDB.

## Staging files (PUT / GET / LIST / REMOVE / COPY INTO)

The emulator implements Snowflake's local stage workflow for JSON and CSV files.
Stages are backed by a directory tree on disk (see `[staging]` config above), and
the official `snowflake-connector-python` driver performs the actual file transfer
through its built-in `local` storage client — no cloud credentials needed.

```python
import snowflake.connector

con = snowflake.connector.connect(
    user="u", password="p", account="a", host="localhost", port=8000,
    protocol="http", disable_ocsp_checks=True,
)
cur = con.cursor()

cur.execute("CREATE STAGE my_stage")
cur.execute("PUT file:///tmp/data.csv @my_stage")            # stage a CSV
cur.execute("PUT file:///tmp/data.json @~/staged")           # user stage @~
cur.execute("LIST @~/staged")
cur.execute("CREATE TABLE t (a INT, b VARCHAR)")
cur.execute("COPY INTO t FROM @~/staged FILE_FORMAT = (TYPE = CSV)")
cur.execute("COPY INTO t FROM @my_stage FILE_FORMAT = (TYPE = JSON)")
cur.execute("GET @my_stage file:///tmp/downloads")
cur.execute("REMOVE @my_stage")
```

Supported stage commands:

| Command | Notes |
| --- | --- |
| `PUT file://... @stage [OVERWRITE=TRUE]` | Stage local JSON/CSV files (globs and multiple files supported). |
| `GET @stage file://... [PATTERN='...']` | Download staged files to a local directory. |
| `LIST @stage [PATTERN='...']` | List staged files (name/size/md5/last_modified). |
| `REMOVE @stage [PATTERN='...']` | Delete staged files. |
| `COPY INTO t FROM @stage [FILE_FORMAT=(TYPE=CSV\|JSON, ...)]` | Load staged files into an existing table via DuckDB. |
| `CREATE STAGE` / `DROP STAGE` / `ALTER STAGE` / `SHOW STAGES` | Stage metadata; `@name` resolves in the session's current database/schema. |

Stage references supported: `@~` (user stage, private per session), `@name`,
`@db.schema.name`, `@%table` (table stage), `@%` (internal stage).

## Development

```bash
uv run pytest        # run the whole test suite
uv run pytest tests/test_statements_api.py           # SQL API v2 tests (FastAPI TestClient)
uv run pytest tests/test_connector_integration.py     # real snowflake-connector-python driver
uv run pytest tests/test_settings.py                  # TOML/env configuration tests
uv run pytest tests/test_stages.py                    # stage command parsing unit tests
```

The connector integration tests spin up a real `uvicorn` server on a background
thread and drive it with the official `snowflake-connector-python` package, which
speaks Snowflake's internal wire protocol (`/session/v1/login-request`,
`/queries/v1/query-request`, `/session?delete=true`) rather than the SQL API v2. Both
protocols are implemented by this emulator and share the same translation/execution
pipeline.

Note: unlike real Snowflake, the emulator does not uppercase unquoted identifiers
(DuckDB preserves case as written), so column names in results are lowercase unless
quoted/aliased otherwise.

## Project layout

```
src/snowflake_emulator/
  main.py              # FastAPI app + routers
  routers/
    statements.py      # POST/GET /api/v2/statements
    auth.py            # session/login endpoints (SQL API v2 + wire protocol)
    queries.py           # /queries/v1/query-request etc. (connector wire protocol)
  translator.py         # sqlglot-based Snowflake -> DuckDB translation
  executor.py            # statement execution against DuckDB
  database.py             # DuckDB connection/catalog management
  sessions.py             # in-memory session (token -> db/schema/warehouse/role)
  stages.py               # local stage tree: PUT/GET/LIST/REMOVE/COPY INTO @stage
  statement_store.py      # cache of executed statement results, by handle
  type_mapping.py          # DuckDB type -> Snowflake rowType mapping
  wire_format.py            # DuckDB value -> connector wire-format string encoding
  middleware.py              # gzip request-body decompression (connector always gzips)
  schemas.py                # Pydantic request/response models
  settings.py                # pydantic-settings config: ~/.sfemu/config.toml + env vars
tests/
  test_statements_api.py           # SQL API v2 tests via FastAPI TestClient
  test_connector_integration.py    # real snowflake-connector-python driver tests
  test_settings.py                 # TOML config file + env var override tests
  test_stages.py                   # stage command parsing unit tests
```
