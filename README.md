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
uv run uvicorn snowflake_emulator.main:app
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
# Anchor path for durable storage across restarts. When set to a file path, a
# directory "<database_path>.d" is created next to it holding one DuckDB file per
# Snowflake database, e.g. emulator.duckdb.d/MY_DB.duckdb. Use ":memory:" (the
# default) for an ephemeral, in-process database.
database_path = ":memory:"

[statements]
# Maximum number of cached statement results kept in memory.
max_cached_statements = 500

[sessions]
# Maximum number of live bearer-token sessions. Above this the least recently used
# session is evicted, releasing its connection and any temporary tables it held.
max_sessions = 1000

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

DuckDB maps one catalog to one file, so a single file cannot hold several
Snowflake databases. When `persistence.database_path` is a file path:

- the emulator creates a dedicated directory `<database_path>.d` next to it and
  stores each Snowflake database there as `<DATABASE>.duckdb` — for
  `/var/lib/sfemu/emulator.duckdb`, `MY_DB` lives in
  `/var/lib/sfemu/emulator.duckdb.d/MY_DB.duckdb`;
- the path named by `database_path` itself is only the anchor for that directory
  name and is not created;
- only that directory is scanned, so unrelated files (including other DuckDB
  files in the parent directory) are never attached;
- existing database files are re-attached on startup, so tables and
  `SHOW DATABASES` survive a restart; a database file that cannot be read is
  skipped with a warning rather than preventing the emulator from starting.

`CREATE OR REPLACE DATABASE` deletes the database's backing file so the replaced
database starts empty. Database names are matched case-insensitively (a DuckDB
catalog constraint), so `"my_db"` and `"MY_DB"` are the same database. With
`":memory:"` nothing is written to disk, and with
`persistence.isolation = "isolated"` each session gets a private in-memory
database regardless of `database_path` (the emulator warns when both are set).

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

Multiple semicolon-separated statements are supported, and only the final statement's
result set is returned (matching Snowflake's own behavior). Like Snowflake, the request has
to opt in: `MULTI_STATEMENT_COUNT` defaults to `1`, so a chained body is rejected unless the
request sends `0` (any number) or the exact count.

```bash
curl -X POST http://localhost:8000/api/v2/statements \
  -H 'Content-Type: application/json' \
  -d '{"statement": "CREATE TABLE t (a INT, b VARCHAR); INSERT INTO t VALUES (1, '\''x'\''); SELECT * FROM t",
       "parameters": {"MULTI_STATEMENT_COUNT": "0"}}'
```

The `parameters` object also honours `TIMEZONE` (an IANA zone name such as
`America/Los_Angeles`, applied to the request; a numeric offset is rejected because DuckDB's
session timezone only knows zone names) and the rowset's text formats:
`DATE_OUTPUT_FORMAT` (`YYYY-MM-DD`), `TIME_OUTPUT_FORMAT` (`HH24:MI:SS`),
`TIMESTAMP_OUTPUT_FORMAT` (`YYYY-MM-DD HH24:MI:SS.FF3`) and `BINARY_OUTPUT_FORMAT`
(`HEX` or `BASE64`) — the defaults in parentheses are what an absent parameter produces.
The pattern tokens `YYYY`, `YY`, `MM`, `DD`, `HH24`, `HH12`, `MI`, `SS`, `AM`/`PM` and
`FF0`-`FF9` are translated; any other token, an unknown timezone, or a negative
`MULTI_STATEMENT_COUNT` is a `422` rather than a silent default.

```bash
curl -X POST http://localhost:8000/api/v2/statements \
  -H 'Content-Type: application/json' \
  -d '{"statement": "SELECT CURRENT_DATE() AS d",
       "parameters": {"DATE_OUTPUT_FORMAT": "DD.MM.YYYY"}}'
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

Snowflake's array functions work over a native array, one the emulator stores as JSON
(`col:names`, `$1:names`, `PARSE_JSON('[...]')`), and a VARIANT-typed array (`v::VARIANT`,
`TO_VARIANT(ARRAY_CONSTRUCT(...))`, `col:path::VARIANT`, or a bare VARIANT/ARRAY column):
`ARRAY_CONTAINS`, `ARRAY_SIZE`, `ARRAY_SLICE`/`SLICE`, `ARRAY_APPEND`, `ARRAY_CAT`,
`ARRAY_COMPACT`, `ARRAY_DISTINCT`, `ARRAY_EXCEPT`, `ARRAY_INSERT`, `ARRAY_INTERSECTION`,
`ARRAY_POSITION`, `ARRAY_PREPEND`, `ARRAY_REMOVE`, `ARRAY_REMOVE_AT`, `ARRAY_REPEAT`,
`ARRAY_TO_STRING`, `ARRAY_UNIQUE`, `ARRAYS_OVERLAP`, `ARRAY_AGG`,
`ARRAY_FLATTEN(arr)` and the `FILTER`/`TRANSFORM`/`REDUCE` lambdas — a JSON array
is cast to DuckDB's `JSON[]`, a VARIANT array to `VARIANT[]`, and a searched value is
encoded as a VARIANT so both sides compare the way Snowflake compares them (the string
`'1'` is not the number `1`).
A pair that mixes the two (`ARRAY_EXCEPT(j:n, ARRAY_CONSTRUCT('a'))`,
`ARRAY_CAT(nums, ARRAY_CONSTRUCT(4))`) is coerced element-wise onto one kind, as Snowflake
coerces the two arrays, so a native `ARRAY_CONSTRUCT` list combines with a JSON or VARIANT
one.
`WHERE ARRAY_CONTAINS('admin', payload:tags)` therefore filters VARIANT rows and staged
records alike, and `LATERAL FLATTEN(input => ...)` / `TABLE(FLATTEN(...))` read a VARIANT
column, a `col:path` or JSON.

Not translated, and left as a loud error rather than a guess:

- **ordering a VARIANT array** (`ARRAY_SORT`, `ARRAY_MAX`, `ARRAY_MIN` over `v`,
  `v::VARIANT`, `PARSE_JSON('[...]')::VARIANT`, or `ARRAY_AGG(v)` of a VARIANT column):
  DuckDB's `list_sort`/`list_max`/`list_min` do not order `VARIANT[]` the way their own `<`
  operator does — for strings it goes by length, so `['aa','ab','b']` comes back
  `['b','aa','ab']` where Snowflake sorts it lexicographically. Write the array as an
  `ARRAY_CONSTRUCT` list, or read it through a JSON path (`j:path`) or a `JSON` column, which
  order by a type-aware key: numbers numerically (`[10,2,33]` → `[2,10,33]`,
  `ARRAY_MAX([9,100])` → `100`), strings lexicographically, compound values by their text,
  and NULL elements last for an ascending sort (first with `nulls_first`, or with
  `ARRAY_SORT(arr, FALSE)`; both direction arguments must be literal TRUE/FALSE, and a NULL
  argument makes the call NULL, as Snowflake documents). A bare
  column is decided by its catalog type: a real list column (`CREATE TABLE t AS SELECT
  ARRAY_CONSTRUCT(...) AS n` stores `VARCHAR[]`/`INTEGER[]`) orders correctly too. Only a
  VARIANT column — or one whose name the catalog cannot attribute to a single type — is
  rejected;
- a **bound array operand**: a positional `?` that holds the array (or the searched value) is
  named in more than one sub-expression by the rebuilt call, so it would need several values;
  bind it as a named parameter (`:arr`) or pass a column. An index bind is fine
  (`ARRAY_SLICE(arr, ?, ?)` names each index once);
- `ARRAY_FLATTEN(arr, depth)` — the emulator's legacy depth form, where Snowflake's
  documented signature is `ARRAY_FLATTEN(arr)` (one level). The depth must be a constant
  between 1 and 10; a computed depth is rejected.

A `FILTER`/`TRANSFORM`/`REDUCE` lambda over a JSON or VARIANT array receives a VARIANT
element, as in Snowflake, so `FILTER(j:n, x -> x = 'a')` filters without an encoding helper
and `TRANSFORM(j:nums, x -> x * 2)` does arithmetic on the element. `TO_JSON` returns VARCHAR
(JSON text) in Snowflake, so `x -> x = TO_JSON('a')` compares against `'"a"'` and matches
nothing — use the plain literal.

A comparison spans element types instead of raising: `ARRAY_INTERSECTION`/`ARRAY_EXCEPT`/
`ARRAYS_OVERLAP` over a VARIANT array compare in JSON space (where a number and the string that
spells it are simply unequal, rather than DuckDB's `Can't compare values of type VARCHAR and type
UBIGINT`), and a VARIANT *value* searched in a JSON array (`ARRAY_CONTAINS`, `ARRAY_POSITION`,
`ARRAY_REMOVE`) compares in VARIANT space, where both sides normalize the same way.
`ARRAY_REMOVE` removes the value rather than every NULL element (a NULL element survives; a NULL
value returns NULL, as documented).

`HASH_AGG(...)` is a `SUM(HASH(...))` folded into a signed 64-bit value: order-independent,
deterministic, influenced by NULLs and duplicates, `0` for an empty input, and never NULL. It is
a `GROUP BY` aggregate here — the window form and `HASH_AGG(*)` are rejected rather than
guessed. `RATIO_TO_REPORT(x) OVER (PARTITION BY ...)` is `x / SUM(x) OVER (...)`: an `ORDER BY`
inside `OVER` is ignored as Snowflake documents, and a partition that sums to zero yields NULL.

`ARRAY_TO_STRING(array, separator)` is the documented two-argument form: a NULL element becomes
an empty string, a NULL separator returns NULL, and a third argument is rejected (sqlglot would
drop it silently) rather than answered differently. A `?` separator is supported too; because
the placeholder may appear only once, a NULL value bound to it joins with an empty separator
instead of returning NULL.

Verified against Snowflake's documentation, and covered by
`test_documented_array_semantics`: `ARRAY_INTERSECTION` and `ARRAY_EXCEPT` use multi-set
("bag") semantics rather than set semantics — a value keeps `min(N, M)` copies after an
intersection and `N - M` copies after an except, so `['a','b','b','b','c']` except
`['b']` is `['a','b','b','c']`; `ARRAY_POSITION` is 0-based and returns NULL, not 0, when
the value is absent (and the position of the first NULL for a NULL value); `ARRAY_AGG`
omits NULL values.

### User-defined functions (`CREATE FUNCTION`)

SQL-bodied scalar UDFs work: `LANGUAGE SQL` (the default) with either a bare expression or
a single-column `SELECT` as the body. A body is parsed and transpiled as Snowflake SQL
like any other statement, so Snowflake-only calls inside it behave as they would on
Snowflake (`EQUAL_NULL(a, b)` becomes `a IS NOT DISTINCT FROM b`). `CREATE OR REPLACE`,
`IF NOT EXISTS`, `DROP FUNCTION` (with or without a signature), `DESCRIBE FUNCTION` and
`SHOW [USER|ALL|EXTERNAL] FUNCTIONS` (including `LIKE` and `IN`) are all supported, and
the declared signatures are recorded per catalog so they survive a restart.

Each function is emulated as a DuckDB macro under a name mangled with its arity, and
calls are rewritten to match. Two functions that differ only by **argument type** at the
same arity are dispatched at run time from the argument's SQL type, so this works:

```sql
CREATE FUNCTION equal_null_case_insensitive(a STRING, b STRING)
  RETURNS BOOLEAN LANGUAGE SQL AS $$ EQUAL_NULL(UPPER(a), UPPER(b)) $$;
CREATE FUNCTION equal_null_case_insensitive(a ARRAY, b ARRAY)
  RETURNS BOOLEAN LANGUAGE SQL AS $$
    EQUAL_NULL(TRANSFORM(a, x -> UPPER(x)), TRANSFORM(b, x -> UPPER(x))) $$;
```

An overloaded call must be resolvable from the argument types alone. All overloads of one
name/arity have to share a return type (the emulator reaches them through a single DuckDB
function, which needs one declared return type), and a call whose arguments match more
than one overload is an error rather than a guess. An untyped `NULL` is the one exception:
it is accepted when the overloads it could mean agree on the answer, which is what keeps a
NULL-safe function usable on NULL columns.

The trade-offs: a type-overloaded function is dispatched through a Python callback with a
nested query per row rather than an inlined macro, and stays dispatch-backed once
overloaded for the life of the process. A function with a *single* ARRAY parameter
declared inherits the emulator's existing limitation that a bare `ARRAY` column is not
recognised as a JSON array (the same reason `ARRAY_SIZE(arr_col)` fails); once it is
overloaded the dispatcher normalizes the argument and it works.

Rejected, rather than guessed at: non-SQL `LANGUAGE` bodies (no JavaScript/Python/Java/
Scala runtime exists here), `RETURNS TABLE (...)` and multi-column `SELECT` bodies, and a
missing `RETURNS <type>`.

## Staging files (PUT / GET / LIST / REMOVE / COPY INTO)

The emulator implements Snowflake's local stage workflow for JSON, CSV and Parquet files.
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
| `COPY INTO t FROM @stage[/path/file] [PATTERN='...'] [FILE_FORMAT=(TYPE=CSV\|JSON\|PARQUET, ...)]` | Load staged files into an existing table via DuckDB; a path prefix or one exact file both work, and `PATTERN` filters the files read. |
| `CREATE STAGE` / `DROP STAGE` / `ALTER STAGE` / `SHOW STAGES` | Stage metadata; `@name` resolves in the session's current database/schema. |

`COPY INTO` follows Snowflake's CSV defaults (no header row; `SKIP_HEADER` is a line
count) and returns one result row per file with Snowflake's load-report columns
(`file`, `status`, `rows_parsed`, `rows_loaded`, ...). Options the emulator cannot honor
(`ON_ERROR`, `PURGE`, `FORCE`, `MATCH_BY_COLUMN_NAME`, a direct `STRIP_OUTER_ARRAY`, or
an unsupported file-format option) are rejected rather than silently ignored. A JSON load
reads one record per line: a single-line `[{...},{...}]` document is one record whose value
is the array (matching the projected reader, and Snowflake without `STRIP_OUTER_ARRAY`),
and a pretty-printed document is rejected on both paths.

Stage references supported: `@~` (user stage, private per session), `@name`,
`@db.schema.name`, `@%table` (table stage), `@%` (internal stage). A stage reference is a
namespace: a sub-path that escapes its stage (`@stage/../x`, `@..`) is rejected rather
than resolving outside the stage tree.

### Querying staged files directly

A stage file can also be read as a relation, without loading it into a table first. Each
record is a row and its fields are reachable positionally:

```sql
SELECT $1:name, $1:age FROM @~/staged/people.json;          -- a JSON record as `$1`
SELECT $1:name FROM @~/staged/ (FILE_FORMAT => (TYPE = JSON));  -- every file under a prefix
SELECT $1, $2 FROM @~/staged/data.csv
  (FILE_FORMAT => (TYPE = CSV, SKIP_HEADER = 1, FIELD_DELIMITER = '|'));
SELECT $1, $2 FROM @my_stage/data.parquet;                  -- `$n` is the nth column
SELECT $1:name FROM @~/staged/data.json (FILE_FORMAT => 'my_json_format');
```

| Format | `$1` is | Notes |
| --- | --- | --- |
| JSON | one record (`$1:key` extracts a field) | Newline-delimited records; `STRIP_OUTER_ARRAY = TRUE` reads each file as one JSON document and explodes its outer array. |
| CSV | the first field (`$2`, `$3`, ... follow) | Snowflake's CSV defaults apply: `SKIP_HEADER` (not `header`), no quoting unless `FIELD_OPTIONALLY_ENCLOSED_BY`, empty fields and `NULL_IF` (`\N` by default) read as NULL, `COMPRESSION`, `EMPTY_FIELD_AS_NULL`. |
| Parquet | the first column | Columns keep the file's own names; `$n` resolves to the nth of them. |

The file format comes from the query: a `(FILE_FORMAT => ...)` clause wins (inline options
or the name of a created format), otherwise it is inferred from the file extensions, as
`COPY INTO` does here. A path mixing formats cannot be read as one relation, so it is
rejected rather than guessed. `(PATTERN => '<regex>')` filters the files read, a directory
prefix reads everything under it, and only one staged location can be queried per
statement. `gzip`/`zstd`-compressed files are read transparently; `bzip2` is rejected.

Caveats:

- Staged JSON must be newline-delimited: a pretty-printed document is rejected rather
  than read as a set of NULL rows, and a malformed record fails the query instead of
  becoming NULL. A single-line array document is one record; `COPY INTO ... FROM
  (SELECT ...) FILE_FORMAT = (TYPE = JSON, STRIP_OUTER_ARRAY = TRUE)` explodes it instead.
- A location whose files differ in width (`a.csv` with 2 fields, `b.csv` with 3) widens
  into the union, filling the missing fields with NULL.
- `FILE_FORMAT` options the emulator does not translate are rejected (`TRIM_SPACE = TRUE`),
  while a value equal to Snowflake's default (`TRIM_SPACE = FALSE`) is accepted, since that
  default is what DuckDB already does. Silently ignoring an option would report Snowflake
  results the emulator did not produce.
- `ESCAPE_UNENCLOSED_FIELD` (Snowflake's default is `\`) has no DuckDB equivalent and is
  rejected when set explicitly; use `ESCAPE` for enclosed fields.
- a semi-structured value selected *without* a cast is inlined as a JSON value in the SQL
  API (`$1:s` -> `y`, a number as a number, an object as an object), while the connector's
  wire protocol keeps returning VARIANT as its JSON text (`"y"`, `42`) — which is what that
  protocol and the Arrow path decode;
- a **bare VARIANT column** cast to text (`SELECT variant_col::VARCHAR`) uses DuckDB's own
  VARIANT text form: right for a string, but an object shows as the variant's internal
  form, so use `variant_col::JSON::VARCHAR` when you need the document text;
- a JSON value nested inside a **STRUCT/MAP** cell keeps its text
  (`SELECT STRUCT_PACK(a := PARSE_JSON('1'))` -> `{'A': '1'}`, `MAP {'k': PARSE_JSON('1')}`,
  or a DuckDB-flavoured `STRUCT(a JSON)` column): only a top-level JSON cell and the elements
  of a `JSON[]` cell are decoded. Snowflake DDL cannot declare such a column (`OBJECT`,
  `VARIANT` and `ARRAY` are its semi-structured types), so this is reachable only through
  DuckDB-specific SQL and is left as it is;
- A `:` path on a non-JSON column (`$1:key` over CSV/Parquet) yields NULL rather than
  Snowflake's compile error.
- The stage's own `FILE_FORMAT` (`CREATE STAGE ... FILE_FORMAT = ...`) is not modelled,
  so an explicit `(FILE_FORMAT => ...)` clause is required for anything but the inferred
  per-extension format.

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
  core/                 # shared types/errors/constants/AST helpers (no engine imports)
    results.py          # ColumnMeta, FileTransfer, ExecutionResult
    errors.py           # ExecutionError + DuckDB -> Snowflake error classification
    constants.py        # INFORMATION_SCHEMA schema name
    ast.py              # shared sqlglot AST helpers
  engine/
    errors.py           # TranslationError -> ExecutionError wrapping
    handlers/
      common.py         # result-column and temp-table helpers
      show.py           # Snowflake-shaped SHOW handlers
      describe.py       # DESCRIBE handlers and describeOnly
      ddl.py            # database/function/file-format DDL
      stage.py          # PUT/GET/LIST/REMOVE and stage DDL
  translator/           # sqlglot-based Snowflake -> DuckDB translation
    __init__.py         # public API, core transpile and DDL rewrites
    arrays.py           # array-function rewrites over JSON/VARIANT arrays
    functions.py        # SQL UDF translation and macro rendering
    parsing.py          # statement splitting and ARRAY_FLATTEN preprocessing
    semi_structured.py  # JSON/VARIANT, FLATTEN and Snowflake-specific rewrites
    transforms.py       # rewrite pipeline shared by statements and UDF bodies
    constants.py        # dialect names and parser markers
    errors.py           # TranslationError
  executor.py            # statement execution against DuckDB
  database.py             # DuckDB connection/catalog management
  sessions.py             # in-memory session (token -> db/schema/warehouse/role)
  stages/                # local stage tree: PUT/GET/LIST/REMOVE/COPY INTO @stage
    __init__.py          # staged-file query path and the stage-manager singleton
    refs.py              # stage references, command parsing and StageError
    manager.py           # on-disk stage tree and stage DDL
    copy.py              # COPY INTO source resolution and DuckDB load SQL
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

## Implementation coverage

Legend: ✅ fully supported, ⚠️ partial / approximate, ❌ not supported (usually
rejected with a Snowflake-shaped error rather than failing silently).

### Protocol and sessions

| Feature | Status |
| --- | --- |
| Snowflake SQL API v2 (`/api/v2/statements`, statement handles, cancel) | ✅ |
| Async submission (`?async=true`) and `?partition=` polling | ✅ (one materialized partition) |
| Connector wire protocol (`/session/v1/login-request`, `/queries/v1/query-request`) | ✅ |
| `statementTypeId`, `describeOnly`, `finalDatabaseName/Schema/Warehouse/Role` | ✅ |
| `GET /monitoring/queries/{sfqid}` | ✅ |
| Typed / named / array bind parameters (`executemany`) | ✅ |
| Arrow result format (`fetch_arrow_all`, `fetch_pandas_all`) | ✅ |
| Error `errno`/`sqlState` classification | ✅ |
| `USE DATABASE/SCHEMA/WAREHOUSE/ROLE`, `ALTER SESSION` no-op | ✅ |
| `CURRENT_USER/ROLE/WAREHOUSE/ACCOUNT/REGION/TIMEZONE` | ✅ |
| `BEGIN`/`COMMIT`/`ROLLBACK` on the session connection | ✅ |
| Session variables (`SET`, `SHOW VARIABLES`, `$name`) | ✅ |
| `RESULT_SCAN(...)`, `LAST_QUERY_ID()` | ✅ (simple `SELECT *` form) |
| Per-session isolation mode; configurable no-op regexes | ✅ |

### SQL and objects

| Feature | Status |
| --- | --- |
| Standard SQL, multi-statement requests, `QUALIFY`, `ILIKE`, `DATEADD`/`DATEDIFF` | ✅ |
| `MERGE` (including qualified `UPDATE SET t.col`), `INSERT OVERWRITE` | ✅ |
| `CREATE TABLE ... CLONE` | ✅ (rows copied, not zero-copy) |
| `AUTOINCREMENT`/`IDENTITY`, `CREATE SEQUENCE`, `seq.NEXTVAL` | ✅ |
| `VARIANT`/`OBJECT`/`ARRAY`, `PARSE_JSON`, `OBJECT_CONSTRUCT`, `ARRAY_CONSTRUCT` | ✅ |
| `LATERAL FLATTEN(...)`, `TABLE(FLATTEN(...))`, including `outer`/`path` | ✅ |
| `FLATTEN(... recursive => TRUE)` | ❌ rejected |
| `DESCRIBE TABLE`; `{db}.INFORMATION_SCHEMA` tables/columns/views/sequences/constraints/functions/databases | ✅ |
| `CREATE [OR REPLACE] DATABASE`, `DROP [IF EXISTS] DATABASE` | ✅ |
| `SHOW` schemas/tables/views/objects/columns/keys/stages/databases/sequences/… | ✅ |
| `CREATE FUNCTION` (SQL-bodied scalar UDFs), `DESCRIBE`/`DROP FUNCTION` | ✅ |
| UDF overloading by arity (always) and by argument type (dispatched at run time) | ✅ |
| `SHOW FUNCTIONS` of user-defined functions | ✅ |
| Semi-structured function breadth beyond the list above | ⚠️ |
| `VARCHAR(n)` lengths in `DESCRIBE`/`SHOW COLUMNS` | ⚠️ reported as `VARCHAR` |
| `TIMESTAMP_LTZ` precision | ⚠️ mapped to `TIMESTAMP_TZ` at microsecond precision |
| `SHOW FUNCTIONS` built-in catalogue | ⚠️ user-defined functions only |

`PRIMARY KEY`, `UNIQUE`, `FOREIGN KEY` and `CHECK` declarations are recorded and
shown by `SHOW ... KEYS`, but not enforced, matching Snowflake; `NOT NULL` is
enforced. This covers both `CREATE TABLE` and `ALTER TABLE ... ADD CONSTRAINT`.
`COMMIT`/`ROLLBACK` outside a transaction are accepted as no-ops.
| Type-overloaded UDFs with differing return types | ⚠️ rejected (one declared return type per name/arity) |

### Staging and files

| Feature | Status |
| --- | --- |
| `PUT`/`GET`/`LIST`/`REMOVE` for `@~`, `@name`, `@db.schema.name`, `@%table` | ✅ |
| `COPY INTO` from a stage with CSV/TSV/JSON and inline `FILE_FORMAT` | ✅ |
| Named file formats (`CREATE FILE FORMAT`, `FORMAT_NAME=`) | ✅ |
| HTTP (non-shared-filesystem) stage transfer | ❌ local filesystem only |
| External stages (`s3://`, `gcs://`, `azure://`) | ❌ |

### Not supported

| Feature | Behaviour |
| --- | --- |
| JavaScript / Python / Java / Scala UDFs | ❌ rejected with a clear error (no runtime) |
| `RETURNS TABLE (...)` UDFs, and multi-column `SELECT` bodies | ❌ rejected with a clear error |
| `CREATE TASK`, `CREATE STREAM` | ❌ syntax error |
| `CREATE MATERIALIZED VIEW`, `CREATE INDEX`, `CREATE RECURSIVE VIEW` | ❌ rejected with a clear error |
| `TRANSIENT` tables, `CLUSTER BY`, `SECURE` | ⚠️ accepted without operational effect |
| Time travel (`AT`/`BEFORE`, `RESULT_SCAN` of a past state) | ❌ rejected |
| `CLONE` of a schema or database | ❌ rejected with a clear error |
| Stored procedures | ❌ |
| Access control (roles are cosmetic; any credential is accepted) | ❌ by design |

## Test layout

```
tests/
  test_statements_api.py        # SQL API v2 surface (FastAPI TestClient)
  test_connector_integration.py # real snowflake-connector-python driver
  test_sqlalchemy.py            # real snowflake-sqlalchemy dialect
  test_jdbc.py                  # real snowflake-jdbc driver (gated on SNOWFLAKE_JDBC_JAR)
  test_features.py              # CLONE, RESULT_SCAN, SET/SHOW VARIABLES, file formats
  test_metadata.py              # SHOW COLUMNS / KEYS / DATABASES / SEQUENCES / …
  test_bindings.py              # typed / named / array bind parameters
  test_errors.py                # errno / sqlState classification
  test_wire_responses.py        # statementTypeId, describeOnly, monitoring
  test_server_ops.py            # isolation mode, no-op regexes, CLI flags
  test_snowflake_compat.py      # USE IDENTIFIER, INFORMATION_SCHEMA, MERGE, FLATTEN, …
  test_autoincrement.py         # AUTOINCREMENT / IDENTITY
  test_type_mapping.py          # DuckDB -> Snowflake type mapping
  test_stages.py                # stage command parsing
  test_settings.py              # TOML/env configuration
```
