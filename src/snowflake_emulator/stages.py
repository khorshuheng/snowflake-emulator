"""Local file-stage emulation: ``PUT``/``GET``/``LIST``/``REMOVE`` and ``COPY INTO ... FROM @stage``.

Snowflake stages are mapped onto a directory tree under a configurable root::

    <stage_root>/
      user/<token>/                         # @~ user stage, private per session
      named/<db>/<schema>/<name>/           # named stages, resolved in session namespace
      table/<db>/<schema>/<table>/          # @%<table> internal stage
      internal/                             # @% unnamed internal stage

The official ``snowflake-connector-python`` driver implements PUT/GET as a client-side
file transfer: it POSTs the ``PUT ...``/``GET ...`` text to the query endpoint, and the
server answers with a "UPLOAD"/"DOWNLOAD" command describing a storage location. The
connector supports a ``local`` storage location type (``SnowflakeLocalStorageClient``),
which is what this module uses: the emulator hands back a directory on disk and the
connector copies files into/out of it directly.
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any

from sqlglot import exp

from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.settings import settings

# -- Command detection (mirrors snowflake.connector._sql_util.get_file_transfer_type) --

_PUT_RE = re.compile(r"^\s*PUT\b", re.IGNORECASE)
_GET_RE = re.compile(r"^\s*GET\b", re.IGNORECASE)
_LIST_RE = re.compile(r"^\s*LIST\b", re.IGNORECASE)
_REMOVE_RE = re.compile(r"^\s*REMOVE\b", re.IGNORECASE)


class StageError(ValueError):
    """Raised for malformed stage commands or stage I/O failures."""


@dataclass(frozen=True)
class StageRef:
    """A parsed Snowflake stage reference such as ``@~/data`` or ``@db.schema.stage/x``."""

    kind: str  # user | table | named | internal
    name: str = ""
    database: str | None = None
    schema: str | None = None
    path: str = ""  # sub-path within the stage ("/" separated), may be empty


@dataclass
class PutCommand:
    """A parsed ``PUT <files> @<stage> [options]`` command."""

    files: list[str] = field(default_factory=list)  # raw source specs (may contain globs)
    stage: StageRef | None = None
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class GetCommand:
    """A parsed ``GET @<stage> <local_dir> [options]`` command."""

    stage: StageRef | None = None
    local_dir: str = ""
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class ListCommand:
    """A parsed ``LIST @<stage> [pattern]`` command."""

    stage: StageRef | None = None
    pattern: str | None = None


@dataclass
class RemoveCommand:
    """A parsed ``REMOVE @<stage> [pattern]`` command."""

    stage: StageRef | None = None
    pattern: str | None = None


@dataclass
class StagedFile:
    """Metadata about a single file as it lives on (or will live on) a stage."""

    name: str  # basename within the target directory
    rel_path: str  # path relative to the stage base directory
    size: int
    md5: str = ""
    last_modified: str = ""
    status: str = "UPLOADED"
    message: str = ""


class StageManager:
    """Owns the on-disk stage tree."""

    def __init__(self, root: str | None = None) -> None:
        self._lock = threading.Lock()
        if root:
            self.root = os.path.abspath(os.path.expanduser(root))
            os.makedirs(self.root, exist_ok=True)
        else:
            self.root = tempfile.mkdtemp(prefix="sfemu_stages_")

    # -- stage resolution ---------------------------------------------------

    def stage_base_dir(self, session: SessionContext, stage: StageRef) -> str:
        """Absolute directory backing the given stage reference (sub-path excluded)."""
        if stage.kind == "user":
            return os.path.join(self.root, "user", session.token)
        if stage.kind == "internal":
            return os.path.join(self.root, "internal")
        if stage.kind == "table":
            return os.path.join(
                self.root, "table", session.database, session.schema, stage.name
            )
        db = stage.database or session.database
        schema = stage.schema or session.schema
        return os.path.join(self.root, "named", db, schema, stage.name)

    def stage_dir(self, session: SessionContext, stage: StageRef) -> str:
        """Absolute directory for a stage reference including its sub-path."""
        base = self.stage_base_dir(session, stage)
        return os.path.join(base, stage.path) if stage.path else base

    # -- PUT ----------------------------------------------------------------

    def put_files(
        self, session: SessionContext, command: PutCommand
    ) -> tuple[list[StagedFile], str]:
        """Copy the source files into the stage; returns (files, target dir).

        The target directory (including sub-paths) is created here so the connector's
        local storage client can write into it without further setup.
        """
        if command.stage is None:
            raise StageError("Missing stage location in PUT command.")
        overwrite = _parse_bool(command.options.get("OVERWRITE", "FALSE"))
        target_dir = self.stage_dir(session, command.stage)
        os.makedirs(target_dir, exist_ok=True)

        staged: list[StagedFile] = []
        with self._lock:
            for spec in command.files:
                for src in _expand_local_files(spec):
                    if not os.path.isfile(src):
                        raise StageError(f"File doesn't exist: {src}")
                    dst = os.path.join(target_dir, os.path.basename(src))
                    if os.path.exists(dst) and not overwrite:
                        raise StageError(
                            f"File already exists: {os.path.basename(src)}. "
                            "Use OVERWRITE=TRUE to replace it."
                        )
                    shutil.copy2(src, dst)
                    staged.append(
                        StagedFile(
                            name=os.path.basename(dst),
                            rel_path=os.path.relpath(
                                dst, self.stage_base_dir(session, command.stage)
                            ),
                            size=os.path.getsize(dst),
                            status="UPLOADED",
                        )
                    )
        return staged, target_dir

    # -- GET ----------------------------------------------------------------

    def get_files(
        self, session: SessionContext, command: GetCommand
    ) -> tuple[list[StagedFile], str]:
        """Resolve the files to download from the stage; returns (files, stage base dir).

        The connector copies the files itself once it receives the DOWNLOAD response;
        the emulator also copies them so the SQL API v2 path behaves the same way.
        """
        if command.stage is None:
            raise StageError("Missing stage location in GET command.")
        if not command.local_dir:
            raise StageError("GET requires a local target directory.")
        local_dir = os.path.abspath(os.path.expanduser(command.local_dir))
        if not os.path.isdir(local_dir):
            raise StageError(f"The local path is not a directory: {local_dir}")

        base = self.stage_base_dir(session, command.stage)
        pattern = _compile_pattern(command.options.get("PATTERN"))
        matches = self._list_matches(base, command.stage.path, pattern)

        files: list[StagedFile] = []
        for rel, abs_path in matches:
            dst = os.path.join(local_dir, os.path.basename(rel))
            shutil.copy2(abs_path, dst)
            files.append(
                StagedFile(
                    name=os.path.basename(rel),
                    rel_path=rel,
                    size=os.path.getsize(abs_path),
                    status="DOWNLOADED",
                )
            )
        return files, base

    # -- LIST ---------------------------------------------------------------

    def list_files(
        self, session: SessionContext, command: ListCommand
    ) -> list[StagedFile]:
        if command.stage is None:
            raise StageError("Missing stage location in LIST command.")
        base = self.stage_base_dir(session, command.stage)
        pattern = _compile_pattern(command.pattern)
        files: list[StagedFile] = []
        for rel, abs_path in self._list_matches(base, command.stage.path, pattern):
            stat = os.stat(abs_path)
            files.append(
                StagedFile(
                    name=os.path.basename(rel),
                    rel_path=rel,
                    size=stat.st_size,
                    md5=_md5_hex(abs_path),
                    last_modified=datetime.datetime.fromtimestamp(
                        stat.st_mtime, datetime.timezone.utc
                    ).strftime("%a, %d %b %Y %H:%M:%S %z"),
                )
            )
        return sorted(files, key=lambda f: f.rel_path)

    # -- REMOVE -------------------------------------------------------------

    def remove_files(
        self, session: SessionContext, command: RemoveCommand
    ) -> list[StagedFile]:
        if command.stage is None:
            raise StageError("Missing stage location in REMOVE command.")
        base = self.stage_base_dir(session, command.stage)
        pattern = _compile_pattern(command.pattern)
        removed: list[StagedFile] = []
        with self._lock:
            for rel, abs_path in self._list_matches(base, command.stage.path, pattern):
                os.remove(abs_path)
                removed.append(
                    StagedFile(
                        name=os.path.basename(rel),
                        rel_path=rel,
                        size=0,
                        status="removed",
                    )
                )
        return sorted(removed, key=lambda f: f.rel_path)

    # -- helpers ------------------------------------------------------------

    def _list_matches(
        self, base: str, sub_path: str, pattern: re.Pattern[str] | None
    ) -> list[tuple[str, str]]:
        """Recursively list files under ``base/sub_path`` matching the optional regex."""
        if not os.path.isdir(base):
            return []
        search_root = os.path.join(base, sub_path) if sub_path else base
        matches: list[tuple[str, str]] = []
        for dirpath, _dirnames, filenames in os.walk(search_root):
            for filename in sorted(filenames):
                abs_path = os.path.join(dirpath, filename)
                rel = os.path.relpath(abs_path, base).replace(os.sep, "/")
                if pattern is None or pattern.search(rel):
                    matches.append((rel, abs_path))
        return sorted(matches)


# -- Parsing helpers ---------------------------------------------------------


def parse_stage_ref(ref: str) -> StageRef:
    """Parse ``@~/path``, ``@%table/path``, ``@name/path``, ``@db.schema.name/path``."""
    ref = ref.strip()
    if not ref.startswith("@"):
        raise StageError(f"Invalid stage reference: {ref!r} (expected it to start with '@')")
    body = ref[1:]
    if "/" in body:
        stage_part, _, path_part = body.partition("/")
    else:
        stage_part, path_part = body, ""

    if stage_part in ("~", ""):
        return StageRef(kind="user", path=path_part)
    if stage_part == "%":
        return StageRef(kind="internal", path=path_part)
    if stage_part.startswith("%"):
        return StageRef(kind="table", name=stage_part[1:], path=path_part)

    parts = stage_part.split(".")
    if len(parts) == 1:
        return StageRef(kind="named", name=stage_part, path=path_part)
    if len(parts) == 2:
        return StageRef(kind="named", schema=parts[0], name=parts[1], path=path_part)
    if len(parts) == 3:
        return StageRef(
            kind="named", database=parts[0], schema=parts[1], name=parts[2], path=path_part
        )
    raise StageError(f"Invalid stage reference: {ref!r}")


def match_stage_command(sql: str) -> str | None:
    """Classify a statement as a stage command: PUT, GET, LIST, REMOVE (else None)."""
    stripped = sql.lstrip()
    if _PUT_RE.match(stripped):
        return "PUT"
    if _GET_RE.match(stripped):
        return "GET"
    if _LIST_RE.match(stripped):
        return "LIST"
    if _REMOVE_RE.match(stripped):
        return "REMOVE"
    return None


def parse_put(sql: str) -> PutCommand:
    """Parse ``PUT file:///a.csv [file:///b.csv] @<stage> [PARALLEL=4] [OVERWRITE=TRUE]``."""
    rest = sql[3:].strip()
    at_index = rest.find("@")
    if at_index < 0:
        raise StageError("PUT requires a stage location, e.g. `PUT file:///x.csv @~/dir`.")
    src_spec = rest[:at_index].strip()
    stage_and_opts = rest[at_index:].strip()

    files = [f for f in re.split(r"[\s,]+", src_spec) if f]
    if not files:
        raise StageError("PUT requires at least one source file.")

    parts = stage_and_opts.split()
    stage = parse_stage_ref(parts[0])
    options: dict[str, str] = {}
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if not sep:
            options[part.upper()] = "TRUE"
        else:
            options[key.strip().upper()] = value.strip().strip("'\"")
    return PutCommand(files=files, stage=stage, options=options)


def parse_get(sql: str) -> GetCommand:
    """Parse ``GET @<stage> file:///local/dir [PATTERN='...'] [PARALLEL=4]``."""
    rest = sql[3:].strip()
    parts = rest.split()
    if not parts:
        raise StageError("GET requires a stage location.")
    stage = parse_stage_ref(parts[0])
    local_dir = parts[1] if len(parts) > 1 else ""
    if local_dir.startswith("file://"):
        local_dir = local_dir[len("file://") :]
    options: dict[str, str] = {}
    for part in parts[2:]:
        key, sep, value = part.partition("=")
        options[key.strip().upper()] = value.strip().strip("'\"") if sep else "TRUE"
    return GetCommand(stage=stage, local_dir=local_dir, options=options)


def parse_list(sql: str) -> ListCommand:
    """Parse ``LIST @<stage> [PATTERN='...']``."""
    rest = sql[4:].strip()
    parts = rest.split()
    if not parts:
        raise StageError("LIST requires a stage location.")
    stage = parse_stage_ref(parts[0])
    pattern = None
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if key.strip().upper() == "PATTERN" and sep:
            pattern = value.strip().strip("'\"")
    return ListCommand(stage=stage, pattern=pattern)


def parse_remove(sql: str) -> RemoveCommand:
    """Parse ``REMOVE @<stage> [PATTERN='...']``."""
    rest = sql[6:].strip()
    parts = rest.split()
    if not parts:
        raise StageError("REMOVE requires a stage location.")
    stage = parse_stage_ref(parts[0])
    pattern = None
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if key.strip().upper() == "PATTERN" and sep:
            pattern = value.strip().strip("'\"")
    return RemoveCommand(stage=stage, pattern=pattern)


def match_stage_ddl(statement: exp.Expression) -> str | None:
    """Classify stage DDL: CREATE_STAGE / DROP_STAGE / SHOW_STAGES / ALTER_STAGE."""
    if isinstance(statement, exp.Create) and str(statement.kind).upper() == "STAGE":
        return "CREATE_STAGE"
    if isinstance(statement, exp.Drop) and str(statement.kind).upper() == "STAGE":
        return "DROP_STAGE"
    if isinstance(statement, exp.Show) and str(statement.this).upper() == "STAGES":
        return "SHOW_STAGES"
    if isinstance(statement, exp.Command) and statement.sql(
        dialect="snowflake"
    ).upper().lstrip().startswith("ALTER STAGE"):
        return "ALTER_STAGE"
    return None


def create_stage(manager: StageManager, session: SessionContext, statement: exp.Create) -> None:
    """Stage directories are created lazily; CREATE STAGE is a metadata no-op."""
    stage = parse_stage_ref("@" + statement.this.name)
    manager.stage_dir(session, stage)  # ensure the directory exists


def drop_stage(manager: StageManager, session: SessionContext, statement: exp.Drop) -> None:
    """Remove the named stage's directory tree."""
    stage = parse_stage_ref("@" + statement.this.name)
    base = manager.stage_base_dir(session, stage)
    shutil.rmtree(base, ignore_errors=True)


def show_stages(manager: StageManager, session: SessionContext) -> list[tuple[str, str, str, str]]:
    """List known named stages as (name, database_name, schema_name, url)."""
    named_root = os.path.join(manager.root, "named")
    rows: list[tuple[str, str, str, str]] = []
    if os.path.isdir(named_root):
        for db in sorted(os.listdir(named_root)):
            db_dir = os.path.join(named_root, db)
            if not os.path.isdir(db_dir):
                continue
            for schema in sorted(os.listdir(db_dir)):
                schema_dir = os.path.join(db_dir, schema)
                if not os.path.isdir(schema_dir):
                    continue
                for name in sorted(os.listdir(schema_dir)):
                    if os.path.isdir(os.path.join(schema_dir, name)):
                        rows.append((name, db, schema, f"s3://sfemu/{db}.{schema}.{name}"))
    return rows


def copy_stage_source(copy_node: exp.Copy) -> tuple[str, str, dict[str, Any]] | None:
    """If ``COPY INTO t FROM @stage``, return (table_sql, stage_ref, file_format dict).

    Returns None when the COPY does not reference a stage.
    """
    files = copy_node.args.get("files") or []
    if not files:
        return None
    source_sql = files[0].sql(dialect="snowflake").strip()
    if not source_sql.startswith("@"):
        return None
    table_sql = copy_node.this.sql(dialect="duckdb")
    file_format = _extract_file_format(copy_node)
    return table_sql, source_sql, file_format


def build_copy_sql(
    session: SessionContext,
    table_sql: str,
    stage_ref_sql: str,
    file_format: dict[str, Any],
) -> str:
    """Build DuckDB SQL that loads the staged files into ``table_sql``."""
    stage = parse_stage_ref(stage_ref_sql)
    manager = get_stage_manager()
    base = manager.stage_base_dir(session, stage)
    fmt_type = file_format.get("TYPE", "").upper()

    if fmt_type in ("JSON", "NDJSON"):
        extensions = (".json", ".ndjson")
    elif fmt_type in ("CSV", "TSV"):
        extensions = (".csv", ".tsv", ".txt")
    elif fmt_type:
        raise StageError(f"Unsupported FILE_FORMAT TYPE: {fmt_type} (expected CSV or JSON).")
    else:
        extensions = None  # infer per file below

    matches = manager._list_matches(base, stage.path, None)
    if not matches:
        raise StageError(f"No files found in stage location: {stage_ref_sql}")

    # Group files by inferred format when TYPE was not specified.
    json_files, csv_files = [], []
    for rel, abs_path in matches:
        if extensions is not None:
            if not abs_path.lower().endswith(extensions):
                continue
            (json_files if fmt_type in ("JSON", "NDJSON") else csv_files).append(abs_path)
        else:
            if abs_path.lower().endswith((".json", ".ndjson")):
                json_files.append(abs_path)
            else:
                csv_files.append(abs_path)

    statements: list[str] = []
    if csv_files:
        statements.append(_copy_csv_sql(table_sql, csv_files, file_format))
    if json_files:
        statements.append(_copy_json_sql(table_sql, json_files))
    if not statements:
        raise StageError(
            f"No {fmt_type or 'matching'} files found in stage location: {stage_ref_sql}"
        )
    return ";\n".join(statements)


def _copy_csv_sql(table_sql: str, files: list[str], file_format: dict[str, Any]) -> str:
    # read_csv_auto is auto-detecting by default; pass explicit options only when
    # the FILE_FORMAT overrides the defaults.
    options: list[str] = []
    skip_header = file_format.get("SKIP_HEADER")
    if skip_header is not None:
        try:
            options.append(f"header={bool(int(skip_header))}".lower())
        except ValueError:
            pass
    if file_format.get("FIELD_DELIMITER"):
        options.append(f"delim='{_sql_escape(file_format['FIELD_DELIMITER'])}'")
    suffix = f", {', '.join(options)}" if options else ""
    read_call = f"read_csv_auto({_format_file_list(files)}{suffix})"
    return f"INSERT INTO {table_sql} SELECT * FROM {read_call}"


def _copy_json_sql(table_sql: str, files: list[str]) -> str:
    return f"INSERT INTO {table_sql} SELECT * FROM read_json_auto({_format_file_list(files)})"


def _format_file_list(files: list[str]) -> str:
    if len(files) == 1:
        return _sql_string(files[0])
    return "[" + ", ".join(_sql_string(f) for f in files) + "]"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _extract_file_format(copy_node: exp.Copy) -> dict[str, Any]:
    """Pull FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1, ...) out of a COPY node."""
    fmt: dict[str, Any] = {}
    for param in copy_node.args.get("params") or []:
        name = param.this.sql(dialect="snowflake").upper() if param.this else ""
        if name != "FILE_FORMAT":
            continue
        for prop in param.expressions or []:
            if not isinstance(prop, exp.Property):
                continue
            key = prop.this.sql(dialect="snowflake").upper()
            value_node = prop.args.get("value")
            if value_node is None:
                continue
            fmt[key] = value_node.sql(dialect="snowflake").strip("'\"")
    return fmt


def _compile_pattern(pattern: str | None) -> re.Pattern[str] | None:
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise StageError(f"Invalid PATTERN regex: {exc}") from exc


def _expand_local_files(spec: str) -> list[str]:
    """Expand ``file://``/``~``/globs in a PUT source spec to concrete paths."""
    src = spec.strip()
    if src.startswith("file://"):
        src = src[len("file://") :]
    src = os.path.expanduser(src)
    if not os.path.isabs(src):
        src = os.path.abspath(src)
    return sorted(glob.glob(src))


def _parse_bool(value: str) -> bool:
    return value.strip().upper() in ("TRUE", "1", "YES", "ON")


def _md5_hex(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -- Singleton ----------------------------------------------------------------

_manager: StageManager | None = None
_manager_lock = threading.Lock()


def get_stage_manager() -> StageManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = StageManager(settings.stage_root or None)
    return _manager
