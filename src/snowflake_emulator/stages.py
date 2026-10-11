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
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

import sqlglot
from sqlglot import exp

from snowflake_emulator.file_formats import resolve_file_format
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
        """Absolute directory backing the given stage reference (sub-path excluded).

        Every stage lives under its kind's root (``named``/``table``/``user``/``internal``),
        and the namespace components come from client input (``USE DATABASE``/``USE
        SCHEMA`` and the stage name), so the base itself is contained in that root. Callers
        that append a client-supplied sub-path contain that against the base in turn.
        """
        user_root = os.path.join(self.root, "user")
        table_root = os.path.join(self.root, "table")
        named_root = os.path.join(self.root, "named")
        if stage.kind == "user":
            return self.contained(os.path.join(user_root, session.token), user_root)
        if stage.kind == "internal":
            return os.path.join(self.root, "internal")
        if stage.kind == "table":
            return self.contained(
                os.path.join(table_root, session.database, session.schema, stage.name),
                table_root,
            )
        db = stage.database or session.database
        schema = stage.schema or session.schema
        return self.contained(os.path.join(named_root, db, schema, stage.name), named_root)

    def stage_dir(self, session: SessionContext, stage: StageRef) -> str:
        """Absolute directory for a stage reference including its sub-path."""
        base = self.stage_base_dir(session, stage)
        return self.contained(os.path.join(base, stage.path), base) if stage.path else base

    @staticmethod
    def contained(path: str, root: str) -> str:
        """Return `path` resolved inside `root`, or fail when it escapes the stage tree.

        Stage references are client-supplied SQL (``@stage/../../../etc`` and ``@../x``
        both resolve to a real path), and a stage is a namespace rather than a way to
        reach the rest of the filesystem -- otherwise a staged query could read, stream
        or delete arbitrary files at the server's privileges.
        """
        resolved_root = os.path.realpath(root)
        resolved = os.path.realpath(path)
        if not PurePath(resolved).is_relative_to(resolved_root):
            raise StageError(f"Stage path escapes its stage location: {path}")
        return resolved

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
        """List the files under ``base/sub_path`` matching the optional regex.

        The sub-path may name one file (`dir/data.csv`) or a directory prefix (`dir/`), and
        is contained within the stage it names: `LIST`/`GET`/`REMOVE`/`COPY INTO` all take
        it from client SQL, so a `..` must not reach outside the stage.
        """
        if not os.path.isdir(base):
            return []
        search_root = self.contained(os.path.join(base, sub_path), base) if sub_path else base
        if os.path.isfile(search_root):
            rel = os.path.relpath(search_root, base).replace(os.sep, "/")
            if pattern is not None and not pattern.search(rel):
                return []
            return [(rel, search_root)]
        matches: list[tuple[str, str]] = []
        for dirpath, _dirnames, filenames in os.walk(search_root):
            for filename in sorted(filenames):
                abs_path = os.path.join(dirpath, filename)
                rel = os.path.relpath(abs_path, base).replace(os.sep, "/")
                if pattern is None or pattern.search(rel):
                    matches.append((rel, abs_path))
        return sorted(matches)


# -- Parsing helpers ---------------------------------------------------------


def _check_stage_segments(segments: list[str], ref: str) -> None:
    """Reject stage/namespace components that would walk the filesystem.

    ``@..`` and ``@db..stage`` are not stages in Snowflake (a stage name is an identifier),
    and letting them through turns the stage root into a relative path into the rest of the
    filesystem -- ``@..`` in particular collapses to three empty segments, which then fall
    back to the session's database/schema and make the whole schema directory the stage.
    """
    for segment in segments:
        if segment in ("", ".", ".."):
            raise StageError(f"Invalid stage reference: {ref!r}")


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
        _check_stage_segments([stage_part[1:]], ref)
        return StageRef(kind="table", name=stage_part[1:], path=path_part)

    parts = stage_part.split(".")
    _check_stage_segments(parts, ref)
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


def copy_stage_source(copy_node: exp.Copy) -> tuple[str, str, dict[str, Any], str | None] | None:
    """If ``COPY INTO t FROM @stage``, return (table_sql, stage_ref, file_format, pattern).

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
    return table_sql, source_sql, file_format, _copy_param(copy_node, "PATTERN")


def _filter_matches(
    matches: list[tuple[str, str]], pattern: str | None
) -> list[tuple[str, str]]:
    """Keep the files whose stage-relative path matches Snowflake's ``PATTERN`` regex."""
    if not pattern:
        return matches
    compiled = _compile_pattern(str(pattern))
    return [match for match in matches if compiled.search(match[0])]


def _check_file_format_options(options: dict[str, Any], file_format: str, ref: str) -> None:
    """Reject file format options the emulator does not translate for `file_format`."""
    supported = _SUPPORTED_FILE_FORMAT_OPTIONS[file_format]
    for name, value in options.items():
        if name in supported:
            continue
        allowed = _DEFAULTS_ONLY_OPTIONS.get(name)
        normalised = str(value).strip().strip("'\"").upper()
        if allowed is not None and normalised in {option.upper() for option in allowed}:
            continue
        raise StageError(
            f"FILE_FORMAT option {name} = {value} is not supported for {file_format} "
            f"staged files ({ref})."
        )


def _check_staged_compression(files: list[str], ref: str) -> None:
    """Fail loudly for staged files whose compression the emulator cannot read.

    bzip2 has no reader here, and DuckDB's CSV reader reports *zero rows* for a truncated
    gzip stream instead of raising, which would silently turn a corrupt file into an empty
    result -- so gzip files are partially decompressed here to prove the stream is intact.
    """
    unsupported = [path for path in files if path.lower().endswith(_UNSUPPORTED_COMPRESSION)]
    if unsupported:
        raise StageError(
            "Bzip2-compressed staged files are not supported by the emulator "
            f"({os.path.basename(unsupported[0])}); use gzip or zstd instead."
        )
    for path in files:
        if not path.lower().endswith(".gz"):
            continue  # zstd is skipped: the stdlib cannot decompress it
        try:
            with gzip.open(path, "rb") as handle:
                handle.read(_COMPRESSION_SNIFF_BYTES)
        except (OSError, EOFError) as exc:  # gzip raises EOFError for a truncated stream
            raise StageError(f"Failed to decompress staged file {path}: {exc}") from exc


def build_copy_sql(
    session: SessionContext,
    table_sql: str,
    stage_ref_sql: str,
    file_format: dict[str, Any],
    pattern: str | None = None,
) -> str:
    """Build DuckDB SQL that loads the staged files into ``table_sql``."""
    stage = parse_stage_ref(stage_ref_sql)
    manager = get_stage_manager()
    fmt_type = file_format.get("TYPE", "").upper()

    if fmt_type in ("JSON", "NDJSON"):
        stated_format = "JSON"
    elif fmt_type in ("CSV", "TSV"):
        stated_format = "CSV"
    elif fmt_type == "PARQUET":
        stated_format = "PARQUET"
    elif fmt_type:
        raise StageError(
            f"Unsupported FILE_FORMAT TYPE: {fmt_type} (expected CSV, JSON or PARQUET)."
        )
    else:
        # TYPE omitted: each file's extension decides (see _extension_file_format).
        stated_format = None

    matches = _filter_matches(_stage_matches(manager, session, stage), pattern)
    if not matches:
        raise StageError(f"No files found in stage location: {stage_ref_sql}")

    groups: dict[str, list[str]] = {"CSV": [], "JSON": [], "PARQUET": []}
    for _rel, abs_path in matches:
        # The extension decides the reader, and pins the files a stated TYPE covers: the
        # extension helper strips compression suffixes, so a `.csv.gz` still counts as CSV.
        file_format_name = _extension_file_format(abs_path)
        if stated_format is not None and file_format_name != stated_format:
            continue
        groups[stated_format or file_format_name].append(abs_path)

    _check_staged_compression(
        sorted(groups["CSV"] + groups["JSON"] + groups["PARQUET"]), stage_ref_sql
    )

    statements: list[str] = []
    if groups["CSV"]:
        statements.append(_copy_csv_sql(table_sql, groups["CSV"], file_format))
    if groups["JSON"]:
        statements.append(_copy_json_sql(table_sql, groups["JSON"]))
    if groups["PARQUET"]:
        statements.append(_copy_parquet_sql(table_sql, groups["PARQUET"]))
    if not statements:
        raise StageError(
            f"No {fmt_type or 'matching'} files found in stage location: {stage_ref_sql}"
        )
    return ";\n".join(statements)


def _stage_matches(
    manager: StageManager, session: SessionContext, stage: StageRef
) -> list[tuple[str, str]]:
    """List the files a stage reference resolves to, as (stage-relative, absolute) pairs."""
    return manager._list_matches(manager.stage_base_dir(session, stage), stage.path, None)


def _extension_file_format(path: str) -> str:
    """Infer a staged file's format from its extension (unknown extensions read as CSV)."""
    lower = path.lower()
    for suffix in _COMPRESSION_EXTENSIONS:
        if lower.endswith(suffix):
            lower = lower[: -len(suffix)]
    if lower.endswith(_PARQUET_EXTENSIONS):
        return "PARQUET"
    if lower.endswith(_JSON_EXTENSIONS):
        return "JSON"
    return "CSV"


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


def _copy_parquet_sql(table_sql: str, files: list[str]) -> str:
    return f"INSERT INTO {table_sql} SELECT * FROM read_parquet({_format_file_list(files)})"


def _format_file_list(files: list[str]) -> str:
    if len(files) == 1:
        return _sql_string(files[0])
    return "[" + ", ".join(_sql_string(f) for f in files) + "]"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _unquote_literal(value: str) -> str:
    """Strip the single pair of quotes enclosing a SQL literal, if any.

    Only one pair is removed (unlike ``str.strip``), so a value that *is* a quote
    character -- ``FIELD_OPTIONALLY_ENCLOSED_BY = '"'`` -- survives intact.
    """
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _copy_param(copy_node: exp.Copy, name: str) -> str | None:
    """Return a scalar COPY INTO parameter's value, e.g. ``PATTERN = '...'``."""
    for param in copy_node.args.get("params") or []:
        param_name = param.this.sql(dialect="snowflake").upper() if param.this else ""
        if param_name == name.upper() and param.expression is not None:
            return _unquote_literal(param.expression.sql(dialect="snowflake"))
    return None


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
            fmt[key] = _unquote_literal(value_node.sql(dialect="snowflake"))
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


# -- Querying staged files directly (SELECT ... FROM @stage[/path][/file]) -----
#
# Snowflake exposes a staged file like a table: every record is a row and its fields are
# reachable positionally. A JSON record collapses into a single ``$1`` value, while CSV
# and Parquet keep one column per field (``$1``..``$n`` for CSV, the file's own column
# names for Parquet, with ``$n`` referring to the nth of them).
#
# The file format comes from the query, not from the file name: an explicit
# ``(FILE_FORMAT => ...)`` clause wins (inline options or a created format's name) and
# otherwise the format is inferred from the file extensions, exactly as ``COPY INTO``
# already does here. A location mixing formats cannot be read as one relation, so it is
# rejected instead of guessed.

_JSON_EXTENSIONS = (".json", ".ndjson")
_PARQUET_EXTENSIONS = (".parquet",)
_COMPRESSION_EXTENSIONS = (".gz", ".zst", ".bz2")
# DuckDB decompresses gzip and zstd itself; Snowflake's other codec (bzip2) has no
# reader here, so such files are rejected rather than read as compressed bytes.
_UNSUPPORTED_COMPRESSION = (".bz2",)
# Snowflake caps a semi-structured value at 16MB, so the staged readers must accept records
# longer than DuckDB's 2MB default CSV line limit (otherwise a long record is an error).
_MAX_RECORD_BYTES = 16 * 1024 * 1024
_COMPRESSION_SNIFF_BYTES = 65536
# Snowflake file format types -> the reader family used to query them.
_QUERYABLE_TYPES = {
    "JSON": "JSON",
    "NDJSON": "JSON",
    "CSV": "CSV",
    "TSV": "CSV",
    "PARQUET": "PARQUET",
}

# File format options translated for a staged query, per format. Anything else is rejected
# below rather than silently ignored: quietly dropping an option reports Snowflake results
# the emulator did not actually produce.
_SUPPORTED_FILE_FORMAT_OPTIONS = {
    "CSV": frozenset(
        {
            "TYPE",
            "PATTERN",
            "FORMAT_NAME",
            "SKIP_HEADER",
            "FIELD_DELIMITER",
            "FIELD_OPTIONALLY_ENCLOSED_BY",
            "ESCAPE",
            "NULL_IF",
            "EMPTY_FIELD_AS_NULL",
            "COMPRESSION",
        }
    ),
    "JSON": frozenset(
        {"TYPE", "PATTERN", "FORMAT_NAME", "STRIP_OUTER_ARRAY", "COMPRESSION"}
    ),
    "PARQUET": frozenset({"TYPE", "PATTERN", "FORMAT_NAME"}),
}
# Options accepted only at Snowflake's default value: that default is what DuckDB already
# does, while any other value would change the data read and cannot be honoured here.
_DEFAULTS_ONLY_OPTIONS = {
    "TRIM_SPACE": ("FALSE", "0", "NO", "OFF"),
    "SKIP_BLANK_LINES": ("FALSE", "0", "NO", "OFF"),
    "ENCODING": ("UTF8", "UTF-8"),
    "VALIDATE_UTF8": ("TRUE", "1", "YES", "ON"),
    "REPLACE_INVALID_CHARACTERS": ("FALSE", "0", "NO", "OFF"),
    "RECORD_DELIMITER": ("\\n", "\n", "\\r\\n", "\r\n"),
    # The emulator always widens a ragged location (union_by_name), so neither value errors.
    "ERROR_ON_COLUMN_COUNT_MISMATCH": ("TRUE", "FALSE"),
    # Rejected separately when set to anything but NONE (see _stage_csv_reader).
    "ESCAPE_UNENCLOSED_FIELD": ("NONE",),
}

# sqlglot has no grammar for a parenthesized option list after a stage reference in a
# FROM clause, so the clause is lifted out of the SQL text before parsing and re-attached
# on the parseable ``(FILE_FORMAT => '<name>')`` form as a marker string, then decoded
# when the AST is rewritten.
_OPTIONS_MARKER = "sfemu:staged-file-options:"
_STAGE_CLAUSE_RE = re.compile(r"@[\w~%./\-]+\s*\(")

# How much of a JSON file is inspected to tell newline-delimited records from a single
# pretty-printed document (which a line-oriented reader would turn into NULL rows).
_JSON_SNIFF_BYTES = 65536
_JSON_SNIFF_LINES = 5


@dataclass(frozen=True)
class StageFileSource:
    """A staged location resolved into a DuckDB-readable relation."""

    ref: str  # the `@stage/path` text as written
    file_format: str  # JSON | CSV | PARQUET
    files: list[str]  # absolute on-disk paths of the staged files
    relation_sql: str  # DuckDB SELECT exposing the records of those files
    positional: dict[int, str]  # `$N` index -> column name in `relation_sql`


def _encode_options(options: dict[str, Any]) -> str:
    return _OPTIONS_MARKER + json.dumps(options, sort_keys=True)


def _decode_options(value: str | None) -> dict[str, Any] | None:
    """Return the options encoded by `_encode_options`, or None for any other value."""
    if not value or not value.startswith(_OPTIONS_MARKER):
        return None
    try:
        decoded = json.loads(value[len(_OPTIONS_MARKER) :])
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _quoted_spans(sql: str) -> list[tuple[int, int]]:
    """Return the [start, end) ranges of single- and double-quoted strings in `sql`.

    Used so a stage reference or `)` inside a string literal is not mistaken for SQL.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(sql):
        quote = sql[i]
        if quote not in ("'", '"'):
            i += 1
            continue
        start = i
        i += 1
        while i < len(sql):
            if sql[i] == quote:
                if sql[i + 1 : i + 2] == quote:  # a doubled quote is an escape
                    i += 2
                    continue
                break
            i += 1
        spans.append((start, min(i + 1, len(sql))))
        i += 1
    return spans


def _matching_paren(sql: str, open_index: int, quoted: list[tuple[int, int]]) -> str:
    """Return the balanced ``(...)`` group starting at `open_index`."""

    def in_string(index: int) -> bool:
        return any(start <= index < end for start, end in quoted)

    depth = 0
    for i in range(open_index, len(sql)):
        if in_string(i):
            continue
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
            if depth == 0:
                return sql[open_index : i + 1]
    raise StageError("Unbalanced parentheses in a staged file FILE_FORMAT clause.")


def _split_top_level(text: str) -> list[str]:
    """Split a comma-separated option list on its top-level commas."""
    quoted = _quoted_spans(text)

    def in_string(index: int) -> bool:
        return any(start <= index < end for start, end in quoted)

    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if in_string(index):
            continue
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def parse_stage_file_options(clause: str) -> dict[str, Any]:
    """Parse ``(FILE_FORMAT => (TYPE = CSV, ...), PATTERN => '...')`` into a flat dict."""
    inner = clause.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()
    if not inner:
        return {}
    # COPY INTO takes the same options with `=`, space-separated, so its grammar parses
    # this list too once the top-level commas are turned into separators.
    items = [item.replace("=>", "=") for item in _split_top_level(inner)]
    try:
        node = sqlglot.parse_one(
            f"COPY INTO __sfemu__ FROM @__sfemu__ {' '.join(items)}", read="snowflake"
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a stage error
        raise StageError(f"Invalid staged file options {clause}: {exc}") from exc
    if not isinstance(node, exp.Copy):
        raise StageError(f"Invalid staged file options: {clause}")
    options: dict[str, Any] = {}
    for param in node.args.get("params") or []:
        name = param.this.sql(dialect="snowflake").upper() if param.this else ""
        if name == "FILE_FORMAT":
            if param.expressions:
                options.update(
                    {key.upper(): value for key, value in _extract_file_format(node).items()}
                )
            elif param.expression is not None:
                # `(FILE_FORMAT => 'name')` names a created file format.
                options["FORMAT_NAME"] = param.expression.sql(dialect="snowflake").strip("'\"")
            else:
                raise StageError(f"FILE_FORMAT needs options or a format name: {clause}")
        elif name == "PATTERN" and param.expression is not None:
            options["PATTERN"] = param.expression.sql(dialect="snowflake").strip("'\"")
    return options


def _in_copy_statement(sql: str, index: int, quoted: list[tuple[int, int]]) -> bool:
    """True when `index` falls in a statement that starts with ``COPY INTO``.

    ``COPY INTO`` takes its options as space-separated parameters and is handled by
    ``build_copy_sql``, so its stage references must be left for that path.
    """
    start = 0
    for i in range(index - 1, -1, -1):
        if sql[i] == ";" and not any(span_start <= i < span_end for span_start, span_end in quoted):
            start = i + 1
            break
    return re.match(r"\s*COPY\s+INTO\b", sql[start:index], re.IGNORECASE) is not None


def preprocess_stage_file_clauses(sql: str) -> str:
    """Rewrite ``@stage/file (FILE_FORMAT => ...)`` clauses into parseable SQL.

    Can't be done on the AST because sqlglot rejects the clause outright, so the options
    are captured from the statement text and re-attached as an encoded ``FILE_FORMAT``
    literal that the AST rewrite decodes. Plain ``(FILE_FORMAT => 'name')`` and
    ``(PATTERN => '...')`` clauses are captured the same way, leaving one form to handle.
    """
    quoted = _quoted_spans(sql)
    parts: list[str] = []
    cursor = 0
    for match in _STAGE_CLAUSE_RE.finditer(sql):
        if match.start() < cursor or any(start <= match.start() < end for start, end in quoted):
            continue
        if _in_copy_statement(sql, match.start(), quoted):
            continue
        open_index = match.end() - 1
        clause = _matching_paren(sql, open_index, quoted)
        options = parse_stage_file_options(clause)
        if not options:
            continue  # leave it for sqlglot to reject, as Snowflake would
        parts.append(sql[cursor:open_index])
        parts.append(f"(FILE_FORMAT => {_sql_string(_encode_options(options))})")
        cursor = open_index + len(clause)
    if not parts:
        return sql
    parts.append(sql[cursor:])
    return "".join(parts)


def is_stage_file_table(table: exp.Table) -> bool:
    """True when a FROM entry references a stage location (``FROM @stage/path``).

    ``COPY INTO`` also names its source (or target) with a stage reference, but that
    statement is handled by ``build_copy_sql``, so stage references inside a COPY are not
    query results and are left alone here.
    """
    if not isinstance(table.this, exp.Var) or not (table.this.name or "").startswith("@"):
        return False
    node: exp.Expression | None = table
    while node is not None:
        if isinstance(node, exp.Copy):
            return False
        node = node.parent
    return True


def has_stage_file_table(statement: exp.Expression) -> bool:
    """True when any FROM entry in the statement queries a staged file."""
    return any(is_stage_file_table(table) for table in statement.find_all(exp.Table))


def _stage_file_options(session: SessionContext, table: exp.Table) -> dict[str, Any]:
    """Collect the FILE_FORMAT/PATTERN options attached to a stage reference."""
    options: dict[str, Any] = {}
    pattern = table.args.get("pattern")
    if isinstance(pattern, exp.Literal):
        options["PATTERN"] = pattern.name
    format_arg = table.args.get("format")
    if isinstance(format_arg, exp.Literal):
        inline = _decode_options(format_arg.name)
        if inline is not None:
            options.update(inline)
        else:
            # A bare `(FILE_FORMAT => 'name')` names a created file format.
            options["FORMAT_NAME"] = format_arg.name
    options = {key.upper(): value for key, value in options.items()}
    if options.get("FORMAT_NAME"):
        # Explicit options win over the named format's defaults.
        options = {
            key.upper(): value for key, value in resolve_file_format(session, options).items()
        }
    return options


def _int_option(options: dict[str, Any], key: str) -> int | None:
    value = options.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        raise StageError(f"Invalid {key} value in a staged file query: {value!r}") from None


def _bool_option(options: dict[str, Any], key: str) -> bool:
    value = options.get(key)
    return value is not None and str(value).strip().upper() in ("TRUE", "1", "YES", "ON")


def _bool_option_explicitly_false(options: dict[str, Any], key: str) -> bool:
    """True only when the option is present and set to a false value."""
    value = options.get(key)
    return value is not None and str(value).strip().upper() in ("FALSE", "0", "NO", "OFF")


def _string_list_option(value: Any) -> list[str]:
    """Reduce a Snowflake single-or-list option value to a list of plain strings."""
    inner = str(value).strip()
    if inner[:1] in ("(", "["):
        inner = inner[1:-1]
    return [item.strip("'\"") for item in _split_top_level(inner)]


def _quoted_identifier(name: str) -> str:
    """Quote an identifier for DuckDB SQL (the reader relation is built as SQL text)."""
    return '"' + name.replace('"', '""') + '"'


def _infer_file_format(files: list[str], ref: str) -> str:
    """Infer a staged location's file format from its files' extensions.

    A location may hold files of several formats, but they cannot be read as one
    relation (JSON records expose ``$1`` while CSV/Parquet expose ``$1``..``$n``), so a
    mixed location is rejected. An explicit ``(FILE_FORMAT => (TYPE = ...))`` pins one.
    """
    formats = {_extension_file_format(path) for path in files}
    if len(formats) > 1:
        raise StageError(
            f"Staged location {ref} mixes file formats ({', '.join(sorted(formats))}); "
            "query it with an explicit (FILE_FORMAT => (TYPE = ...)) clause."
        )
    return formats.pop()


def _check_newline_delimited_json(files: list[str], ref: str) -> None:
    """Reject staged JSON that is not one record per line.

    Snowflake reads staged JSON as newline-delimited records; a pretty-printed document
    would silently become a set of NULL rows under a line-oriented reader, so the leading
    lines are validated up front and rejected with a clear message instead. Gzipped files
    are decompressed for the check; zstd files are skipped, as Python cannot read them.
    """
    for path in files:
        if path.lower().endswith(".zst"):
            continue
        opener = gzip.open if path.lower().endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
                sample = handle.read(_JSON_SNIFF_BYTES)
        except OSError as exc:
            raise StageError(f"Failed to read staged file {path}: {exc}") from exc
        lines = sample.splitlines()
        if len(sample) == _JSON_SNIFF_BYTES and not sample.endswith(("\n", "\r")):
            lines = lines[:-1]  # the sample cut a record in half
        checked = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                raise StageError(
                    f"{os.path.basename(path)} is not newline-delimited JSON: the "
                    f"emulator reads staged JSON as one record per line ({ref})."
                ) from None
            checked += 1
            if checked >= _JSON_SNIFF_LINES:
                break


def _stage_csv_reader(files: list[str], options: dict[str, Any]) -> str:
    """Build the DuckDB CSV reader call for a staged file query.

    The options follow Snowflake's CSV file format defaults, which differ from DuckDB's:
    no header row (``header=false``: a header is skipped via ``SKIP_HEADER`` rather than
    becoming column names), no quoting (``FIELD_OPTIONALLY_ENCLOSED_BY`` defaults to
    NONE) and empty fields reading as NULL (``EMPTY_FIELD_AS_NULL`` with ``NULL_IF``
    defaulting to ``\\N``). ``union_by_name`` widens a location whose files differ in width
    instead of silently dropping the extra fields of the narrower file.
    """
    args = ["header=false", "union_by_name=true", f"max_line_size={_MAX_RECORD_BYTES}"]
    skip_header = _int_option(options, "SKIP_HEADER")
    if skip_header:
        args.append(f"skip={skip_header}")
    if options.get("FIELD_DELIMITER") is not None:
        args.append(f"delim={_sql_string(str(options['FIELD_DELIMITER']))}")

    enclosed_by = options.get("FIELD_OPTIONALLY_ENCLOSED_BY")
    if enclosed_by is None or str(enclosed_by).strip().upper() == "NONE":
        args.append("quote=''")
    else:
        args.append(f"quote={_sql_string(str(enclosed_by))}")
    if options.get("ESCAPE") is not None:
        args.append(f"escape={_sql_string(str(options['ESCAPE']))}")
    unenclosed = options.get("ESCAPE_UNENCLOSED_FIELD")
    if unenclosed is not None and str(unenclosed).strip().upper() != "NONE":
        # DuckDB only escapes enclosed fields, so honouring this would change results.
        raise StageError(
            "ESCAPE_UNENCLOSED_FIELD is not supported by the emulator (DuckDB escapes "
            "enclosed fields only); drop the option or set it to NONE."
        )

    null_if = _string_list_option(options["NULL_IF"]) if options.get("NULL_IF") else ["\\N"]
    if not _bool_option_explicitly_false(options, "EMPTY_FIELD_AS_NULL"):
        null_if = ["", *null_if]
    if null_if:  # DuckDB rejects an empty nullstr list
        args.append("nullstr=[" + ", ".join(_sql_string(v) for v in null_if) + "]")

    compression = str(options.get("COMPRESSION", "")).strip().upper()
    if compression in ("GZIP", "ZSTD", "NONE"):
        args.append(f"compression={_sql_string(compression.lower())}")
    elif compression and compression not in ("AUTO", "AUTO_DETECT"):
        raise StageError(
            f"COMPRESSION = {compression} is not supported by the emulator "
            "(use AUTO, GZIP, ZSTD or NONE)."
        )
    return f"read_csv_auto({_format_file_list(files)}, {', '.join(args)})"


def _reader_columns(cursor: Any, reader: str, ref: str) -> list[str]:
    """Column names of a DuckDB file reader (metadata only; no rows are read)."""
    try:
        cursor.execute(f"DESCRIBE SELECT * FROM {reader}")
        described = cursor.fetchall()
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a stage error
        raise StageError(f"Failed to read staged files at {ref}: {exc}") from exc
    return [str(row[0]) for row in described]


def _positional_relation(
    cursor: Any, reader: str, ref: str, *, rename: bool
) -> tuple[str, dict[int, str]]:
    """Build ``SELECT <cols> FROM <reader>`` plus its ``$N`` -> column mapping.

    CSV is read positionally, so the reader's own names are hidden behind the ``$1``..
    ``$n`` names Snowflake exposes. Parquet files carry their column names, and ``$n``
    resolves to the nth of them, as Snowflake does.
    """
    columns = _reader_columns(cursor, reader, ref)
    if not columns:  # an empty file has no schema to expose positionally
        return f"SELECT * FROM {reader}", {}
    if not rename:
        return f"SELECT * FROM {reader}", {
            index: name for index, name in enumerate(columns, start=1)
        }
    selected = ", ".join(
        f"{_quoted_identifier(name)} AS {_quoted_identifier(f'${index}')}"
        for index, name in enumerate(columns, start=1)
    )
    return f"SELECT {selected} FROM {reader}", {
        index: f"${index}" for index in range(1, len(columns) + 1)
    }


def resolve_stage_file_source(
    session: SessionContext, cursor: Any, table: exp.Table, ref: str
) -> StageFileSource:
    """Resolve a ``FROM @stage/path`` reference into a DuckDB-readable relation."""
    options = _stage_file_options(session, table)
    stage = parse_stage_ref(ref)
    manager = get_stage_manager()

    matches = _filter_matches(_stage_matches(manager, session, stage), options.get("PATTERN"))
    if not matches:
        raise StageError(f"No files found in stage location: {ref}")
    files = [path for _rel, path in matches]
    _check_staged_compression(files, ref)

    # An explicit TYPE is authoritative for every file under the location; without one,
    # the format is inferred from the extensions (see _infer_file_format).
    type_option = str(options.get("TYPE", "")).strip().upper()
    if type_option:
        file_format = _QUERYABLE_TYPES.get(type_option)
        if file_format is None:
            raise StageError(
                f"Querying staged {type_option} files is not supported "
                "(expected CSV, JSON or PARQUET)."
            )
        if type_option == "TSV":
            options.setdefault("FIELD_DELIMITER", "\t")
    else:
        file_format = _infer_file_format(files, ref)
    _check_file_format_options(options, file_format, ref)

    if file_format == "JSON":
        # STRIP_OUTER_ARRAY=TRUE reads each file as one JSON document and explodes the
        # outer array; the default (FALSE) keeps one record per line.
        if _bool_option(options, "STRIP_OUTER_ARRAY"):
            compressed = [path for path in files if path.lower().endswith(_COMPRESSION_EXTENSIONS)]
            if compressed:
                raise StageError(
                    "STRIP_OUTER_ARRAY is not supported for compressed staged files "
                    f"({os.path.basename(compressed[0])})."
                )
            relation_sql = (
                'SELECT UNNEST(CAST(decode(content) AS JSON[])) AS "$1" '
                f"FROM read_blob({_format_file_list(files)})"
            )
        else:
            _check_newline_delimited_json(files, ref)
            # `CAST` (not `TRY_CAST`): a malformed record must fail loudly rather than
            # silently becoming a NULL row. Blank lines are removed by the filter, which
            # is evaluated before the projection.
            relation_sql = (
                'SELECT CAST(column0 AS JSON) AS "$1" '
                f"FROM read_csv({_format_file_list(files)}, header=false, quote='', "
                "delim='\\x01', columns={'column0': 'VARCHAR'}, auto_detect=false, "
                f"max_line_size={_MAX_RECORD_BYTES}) "
                "WHERE trim(column0) <> ''"
            )
        return StageFileSource(ref, file_format, files, relation_sql, {1: "$1"})

    if file_format == "CSV":
        reader = _stage_csv_reader(files, options)
        relation_sql, positional = _positional_relation(cursor, reader, ref, rename=True)
    else:
        reader = f"read_parquet({_format_file_list(files)})"
        relation_sql, positional = _positional_relation(cursor, reader, ref, rename=False)
    return StageFileSource(ref, file_format, files, relation_sql, positional)


def _positional_index(node: exp.Expression) -> int | None:
    """Return N for a positional stage reference ``$N``, or None for anything else.

    ``$name`` is a session variable rather than a stage column, so only integer
    parameters map to a position.
    """
    if not isinstance(node, exp.Parameter) or not isinstance(node.this, exp.Literal):
        return None
    if node.this.is_string:
        return None
    try:
        return int(node.this.name)
    except (TypeError, ValueError):
        return None


def rewrite_stage_file_tables(
    statement: exp.Expression, session: SessionContext, cursor: Any
) -> exp.Expression:
    """Replace ``FROM @stage/path`` entries with DuckDB readers and resolve ``$1``/``$2``.

    Returns the statement unchanged when it does not query a staged file.
    """
    tables = [table for table in statement.find_all(exp.Table) if is_stage_file_table(table)]
    if not tables:
        return statement
    if len(tables) > 1:
        raise StageError(
            "A statement can query only one staged file location "
            f"({len(tables)} were found)."
        )
    statement = statement.copy()
    table = next(table for table in statement.find_all(exp.Table) if is_stage_file_table(table))
    source = resolve_stage_file_source(session, cursor, table, table.this.name)
    alias = table.args.get("alias")
    relation = sqlglot.parse_one(source.relation_sql, read="duckdb")
    replacement = exp.Subquery(this=relation)
    if alias is not None:
        replacement.set("alias", alias.copy())

    def _substitute(node: exp.Expression) -> exp.Expression:
        if node is table:
            return replacement
        index = _positional_index(node)
        if index is None:
            return node
        column = source.positional.get(index)
        if column is None:
            raise StageError(
                f"${index} does not exist in {source.ref}: {source.file_format} files "
                f"expose {len(source.positional)} positional column(s)."
            )
        return exp.column(column, quoted=True)

    return statement.transform(_substitute, copy=False)


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
