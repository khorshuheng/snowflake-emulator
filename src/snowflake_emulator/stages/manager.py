"""The on-disk stage tree: PUT/GET/LIST/REMOVE storage and stage DDL."""

from __future__ import annotations

import datetime
import glob
import gzip
import hashlib
import os
import re
import shutil
import tempfile
import threading
from pathlib import PurePath
from typing import Any

from sqlglot import exp

from snowflake_emulator import metadata

from snowflake_emulator.core.ast import drop_target_table
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.stages.refs import (
    GetCommand,
    ListCommand,
    PutCommand,
    RemoveCommand,
    StageError,
    StageRef,
    StagedFile,
    _compile_pattern,
    parse_stage_ref,
)


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
        auto_compress = _parse_bool(command.options.get("AUTO_COMPRESS", "TRUE"))
        # SOURCE_COMPRESSION tells Snowflake how a source whose name does not say is
        # compressed; AUTO_DETECT and NONE defer to the file name's extension.
        source_override = command.options.get("SOURCE_COMPRESSION", "AUTO_DETECT").upper()
        if source_override in ("AUTO_DETECT", "NONE", ""):
            source_override = ""
        target_dir = self.stage_dir(session, command.stage)
        os.makedirs(target_dir, exist_ok=True)

        sources: list[str] = []
        for spec in command.files:
            matches = _expand_local_files(spec)
            if not matches:
                if os.path.isdir(os.path.expanduser(_strip_file_uri(spec))):
                    raise StageError(
                        "PUT does not stage a directory: give a file name or a wildcard, "
                        "e.g. `PUT file:///tmp/data/*.csv @stage`."
                    )
                raise StageError(f"File doesn't exist: {spec}")
            for src in matches:
                if os.path.isdir(src):
                    raise StageError(
                        "PUT does not stage a directory (or the files inside nested "
                        "subdirectories): give a file name or a wildcard for one directory, "
                        "e.g. `PUT file:///tmp/data/*.csv @stage`."
                    )
                sources.append(src)
        # Snowflake does not preserve the local directory structure, so it rejects a PUT
        # that would stage files from more than one directory (docs).
        directories = {os.path.dirname(src) for src in sources}
        if len(directories) > 1:
            raise StageError(
                "PUT does not support multiple files with divergent directory paths, "
                "because Snowflake does not preserve the file system directory structure."
            )

        staged: list[StagedFile] = []
        with self._lock:
            for src in sources:
                name = os.path.basename(src)
                source_size = os.path.getsize(src)
                source_compression = source_override or _compression_of(name)
                target_compression = source_compression
                if auto_compress and source_compression == "NONE":
                    name += ".gz"
                    target_compression = "GZIP"
                dst = os.path.join(target_dir, name)
                if os.path.exists(dst) and not overwrite:
                    raise StageError(
                        f"File already exists: {name}. "
                        "Use OVERWRITE=TRUE to replace it."
                    )
                if target_compression == "GZIP" and source_compression == "NONE":
                    with open(src, "rb") as reader, gzip.open(dst, "wb") as writer:
                        shutil.copyfileobj(reader, writer)
                else:
                    shutil.copy2(src, dst)
                staged.append(
                    StagedFile(
                        name=name,
                        rel_path=os.path.relpath(
                            dst, self.stage_base_dir(session, command.stage)
                        ),
                        size=os.path.getsize(dst),
                        source_size=source_size,
                        status="UPLOADED",
                        source_compression=source_compression,
                        target_compression=target_compression,
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
def stage_file_format(session: SessionContext, stage: StageRef) -> dict[str, Any]:
    """The parsing options a named stage was created with (empty for other stage kinds).

    ``CREATE STAGE ... FILE_FORMAT = (...)`` sets the stage's default format, which a read
    uses when it does not name one; only a *named* stage has metadata to look up.
    """
    if stage.kind != "named" or not stage.name:
        return {}
    from snowflake_emulator.database import get_manager

    options = metadata.stage_file_format(
        get_manager(session),
        stage.database or session.database,
        stage.schema or session.schema,
        stage.name,
    )
    if options.get("FORMAT_NAME"):
        # ``ALTER/CREATE STAGE ... FILE_FORMAT = (FORMAT_NAME = ...)`` names a format.
        from snowflake_emulator.file_formats import resolve_file_format

        options = {
            key.upper(): value for key, value in resolve_file_format(session, options).items()
        }
    return options


def _table_parts(
    table: exp.Table | None, session: SessionContext
) -> tuple[str, str, str]:
    """The ``(database, schema, name)`` a DDL node names, defaulting to the session."""
    if table is None:
        return session.database, session.schema, ""
    return (
        table.text("catalog") or session.database,
        table.text("db") or session.schema,
        table.name,
    )


def _ensure_stage_namespace(
    duckdb_manager: Any, database: str, schema: str
) -> None:
    """Reject a stage whose database/schema does not exist, as Snowflake does.

    The catalog is attached first (without creating a schema) so a persisted database that
    has not been touched yet still resolves, while a missing schema stays an error instead
    of being created by the stage DDL.
    """
    if not duckdb_manager.has_database(database):
        raise StageError(f"Database '{database.upper()}' does not exist.")
    duckdb_manager.attach_database(database)
    cursor = duckdb_manager.cursor(None)
    found = cursor.execute(
        "SELECT 1 FROM duckdb_schemas() WHERE lower(database_name) = lower(?) "
        "AND lower(schema_name) = lower(?)",
        [database, schema],
    ).fetchone()
    if found is None:
        raise StageError(f"Schema '{schema.upper()}' does not exist.")


def _stage_for_parts(database: str, schema: str, name: str) -> StageRef:
    """A named-stage reference built from identifiers (never re-split on dots)."""
    return StageRef(kind="named", database=database, schema=schema, name=name)


def _split_qualified_name(text: str) -> list[str]:
    """Split ``db."My Stage".s`` on dots outside double quotes."""
    parts: list[str] = []
    current: list[str] = []
    quoted = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            if quoted and index + 1 < len(text) and text[index + 1] == '"':
                current.append('"')
                index += 2
                continue
            quoted = not quoted
            index += 1
            continue
        if char == "." and not quoted:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    parts.append("".join(current).strip())
    return parts


def _parenthesised(text: str, start: int) -> str | None:
    """The contents of the balanced ``( ... )`` starting at `start`, quotes respected."""
    depth = 0
    index = start
    quote: str | None = None
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 2
                    continue
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
        index += 1
    return None


def _stage_file_format(statement: exp.Expression) -> dict[str, str]:
    """The ``FILE_FORMAT = (...)`` options of a ``CREATE STAGE`` (empty when absent)."""
    options: dict[str, str] = {}
    properties = statement.args.get("properties") or []
    for prop in properties:
        if not isinstance(prop, exp.FileFormatProperty):
            continue
        for item in prop.expressions:
            if not isinstance(item, exp.Property):
                continue
            value = item.args.get("value")
            if value is None:
                continue
            if isinstance(value, exp.Literal) and value.is_string:
                options[str(item.this).upper()] = str(value.this)
            else:
                options[str(item.this).upper()] = value.sql(dialect="snowflake")
    return options


def _stage_comment(statement: exp.Expression) -> str | None:
    """The ``COMMENT = '...'`` of a ``CREATE STAGE``, when given."""
    for prop in statement.args.get("properties") or []:
        if isinstance(prop, exp.SchemaCommentProperty) and isinstance(
            prop.this, exp.Literal
        ):
            return str(prop.this.this)
    return None


def _stage_url_property(statement: exp.Expression) -> str | None:
    """The external ``URL = '...'`` of a ``CREATE STAGE`` (None when the property is absent)."""
    for prop in statement.args.get("properties") or []:
        if isinstance(prop, exp.Property) and str(prop.this).upper() == "URL":
            value = prop.args.get("value")
            if isinstance(value, exp.Literal) and value.is_string:
                return str(value.this)
            return ""
    return None


_ALTER_STAGE_RE = re.compile(
    r"(?is)^\s*STAGE\s+(\"[^\"]+\"|\S+)\s+SET\s+(.+?)\s*$"
)


def alter_stage(
    duckdb_manager: Any, session: SessionContext, statement: exp.Expression
) -> str:
    """Apply ``ALTER STAGE ... SET FILE_FORMAT = (...)`` / ``SET COMMENT = '...'``.

    Only those two assignments are modelled; any other ``SET`` form is a loud error
    instead of a silent no-op, because Snowflake would either apply it or reject it.
    The FILE_FORMAT parentheses are scanned (quotes respected), so a value containing
    ``)`` is kept whole.
    """
    text = str(statement.args.get("expression") or "")
    match = re.match(
        r'(?is)^\s*STAGE\s+((?:"(?:[^"]|"")*"|[^\s"]+)(?:\.(?:"(?:[^"]|"")*"|[^\s"]+))*)\s+SET\s+(.+?)\s*$',
        text,
    )
    if match is None:
        raise StageError(f"Unsupported ALTER STAGE form: {text.strip()}")
    parts = _split_qualified_name(match.group(1))
    name = parts[-1]
    database = parts[0] if len(parts) == 3 else session.database
    schema = parts[-2] if len(parts) > 1 else session.schema
    current = metadata.get_stage(duckdb_manager, database, schema, name)
    if current is None:
        raise StageError(f"Stage '{name.upper()}' does not exist.")
    file_format = dict(current.get("file_format") or {})
    comment = current.get("comment")
    assignments = match.group(2)
    applied = False
    format_match = re.search(r"(?is)FILE_FORMAT\s*=\s*\(", assignments)
    if format_match is not None:
        inner = _parenthesised(assignments, format_match.end() - 1)
        if inner is None:
            raise StageError("Unterminated FILE_FORMAT in ALTER STAGE.")
        from snowflake_emulator.stages.refs import _parse_stage_options

        file_format = {
            key.upper(): value for key, value in _parse_stage_options(inner).items()
        }
        applied = True
    comment_match = re.search(r"(?is)COMMENT\s*=\s*'((?:[^']|'')*)'", assignments)
    if comment_match is not None:
        comment = comment_match.group(1).replace("''", "'")
        applied = True
    if not applied:
        raise StageError(
            f"Unsupported ALTER STAGE SET form: {assignments.strip()}; the emulator "
            "supports SET FILE_FORMAT = (...) and SET COMMENT = '...'."
        )
    metadata.update_stage(
        duckdb_manager,
        database,
        schema,
        name,
        file_format=file_format,
        comment=comment,
    )
    return name.upper()


def create_stage(
    duckdb_manager: Any,
    manager: StageManager,
    session: SessionContext,
    statement: exp.Create,
) -> None:
    """Create the stage directory and record its metadata.

    The ``FILE_FORMAT`` options become the stage's default parsing format (used when a
    read does not name one), and the URL/owner/comment/type are what ``SHOW STAGES`` and
    ``INFORMATION_SCHEMA.STAGES`` report.
    """
    url = _stage_url_property(statement)
    if url is not None:
        if not url.strip():
            raise StageError("URL must not be empty.")
        raise StageError(
            "External stages are not supported by the emulator; only internal stages "
            "(@name, @~, @%table) can be created."
        )
    table = statement.this if isinstance(statement.this, exp.Table) else None
    database, schema, name = _table_parts(table, session)
    _ensure_stage_namespace(duckdb_manager, database, schema)
    stage = _stage_for_parts(database, schema, name)
    exists = (
        metadata.get_stage(duckdb_manager, database, schema, name) is not None
        or os.path.isdir(manager.stage_base_dir(session, stage))
    )
    if exists:
        if statement.args.get("exists"):  # IF NOT EXISTS: keep the existing stage
            return
        if not statement.args.get("replace"):
            raise StageError(f"Stage '{name.upper()}' already exists.")
    os.makedirs(manager.stage_dir(session, stage), exist_ok=True)
    metadata.record_stage(
        duckdb_manager,
        session,
        name,
        database=database,
        schema=schema,
        file_format=_stage_file_format(statement),
        comment=_stage_comment(statement),
    )


def drop_stage(
    duckdb_manager: Any,
    manager: StageManager,
    session: SessionContext,
    statement: exp.Drop,
) -> str | None:
    """Remove the named stage's directory tree and metadata, returning the stage name."""
    target = drop_target_table(statement)
    if target is None:
        return None
    database, schema, name = _table_parts(target, session)
    stage = _stage_for_parts(database, schema, name)
    recorded = metadata.get_stage(duckdb_manager, database, schema, name)
    base = manager.stage_base_dir(session, stage)
    if recorded is None and not os.path.isdir(base) and not statement.args.get("exists"):
        raise StageError(f"Stage '{name.upper()}' does not exist.")
    shutil.rmtree(base, ignore_errors=True)
    metadata.drop_stage(duckdb_manager, database, schema, name)
    return name


def show_stages(
    duckdb_manager: Any,
    manager: StageManager,
    session: SessionContext,
    database: str | None = None,
    schema: str | None = None,
) -> list[tuple[str, str, str, str]]:
    """List named stages as (name, database_name, schema_name, url).

    The recorded metadata is the source of truth; a stage directory created by an older
    emulator (or by hand under the stage root) is still listed with the synthetic URL.
    ``database``/``schema`` narrow the listing, and neither means every database.
    """
    rows: list[tuple[str, str, str, str]] = [
        (stage["name"], stage["database"], stage["schema"], stage["url"])
        for stage in metadata.stages(duckdb_manager, database, schema)
    ]
    known = {(name.lower(), db.lower(), schema_name.lower()) for name, db, schema_name, _ in rows}
    named_root = os.path.join(manager.root, "named")
    if os.path.isdir(named_root):
        for db in sorted(os.listdir(named_root)):
            if database is not None and db.lower() != database.lower():
                continue
            db_dir = os.path.join(named_root, db)
            if not os.path.isdir(db_dir):
                continue
            for schema_name in sorted(os.listdir(db_dir)):
                if schema is not None and schema_name.lower() != schema.lower():
                    continue
                schema_dir = os.path.join(db_dir, schema_name)
                if not os.path.isdir(schema_dir):
                    continue
                for name in sorted(os.listdir(schema_dir)):
                    key = (name.lower(), db.lower(), schema_name.lower())
                    if key in known or not os.path.isdir(os.path.join(schema_dir, name)):
                        continue
                    rows.append(
                        (name, db, schema_name, f"s3://sfemu/{db}.{schema_name}.{name}")
                    )
    return sorted(rows)
def _expand_local_files(spec: str) -> list[str]:
    """Expand ``file://``/``~``/globs in a PUT source spec to concrete paths."""
    src = spec.strip()
    if src.startswith("file://"):
        src = src[len("file://") :]
    src = os.path.expanduser(src)
    if not os.path.isabs(src):
        src = os.path.abspath(src)
    return sorted(glob.glob(src))


# Extensions Snowflake treats as already compressed (it then skips AUTO_COMPRESS).
_COMPRESSED_EXTENSIONS = {
    ".gz": "GZIP",
    ".gzip": "GZIP",
    ".bz2": "BZIP2",
    ".zst": "ZSTD",
    ".br": "BROTLI",
    ".zip": "DEFLATE",
    ".deflate": "DEFLATE",
    ".lz4": "LZ4",
    ".snappy": "SNAPPY",
}


def _compression_of(name: str) -> str:
    """The compression a file name already carries, or ``NONE``."""
    _, extension = os.path.splitext(name.lower())
    return _COMPRESSED_EXTENSIONS.get(extension, "NONE")


def _strip_file_uri(spec: str) -> str:
    """The local path of a PUT source spec (drop ``file://``)."""
    return spec[len("file://") :] if spec.startswith("file://") else spec


def _parse_bool(value: str) -> bool:
    return value.strip().upper() in ("TRUE", "1", "YES", "ON")


def _md5_hex(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
