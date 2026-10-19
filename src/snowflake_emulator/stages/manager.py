"""The on-disk stage tree: PUT/GET/LIST/REMOVE storage and stage DDL."""

from __future__ import annotations

import datetime
import glob
import hashlib
import os
import shutil
import tempfile
import threading
from pathlib import PurePath

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
def create_stage(manager: StageManager, session: SessionContext, statement: exp.Create) -> None:
    """Create the stage directory so an empty stage is visible to SHOW STAGES."""
    stage = parse_stage_ref("@" + statement.this.name)
    os.makedirs(manager.stage_dir(session, stage), exist_ok=True)


def drop_stage(
    manager: StageManager, session: SessionContext, statement: exp.Drop
) -> str | None:
    """Remove the named stage's directory tree, returning the stage name."""
    target = drop_target_table(statement)
    if target is None:
        return None
    stage = parse_stage_ref("@" + target.name)
    base = manager.stage_base_dir(session, stage)
    shutil.rmtree(base, ignore_errors=True)
    return target.name


def show_stages(
    manager: StageManager,
    session: SessionContext,
    database: str | None = None,
    schema: str | None = None,
) -> list[tuple[str, str, str, str]]:
    """List named stages as (name, database_name, schema_name, url).

    ``database``/``schema`` narrow the listing to one namespace; neither means every
    named stage on disk, which is what an account-level listing asks for.
    """
    named_root = os.path.join(manager.root, "named")
    rows: list[tuple[str, str, str, str]] = []
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
                    if os.path.isdir(os.path.join(schema_dir, name)):
                        rows.append(
                            (
                                name,
                                db,
                                schema_name,
                                f"s3://sfemu/{db}.{schema_name}.{name}",
                            )
                        )
    return rows
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
