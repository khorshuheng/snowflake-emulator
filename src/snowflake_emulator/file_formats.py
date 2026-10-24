"""Named file formats (``CREATE FILE FORMAT``) and their use by ``COPY INTO``.

A named file format is a schema-scoped object holding parsing options; ``COPY INTO``
references it with ``FILE_FORMAT = (FORMAT_NAME = '...')``. DuckDB has no equivalent
object, so the options are persisted in the database's ``_sf_file_formats_ext`` side table
(see ``metadata``) and surfaced by ``SHOW FILE FORMATS`` and
``INFORMATION_SCHEMA.FILE_FORMATS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from snowflake_emulator import metadata
from snowflake_emulator.sessions import SessionContext


class FileFormatError(ValueError):
    """Raised for missing/duplicate file formats or malformed FILE_FORMAT clauses."""


@dataclass
class FileFormat:
    name: str
    database: str
    schema: str
    options: dict[str, str] = field(default_factory=dict)


class FileFormatManager:
    """The named file formats, persisted per (database, schema, name)."""

    def create(
        self,
        session: SessionContext,
        name: str,
        options: dict[str, str],
        *,
        replace: bool = False,
        if_not_exists: bool = False,
        database: str | None = None,
        schema: str | None = None,
    ) -> bool:
        manager = _duckdb_manager(session)
        try:
            return metadata.record_file_format(
                manager,
                session,
                name,
                options,
                replace=replace,
                if_not_exists=if_not_exists,
                database=database,
                schema=schema,
            )
        except metadata.FileFormatExists:
            raise FileFormatError(f"File format '{name.upper()}' already exists.") from None

    def get(
        self,
        session: SessionContext,
        name: str,
        *,
        database: str | None = None,
        schema: str | None = None,
    ) -> FileFormat | None:
        options = metadata.file_format(
            _duckdb_manager(session),
            database or session.database,
            schema or session.schema,
            name,
        )
        if options is None:
            return None
        return FileFormat(
            name=name.upper(),
            database=session.database,
            schema=session.schema,
            options=options,
        )

    def drop(
        self,
        session: SessionContext,
        name: str,
        *,
        if_exists: bool = False,
        database: str | None = None,
        schema: str | None = None,
    ) -> bool:
        try:
            return metadata.drop_file_format(
                _duckdb_manager(session),
                database or session.database,
                schema or session.schema,
                name,
                if_exists=if_exists,
            )
        except metadata.FileFormatNotFound:
            raise FileFormatError(f"File format '{name.upper()}' does not exist.") from None

    def list(self, session: SessionContext) -> list[FileFormat]:
        return [
            FileFormat(
                name=row["name"],
                database=session.database,
                schema=row["schema"],
                options=row["options"],
            )
            for row in metadata.file_formats(
                _duckdb_manager(session), session.database, session.schema
            )
        ]


# The manager is a stateless facade over the catalog side table; the variable exists so a
# test (or a reset) can drop it the way the other singletons are dropped.
_manager: FileFormatManager | None = None


def get_file_format_manager() -> FileFormatManager:
    """The process-wide file-format manager (the rows live in the catalog side table)."""
    global _manager
    if _manager is None:
        _manager = FileFormatManager()
    return _manager


def _duckdb_manager(session: SessionContext) -> Any:
    """The DuckDB manager holding the session's catalogs (isolated mode has its own)."""
    from snowflake_emulator.database import get_manager

    return get_manager(session)


def resolve_file_format(session: SessionContext, file_format: dict[str, Any]) -> dict[str, Any]:
    """Expand ``FORMAT_NAME`` into the referenced format's options.

    Explicit options in the ``COPY INTO`` clause override the named format's defaults.
    """
    name = file_format.get("FORMAT_NAME")
    if not name:
        return file_format
    stored = get_file_format_manager().get(session, str(name))
    if stored is None:
        raise FileFormatError(f"File format '{name}' does not exist.")
    merged: dict[str, Any] = dict(stored.options)
    merged.update({key: value for key, value in file_format.items() if key != "FORMAT_NAME"})
    return merged
