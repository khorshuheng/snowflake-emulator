"""Named file formats (``CREATE FILE FORMAT``) and their use by ``COPY INTO``.

A named file format is a schema-scoped object holding parsing options; ``COPY INTO``
references it with ``FILE_FORMAT = (FORMAT_NAME = '...')``. The emulator keeps them in
memory (like session state) rather than in DuckDB, which has no equivalent object.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

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
    """Owns the named file formats, keyed by (database, schema, name)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._formats: dict[tuple[str, str, str], FileFormat] = {}

    def _key(self, session: SessionContext, name: str) -> tuple[str, str, str]:
        return (session.database, session.schema, name.upper())

    def create(
        self,
        session: SessionContext,
        name: str,
        options: dict[str, str],
        *,
        replace: bool = False,
        if_not_exists: bool = False,
    ) -> bool:
        key = self._key(session, name)
        with self._lock:
            if key in self._formats and not replace:
                if if_not_exists:
                    return False
                raise FileFormatError(f"File format '{name.upper()}' already exists.")
            self._formats[key] = FileFormat(
                name=name.upper(),
                database=session.database,
                schema=session.schema,
                options=dict(options),
            )
        return True

    def get(self, session: SessionContext, name: str) -> FileFormat | None:
        with self._lock:
            return self._formats.get(self._key(session, name))

    def drop(self, session: SessionContext, name: str, *, if_exists: bool = False) -> bool:
        key = self._key(session, name)
        with self._lock:
            existed = self._formats.pop(key, None) is not None
        if not existed and not if_exists:
            raise FileFormatError(f"File format '{name.upper()}' does not exist.")
        return existed

    def list(self, session: SessionContext) -> list[FileFormat]:
        with self._lock:
            return [
                fmt
                for fmt in self._formats.values()
                if fmt.database == session.database and fmt.schema == session.schema
            ]


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


_manager: FileFormatManager | None = None
_manager_lock = threading.Lock()


def get_file_format_manager() -> FileFormatManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = FileFormatManager()
    return _manager
