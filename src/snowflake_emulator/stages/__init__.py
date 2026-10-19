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

import base64
import gzip
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp

from snowflake_emulator.file_formats import resolve_file_format
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.settings import settings
from snowflake_emulator.stages.copy import (
    _check_file_format_options,
    _check_staged_compression,
    _extension_file_format,
    _extract_file_format,
    _filter_matches,
    _MAX_RECORD_BYTES,
    _bool_option,
    _format_file_list,
    _option_value_text,
    _positional_index,
    _raw_json_relation,
    _sql_string,
    _stage_matches,
    build_copy_sql,
    copy_stage_source,
)
from snowflake_emulator.stages.manager import (
    StageManager,
    create_stage,
    drop_stage,
    show_stages,
)
from snowflake_emulator.stages.refs import (
    GetCommand,
    ListCommand,
    PutCommand,
    RemoveCommand,
    StageError,
    StageRef,
    StagedFile,
    _compile_pattern,
    _parse_stage_options,
    _unquote_literal,
    match_stage_command,
    match_stage_ddl,
    parse_get,
    parse_list,
    parse_put,
    parse_remove,
    parse_stage_ref,
)

# Re-exported stage API: the engine and tests import these from this package.


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

# Snowflake file format types -> the reader family used to query them.
_QUERYABLE_TYPES = {
    "JSON": "JSON",
    "NDJSON": "JSON",
    "CSV": "CSV",
    "TSV": "CSV",
    "PARQUET": "PARQUET",
}

# sqlglot has no grammar for a parenthesized option list after a stage reference in a
# FROM clause, so the clause is lifted out of the SQL text before parsing and re-attached
# on the parseable ``(FILE_FORMAT => '<name>')`` form as a marker string, then decoded
# when the AST is rewritten.
_OPTIONS_MARKER = "sfemu:staged-file-options:"
_STAGE_CLAUSE_RE = re.compile(r"@[\w~%./\-]+\s*\(")



@dataclass(frozen=True)
class StageFileSource:
    """A staged location resolved into a DuckDB-readable relation."""

    ref: str  # the `@stage/path` text as written
    file_format: str  # JSON | CSV | PARQUET
    files: list[str]  # absolute on-disk paths of the staged files
    relation_sql: str  # DuckDB SELECT exposing the records of those files
    positional: dict[int, str]  # `$N` index -> column name in `relation_sql`


def _encode_options(options: dict[str, Any]) -> str:
    """Encode staged-file options into a marker usable inside a SQL string literal.

    The JSON is base64-encoded rather than embedded directly: sqlglot 30.21+ treats a
    backslash as a string escape in the Snowflake dialect, so the ``\"`` JSON escapes
    would be eaten on the way back through the parser. Base64's alphabet has no quotes
    or backslashes, so the marker survives any such escape handling.
    """
    payload = json.dumps(options, sort_keys=True).encode("utf-8")
    return _OPTIONS_MARKER + base64.urlsafe_b64encode(payload).decode("ascii")


def _decode_options(value: str | None) -> dict[str, Any] | None:
    """Return the options encoded by `_encode_options`, or None for any other value."""
    if not value or not value.startswith(_OPTIONS_MARKER):
        return None
    try:
        payload = base64.urlsafe_b64decode(value[len(_OPTIONS_MARKER) :].encode("ascii"))
        decoded = json.loads(payload)
    except (ValueError, json.JSONDecodeError):
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
                options["FORMAT_NAME"] = _option_value_text(param.expression)
            else:
                raise StageError(f"FILE_FORMAT needs options or a format name: {clause}")
        elif name == "PATTERN" and param.expression is not None:
            options["PATTERN"] = _option_value_text(param.expression)
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
    session: SessionContext,
    cursor: Any,
    table: exp.Table,
    ref: str,
    manager: StageManager,
) -> StageFileSource:
    """Resolve a ``FROM @stage/path`` reference into a DuckDB-readable relation."""
    options = _stage_file_options(session, table)
    stage = parse_stage_ref(ref)

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
        relation_sql = _raw_json_relation(
            files, ref, _bool_option(options, "STRIP_OUTER_ARRAY")
        )
        return StageFileSource(ref, file_format, files, relation_sql, {1: "$1"})

    if file_format == "CSV":
        reader = _stage_csv_reader(files, options)
        relation_sql, positional = _positional_relation(cursor, reader, ref, rename=True)
    else:
        reader = f"read_parquet({_format_file_list(files)})"
        relation_sql, positional = _positional_relation(cursor, reader, ref, rename=False)
    return StageFileSource(ref, file_format, files, relation_sql, positional)




def rewrite_stage_file_tables(
    statement: exp.Expression,
    session: SessionContext,
    cursor: Any,
    manager: StageManager,
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
    source = resolve_stage_file_source(session, cursor, table, table.this.name, manager)
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
