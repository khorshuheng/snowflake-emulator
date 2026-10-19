"""COPY INTO from a stage: source resolution and DuckDB load SQL."""

from __future__ import annotations

import gzip
import json
import os
import re
from typing import Any

from sqlglot import exp

from snowflake_emulator.file_formats import resolve_file_format
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.stages.manager import StageManager
from snowflake_emulator.stages.refs import (
    StageError,
    StageRef,
    _compile_pattern,
    _split_top_level,
    _unquote_literal,
    parse_stage_ref,
)


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


def copy_stage_source(
    copy_node: exp.Copy,
) -> tuple[str, str, dict[str, Any], str | None, str | None] | None:
    """Resolve ``COPY INTO t FROM <source>`` into the pieces ``build_copy_sql`` needs.

    Returns ``(table_sql, stage_ref, file_format, pattern, projection)``. ``projection``
    is the SELECT list of a ``FROM (SELECT ... FROM @stage)`` source, carried so the
    staged rows are transformed on load rather than copied column-for-column; it is
    ``None`` for a plain ``FROM @stage`` source. Returns ``None`` when the COPY does not
    reference a stage.
    """
    files = copy_node.args.get("files") or []
    if not files:
        return None
    source = files[0]
    projection: str | None
    if isinstance(source, exp.Subquery):
        resolved = _subquery_stage_source(source)
        if resolved is None:
            return None
        stage_ref, projection = resolved
    else:
        stage_ref = source.sql(dialect="snowflake").strip()
        if not stage_ref.startswith("@"):
            return None
        projection = None
    if len(files) > 1:
        raise StageError(
            "COPY INTO with multiple source references is not supported by the emulator."
        )
    _check_copy_options(copy_node)
    table_sql = copy_node.this.sql(dialect="duckdb")
    file_format = _extract_file_format(copy_node)
    return table_sql, stage_ref, file_format, _copy_param(copy_node, "PATTERN"), projection


# Options the emulator does not implement. A value equal to Snowflake's default is
# accepted (it changes nothing); anything else is rejected instead of silently ignored.
_COPY_OPTION_DEFAULTS = {
    "ON_ERROR": "ABORT_STATEMENT",
    "PURGE": "FALSE",
    "FORCE": "FALSE",
    "LOAD_UNCERTAIN_FILES": "FALSE",
    "MATCH_BY_COLUMN_NAME": "NONE",
}


def _check_copy_options(copy_node: exp.Copy) -> None:
    for name, default in _COPY_OPTION_DEFAULTS.items():
        value = _copy_param(copy_node, name)
        if value is not None and value.strip().upper() != default:
            raise StageError(
                f"COPY INTO option {name} = {value} is not supported by the emulator."
            )
    for name in ("INCLUDE_METADATA", "VALIDATION_MODE"):
        value = _copy_param(copy_node, name)
        if value is not None:
            raise StageError(
                f"COPY INTO option {name} = {value} is not supported by the emulator."
            )


# Only these SELECT clauses are carried into the staged reader. Every other clause
# (WHERE, DISTINCT, joins, ...) changes which rows are read or how they are shaped, and
# build_copy_sql rebuilds the FROM from scratch, so silently dropping one would load the
# wrong data; they are rejected instead, naming the SQL keyword.
_SUPPORTED_SUBQUERY_CLAUSES = frozenset({"expressions", "from_"})
_SUBQUERY_CLAUSE_NAMES = {
    "with_": "WITH",
    "distinct": "DISTINCT",
    "where": "WHERE",
    "group": "GROUP BY",
    "having": "HAVING",
    "qualify": "QUALIFY",
    "order": "ORDER BY",
    "limit": "LIMIT",
    "offset": "OFFSET",
    "joins": "JOIN",
    "windows": "WINDOW",
    "cluster": "CLUSTER BY",
    "distribute": "DISTRIBUTE BY",
    "sort": "SORT BY",
    "sample": "SAMPLE",
    "locks": "LOCK",
    "into": "INTO",
    "top": "TOP",
}


def _subquery_stage_source(subquery: exp.Subquery) -> tuple[str, str] | None:
    """Resolve ``(SELECT <projection> FROM @stage)`` to ``(stage_ref, projection_sql)``.

    Only the SELECT list is carried: it is rewritten from Snowflake's positional ``$N``
    references to the quoted ``"$N"`` column the raw staged relation exposes, then
    rendered as DuckDB SQL for ``build_copy_sql`` to place over that relation. The FROM
    entry must be a bare stage reference; anything else is not a stage COPY source.
    """
    select = subquery.this
    if not isinstance(select, exp.Select):
        return None
    unsupported = [
        key
        for key, value in select.args.items()
        if value and key not in _SUPPORTED_SUBQUERY_CLAUSES
    ]
    if unsupported:
        clauses = ", ".join(
            _SUBQUERY_CLAUSE_NAMES.get(key, key.rstrip("_").upper())
            for key in unsupported
        )
        raise StageError(
            "COPY INTO ... FROM (SELECT ...) supports only a SELECT list over a single "
            f"stage reference; unsupported clause(s): {clauses}."
        )
    from_ = select.args.get("from_")
    table = from_.this if from_ is not None else None
    if not isinstance(table, exp.Table) or not isinstance(table.this, exp.Var):
        return None
    stage_ref = table.this.name or ""
    if not stage_ref.startswith("@"):
        return None
    if not select.expressions:
        raise StageError(
            "COPY INTO ... FROM (SELECT ...) requires a select list over the stage."
        )
    projection = ", ".join(
        _stage_positional_expression(expression, stage_ref).sql(dialect="duckdb")
        for expression in select.expressions
    )
    return stage_ref, projection


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


def _stage_positional_expression(expression: exp.Expression, ref: str) -> exp.Expression:
    """Rewrite a staged-file projection's ``$N`` references to the raw ``"$1"`` column."""

    def replace(node: exp.Expression) -> exp.Expression:
        # A qualified `alias.$N` nests the parameter in a Column; matching the Column
        # itself drops the qualifier, which the rebuilt COPY relation cannot resolve.
        inner = node.this if isinstance(node, exp.Column) else node
        index = _positional_index(inner)
        return node if index is None else _projected_positional_column(index, ref)

    return expression.transform(replace, copy=True)


def _projected_positional_column(index: int, ref: str) -> exp.Column:
    """The ``"$1"`` column for ``$index``, rejecting positions the reader lacks.

    A projected staged source exposes one raw record value (``$1``), so ``$2`` and up do
    not exist. Failing here reproduces the direct ``SELECT ... FROM @stage`` message
    instead of a raw DuckDB binder error; a table qualifier on ``$N`` is dropped because
    the COPY rebuilds the relation and the subquery's own alias has no meaning here.
    """
    if index != 1:
        raise StageError(
            f"${index} does not exist in {ref}: a projected staged source exposes "
            "only $1."
        )
    return exp.column("$1", quoted=True)


def _filter_matches(
    matches: list[tuple[str, str]], pattern: str | None
) -> list[tuple[str, str]]:
    """Keep the files whose stage-relative path matches Snowflake's ``PATTERN`` regex."""
    if not pattern:
        return matches
    compiled = _compile_pattern(str(pattern))
    return [match for match in matches if compiled.search(match[0])]


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
    manager: StageManager,
    pattern: str | None = None,
    projection: str | None = None,
) -> list[tuple[str, str]]:
    """Build one DuckDB load statement per staged file.

    Returns ``(stage_relative_path, sql)`` pairs, so the caller can report one
    Snowflake-shaped COPY result row per file.
    """
    stage = parse_stage_ref(stage_ref_sql)
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

    groups: dict[str, list[tuple[str, str]]] = {"CSV": [], "JSON": [], "PARQUET": []}
    for rel, abs_path in matches:
        # The extension decides the reader, and pins the files a stated TYPE covers: the
        # extension helper strips compression suffixes, so a `.csv.gz` still counts as CSV.
        file_format_name = _extension_file_format(abs_path)
        if stated_format is not None and file_format_name != stated_format:
            continue
        groups[stated_format or file_format_name].append((rel, abs_path))

    _check_staged_compression(
        sorted(abs_path for group in groups.values() for _rel, abs_path in group),
        stage_ref_sql,
    )

    # A projection is carried only for JSON, whose raw reader exposes it as `$1`; the
    # CSV/Parquet readers are copy-only, so accepting one there would silently drop it.
    if projection is not None and (groups["CSV"] or groups["PARQUET"]):
        raise StageError(
            "COPY INTO ... FROM (SELECT ...) supports a projection only for staged "
            "JSON files; CSV and Parquet sources must be copied without one."
        )

    for format_name, entries in groups.items():
        if entries:
            _check_file_format_options(file_format, format_name, stage_ref_sql)

    plan: list[tuple[str, str]] = []
    for rel, abs_path in groups["CSV"]:
        plan.append((rel, _copy_csv_sql(table_sql, [abs_path], file_format)))
    for rel, abs_path in groups["JSON"]:
        plan.append(
            (
                rel,
                _copy_json_sql(
                    table_sql,
                    [abs_path],
                    stage_ref_sql,
                    projection,
                    _bool_option(file_format, "STRIP_OUTER_ARRAY"),
                ),
            )
        )
    for rel, abs_path in groups["PARQUET"]:
        plan.append((rel, _copy_parquet_sql(table_sql, [abs_path])))
    if not plan:
        raise StageError(
            f"No {fmt_type or 'matching'} files found in stage location: {stage_ref_sql}"
        )
    return plan


def _stage_matches(
    manager: StageManager, session: SessionContext, stage: StageRef
) -> list[tuple[str, str]]:
    """List the files a stage reference resolves to, as (stage-relative, absolute) pairs."""
    return manager._list_matches(manager.stage_base_dir(session, stage), stage.path, None)


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


def _bool_option(options: dict[str, Any], key: str) -> bool:
    value = options.get(key)
    return value is not None and str(value).strip().upper() in ("TRUE", "1", "YES", "ON")


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
    return f"INSERT INTO {table_sql} SELECT * FROM {_stage_csv_reader(files, file_format)}"


def _copy_json_sql(
    table_sql: str,
    files: list[str],
    ref: str,
    projection: str | None = None,
    strip_outer_array: bool = False,
) -> str:
    if projection is None:
        if strip_outer_array:
            raise StageError(
                "STRIP_OUTER_ARRAY is only supported for "
                "COPY INTO ... FROM (SELECT ... FROM @stage) by the emulator."
            )
        return (
            f"INSERT INTO {table_sql} SELECT * "
            f"FROM read_json_auto({_format_file_list(files)})"
        )
    # A projected `FROM (SELECT ... FROM @stage)` addresses each record as `$1`, so the
    # reader must expose the raw document as that single value (read_json_auto flattens
    # the keys into columns instead).
    relation = _raw_json_relation(files, ref, strip_outer_array)
    return f"INSERT INTO {table_sql} SELECT {projection} FROM ({relation})"


# How much of a JSON file is inspected to tell newline-delimited records from a single
# pretty-printed document (which a line-oriented reader would turn into NULL rows).
_JSON_SNIFF_BYTES = 65536
_JSON_SNIFF_LINES = 5

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


def _raw_json_relation(files: list[str], ref: str, strip_outer_array: bool = False) -> str:
    """DuckDB relation exposing each staged JSON record as a single ``$1`` value.

    With ``strip_outer_array`` each file is one JSON document whose outer array is
    exploded; otherwise the files are read as newline-delimited records.
    """
    if strip_outer_array:
        compressed = [
            path for path in files if path.lower().endswith(_COMPRESSION_EXTENSIONS)
        ]
        if compressed:
            raise StageError(
                "STRIP_OUTER_ARRAY is not supported for compressed staged files "
                f"({os.path.basename(compressed[0])})."
            )
        return (
            'SELECT UNNEST(CAST(decode(content) AS JSON[])) AS "$1" '
            f"FROM read_blob({_format_file_list(files)})"
        )
    _check_newline_delimited_json(files, ref)
    # `CAST` (not `TRY_CAST`): a malformed record must fail loudly rather than silently
    # becoming a NULL row. Blank lines are removed by the filter, which is evaluated
    # before the projection.
    return (
        'SELECT CAST(column0 AS JSON) AS "$1" '
        f"FROM read_csv({_format_file_list(files)}, header=false, quote='', "
        "delim='\\x01', columns={'column0': 'VARCHAR'}, auto_detect=false, "
        f"max_line_size={_MAX_RECORD_BYTES}) "
        "WHERE trim(column0) <> ''"
    )


def _copy_parquet_sql(table_sql: str, files: list[str]) -> str:
    return f"INSERT INTO {table_sql} SELECT * FROM read_parquet({_format_file_list(files)})"


def _format_file_list(files: list[str]) -> str:
    if len(files) == 1:
        return _sql_string(files[0])
    return "[" + ", ".join(_sql_string(f) for f in files) + "]"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _option_value_text(value_node: exp.Expression) -> str:
    """Return a scalar staged-file option's text, without re-escaping it.

    ``.sql()`` re-serializes a string literal with escapes (and sqlglot escapes a literal
    backslash as ``\\``), so round-tripping a value through ``.sql()`` mangles regex and
    path values. ``Literal.name`` is the decoded value already, so prefer it for literals
    and fall back to the serialized form for non-literal operands.
    """
    if isinstance(value_node, exp.Literal):
        return value_node.name
    return _unquote_literal(value_node.sql(dialect="snowflake"))


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
            fmt[key] = _option_value_text(value_node)
    return fmt
