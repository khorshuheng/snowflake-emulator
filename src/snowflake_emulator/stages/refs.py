"""Stage references, command/DDL parsing and the shared stage error type."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlglot import exp


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


# A PUT/GET/LIST/REMOVE option: `KEY=value`, `KEY = value` (Snowflake's own spelling) or a
# bare flag. A quoted value may contain spaces, so options cannot be split on whitespace.
_OPTION_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s*=\s*"
    r"(?P<value>'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|[^\s,]+)"
    r")?"
)


def _parse_stage_options(text: str) -> dict[str, str]:
    """Parse stage-command options, tolerating whitespace around ``=``.

    Snowflake writes these as ``PATTERN = '<regex>'`` / ``PARALLEL = 4``, and a pattern may
    contain spaces, so a whitespace split would lose the option (or the pattern) rather than
    report it. A bare key is a flag, as ``OVERWRITE`` is for PUT.
    """
    options: dict[str, str] = {}
    for match in _OPTION_RE.finditer(text):
        value = match.group("value")
        options[match.group("key").upper()] = (
            "TRUE" if value is None else _unquote_literal(value)
        )
    return options


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

    parts = stage_and_opts.split(maxsplit=1)
    stage = parse_stage_ref(parts[0])
    options = _parse_stage_options(parts[1]) if len(parts) > 1 else {}
    return PutCommand(files=files, stage=stage, options=options)


def parse_get(sql: str) -> GetCommand:
    """Parse ``GET @<stage> file:///local/dir [PATTERN='...'] [PARALLEL=4]``."""
    rest = sql[3:].strip()
    parts = rest.split(maxsplit=2)
    if not parts:
        raise StageError("GET requires a stage location.")
    stage = parse_stage_ref(parts[0])
    local_dir = parts[1] if len(parts) > 1 else ""
    if local_dir.startswith("file://"):
        local_dir = local_dir[len("file://") :]
    options = _parse_stage_options(parts[2]) if len(parts) > 2 else {}
    return GetCommand(stage=stage, local_dir=local_dir, options=options)


def parse_list(sql: str) -> ListCommand:
    """Parse ``LIST @<stage> [PATTERN='...']``."""
    rest = sql[4:].strip()
    parts = rest.split(maxsplit=1)
    if not parts:
        raise StageError("LIST requires a stage location.")
    stage = parse_stage_ref(parts[0])
    options = _parse_stage_options(parts[1]) if len(parts) > 1 else {}
    return ListCommand(stage=stage, pattern=options.get("PATTERN"))


def parse_remove(sql: str) -> RemoveCommand:
    """Parse ``REMOVE @<stage> [PATTERN='...']``."""
    rest = sql[6:].strip()
    parts = rest.split(maxsplit=1)
    if not parts:
        raise StageError("REMOVE requires a stage location.")
    stage = parse_stage_ref(parts[0])
    options = _parse_stage_options(parts[1]) if len(parts) > 1 else {}
    return RemoveCommand(stage=stage, pattern=options.get("PATTERN"))
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
def _unquote_literal(value: str) -> str:
    """Strip the single pair of quotes enclosing a SQL literal, and unescape its quotes.

    Only one pair is removed (unlike ``str.strip``), so a value that *is* a quote character
    -- ``FIELD_OPTIONALLY_ENCLOSED_BY = '"'`` -- survives intact, while a doubled quote
    inside the literal (SQL's escape for one) collapses back to a single character.
    """
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1].replace(text[0] * 2, text[0])
    return text
def _compile_pattern(pattern: str | None) -> re.Pattern[str] | None:
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise StageError(f"Invalid PATTERN regex: {exc}") from exc

