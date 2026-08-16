"""Unit tests for stage command parsing (stages.py)."""

from __future__ import annotations

import pytest

from snowflake_emulator.stages import (
    StageError,
    match_stage_command,
    parse_get,
    parse_list,
    parse_put,
    parse_remove,
    parse_stage_ref,
)


def test_parse_stage_ref_user_stage():
    ref = parse_stage_ref("@~/data/sub")
    assert ref.kind == "user"
    assert ref.path == "data/sub"


def test_parse_stage_ref_bare_user_stage():
    ref = parse_stage_ref("@~")
    assert ref.kind == "user"
    assert ref.path == ""


def test_parse_stage_ref_named_stage():
    ref = parse_stage_ref("@my_stage/data.csv")
    assert ref.kind == "named"
    assert ref.name == "my_stage"
    assert ref.path == "data.csv"
    assert ref.database is None
    assert ref.schema is None


def test_parse_stage_ref_qualified_named_stage():
    ref = parse_stage_ref("@db.schema.stage/data")
    assert ref.kind == "named"
    assert (ref.database, ref.schema, ref.name) == ("db", "schema", "stage")
    assert ref.path == "data"


def test_parse_stage_ref_table_stage():
    ref = parse_stage_ref("@%my_table/path")
    assert ref.kind == "table"
    assert ref.name == "my_table"
    assert ref.path == "path"


def test_parse_stage_ref_internal_stage():
    ref = parse_stage_ref("@%/dir")
    assert ref.kind == "internal"
    assert ref.path == "dir"


def test_parse_stage_ref_requires_at():
    with pytest.raises(StageError):
        parse_stage_ref("~/data")


def test_match_stage_command():
    assert match_stage_command("PUT file:///tmp/x.csv @~") == "PUT"
    assert match_stage_command("  get @stage file:///tmp") == "GET"
    assert match_stage_command("LIST @~") == "LIST"
    assert match_stage_command("REMOVE @~") == "REMOVE"
    assert match_stage_command("SELECT 1") is None


def test_parse_put_basic():
    cmd = parse_put("PUT file:///tmp/data.csv @~/staged")
    assert cmd.files == ["file:///tmp/data.csv"]
    assert cmd.stage.kind == "user"
    assert cmd.stage.path == "staged"
    assert cmd.options == {}


def test_parse_put_multiple_files_and_options():
    cmd = parse_put(
        "PUT file:///a.csv file:///b.csv @my_stage PARALLEL=4 OVERWRITE=TRUE AUTO_COMPRESS=FALSE"
    )
    assert cmd.files == ["file:///a.csv", "file:///b.csv"]
    assert cmd.stage.name == "my_stage"
    assert cmd.options == {
        "PARALLEL": "4",
        "OVERWRITE": "TRUE",
        "AUTO_COMPRESS": "FALSE",
    }


def test_parse_put_requires_stage():
    with pytest.raises(StageError):
        parse_put("PUT file:///tmp/data.csv")


def test_parse_get():
    cmd = parse_get("GET @~/staged file:///tmp/out PATTERN='.*\\.csv'")
    assert cmd.stage.kind == "user"
    assert cmd.stage.path == "staged"
    assert cmd.local_dir == "/tmp/out"
    assert cmd.options["PATTERN"] == ".*\\.csv"


def test_parse_list_pattern():
    cmd = parse_list("LIST @~/staged PATTERN='.*data.*'")
    assert cmd.stage.path == "staged"
    assert cmd.pattern == ".*data.*"


def test_parse_list_no_pattern():
    cmd = parse_list("LIST @my_stage")
    assert cmd.pattern is None
    assert cmd.stage.name == "my_stage"


def test_parse_remove():
    cmd = parse_remove("REMOVE @~/staged/data.csv")
    assert cmd.stage.path == "staged/data.csv"
    assert cmd.pattern is None
