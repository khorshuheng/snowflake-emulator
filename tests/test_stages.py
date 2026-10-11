"""Unit tests for stage command parsing (stages.py)."""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from snowflake_emulator.stages import (
    StageError,
    _decode_options,
    copy_stage_source,
    has_stage_file_table,
    is_stage_file_table,
    match_stage_command,
    parse_get,
    parse_list,
    parse_put,
    parse_remove,
    parse_stage_file_options,
    parse_stage_ref,
    preprocess_stage_file_clauses,
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


def test_parse_stage_ref_rejects_relative_components():
    """`..` is not a stage name, and must not become a path into the filesystem."""
    for ref in ("@..", "@../data.csv", "@../../etc/passwd", "@db..stage/x", "@%.."):
        with pytest.raises(StageError):
            parse_stage_ref(ref)


def test_stage_manager_contained_rejects_escapes(tmp_path):
    """The containment helper is what keeps every stage path inside its stage root."""
    from snowflake_emulator.stages import StageManager

    root = tmp_path / "stages"
    stage = root / "named" / "DB" / "PUBLIC" / "s"
    stage.mkdir(parents=True)
    (root / "outside.csv").write_text("x\n")
    manager = StageManager(str(root))

    assert manager.contained(str(stage / "nested" / "f.csv"), str(stage)) == str(
        stage / "nested" / "f.csv"
    )
    assert manager.contained(str(stage), str(stage)) == str(stage)
    for escaped in (stage / ".." / "outside.csv", stage / ".." / ".." / "..", root):
        with pytest.raises(StageError):
            manager.contained(str(escaped), str(stage))


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


def test_parse_stage_options_tolerate_whitespace_around_equals():
    """Snowflake writes `PATTERN = '...'` / `PARALLEL = 4`; a whitespace split loses them."""
    assert parse_get("GET @~/s file:///tmp/o PATTERN = '.*[.]parquet'").options == {
        "PATTERN": ".*[.]parquet"
    }
    assert parse_get("GET @~/s file:///tmp/o PARALLEL = 4 PATTERN='.*x.*'").options == {
        "PATTERN": ".*x.*",
        "PARALLEL": "4",
    }
    assert parse_put("PUT file:///a.csv @~/s OVERWRITE = TRUE").options == {"OVERWRITE": "TRUE"}
    assert parse_put("PUT file:///a.csv @~/s OVERWRITE").options == {"OVERWRITE": "TRUE"}
    assert parse_put("PUT file:///a.csv @~/s PARALLEL = 8").options == {"PARALLEL": "8"}


def test_parse_stage_options_keep_quoted_values_with_spaces():
    assert parse_list("LIST @~/s PATTERN = '.*a b.*'").pattern == ".*a b.*"
    assert parse_list("LIST @~/s PATTERN='.*a b.*'").pattern == ".*a b.*"
    assert parse_remove("REMOVE @~/s PATTERN = '.*it''s.*'").pattern == ".*it's.*"
    assert parse_remove("REMOVE @~/s PATTERN = '.*[.]csv'").pattern == ".*[.]csv"


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


# -- Querying staged files (SELECT ... FROM @stage/file) ----------------------


def test_parse_stage_file_options_inline_format_and_pattern():
    options = parse_stage_file_options(
        "(FILE_FORMAT => (TYPE = CSV, FIELD_DELIMITER = '|', SKIP_HEADER = 1), "
        "PATTERN => '.*[.]csv')"
    )
    assert options == {
        "TYPE": "CSV",
        "FIELD_DELIMITER": "|",
        "SKIP_HEADER": "1",
        "PATTERN": ".*[.]csv",
    }


def test_parse_stage_file_options_named_format():
    assert parse_stage_file_options("(FILE_FORMAT => 'my_fmt')") == {"FORMAT_NAME": "my_fmt"}


def test_parse_stage_file_options_pattern_only():
    assert parse_stage_file_options("(PATTERN => 'part.*')") == {"PATTERN": "part.*"}


def test_parse_stage_file_options_pattern_keeps_backslash():
    r"""A regex escape such as ``\.`` must survive option parsing.

    sqlglot >= 30.21 decodes it to ``.``, which silently broadens the pattern (``.*.csv``
    also matches ``-csv``); the emulator pins the tokenizer so the backslash is kept.
    """
    assert parse_stage_file_options(r"(PATTERN => '.*\.csv')") == {"PATTERN": r".*\.csv"}


def test_parse_stage_file_options_empty_clause():
    assert parse_stage_file_options("()") == {}


def test_preprocess_stage_file_clauses_encodes_options():
    sql = "SELECT $1 FROM @s/f.csv (FILE_FORMAT => (TYPE = CSV, SKIP_HEADER = 1))"
    rewritten = preprocess_stage_file_clauses(sql)
    assert rewritten.startswith("SELECT $1 FROM @s/f.csv (FILE_FORMAT => '")
    # sqlglot rejects the original clause but parses the rewritten one.
    parsed = sqlglot.parse_one(rewritten, read="snowflake")
    marker = parsed.find(exp.Table).args["format"]
    assert marker.name.startswith("sfemu:staged-file-options:")
    assert _decode_options(marker.name) == {"TYPE": "CSV", "SKIP_HEADER": "1"}


def test_preprocess_stage_file_clauses_leaves_other_sql_alone():
    for sql in (
        "SELECT $1 FROM @s/f.csv",
        "COPY INTO t FROM @s FILE_FORMAT = (TYPE = CSV)",
        "COPY INTO t FROM @s (PATTERN => 'x')",
        "SELECT '@a (b' AS x FROM t",
    ):
        assert preprocess_stage_file_clauses(sql) == sql


def test_preprocess_stage_file_clauses_keeps_rest_of_statement():
    sql = "SELECT $1, x FROM t, @s/f.json (PATTERN => 'p') WHERE x > 1"
    rewritten = preprocess_stage_file_clauses(sql)
    assert rewritten.startswith("SELECT $1, x FROM t, @s/f.json (FILE_FORMAT => '")
    assert rewritten.endswith(" WHERE x > 1")


def test_is_stage_file_table_detects_from_clause_reference():
    parsed = sqlglot.parse_one("SELECT $1 FROM @s/f.csv", read="snowflake")
    assert has_stage_file_table(parsed)
    assert all(is_stage_file_table(table) for table in parsed.find_all(exp.Table))


def test_is_stage_file_table_ignores_copy_into_source():
    """COPY INTO names its source with a stage reference, but that is not a query."""
    parsed = sqlglot.parse_one("COPY INTO t FROM @s FILE_FORMAT = (TYPE = CSV)", read="snowflake")
    assert not has_stage_file_table(parsed)


def test_is_stage_file_table_ignores_plain_tables():
    parsed = sqlglot.parse_one("SELECT * FROM t", read="snowflake")
    assert not has_stage_file_table(parsed)


def test_parse_stage_file_options_rejects_garbage():
    with pytest.raises(StageError):
        parse_stage_file_options("(FILE_FORMAT => )")


def test_copy_stage_source_extracts_file_format_and_pattern():
    node = sqlglot.parse_one(
        "COPY INTO t FROM @stage FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1) "
        "PATTERN = '.*part.*'",
        read="snowflake",
    )
    table_sql, stage_ref, file_format, pattern = copy_stage_source(node)
    assert table_sql == "t"
    assert stage_ref == "@stage"
    assert file_format == {"TYPE": "CSV", "SKIP_HEADER": "1"}
    assert pattern == ".*part.*"


def test_copy_stage_source_without_pattern_or_stage():
    plain = sqlglot.parse_one("COPY INTO t FROM @stage", read="snowflake")
    assert copy_stage_source(plain)[3] is None
    assert copy_stage_source(sqlglot.parse_one("COPY INTO t FROM 's3://b/x'", read="snowflake")) is None
