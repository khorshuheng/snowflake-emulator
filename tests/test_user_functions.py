"""User-defined functions: ``CREATE FUNCTION`` emulated as DuckDB macros.

A SQL UDF is translated into ``CREATE MACRO`` under an arity-mangled name, with the
Snowflake-side metadata recorded in the catalog's ``_sf_functions_ext`` side table. The
body is transpiled as Snowflake SQL (not passed through as text), so a body using a
Snowflake-only function such as ``EQUAL_NULL`` behaves as it would on Snowflake --
previously it silently created a macro whose value was the body *text*.
"""

from __future__ import annotations

import json

import pytest

from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.settings import settings
from snowflake_emulator.translator import (
    INFORMATION_SCHEMA_SCHEMA,
    split_statements,
    translate_create_function,
)
from snowflake_emulator.user_functions import execute_create_function

CASE_INSENSITIVE_EQUAL = (
    "CREATE OR REPLACE FUNCTION EQUAL_NULL_CASE_INSENSITIVE(str STRING, str2 STRING) "
    "RETURNS BOOLEAN "
    "LANGUAGE SQL "
    "AS $$ EQUAL_NULL(UPPER(str), UPPER(str2)) $$"
)

# The same name and arity as CASE_INSENSITIVE_EQUAL, so the two overloads can only be
# told apart from their argument types.
CASE_INSENSITIVE_ARRAY_EQUAL = (
    "CREATE OR REPLACE FUNCTION EQUAL_NULL_CASE_INSENSITIVE(a ARRAY, b ARRAY) "
    "RETURNS BOOLEAN "
    "LANGUAGE SQL "
    "AS $$ EQUAL_NULL(TRANSFORM(a, x -> UPPER(x)), TRANSFORM(b, x -> UPPER(x))) $$"
)


def _post(client, statement, headers=None):
    return client.post("/api/v2/statements", json={"statement": statement}, headers=headers)


def _auth_headers(client) -> dict:
    """A bearer token, so several statements share one session.

    Without it every request gets its own session, so a ``USE SCHEMA`` would not apply
    to the statements that follow.
    """
    token = client.post("/session/token").json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _data(client, statement):
    resp = _post(client, statement)
    assert resp.status_code == 200, resp.json()
    return resp.json()["data"]


def _columns(client, statement):
    resp = _post(client, statement)
    assert resp.status_code == 200, resp.json()
    return [col["name"] for col in resp.json()["resultSetMetaData"]["rowType"]]


def _error(client, statement):
    resp = _post(client, statement)
    assert resp.status_code == 422, resp.json()
    return resp.json()["detail"]["message"]


def test_sql_udf_body_is_transpiled_not_passed_through(client):
    # ``EQUAL_NULL`` has no DuckDB equivalent; it only works if the body is translated
    # as Snowflake SQL. The old behaviour created a macro returning the body *text*.
    assert _post(client, CASE_INSENSITIVE_EQUAL).status_code == 200
    assert _data(
        client,
        "SELECT EQUAL_NULL_CASE_INSENSITIVE('aBc', 'ABC') AS eq, "
        "EQUAL_NULL_CASE_INSENSITIVE('abc', 'xyz') AS ne, "
        "EQUAL_NULL_CASE_INSENSITIVE(NULL, NULL) AS both_null, "
        "EQUAL_NULL_CASE_INSENSITIVE('a', NULL) AS one_null",
    ) == [[True, False, True, False]]


def test_sql_udf_is_usable_in_a_where_clause(client):
    _post(client, CASE_INSENSITIVE_EQUAL)
    assert _data(
        client,
        "SELECT a, b FROM (VALUES ('x', 'X'), ('y', 'z'), (NULL, NULL), (NULL, 'q')) "
        "AS t(a, b) WHERE EQUAL_NULL_CASE_INSENSITIVE(a, b)",
    ) == [["x", "X"], [None, None]]


def test_create_function_returns_snowflake_status(client):
    resp = _post(client, CASE_INSENSITIVE_EQUAL)
    assert resp.status_code == 200
    assert resp.json()["data"] == [
        ["Function EQUAL_NULL_CASE_INSENSITIVE successfully created."]
    ]


def test_create_or_replace_swaps_the_body(client):
    assert _post(
        client,
        "CREATE FUNCTION f_replace(x NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ x + 1 $$",
    ).status_code == 200
    assert _data(client, "SELECT F_REPLACE(1)") == [[2]]
    assert _post(
        client,
        "CREATE OR REPLACE FUNCTION f_replace(x NUMBER) RETURNS NUMBER LANGUAGE SQL "
        "AS $$ x * 10 $$",
    ).status_code == 200
    assert _data(client, "SELECT F_REPLACE(2)") == [[20]]


def test_duplicate_function_without_replace_is_rejected(client):
    _post(client, "CREATE FUNCTION f_dup(x NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ x $$")
    message = _error(
        client, "CREATE FUNCTION f_dup(x NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ x + 1 $$"
    )
    assert "F_DUP" in message and "already exists" in message


def test_if_not_exists_is_a_no_op(client):
    _post(client, "CREATE FUNCTION f_ine(x NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ x + 1 $$")
    resp = _post(
        client,
        "CREATE FUNCTION IF NOT EXISTS f_ine(x NUMBER) RETURNS NUMBER LANGUAGE SQL "
        "AS $$ x * 100 $$",
    )
    assert resp.status_code == 200
    # The original definition is kept, as on Snowflake.
    assert _data(client, "SELECT F_INE(1)") == [[2]]


def test_overloading_by_arity_is_supported(client):
    _post(client, "CREATE FUNCTION f_ov(a NUMBER, b NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ a + b $$")
    _post(client, "CREATE FUNCTION f_ov(a NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ a * 100 $$")
    assert _data(client, "SELECT F_OV(1, 2) AS two, F_OV(3) AS one") == [[3, 300]]
    # Each arity is recorded separately, so both show up in the catalog.
    assert [row[6] for row in _data(client, "SHOW USER FUNCTIONS")] == [
        "F_OV(NUMBER(38,0)) RETURN NUMBER(38,0)",
        "F_OV(NUMBER(38,0), NUMBER(38,0)) RETURN NUMBER(38,0)",
    ]


def test_overloading_by_argument_type_dispatches_on_the_value(client):
    """One name/arity, two argument types: each call reaches the matching body.

    A macro cannot express this -- every branch of a ``CASE`` over ``typeof()`` has to
    bind for both types, and ``upper(LIST)``/``list_reverse(VARCHAR)`` do not. So the
    overloads are dispatched from the runtime Python value type instead.
    """
    assert _post(client, CASE_INSENSITIVE_EQUAL).status_code == 200
    assert _post(client, CASE_INSENSITIVE_ARRAY_EQUAL).status_code == 200

    # Scalar strings use the STRING overload...
    assert _data(client, "SELECT EQUAL_NULL_CASE_INSENSITIVE('aBc', 'ABC')") == [[True]]
    assert _data(client, "SELECT EQUAL_NULL_CASE_INSENSITIVE('a', 'z')") == [[False]]
    # ...and arrays use the ARRAY overload, element-wise.
    assert _data(
        client,
        "SELECT EQUAL_NULL_CASE_INSENSITIVE(ARRAY_CONSTRUCT('a','B'), "
        "ARRAY_CONSTRUCT('A','b'))",
    ) == [[True]]
    assert _data(
        client,
        "SELECT EQUAL_NULL_CASE_INSENSITIVE(ARRAY_CONSTRUCT('a'), ARRAY_CONSTRUCT('z'))",
    ) == [[False]]


def test_both_overloads_appear_in_the_catalog(client):
    _post(client, CASE_INSENSITIVE_EQUAL)
    _post(client, CASE_INSENSITIVE_ARRAY_EQUAL)
    signatures = [row[6] for row in _data(client, "SHOW USER FUNCTIONS")]
    assert signatures == [
        "EQUAL_NULL_CASE_INSENSITIVE(ARRAY, ARRAY) RETURN BOOLEAN",
        "EQUAL_NULL_CASE_INSENSITIVE(VARCHAR, VARCHAR) RETURN BOOLEAN",
    ]
    assert len(_data(client, "DESCRIBE FUNCTION EQUAL_NULL_CASE_INSENSITIVE")) == 2


def test_null_arguments_reach_both_overloads(client):
    """A NULL carries no type, so it matches both overloads; they agree here.

    This is the case that makes NULL-safe functions work on NULL columns. It also covers
    the DuckDB null-handling setting: with the default, a NULL argument would skip the
    Python dispatcher entirely and the whole call would answer NULL rather than TRUE.
    """
    _post(client, CASE_INSENSITIVE_EQUAL)
    _post(client, CASE_INSENSITIVE_ARRAY_EQUAL)
    assert _data(client, "SELECT EQUAL_NULL_CASE_INSENSITIVE(NULL, NULL)") == [[True]]
    assert _data(client, "SELECT EQUAL_NULL_CASE_INSENSITIVE('a', NULL)") == [[False]]
    assert _data(
        client,
        "SELECT EQUAL_NULL_CASE_INSENSITIVE(ARRAY_CONSTRUCT('a'), NULL)",
    ) == [[False]]


def test_dispatched_function_works_over_table_columns(client):
    _post(client, CASE_INSENSITIVE_EQUAL)
    _post(client, CASE_INSENSITIVE_ARRAY_EQUAL)
    _post(client, "CREATE TABLE ci_t (id INT, s1 VARCHAR, s2 VARCHAR, a1 ARRAY, a2 ARRAY)")
    _post(
        client,
        "INSERT INTO ci_t VALUES "
        "(1, 'aB', 'Ab', ARRAY_CONSTRUCT('x'), ARRAY_CONSTRUCT('X')), "
        "(2, 'a', 'z', ARRAY_CONSTRUCT('x'), ARRAY_CONSTRUCT('Y')), "
        "(3, NULL, NULL, NULL, NULL)",
    )
    # Each row is dispatched separately, so one query mixes both overloads -- including
    # an all-NULL row, where the overloads agree and the call still answers TRUE.
    assert _data(
        client,
        "SELECT id, EQUAL_NULL_CASE_INSENSITIVE(s1, s2) AS s, "
        "EQUAL_NULL_CASE_INSENSITIVE(a1, a2) AS a FROM ci_t ORDER BY id",
    ) == [[1, True, True], [2, False, False], [3, True, True]]


def test_unmatchable_arguments_are_reported(client):
    _post(client, CASE_INSENSITIVE_EQUAL)
    _post(client, CASE_INSENSITIVE_ARRAY_EQUAL)
    message = _error(client, "SELECT EQUAL_NULL_CASE_INSENSITIVE(1, 2)")
    # The passed values are described by category, the overloads by declared type, so the
    # two halves of the message cannot be confused for each other.
    assert "no overload matching (number, number)" in message
    assert "ARRAY, ARRAY" in message and "VARCHAR, VARCHAR" in message


def test_overloads_must_share_a_return_type(client):
    # The dispatcher is one DuckDB function, which needs a single declared return type
    # (create_function has no ANY), so differing returns cannot be dispatched.
    _post(
        client,
        "CREATE FUNCTION f_ret(a NUMBER, b NUMBER) RETURNS BOOLEAN LANGUAGE SQL AS $$ a = b $$",
    )
    message = _error(
        client,
        "CREATE FUNCTION f_ret(a ARRAY, b ARRAY) RETURNS NUMBER LANGUAGE SQL "
        "AS $$ ARRAY_SIZE(a) $$",
    )
    assert "must share a return type" in message
    assert "BOOLEAN" in message and "NUMBER(38,0)" in message


def test_udf_parameters_are_coerced_to_their_declared_type(client):
    """A UDF call coerces each argument to the parameter's declared type, as Snowflake does.

    A DuckDB macro is untyped, so the body used to see the caller's type: a ``VARCHAR``
    parameter received a DATE from ``shout(CURRENT_DATE())`` and ``UPPER`` failed with a binder
    error. The parameter reference is now cast inside the macro body, and a NUMBER cell comes
    back as a JSON number rather than the string a Decimal serializes to.
    """
    import datetime

    def created(statement):
        resp = _post(client, statement)
        assert resp.status_code == 200, resp.json()
        return resp

    created(
        "CREATE OR REPLACE FUNCTION shout(x VARCHAR) RETURNS VARCHAR AS $$ UPPER(x) $$"
    )
    assert _data(client, "SELECT shout('ada')") == [["ADA"]]
    assert _data(client, "SELECT shout(42)") == [["42"]]
    assert _data(client, "SELECT shout(CURRENT_DATE())") == [
        [datetime.date.today().isoformat()]
    ]

    # A NUMBER parameter keeps Snowflake's NUMBER(38,0) coercion (a fraction rounds), and the
    # result is a JSON number.
    created("CREATE OR REPLACE FUNCTION trunc_num(n NUMBER) RETURNS NUMBER AS $$ n $$")
    truncated = _data(client, "SELECT trunc_num(3.7)")
    assert truncated == [[4]]
    assert isinstance(truncated[0][0], int)

    created("CREATE OR REPLACE FUNCTION twice_num(n NUMBER) RETURNS NUMBER AS $$ n * 2 $$")
    assert _data(client, "SELECT twice_num(3)") == [[6]]

    # A type with no usable DuckDB spelling (a bare ARRAY) still creates: it is not coerced.
    created(
        "CREATE OR REPLACE FUNCTION arr_len(a ARRAY) RETURNS NUMBER AS $$ ARRAY_SIZE(a) $$"
    )
    assert _data(client, "SELECT arr_len(ARRAY_CONSTRUCT('a','b'))") == [[2]]


def test_udf_text_parameters_unify_the_call_temporal_types(client):
    """Text parameters are unified with their siblings before becoming strings.

    Snowflake resolves a call's argument types together: a DATE next to a TIMESTAMP is
    promoted to a TIMESTAMP (midnight), and a string next to a DATE is read as a DATE.
    Casting each parameter to text independently would give two different strings for the
    same instant.
    """
    def created(statement):
        resp = _post(client, statement)
        assert resp.status_code == 200, resp.json()

    created(
        "CREATE OR REPLACE FUNCTION date_args_equal(a VARCHAR, b VARCHAR) "
        "RETURNS BOOLEAN AS $$ a = b $$"
    )
    # The string is read as a DATE, matching both date spellings.
    assert _data(
        client, "SELECT date_args_equal(DATE '2026-09-30', '2026-09-30 00:00:00.000')"
    ) == [[True]]
    assert _data(
        client, "SELECT date_args_equal('2026-09-30 00:00', DATE '2026-09-30')"
    ) == [[True]]
    # The DATE is promoted to midnight TIMESTAMP, matching the timestamp argument.
    assert _data(
        client,
        "SELECT date_args_equal(DATE '2026-09-30', TIMESTAMP '2026-09-30 00:00:00')",
    ) == [[True]]
    assert _data(
        client,
        "SELECT date_args_equal('2026-09-30 00:00:00', "
        "TIMESTAMP '2026-09-30 00:00:00')",
    ) == [[True]]

    created(
        "CREATE OR REPLACE FUNCTION date_args_text(a VARCHAR, b VARCHAR) "
        "RETURNS VARCHAR AS $$ a || '|' || b $$"
    )
    assert _data(
        client,
        "SELECT date_args_text(DATE '2026-09-30', '2026-09-30 00:00:00.000')",
    ) == [["2026-09-30|2026-09-30"]]
    assert _data(
        client,
        "SELECT date_args_text(DATE '2026-09-30', TIMESTAMP '2026-09-30 00:00:00')",
    ) == [["2026-09-30 00:00:00.000|2026-09-30 00:00:00.000"]]
    # The common timestamp follows Snowflake's default FF3 text format.
    assert _data(
        client,
        "SELECT date_args_text(DATE '2026-09-30', "
        "TIMESTAMP '2026-09-30 12:34:56.123456')",
    ) == [["2026-09-30 00:00:00.000|2026-09-30 12:34:56.123"]]
    # A timezone-aware sibling keeps its offset in Snowflake's TZHTZM form.
    assert _data(
        client,
        "SELECT date_args_text(DATE '2026-09-30', "
        "TIMESTAMP_TZ '2026-09-30 00:00:00 UTC')",
    ) == [["2026-09-30 00:00:00.000 +0000|2026-09-30 00:00:00.000 +0000"]]


def test_udf_array_parameter_is_not_treated_as_text(client):
    """An array parameter beside a temporal argument keeps its array type."""
    resp = _post(
        client,
        "CREATE OR REPLACE FUNCTION arr_date(a VARCHAR[], b VARCHAR) RETURNS NUMBER "
        "AS $$ ARRAY_SIZE(a) + LENGTH(b) $$",
    )
    assert resp.status_code == 200, resp.json()
    assert _data(
        client, "SELECT arr_date(ARRAY_CONSTRUCT('x','y'), DATE '2026-09-30')"
    ) == [[12]]
    assert _data(client, "SELECT arr_date(ARRAY_CONSTRUCT('x','y'), 'abc')") == [[5]]


def test_dropping_one_overload_leaves_the_other_callable(client):
    _post(client, CASE_INSENSITIVE_EQUAL)
    _post(client, CASE_INSENSITIVE_ARRAY_EQUAL)
    assert _post(
        client, "DROP FUNCTION EQUAL_NULL_CASE_INSENSITIVE(ARRAY, ARRAY)"
    ).status_code == 200
    # The signature picked out exactly one overload: the STRING one still resolves...
    assert _data(client, "SELECT EQUAL_NULL_CASE_INSENSITIVE('aB', 'Ab')") == [[True]]
    # ...and the dropped ARRAY overload is gone, so an array argument now resolves to the
    # remaining STRING overload by coercion (Snowflake casts an ARRAY to VARCHAR for a STRING
    # parameter), rather than reaching the dropped ARRAY body.
    assert _data(
        client,
        "SELECT EQUAL_NULL_CASE_INSENSITIVE(ARRAY_CONSTRUCT('a'), ARRAY_CONSTRUCT('A'))",
    ) == [[True]]
    assert len(_data(client, "SHOW USER FUNCTIONS")) == 1


def test_replacing_one_overload_keeps_the_other(client):
    _post(client, CASE_INSENSITIVE_EQUAL)
    _post(client, CASE_INSENSITIVE_ARRAY_EQUAL)
    # Same signature, new body: replaces just that overload.
    assert _post(
        client,
        "CREATE OR REPLACE FUNCTION EQUAL_NULL_CASE_INSENSITIVE(a ARRAY, b ARRAY) "
        "RETURNS BOOLEAN LANGUAGE SQL AS $$ FALSE $$",
    ).status_code == 200
    assert _data(
        client,
        "SELECT EQUAL_NULL_CASE_INSENSITIVE(ARRAY_CONSTRUCT('a'), ARRAY_CONSTRUCT('A'))",
    ) == [[False]]
    assert _data(client, "SELECT EQUAL_NULL_CASE_INSENSITIVE('aB', 'Ab')") == [[True]]
    assert len(_data(client, "SHOW USER FUNCTIONS")) == 2


def test_array_typed_arguments_are_supported(client):
    assert _post(
        client,
        "CREATE FUNCTION arr_total(arr ARRAY) RETURNS NUMBER LANGUAGE SQL AS $$ ARRAY_SIZE(arr) $$",
    ).status_code == 200
    assert _data(client, "SELECT ARR_TOTAL(ARRAY_CONSTRUCT(1, 2, 3))") == [[3]]


def test_select_form_body_and_zero_arguments(client):
    # Snowflake allows a single-column SELECT as the body of a scalar UDF.
    assert _post(
        client,
        "CREATE FUNCTION f_sel(x NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ SELECT x + 1 $$",
    ).status_code == 200
    assert _data(client, "SELECT F_SEL(41)") == [[42]]

    assert _post(
        client, "CREATE FUNCTION f_none() RETURNS NUMBER LANGUAGE SQL AS $$ 42 $$"
    ).status_code == 200
    assert _data(client, "SELECT F_NONE()") == [[42]]


def test_language_sql_is_optional_and_case_insensitive(client):
    assert _post(
        client, "CREATE FUNCTION f_nolang(x NUMBER) RETURNS NUMBER AS $$ x + 5 $$"
    ).status_code == 200
    assert _data(client, "SELECT F_NOLANG(1)") == [[6]]


def test_non_sql_language_is_still_rejected(client):
    message = _error(
        client,
        "CREATE FUNCTION f_py(x NUMBER) RETURNS NUMBER LANGUAGE PYTHON AS $$ return x $$",
    )
    assert "PYTHON" in message and "not supported" in message


def test_table_returning_function_is_rejected(client):
    message = _error(
        client,
        "CREATE FUNCTION f_tab(x NUMBER) RETURNS TABLE (a NUMBER) LANGUAGE SQL "
        "AS $$ SELECT x AS a $$",
    )
    assert "RETURNS TABLE" in message


def test_multi_column_select_body_is_rejected(client):
    message = _error(
        client,
        "CREATE FUNCTION f_multi(x NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ SELECT x, x $$",
    )
    assert "more than one column" in message


def test_missing_returns_clause_is_rejected(client):
    resp = _post(client, "CREATE FUNCTION f_noret(x NUMBER) LANGUAGE SQL AS $$ x $$")
    assert resp.status_code == 422


def test_drop_function_with_and_without_signature(client):
    _post(client, "CREATE FUNCTION f_drop(a NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ a $$")
    resp = _post(client, "DROP FUNCTION f_drop(NUMBER)")
    assert resp.json()["data"] == [["Function F_DROP successfully dropped."]]
    # The macro is gone too, so the call fails rather than resolving a stale function.
    assert _post(client, "SELECT F_DROP(1)").status_code == 422


def test_drop_function_without_signature_drops_every_overload(client):
    _post(client, "CREATE FUNCTION f_all(a NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ a $$")
    _post(client, "CREATE FUNCTION f_all(a NUMBER, b NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ a $$")
    assert _post(client, "DROP FUNCTION f_all").status_code == 200
    assert _post(client, "SELECT F_ALL(1)").status_code == 422
    assert _post(client, "SELECT F_ALL(1, 2)").status_code == 422


def test_drop_missing_function_respects_if_exists(client):
    message = _error(client, "DROP FUNCTION f_absent(NUMBER)")
    assert "does not exist" in message
    assert _post(client, "DROP FUNCTION IF EXISTS f_absent(NUMBER)").status_code == 200


def test_show_user_functions_lists_recorded_functions(client):
    _post(client, CASE_INSENSITIVE_EQUAL)
    resp = _post(client, "SHOW USER FUNCTIONS IN EMULATOR_DB.PUBLIC")
    assert resp.status_code == 200
    row = resp.json()["data"][0]
    names = [col["name"] for col in resp.json()["resultSetMetaData"]["rowType"]]
    assert names == [
        "created_on",
        "name",
        "schema_name",
        "catalog_name",
        "is_builtin",
        "is_secure",
        "arguments",
        "result_type",
        "description",
    ]
    assert row[1] == "EQUAL_NULL_CASE_INSENSITIVE"
    assert row[2] == "PUBLIC" and row[3] == "EMULATOR_DB"
    assert row[4] == "N"  # is_builtin
    assert row[6] == "EQUAL_NULL_CASE_INSENSITIVE(VARCHAR, VARCHAR) RETURN BOOLEAN"
    assert row[7] == "BOOLEAN"


def test_show_functions_and_like_filter(client):
    _post(client, "CREATE FUNCTION f_like(x NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ x $$")
    _post(client, "CREATE FUNCTION g_like(x NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ x $$")
    assert [row[1] for row in _data(client, "SHOW USER FUNCTIONS")] == ["F_LIKE", "G_LIKE"]
    assert [row[1] for row in _data(client, "SHOW USER FUNCTIONS LIKE 'F%'")] == ["F_LIKE"]
    assert [
        row[1] for row in _data(client, "SHOW USER FUNCTIONS LIKE 'G%' IN EMULATOR_DB.PUBLIC")
    ] == ["G_LIKE"]
    # SHOW FUNCTIONS lists built-ins as well (docs), and LIKE filters them too.
    rows = _data(client, "SHOW FUNCTIONS LIKE 'UPPER'")
    assert rows == [
        [None, "UPPER", None, None, "Y", "N", "(VARCHAR)", "VARCHAR",
         "Converts to upper case."]
    ]


def test_show_functions_lists_builtins_without_user_functions(client):
    assert _data(client, "SHOW USER FUNCTIONS IN EMULATOR_DB.PUBLIC") == []
    # SHOW FUNCTIONS includes the built-in catalogue (docs), so it is never empty.
    builtins = [row for row in _data(client, "SHOW FUNCTIONS") if row[4] == "Y"]
    assert builtins
    assert {row[1] for row in builtins} >= {"UPPER", "PARSE_JSON", "FLATTEN"}


def test_describe_function_by_name_and_signature(client):
    _post(client, CASE_INSENSITIVE_EQUAL)
    resp = _post(client, "DESCRIBE FUNCTION EQUAL_NULL_CASE_INSENSITIVE")
    assert resp.status_code == 200
    row = resp.json()["data"][0]
    idx = {col["name"]: i for i, col in enumerate(resp.json()["resultSetMetaData"]["rowType"])}
    assert row[idx["name"]] == "EQUAL_NULL_CASE_INSENSITIVE"
    assert row[idx["min_num_arguments"]] == 2
    assert row[idx["max_num_arguments"]] == 2
    assert row[idx["arguments"]] == (
        "EQUAL_NULL_CASE_INSENSITIVE(STR VARCHAR, STR2 VARCHAR) RETURN BOOLEAN"
    )
    assert row[idx["language"]] == "SQL"
    assert "EQUAL_NULL" in row[idx["body"]]

    # The signature form resolves the same function.
    by_signature = _post(
        client, "DESCRIBE FUNCTION EQUAL_NULL_CASE_INSENSITIVE(STRING, STRING)"
    )
    assert by_signature.json()["data"] == resp.json()["data"]


def test_describe_function_lists_every_overload(client):
    _post(client, "CREATE FUNCTION f_desc(a NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ a $$")
    _post(client, "CREATE FUNCTION f_desc(a NUMBER, b NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ a $$")
    rows = _data(client, "DESCRIBE FUNCTION f_desc")
    assert [row[7] for row in rows] == [1, 2]
    assert len(_data(client, "DESCRIBE FUNCTION f_desc(NUMBER, NUMBER)")) == 1


def test_describe_missing_function_errors(client):
    message = _error(client, "DESCRIBE FUNCTION f_no_such")
    assert "does not exist" in message


def test_overloaded_function_works_on_array_columns(client):
    """An ARRAY column is stored as DuckDB JSON, which reaches the dispatcher as text.

    Two things have to hold for this to work: the dispatcher must be told each argument's
    *SQL* type (the value alone cannot say), and a JSON argument must be presented to the
    body the same way as an ``ARRAY_CONSTRUCT`` literal, or a comparison of the two fails
    as ``JSON[] = VARCHAR[]``.
    """
    _post(client, CASE_INSENSITIVE_EQUAL)
    _post(client, CASE_INSENSITIVE_ARRAY_EQUAL)
    _post(client, "CREATE TABLE arr_t (id INT, a1 ARRAY, a2 ARRAY)")
    _post(
        client,
        "INSERT INTO arr_t VALUES "
        "(1, ARRAY_CONSTRUCT('a','B'), ARRAY_CONSTRUCT('A','b')), "
        "(2, ARRAY_CONSTRUCT('a'), ARRAY_CONSTRUCT('z')), "
        "(3, NULL, NULL)",
    )
    # Row 3 matches: a NULL in an ARRAY column still has the column's type, so the array
    # overload is chosen and both sides are NULL.
    assert _data(
        client, "SELECT id FROM arr_t WHERE EQUAL_NULL_CASE_INSENSITIVE(a1, a2) ORDER BY id"
    ) == [[1], [3]]
    # A JSON column compared against a plain array literal: both sides must normalize to
    # the same representation.
    assert _data(
        client,
        "SELECT EQUAL_NULL_CASE_INSENSITIVE(a1, ARRAY_CONSTRUCT('a','B')) FROM arr_t "
        "WHERE id = 1",
    ) == [[True]]


def test_function_is_scoped_to_its_schema(client):
    headers = _auth_headers(client)
    _post(client, "CREATE SCHEMA other_s", headers)
    _post(
        client,
        "CREATE FUNCTION other_s.f_scoped(x NUMBER) RETURNS NUMBER LANGUAGE SQL AS $$ x + 1 $$",
        headers,
    )
    # Resolvable while OTHER_S is the current schema...
    assert _post(client, "USE SCHEMA other_s", headers).status_code == 200
    assert _post(client, "SELECT F_SCOPED(1)", headers).json()["data"] == [[2]]
    # ...and not from PUBLIC, where it was never created.
    assert _post(client, "USE SCHEMA PUBLIC", headers).status_code == 200
    assert _post(client, "SELECT F_SCOPED(1)", headers).status_code == 422
    assert _post(client, "SHOW USER FUNCTIONS", headers).json()["data"] == []


def test_unknown_function_call_still_reports_a_duckdb_error(client):
    # A call to a function that is neither built-in nor a UDF must not be rewritten.
    message = _error(client, "SELECT NOT_A_REAL_FUNCTION(1)")
    assert "NOT_A_REAL_FUNCTION" in message


@pytest.fixture()
def persistent_udf_db(tmp_path):
    """Point persistence at a scratch file, restoring the settings afterwards."""
    saved_path = settings.persistence.database_path
    saved_isolation = settings.persistence.isolation
    settings.persistence.database_path = str(tmp_path / "emu.duckdb")
    yield
    settings.persistence.database_path = saved_path
    settings.persistence.isolation = saved_isolation


def test_udf_macro_and_metadata_survive_a_restart(persistent_udf_db):
    """A file-backed UDF must come back both as a macro and as catalog metadata.

    The macro itself lives in the DuckDB catalog, but the Snowflake-side signature is
    only in the emulator's side table; if that were in-memory only, a restart would
    leave a macro that no call site could ever be rewritten to.
    """
    path = settings.persistence.database_path
    session = SessionContext(token="persistence-test")
    manager = DuckDBManager(path)
    manager.ensure_namespace(session.database, session.schema)
    spec = translate_create_function(split_statements(CASE_INSENSITIVE_EQUAL)[0])
    execute_create_function(manager, manager.cursor(), session, spec)
    manager.connection.close()

    reopened = DuckDBManager(path)
    arities = reopened.function_arities(session.database, session.schema)
    assert arities["EQUAL_NULL_CASE_INSENSITIVE"][2] == spec.macro_name
    # The macro survived as well, so the recorded spelling is still resolvable. It is
    # qualified because a fresh connection has no search path into the catalog yet.
    assert reopened.connection.execute(
        f'SELECT "{session.database}"."{session.schema}"."{spec.macro_name}"('
        "'abc', 'ABC')"
    ).fetchall() == [(True,)]
    reopened.connection.close()


def test_overloaded_function_survives_a_restart(persistent_udf_db, client):
    """A dispatched overload pair must be callable again in a fresh process.

    The sub-macros live in the DuckDB catalog, but the Python dispatcher does not, so it
    has to be re-registered from the side table before the first statement resolves it.
    """
    import snowflake_emulator.database as database_module

    assert _post(client, CASE_INSENSITIVE_EQUAL).status_code == 200
    assert _post(client, CASE_INSENSITIVE_ARRAY_EQUAL).status_code == 200
    assert _data(client, "SELECT EQUAL_NULL_CASE_INSENSITIVE('aB', 'Ab')") == [[True]]

    # Drop the manager as a restart would, keeping the file behind it. The old connection
    # has to be closed first, or the new one cannot attach the same catalog file.
    database_module.get_manager().connection.close()
    database_module._manager = None

    assert _data(client, "SELECT EQUAL_NULL_CASE_INSENSITIVE('aB', 'Ab')") == [[True]]
    assert _data(
        client,
        "SELECT EQUAL_NULL_CASE_INSENSITIVE(ARRAY_CONSTRUCT('a'), ARRAY_CONSTRUCT('A'))",
    ) == [[True]]


def test_metadata_table_from_before_overloading_is_migrated(persistent_udf_db):
    """A side table written by the pre-overloading version is upgraded in place.

    That shape had no ``signature`` column, so one name/arity could not hold two
    overloads; the migration derives the signature from the stored JSON argument list.
    """
    path = settings.persistence.database_path
    session = SessionContext(token="migration-test")
    manager = DuckDBManager(path)
    manager.ensure_namespace(session.database, session.schema)
    ext = f'"{session.database}"."{INFORMATION_SCHEMA_SCHEMA}"'
    manager.connection.execute(f"DROP TABLE {ext}._sf_functions_ext")
    manager.connection.execute(
        f"""
        CREATE TABLE {ext}._sf_functions_ext (
            function_catalog VARCHAR,
            function_schema VARCHAR,
            function_name VARCHAR,
            num_arguments INTEGER,
            macro_name VARCHAR,
            arguments VARCHAR,
            return_type VARCHAR,
            language VARCHAR,
            body VARCHAR,
            created_on TIMESTAMP,
            PRIMARY KEY (function_catalog, function_schema, function_name, num_arguments)
        )
        """
    )
    manager.connection.execute(
        f"INSERT INTO {ext}._sf_functions_ext VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, now())",
        [
            session.database,
            session.schema,
            "F_OLD",
            2,
            "__SFUDF_F_OLD__2",
            json.dumps([["A", "VARCHAR"], ["B", "VARCHAR"]]),
            "BOOLEAN",
            "SQL",
            " EQUAL_NULL(UPPER(a), UPPER(b)) ",
        ],
    )
    manager.connection.close()

    reopened = DuckDBManager(path)
    rows = reopened.query_functions(session.database, session.schema)
    assert [(row[0], row[4], row[5], row[6]) for row in rows] == [
        ("F_OLD", "VARCHAR|VARCHAR", "__SFUDF_F_OLD__2", None)
    ]
    # The DuckDB return type was not recorded by the old shape, so promoting it to a
    # dispatcher reports the gap instead of guessing a type.
    assert rows[0][9] is None
    reopened.connection.close()
