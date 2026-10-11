"""User-defined functions: ``CREATE FUNCTION`` emulated as DuckDB macros.

A SQL UDF is translated into ``CREATE MACRO`` under an arity-mangled name, with the
Snowflake-side metadata recorded in the catalog's ``_sf_functions_ext`` side table. The
body is transpiled as Snowflake SQL (not passed through as text), so a body using a
Snowflake-only function such as ``EQUAL_NULL`` behaves as it would on Snowflake --
previously it silently created a macro whose value was the body *text*.
"""

from __future__ import annotations

import pytest

from snowflake_emulator.database import DuckDBManager
from snowflake_emulator.sessions import SessionContext
from snowflake_emulator.settings import settings
from snowflake_emulator.translator import split_statements, translate_create_function
from snowflake_emulator.user_functions import execute_create_function

CASE_INSENSITIVE_EQUAL = (
    "CREATE OR REPLACE FUNCTION EQUAL_NULL_CASE_INSENSITIVE(str STRING, str2 STRING) "
    "RETURNS BOOLEAN "
    "LANGUAGE SQL "
    "AS $$ EQUAL_NULL(UPPER(str), UPPER(str2)) $$"
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
    assert [row[6] for row in _data(client, "SHOW FUNCTIONS")] == [
        "F_OV(NUMBER(38,0)) RETURN NUMBER(38,0)",
        "F_OV(NUMBER(38,0), NUMBER(38,0)) RETURN NUMBER(38,0)",
    ]


def test_overloading_by_argument_type_is_rejected_loudly(client):
    # A DuckDB macro cannot be overloaded by name, so the same arity with a different
    # signature would silently resolve to whichever definition won. Reject instead.
    _post(client, "CREATE FUNCTION f_ty(a NUMBER, b NUMBER) RETURNS BOOLEAN LANGUAGE SQL AS $$ a = b $$")
    message = _error(
        client,
        "CREATE FUNCTION f_ty(a ARRAY, b ARRAY) RETURNS BOOLEAN LANGUAGE SQL "
        "AS $$ ARRAY_SIZE(a) = ARRAY_SIZE(b) $$",
    )
    assert "F_TY" in message and "cannot overload" in message


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
    assert [row[1] for row in _data(client, "SHOW FUNCTIONS")] == ["F_LIKE", "G_LIKE"]
    assert [row[1] for row in _data(client, "SHOW FUNCTIONS LIKE 'F%'")] == ["F_LIKE"]
    assert [
        row[1] for row in _data(client, "SHOW USER FUNCTIONS LIKE 'G%' IN EMULATOR_DB.PUBLIC")
    ] == ["G_LIKE"]


def test_show_functions_is_empty_without_user_functions(client):
    assert _data(client, "SHOW FUNCTIONS") == []
    assert _data(client, "SHOW USER FUNCTIONS IN EMULATOR_DB.PUBLIC") == []


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
    assert _post(client, "SHOW FUNCTIONS", headers).json()["data"] == []


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
