"""Unit tests for wire-protocol bind-parameter decoding."""

from __future__ import annotations

import datetime

import pytest

from snowflake_emulator.bindings import decode_bindings, from_binding


def test_fixed_decodes_to_int():
    assert from_binding({"type": "FIXED", "value": "42"}) == 42
    assert from_binding({"type": "FIXED", "value": 7}) == 7


def test_boolean_decodes():
    assert from_binding({"type": "BOOLEAN", "value": "true"}) is True
    assert from_binding({"type": "BOOLEAN", "value": "FALSE"}) is False


def test_binary_decodes_from_hex():
    assert from_binding({"type": "BINARY", "value": "6869"}) == b"hi"


def test_date_decodes_from_epoch_millis():
    millis = int(
        datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000
    )
    assert from_binding({"type": "DATE", "value": str(millis)}) == datetime.date(2024, 1, 1)


def test_variant_value_passes_through():
    assert from_binding({"type": "VARIANT", "value": {"a": 1}}) == {"a": 1}


def test_none_value_stays_none():
    assert from_binding({"type": "FIXED", "value": None}) is None


def test_positional_bindings_are_ordered_by_index():
    decoded = decode_bindings(
        {
            "2": {"type": "TEXT", "value": "x"},
            "1": {"type": "FIXED", "value": "1"},
        }
    )
    assert decoded.positional == [1, "x"]
    assert decoded.named is None
    assert decoded.batch is None


def test_named_bindings_use_keys():
    decoded = decode_bindings({"a": {"type": "FIXED", "value": "1"}})
    assert decoded.named == {"a": 1}
    assert decoded.positional is None


def test_array_binding_is_transposed_into_rows():
    decoded = decode_bindings(
        {
            "1": {"type": "FIXED", "value": ["1", "2"]},
            "2": {"type": "TEXT", "value": ["a", "b"]},
        }
    )
    assert decoded.batch == [(1, "a"), (2, "b")]


def test_array_binding_with_mismatched_columns_raises():
    with pytest.raises(ValueError, match="mismatched lengths"):
        decode_bindings(
            {
                "1": {"type": "FIXED", "value": ["1", "2"]},
                "2": {"type": "TEXT", "value": ["a"]},
            }
        )


def test_no_bindings_is_empty():
    decoded = decode_bindings(None)
    assert decoded.positional is None and decoded.named is None and decoded.batch is None


def test_named_binding_via_wire_protocol(client):
    """A named (non-numeric-keyed) binding is passed to DuckDB as a named parameter."""
    import base64

    import pyarrow as pa

    token = client.post("/session/token").json()["token"]
    body = client.post(
        "/queries/v1/query-request",
        json={"sqlText": "SELECT $a AS x", "bindings": {"a": {"type": "FIXED", "value": "1"}}},
        headers={"Authorization": f"Bearer {token}"},
    ).json()
    assert body["success"] is True, body
    table = pa.ipc.open_stream(
        pa.BufferReader(base64.b64decode(body["data"]["rowsetBase64"]))
    ).read_all()
    assert [str(value) for value in table.column(0).to_pylist()] == ["1"]


def test_timestamp_ntz_binding_is_naive_wall_clock():
    ns = int(
        datetime.datetime(2024, 1, 1, 10, 20, 30, tzinfo=datetime.timezone.utc).timestamp()
        * 1e9
    )
    decoded = from_binding({"type": "TIMESTAMP_NTZ", "value": str(ns)})
    assert decoded == datetime.datetime(2024, 1, 1, 10, 20, 30)
    assert decoded.tzinfo is None


def test_semi_structured_list_is_a_single_value():
    """A VARIANT/ARRAY list is one value, not an executemany batch."""
    decoded = decode_bindings({"1": {"type": "VARIANT", "value": [1, 2, 3]}})
    assert decoded.positional == [[1, 2, 3]]
    assert decoded.batch is None


def test_array_binding_with_a_non_list_column_is_rejected():
    with pytest.raises(ValueError, match="list value for every column"):
        decode_bindings(
            {
                "1": {"type": "TEXT", "value": "xy"},
                "2": {"type": "FIXED", "value": ["1", "2"]},
            }
        )


def test_malformed_array_binding_is_not_a_500(client):
    bad = {"1": {"type": "FIXED", "value": ["1", "2"]}, "2": {"type": "TEXT", "value": ["a"]}}
    resp = client.post(
        "/api/v2/statements", json={"statement": "SELECT ?", "bindings": bad}
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"]


def test_bad_binary_binding_is_not_a_500(client):
    token = client.post("/session/token").json()["token"]
    resp = client.post(
        "/queries/v1/query-request",
        json={"sqlText": "SELECT ? AS x", "bindings": {"1": {"type": "BINARY", "value": "zz"}}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is False


def test_variant_list_binding_inserts_one_row(client):
    token = client.post("/session/token").json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post(
        "/queries/v1/query-request",
        json={"sqlText": "CREATE TABLE vlist (data VARIANT)"},
        headers=headers,
    ).json()["success"]
    assert client.post(
        "/queries/v1/query-request",
        json={"sqlText": "INSERT INTO vlist VALUES (?)", "bindings": {"1": {"type": "VARIANT", "value": [1, 2, 3]}}},
        headers=headers,
    ).json()["success"]
    sel = client.post("/api/v2/statements", json={"statement": "SELECT COUNT(*) AS n FROM vlist"})
    assert sel.json()["data"] == [[1]]


def test_merge_bindings_split_actions(client):
    """A positional binding on a MERGE runs its per-action count previews with the values."""
    assert client.post(
        "/api/v2/statements", json={"statement": "CREATE TABLE bind_merge (id INT, v INT)"}
    ).status_code == 200
    assert client.post(
        "/api/v2/statements", json={"statement": "INSERT INTO bind_merge VALUES (1, 10)"}
    ).status_code == 200

    resp = client.post(
        "/api/v2/statements",
        json={
            "statement": (
                "MERGE INTO bind_merge t USING (SELECT ? AS id, ? AS v) s ON t.id = s.id "
                "WHEN MATCHED THEN UPDATE SET t.v = s.v "
                "WHEN NOT MATCHED THEN INSERT (id, v) VALUES (s.id, s.v)"
            ),
            "bindings": {
                "1": {"type": "FIXED", "value": "1"},
                "2": {"type": "FIXED", "value": "77"},
            },
        },
    )
    assert resp.status_code == 200, resp.json()
    assert [col["name"] for col in resp.json()["resultSetMetaData"]["rowType"]] == [
        "number of rows inserted",
        "number of rows updated",
    ]
    assert resp.json()["data"] == [[0, 1]]
