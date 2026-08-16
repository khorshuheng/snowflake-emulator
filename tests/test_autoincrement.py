"""Tests for Snowflake AUTOINCREMENT / IDENTITY column support."""

from __future__ import annotations


def test_autoincrement_generates_sequential_ids(client):
    client.post(
        "/api/v2/statements",
        json={"statement": "CREATE TABLE auto_t (id INT AUTOINCREMENT, name VARCHAR)"},
    )
    client.post(
        "/api/v2/statements",
        json={"statement": "INSERT INTO auto_t (name) VALUES ('a')"},
    )
    client.post(
        "/api/v2/statements",
        json={"statement": "INSERT INTO auto_t (name) VALUES ('b')"},
    )

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT * FROM auto_t ORDER BY id"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == [[1, "a"], [2, "b"]]


def test_autoincrement_with_explicit_start_and_increment(client):
    client.post(
        "/api/v2/statements",
        json={
            "statement": (
                "CREATE TABLE auto_bounds_t "
                "(id INT AUTOINCREMENT START 5 INCREMENT 2, name VARCHAR)"
            )
        },
    )
    client.post(
        "/api/v2/statements",
        json={"statement": "INSERT INTO auto_bounds_t (name) VALUES ('a')"},
    )
    client.post(
        "/api/v2/statements",
        json={"statement": "INSERT INTO auto_bounds_t (name) VALUES ('b')"},
    )

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT * FROM auto_bounds_t ORDER BY id"},
    )
    assert resp.json()["data"] == [[5, "a"], [7, "b"]]


def test_identity_syntax_generates_sequential_ids(client):
    client.post(
        "/api/v2/statements",
        json={"statement": "CREATE TABLE identity_t (id INT IDENTITY(10, 5), name VARCHAR)"},
    )
    client.post(
        "/api/v2/statements",
        json={"statement": "INSERT INTO identity_t (name) VALUES ('a')"},
    )
    client.post(
        "/api/v2/statements",
        json={"statement": "INSERT INTO identity_t (name) VALUES ('b')"},
    )

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT * FROM identity_t ORDER BY id"},
    )
    assert resp.json()["data"] == [[10, "a"], [15, "b"]]


def test_create_or_replace_resets_autoincrement_sequence(client):
    client.post(
        "/api/v2/statements",
        json={"statement": "CREATE OR REPLACE TABLE replace_t (id INT AUTOINCREMENT, name VARCHAR)"},
    )
    client.post(
        "/api/v2/statements",
        json={"statement": "INSERT INTO replace_t (name) VALUES ('a')"},
    )

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT * FROM replace_t"},
    )
    assert resp.json()["data"] == [[1, "a"]]

    client.post(
        "/api/v2/statements",
        json={
            "statement": (
                "CREATE OR REPLACE TABLE replace_t "
                "(id INT AUTOINCREMENT START 100 INCREMENT 1, name VARCHAR)"
            )
        },
    )
    client.post(
        "/api/v2/statements",
        json={"statement": "INSERT INTO replace_t (name) VALUES ('b')"},
    )

    resp = client.post(
        "/api/v2/statements",
        json={"statement": "SELECT * FROM replace_t ORDER BY id"},
    )
    assert resp.json()["data"] == [[100, "b"]]
