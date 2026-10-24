"""Snowflake numeric format models: printing and parsing, against the documented examples."""

from __future__ import annotations

import pytest

from snowflake_emulator.number_format import (
    NumberFormatError,
    format_number,
    parse_number,
    validate_model,
)


@pytest.mark.parametrize(
    ("value", "model", "expected"),
    [
        # Docs "Output examples": the default fill mode reserves the sign's blank and
        # replaces leading zeros / trailing fraction zeros with spaces.
        ("1.2", "999,999.999", "      1.2  "),
        ("123.456", "999,999.999", "    123.456"),
        ("1234.56", "999,999.999", "   1,234.56 "),
        ("-123456.789", "999,999.999", "-123,456.789"),
        ("1.2", "S000,000.000", "+000,001.200"),
        ("1.2", "S999,999.999", "     +1.2  "),
        ("-123456.789", "S000,000.000", "-123,456.789"),
        # Fill-mode modifier: leading spaces disappear.
        ("1.2", "FM999,999.999", "1.2"),
        ("123.456", "FM999,999.999", "123.456"),
        ("-123456.789", "FM999,999.999", "-123,456.789"),
        ("1234", "FM999,999.999", "1,234"),
        # Exponents and literals.
        ("1234", "9d999EE", " 1.234E3"),
        ("1234", "9d999EEEE", " 1.234E+03"),
        ("12", '">"99"<"', "> 12<"),
        # Sign and currency placement (docs' sign-position table).
        ("12", "99", " 12"),
        ("-7", "99", " -7"),
        ("12", "S99", "+12"),
        ("-7", "S99", " -7"),
        ("12", "99S", "12+"),
        ("-7", "99S", " 7-"),
        ("12", "MI99", " 12"),
        ("-7", "MI99", "- 7"),
        ("12", "99MI", "12 "),
        ("-7", "99MI", " 7-"),
        ("12", "$99", " $12"),
        ("-7", "$99", " -$7"),
        ("-123.45", "999.99MI", "123.45-"),
        # Text-minimal models.
        ("0.25", "TM9%", "25%"),
        ("123.400", "TM9", "123.400"),
        ("1234", "TME", "1.234E3"),
        ("1234", "TME", "1.234E3"),
        ("1234567.891", "TM9(2,3)", "1,234,567.89"),
        # Zero handling and rounding.
        ("0", "99B", "   "),
        ("0", "990", "   0"),
        ("0", "99.99", "  0.  "),
        ("123.456", "FM999,999.99", "123.46"),
    ],
)
def test_formats_like_snowflake(value, model, expected):
    assert format_number(value, model) == expected


@pytest.mark.parametrize(
    ("text", "model", "expected"),
    [
        ("1,234.50", "9,999.99", "1234.5"),
        ("012,345.67", "999,999.99", "12345.67"),
        (" 12,345.67", "999,999.99", "12345.67"),
        ("-1,234.50", "9,999.99", "-1234.5"),
        ("123.45-", "999.99MI", "-123.45"),
        ("25%", "TM9%", "0.25"),
        ("1.234E3", "TME", "1234"),
    ],
)
def test_parses_like_snowflake(text, model, expected):
    assert parse_number(text, model) == expected


def test_a_zero_padded_model_requires_every_digit():
    """Docs: with ``0`` positions, blanks and missing leading zeros do not convert."""
    for text in ("12345.67", " 12,345.67"):
        with pytest.raises(NumberFormatError):
            parse_number(text, "000,000.00")


def test_input_that_does_not_match_the_model_is_an_error():
    with pytest.raises(NumberFormatError):
        parse_number("bad", "999")
    with pytest.raises(NumberFormatError):
        parse_number("12.345.6", "999.99")


@pytest.mark.parametrize("model", ["V99", "RN", "FX999", "999.99.99", "EE"])
def test_unsupported_models_are_rejected(model):
    with pytest.raises(NumberFormatError):
        validate_model(model)
