"""Snowflake's SQL format models for numbers (``TO_CHAR(number, format)``).

Snowflake's numeric format models are Oracle-style. A *fixed-position* model places each
digit (``9``, ``0``, ``X``), the decimal separator (``.``/``D``), group separators
(``,``/``G``), the sign (``S``/``MI``), the currency symbol (``$``), a percentage (``%``),
an exponent (``EE``..``EEEEEEE``) and the ``FM`` fill-mode modifier. The *text-minimal*
family (``TM``, ``TM9``, ``TME``) prints a value with as few characters as it needs. This
module implements both families for printing and parsing, following the documented rules:
the default reserved sign blank, fill vs compact mode, blank-vs-zero digit positions,
``#`` overflow, ``TM9(n,g)`` parameters, and trailing zeros in text-minimal output.

There is no sqlglot dependency here: the translator maps a numeric format model onto the
``sfemu_format_number``/``sfemu_parse_number`` functions registered on every DuckDB
connection (see ``database.py``), and those functions call in here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

__all__ = ["NumberFormatError", "format_number", "parse_number"]


class NumberFormatError(ValueError):
    """A format model the emulator cannot honour, or input that does not match it."""


@dataclass
class _Digit:
    """One digit position of a fixed-position model."""

    element: str  # "9", "0" or "X"/"x"
    separator: str = ""  # group separator printed before this position


@dataclass
class _Item:
    """A prefix/suffix element, in the order the model places it."""

    kind: str  # "literal", "dollar", "percent", "sign", "exponent"
    text: str = ""


@dataclass
class _Fixed:
    integer: list[_Digit] = field(default_factory=list)
    fraction: list[_Digit] = field(default_factory=list)
    point: str = "."
    fill: bool = True
    blank_zero: bool = False
    prefix: list[_Item] = field(default_factory=list)
    suffix: list[_Item] = field(default_factory=list)


@dataclass
class _Minimal:
    kind: str = "TM"  # "TM", "TM9" or "TME"
    exponent_char: str = "E"
    digits: int | None = None  # TM9's first parameter (None = ALL)
    group: int = 0  # TM9's second parameter
    fill: bool = True
    blank_zero: bool = False
    prefix: list[_Item] = field(default_factory=list)
    suffix: list[_Item] = field(default_factory=list)


def _to_decimal(text: str) -> Decimal:
    try:
        return Decimal(str(text).strip())
    except (InvalidOperation, ValueError) as exc:
        raise NumberFormatError(f"'{text}' is not a number.") from exc


# --------------------------------------------------------------------------------------
# Tokenizing
# --------------------------------------------------------------------------------------


def _read_literal(model: str, index: int) -> tuple[str, int]:
    """Read a double-quoted literal starting at `index`; a doubled quote escapes one."""
    index += 1
    text: list[str] = []
    while index < len(model):
        if model[index] == '"':
            if index + 1 < len(model) and model[index + 1] == '"':
                text.append('"')
                index += 2
                continue
            return "".join(text), index + 1
        text.append(model[index])
        index += 1
    raise NumberFormatError("Unterminated literal in a numeric format model.")


def _parse_fixed(model: str) -> _Fixed:
    state = _Fixed()
    index = 0
    length = len(model)
    seen_point = False
    seen_sign = False
    seen_exponent = False
    digits_started = False
    pending_separator = ""

    def add_item(item: _Item) -> None:
        (state.suffix if digits_started else state.prefix).append(item)

    while index < length:
        char = model[index]
        upper = char.upper()
        if char == '"':
            text, index = _read_literal(model, index)
            add_item(_Item("literal", text))
            continue
        if char == "9" or char == "0":
            digit = _Digit(char, pending_separator)
            pending_separator = ""
            (state.fraction if seen_point else state.integer).append(digit)
            digits_started = True
            index += 1
            continue
        if upper == "X":
            if seen_point:
                raise NumberFormatError("The X element cannot be used in the fractional part.")
            state.integer.append(_Digit(char, pending_separator))
            pending_separator = ""
            digits_started = True
            index += 1
            continue
        if char == "," or upper == "G":
            if not digits_started or seen_point:
                raise NumberFormatError(
                    "A group separator must follow an integer digit position."
                )
            # G is an alternative for "," and prints a comma.
            pending_separator = ","
            index += 1
            continue
        if char == "." or upper == "D":
            if seen_point:
                raise NumberFormatError("A numeric format model can have only one decimal point.")
            seen_point = True
            # The D element is an alternative for "." and always prints a period
            # (docs: "D is the decimal fraction separator; alternative for .").
            state.point = "."
            index += 1
            continue
        if char == "$":
            add_item(_Item("dollar"))
            index += 1
            continue
        if char == "%":
            add_item(_Item("percent"))
            index += 1
            continue
        if upper == "B":
            state.blank_zero = True
            index += 1
            continue
        if upper == "S":
            if seen_sign:
                raise NumberFormatError("A numeric format model can have only one sign element.")
            seen_sign = True
            add_item(_Item("sign", "S"))
            index += 1
            continue
        if model[index : index + 2].upper() == "MI":
            if seen_sign:
                raise NumberFormatError("A numeric format model can have only one sign element.")
            seen_sign = True
            add_item(_Item("sign", "MI"))
            index += 2
            continue
        if model[index : index + 2].upper() == "FM":
            state.fill = not state.fill
            index += 2
            continue
        if model[index : index + 2].upper() == "FX":
            raise NumberFormatError("The FX (exact match) modifier is not supported.")
        if char == "_":
            # The underscore prints nothing; on input it allows one optional space.
            index += 1
            continue
        if upper == "E":
            end = index
            while end < length and model[end].upper() == "E":
                end += 1
            count = end - index
            if count < 2:
                raise NumberFormatError("An exponent element needs at least two E characters.")
            if count > 7:
                raise NumberFormatError("An exponent element allows at most 7 E characters.")
            if seen_exponent:
                raise NumberFormatError("A numeric format model can have only one exponent.")
            seen_exponent = True
            add_item(_Item("exponent", char + model[index + 1 : end]))
            index = end
            continue
        if char.isalpha():
            raise NumberFormatError(
                f"Numeric format element '{model[index : index + 2]}' is not supported."
            )
        add_item(_Item("literal", char))
        index += 1
    if pending_separator:
        raise NumberFormatError("A group separator must be followed by a digit position.")
    if not state.integer and not state.fraction:
        raise NumberFormatError("A numeric format model needs a digit element (9, 0 or X).")
    if not state.integer and state.fraction:
        raise NumberFormatError("A numeric format model needs an integer digit position.")
    return state


def _is_text_minimal(model: str) -> bool:
    upper = model.upper()
    return upper.startswith("TM") or "TM9" in upper or "TME" in upper


def _parse_minimal(model: str) -> _Minimal:
    state = _Minimal()
    index = 0
    while index < len(model):
        token = model[index : index + 3].upper()
        if token == "TM9":
            if state.kind != "TM" or index > 0 and model[:index].strip():
                raise NumberFormatError("A format model can have only one text-minimal element.")
            state.kind = "TM9"
            index += 3
            if index < len(model) and model[index] == "(":
                end = model.find(")", index)
                if end < 0:
                    raise NumberFormatError("Unterminated TM9 parameter list.")
                arguments = [part.strip().upper() for part in model[index + 1 : end].split(",")]
                state.digits = None if arguments[0] == "ALL" else int(arguments[0])
                if len(arguments) > 1:
                    state.group = int(arguments[1])
                index = end + 1
            continue
        if token == "TME":
            state.kind = "TME"
            state.exponent_char = model[index + 2]
            index += 3
            continue
        if token == "TM":
            index += 2
            continue
        char = model[index]
        upper = char.upper()
        if model[index : index + 2].upper() == "FM":
            state.fill = not state.fill
            index += 2
            continue
        if model[index : index + 2].upper() == "FX":
            raise NumberFormatError("The FX (exact match) modifier is not supported.")
        if model[index : index + 2].upper() == "MI":
            state.suffix.append(_Item("sign", "MI"))
            index += 2
            continue
        if upper == "S":
            state.suffix.append(_Item("sign", "S"))
            index += 1
            continue
        if char == "$":
            (state.suffix if state.suffix else state.prefix).append(_Item("dollar"))
            index += 1
            continue
        if char == "%":
            state.suffix.append(_Item("percent"))
            index += 1
            continue
        if upper == "B":
            state.blank_zero = True
            index += 1
            continue
        if char == '"':
            text, index = _read_literal(model, index)
            (state.suffix if state.suffix else state.prefix).append(_Item("literal", text))
            continue
        if char in ";-=/():." or not char.isalnum():
            (state.suffix if state.suffix else state.prefix).append(_Item("literal", char))
            index += 1
            continue
        raise NumberFormatError(
            f"Text-minimal format element '{model[index : index + 2]}' is not supported."
        )
    return state


# --------------------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------------------


def _round(value: Decimal, places: int) -> tuple[str, str]:
    """The absolute value's integer and fraction digits, half-up rounded to `places`."""
    quantum = Decimal(1).scaleb(-places) if places >= 0 else Decimal(1).scaleb(-places)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    text = format(abs(rounded), "f")
    if "." in text:
        integral, fraction = text.split(".", 1)
        return integral, fraction
    return text, ""


def _overflow(fixed: _Fixed) -> str:
    """Every digit position prints ``#`` when the integer part does not fit."""
    integer = "".join("#" if digit.element.upper() != "X" else "#" for digit in fixed.integer)
    if not fixed.fraction:
        return integer
    return integer + fixed.point + "#" * len(fixed.fraction)


def _render_integer(fixed: _Fixed, integral: str, is_zero: bool) -> str:
    if any(digit.element.upper() == "X" for digit in fixed.integer):
        upper = any(digit.element == "X" for digit in fixed.integer)
        rendered = (f"{int(integral):X}" if upper else f"{int(integral):x}")
        if len(rendered) > len(fixed.integer):
            return _overflow(fixed)
        return rendered.rjust(len(fixed.integer))
    if len(integral) > len(fixed.integer):
        return _overflow(fixed)
    padded = integral.rjust(len(fixed.integer))
    characters: list[str] = []
    for index, digit in enumerate(fixed.integer):
        character = padded[index]
        if character == " ":
            if digit.element == "0":
                character = "0"
            elif fixed.blank_zero and is_zero:
                character = " "
            else:
                character = " " if fixed.fill else ""
        characters.append(character)
    if is_zero and not any(digit.element == "0" for digit in fixed.integer):
        # A zero integer part prints as a single 0 unless B blanks it out.
        last = len(characters) - 1
        while last >= 0 and characters[last] == " ":
            characters[last] = ""
            last -= 1
        if last >= 0:
            characters[last] = " " if (fixed.blank_zero and fixed.fill) else "0"
    output: list[str] = []
    for index, digit in enumerate(fixed.integer):
        if digit.separator:
            left = "".join(output).strip()
            right = "".join(characters[index:]).strip()
            if left and right:
                output.append(digit.separator)
        output.append(characters[index])
    return "".join(output)


def _render_fraction(fixed: _Fixed, fraction: str) -> str:
    # "Normally, the ... trailing zeros in the fractional part are replaced with spaces";
    # a 0 position keeps its zero instead.
    remaining = fraction.rstrip("0")
    characters: list[str] = []
    for digit in fixed.fraction:
        if remaining:
            characters.append(remaining[0])
            remaining = remaining[1:]
        elif digit.element == "0":
            characters.append("0")
        else:
            characters.append(" " if fixed.fill else "")
    text = "".join(characters)
    if not fixed.fill:
        text = text.rstrip()
    return text


def _render_exponent(item: _Item, exponent: int) -> str:
    character = item.text[0]
    width = len(item.text)
    sign = "-" if exponent < 0 else ("+" if width > 2 else "")
    digits = str(abs(exponent))
    if width > 2:
        # A fixed-width exponent counts its letter: EEE -> E+9, EEEE -> E+03.
        digits = digits.rjust(width - 2, "0")
    return f"{character}{sign}{digits}"


def _format_fixed(value: Decimal, fixed: _Fixed) -> str:
    """Render a fixed-position model.

    The default sign (and the ``$``) *float* with the digits: they sit immediately before
    the first digit, after the leading blanks a ``9`` position leaves in fill mode -- the
    documented examples show ``'$99'`` -> ``_$12`` / ``_-$7`` and ``'">"99"<"'`` ->
    ``> 12<``. ``S`` floats too, while ``MI`` is a fixed-position element and occupies its
    own slot.
    """
    negative = value < 0
    percent = any(item.kind == "percent" for item in fixed.prefix + fixed.suffix)
    scaled = value * 100 if percent else value
    exponent_item = next(
        (item for item in fixed.prefix + fixed.suffix if item.kind == "exponent"), None
    )
    exponent: int | None = None
    if exponent_item is not None:
        magnitude = abs(scaled)
        exponent = 0 if magnitude == 0 else magnitude.adjusted()
        mantissa = magnitude if magnitude == 0 else magnitude.scaleb(-exponent)
        integral, fraction = _round(mantissa, len(fixed.fraction))
        if len(integral) > 1:  # 9.99 rounded to 10.0: renormalize the mantissa
            shift = len(integral) - 1
            exponent += shift
            integral, fraction = _round(mantissa.scaleb(-shift), len(fixed.fraction))
    else:
        integral, fraction = _round(abs(scaled), len(fixed.fraction))
    integer_field = _render_integer(fixed, integral, integral.strip("0") == "")
    if integer_field.startswith("#"):
        return ("-" if negative else "") + integer_field
    fraction_field = _render_fraction(fixed, fraction) if fixed.fraction else ""

    sign_items = [item for item in fixed.prefix + fixed.suffix if item.kind == "sign"]
    explicit = sign_items[0] if sign_items else None
    if explicit is not None and explicit.text == "S":
        sign_character = "-" if negative else "+"
    elif explicit is not None:
        sign_character = "-" if negative else " "
    else:
        sign_character = ("-" if negative else " ") if fixed.fill else ("-" if negative else "")

    float_prefix = ""
    positional_prefix = ""
    for item in fixed.prefix:
        if item.kind == "literal":
            positional_prefix += item.text
        elif item.kind == "percent":
            positional_prefix += "%"
        elif item.kind == "exponent":
            positional_prefix += _render_exponent(item, exponent or 0)
        elif item.kind == "sign" and item.text == "MI":
            positional_prefix += sign_character
        elif item.kind == "sign":
            float_prefix += sign_character
        elif item.kind == "dollar":
            float_prefix += "$"
    if explicit is None:
        float_prefix = sign_character + float_prefix
    insertion = next(
        (index for index, char in enumerate(integer_field) if char.strip()),
        len(integer_field),
    )
    integer_field = integer_field[:insertion] + float_prefix + integer_field[insertion:]

    # A suffix S floats after the number; MI keeps its fixed slot after the field.
    float_suffix = ""
    positional_suffix = ""
    for item in fixed.suffix:
        if item.kind == "literal":
            positional_suffix += item.text
        elif item.kind == "percent":
            positional_suffix += "%"
        elif item.kind == "dollar":
            positional_suffix += "$"
        elif item.kind == "exponent":
            positional_suffix += _render_exponent(item, exponent or 0)
        elif item.kind == "sign" and item.text == "S":
            float_suffix += sign_character
        elif item.kind == "sign":
            float_suffix += sign_character
    if fixed.fraction and (fraction_field or fixed.fill):
        digits = integer_field + fixed.point + fraction_field
    else:
        digits = integer_field
    return positional_prefix + digits + float_suffix + positional_suffix


def _minimal_positional(value: Decimal, state: _Minimal) -> str:
    text = format(value, "f")
    if state.digits is not None and "." in text:
        integral, fraction = _round(value, state.digits)
        text = integral + ("." + fraction if fraction else "")
    if state.group:
        integral, _, fraction = text.partition(".")
        groups: list[str] = []
        while len(integral) > state.group:
            groups.insert(0, integral[-state.group :])
            integral = integral[: -state.group]
        groups.insert(0, integral)
        text = ",".join(groups) + ("." + fraction if fraction else "")
    return text


def _minimal_scientific(value: Decimal, state: _Minimal) -> str:
    if value == 0:
        return "0" + state.exponent_char + "0"
    exponent = value.adjusted()
    mantissa = value.scaleb(-exponent).normalize()
    return f"{format(mantissa, 'f')}{state.exponent_char}{exponent}"


def _format_minimal(value: Decimal, state: _Minimal) -> str:
    negative = value < 0
    magnitude = abs(value)
    positional = _minimal_positional(magnitude, state)
    scientific = _minimal_scientific(magnitude, state)
    if state.kind == "TME":
        body = scientific
    elif state.kind == "TM9":
        body = positional
    else:
        body = min(positional, scientific, key=len)
    if state.blank_zero and magnitude == 0:
        body = ""
    if any(item.kind == "percent" for item in state.prefix + state.suffix):
        scaled = (magnitude * 100).normalize()
        body = _minimal_scientific(scaled, state) if state.kind == "TME" else _minimal_positional(scaled, state)
    output: list[str] = []
    for item in state.prefix:
        if item.kind == "literal":
            output.append(item.text)
        elif item.kind == "dollar":
            output.append("$")
        elif item.kind == "sign":
            output.append("-" if negative else ("+" if item.text == "S" else " "))
    if negative:
        output.append("-")
    output.append(body)
    for item in state.suffix:
        if item.kind == "literal":
            output.append(item.text)
        elif item.kind == "dollar":
            output.append("$")
        elif item.kind == "percent":
            output.append("%")
        elif item.kind == "sign":
            output.append("-" if negative else ("+" if item.text == "S" else " "))
    return "".join(output)


def format_number(value_text: str, model: str) -> str:
    """Format ``value_text`` with a Snowflake numeric format model."""
    value = _to_decimal(value_text)
    if _is_text_minimal(model):
        return _format_minimal(value, _parse_minimal(model))
    return _format_fixed(value, _parse_fixed(model))


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _fixed_pattern(fixed: _Fixed) -> str:
    parts: list[str] = ["^" if not fixed.fill else r"^\s*"]
    # Lax mode accepts a leading sign/blank even when the model places its sign element
    # after the digits (docs: lax matching is more permissive than exact mode).
    parts.append(r"[-+ ]?")
    for item in fixed.prefix:
        if item.kind == "literal":
            parts.append(re.escape(item.text))
        elif item.kind == "dollar":
            parts.append(r"\$")
        elif item.kind == "percent":
            parts.append("%")
        elif item.kind == "sign":
            parts.append(r"[-+]?" if item.text == "S" else r"[- ]?")
    for index, digit in enumerate(fixed.integer):
        if index:
            previous = fixed.integer[index - 1]
            if digit.separator:
                parts.append(",?")
        elif digit.separator:
            parts.append(",?")
        if digit.element.upper() == "X":
            parts.append(r"[0-9A-Fa-f]")
        else:
            parts.append(r"\d" if digit.element == "0" else r"\d?")
    if fixed.fraction:
        parts.append(re.escape(fixed.point))
        for digit in fixed.fraction:
            parts.append(r"\d" if digit.element == "0" else r"\d?")
    for item in fixed.suffix:
        if item.kind == "literal":
            parts.append(re.escape(item.text))
        elif item.kind == "dollar":
            parts.append(r"\$")
        elif item.kind == "percent":
            parts.append("%")
        elif item.kind == "sign":
            parts.append(r"[-+]?" if item.text == "S" else r"[- ]?")
    parts.append(r"\s*$" if fixed.fill else "$")
    return "".join(parts)


def _minimal_pattern(state: _Minimal) -> str:
    sign = "-?" if not any(item.kind == "sign" for item in state.prefix + state.suffix) else ""
    number = {
        "TM9": r"\d+(?:\.\d+)?",
        "TME": r"\d(?:\.\d+)?[Ee][-+]?\d+",
    }.get(state.kind, r"\d+(?:\.\d+)?|\d(?:\.\d+)?[Ee][-+]?\d+")
    percent = "%" if any(item.kind == "percent" for item in state.prefix + state.suffix) else ""
    return f"^{sign}{number}{percent}$"


def parse_number(text: str, model: str) -> str:
    """Parse ``text`` with a Snowflake numeric format model; return canonical decimal text."""
    if _is_text_minimal(model):
        state = _parse_minimal(model)
        match = re.match(_minimal_pattern(state), text.strip())
        if match is None:
            raise NumberFormatError(
                f"The input string '{text}' does not match the numeric format model '{model}'."
            )
        value = _to_decimal(text.strip().rstrip("%"))
        if any(item.kind == "percent" for item in state.prefix + state.suffix):
            value = value / 100
        return _canonical(value)
    fixed = _parse_fixed(model)
    pattern = _fixed_pattern(fixed)
    match = re.match(pattern, text, re.IGNORECASE)
    if match is None:
        raise NumberFormatError(
            f"The input string '{text}' does not match the numeric format model '{model}'."
        )
    matched = match.group(0)
    negative = "-" in matched
    digits = "".join(char for char in matched if char.isdigit() or char == fixed.point)
    digits = digits.replace(fixed.point, ".")
    if not digits.strip("."):
        raise NumberFormatError(f"The input string '{text}' contains no number.")
    value = _to_decimal(("-" if negative else "") + digits.strip("."))
    if any(item.kind == "percent" for item in fixed.prefix + fixed.suffix):
        value = value / 100
    return _canonical(value)


def _canonical(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".") or "0"
    return text


def validate_model(model: str) -> None:
    """Raise `NumberFormatError` when the model can never be honoured."""
    if _is_text_minimal(model):
        _parse_minimal(model)
        return
    _parse_fixed(model)
