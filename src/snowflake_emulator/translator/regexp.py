"""Snowflake's ``REGEXP_SUBSTR`` semantics that sqlglot's rendering drops.

sqlglot maps the call onto DuckDB's ``regexp_extract``/``regexp_extract_all``, which differ from
Snowflake in three ways this module restores:

* the ``e`` ("extract") parameter: Snowflake returns the part matched by group 1, and an explicit
  ``group_num`` implies ``e``; without either, the whole match is returned. sqlglot always passes
  group 0, so ``REGEXP_SUBSTR(s, '([0-9]+)', 1, 1, 'e')`` returned the whole match;
* a pattern with no group under ``e`` "behaves as if e was not set" (the whole match), while
  DuckDB raises for a group that the pattern does not have;
* no match is NULL, where DuckDB's ``regexp_extract`` answers an empty string.

Known limits: a *non-literal* pattern under ``e`` is assumed to contain a group (the documented
default), so a group-less runtime pattern answers an empty string for the first occurrence and
DuckDB's "Pattern has 0 groups" error for a later one; and named groups (``(?<name>...)``) and
lookarounds are PCRE constructs that DuckDB's RE2 engine rejects.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from snowflake_emulator.translator.constants import WRITE_DIALECT
from snowflake_emulator.translator.errors import TranslationError
from snowflake_emulator.translator.semi_structured import _text_operand

# Constructs whose "(" does not open a capturing group. ``(?<name>`` is capturing, so the two
# lookbehind forms are matched exactly rather than by the ``(?<`` prefix.
_NON_CAPTURING = ("(?:", "(?=", "(?!", "(?<=", "(?<!", "(?i", "(?m", "(?s", "(?c")

# Marks the ``regexp_extract`` nodes this module builds, so a second pass over the same tree
# (a retry, or a caller that reapplies the transform) leaves them alone. Without it each pass
# wrapped the previous result in another CASE and the SQL kept growing.
_REWRITTEN = "sfemu_regexp_rewritten"


def _has_capturing_group(pattern: str) -> bool:
    """True when a POSIX pattern contains a capturing group."""
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\":
            index += 2  # an escaped "(" is a literal parenthesis
            continue
        if character == "(" and not any(
            pattern.startswith(prefix, index) for prefix in _NON_CAPTURING
        ):
            return True
        index += 1
    return False


def _text(node: exp.Expression | None) -> str | None:
    """The value of a string literal, or None when the expression is not one."""
    if isinstance(node, exp.Literal) and node.is_string:
        return str(node.this)
    return None


def _integer(node: exp.Expression | None) -> int | None:
    """The value of an integer literal, or None when the expression is not one."""
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return int(str(node.this))
        except ValueError:
            return None
    return None


def _positive(node: exp.Expression | None, name: str, default: str) -> str:
    """A positive-integer argument as SQL text, defaulting when it was not given."""
    if node is None:
        return default
    literal = _integer(node)
    if literal is not None and literal < 1:
        raise TranslationError(f"REGEXP_SUBSTR {name} must be a positive integer.")
    return node.sql(dialect=WRITE_DIALECT)


def _rewrite_regexp_substr(node: exp.Expression) -> exp.Expression:
    """Rewrite one ``REGEXP_SUBSTR`` call into an explicit group- and NULL-aware form."""
    if node.meta.get(_REWRITTEN):
        return node
    subject, pattern = node.this, node.args.get("expression")
    if not isinstance(subject, exp.Expression) or not isinstance(pattern, exp.Expression):
        return node
    # Snowflake coerces a VARIANT/JSON argument to its text; DuckDB's regexp functions see the
    # VARIANT struct repr (``{'names': [X]}``), which does not match the JSON quotes.
    subject = _text_operand(subject)
    pattern = _text_operand(pattern)
    assert subject is not None and pattern is not None

    parameters_node = node.args.get("parameters")
    if isinstance(parameters_node, exp.Null):
        # "The function returns NULL ... if any argument is NULL."
        return exp.Null()
    parameters = _text(parameters_node)
    if parameters_node is not None and parameters is None:
        # DuckDB cannot apply non-literal regex flags: it warns and ignores them, so 'e' (and
        # 'i', 's', 'm') would silently not apply. Say so instead of answering wrong.
        raise TranslationError(
            "REGEXP_SUBSTR's regex_parameters argument must be a string literal; DuckDB "
            "cannot apply non-literal regex flags."
        )
    parameters = (parameters or "").lower()
    extracts = "e" in parameters
    # DuckDB has no 'e' option; everything else (i, c, m, s) it understands.
    options = "".join(character for character in parameters if character != "e")

    group = _integer(node.args.get("group")) or 0
    if group:
        wanted = group  # an explicit group_num implies 'e'
    elif extracts:
        pattern_text = _text(pattern)
        # A non-literal pattern is assumed to have a group: that is the documented default.
        wanted = 1 if pattern_text is None or _has_capturing_group(pattern_text) else 0
    else:
        wanted = 0

    position = _positive(node.args.get("position"), "position", "1")
    occurrence = _positive(node.args.get("occurrence"), "occurrence", "1")

    subject_sql = subject.sql(dialect=WRITE_DIALECT)
    pattern_sql = pattern.sql(dialect=WRITE_DIALECT)
    searched = (
        subject_sql if position == "1" else f"SUBSTRING({subject_sql}, {position})"
    )
    option_sql = f", '{options}'" if options else ""

    if occurrence == "1":
        # ``regexp_matches`` distinguishes "no match" (NULL in Snowflake) from an empty match.
        rendered = (
            f"CASE WHEN REGEXP_MATCHES({searched}, {pattern_sql}{option_sql}) "
            f"THEN REGEXP_EXTRACT({searched}, {pattern_sql}, {wanted}{option_sql}) "
            f"ELSE NULL END"
        )
    else:
        matches = (
            f"REGEXP_EXTRACT_ALL({searched}, {pattern_sql}, {wanted}{option_sql})"
        )
        rendered = (
            f"CASE WHEN LENGTH({matches}) >= {occurrence} "
            f"THEN LIST_EXTRACT({matches}, {occurrence}) ELSE NULL END"
        )
    rewritten = sqlglot.parse_one(rendered, read=WRITE_DIALECT)
    for inner in rewritten.find_all(exp.RegexpExtract):
        inner.meta[_REWRITTEN] = True
    return rewritten


def _rewrite_regexp_functions(statement: exp.Expression) -> exp.Expression:
    """Apply ``_rewrite_regexp_substr`` to every call in `statement`.

    Innermost first: a rebuild serializes its subject and pattern, so a nested call has to be in
    its final form before the enclosing one reads it. Replacing in place keeps the parent's
    pointer current, and processing children first means no node is detached before its turn.
    The expression itself can be the call -- a bare-expression UDF body, or a statement that is
    just a function call -- and ``replace`` has no parent to rewrite there, so the root is
    returned instead (``replace`` on a parentless node silently does nothing).
    """
    root = statement
    for node in reversed(list(statement.find_all(exp.RegexpExtract))):
        replacement = _rewrite_regexp_substr(node)
        if replacement is node:
            continue
        if node is root:
            root = replacement
        else:
            node.replace(replacement)
    return root
