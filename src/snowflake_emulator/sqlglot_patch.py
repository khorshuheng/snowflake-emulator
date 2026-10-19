"""Tokenizer settings shared by every module that parses Snowflake SQL.

sqlglot 30.21 began decoding backslash escapes in Snowflake string literals: an unknown
escape (``\\d``, ``\\.``, ``\\U``) lost its backslash, and numeric escapes (``\\x41``,
``\\u0041``, octal ``\\101``) were decoded to their character. Snowflake keeps unknown
escapes literal and does not decode those numeric forms, so restoring the pre-30.21
tokenizer settings keeps regex patterns and paths intact (e.g.
``REGEXP_REPLACE(x, '\\d', ...)`` and ``'C:\\Users\\me'``).

Set on the imported ``Snowflake.Tokenizer`` rather than a subclass:
``sqlglot.parse(..., read="snowflake")`` resolves the dialect class, so replacing
``Tokenizer`` afterwards has no effect.
"""

from sqlglot.dialects import Snowflake

Snowflake.Tokenizer.DROP_UNKNOWN_ESCAPES = False
Snowflake.Tokenizer.NUMERIC_ESCAPES = {}
