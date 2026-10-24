"""Dialect names and parser markers shared by the translator modules."""

import re

READ_DIALECT = "snowflake"
WRITE_DIALECT = "duckdb"

# ``ARRAY_FLATTEN`` is renamed before parsing: sqlglot's node for it accepts one argument,
# so the documented two-argument Snowflake form fails to parse outright.
_FLATTEN_MARKER = "SFEMU_ARRAY_FLATTEN"
_ARRAY_FLATTEN_NAME = "ARRAY_FLATTEN"

_SIMPLE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

# Marks the ``json_each``-based subquery ``FLATTEN`` is rewritten to (see
# ``semi_structured._flatten_subquery``), so the derived-column type lookup can tell a flattened
# relation apart from an ordinary derived table and type its VALUE/THIS columns.
FLATTEN_RELATION_META = "sfemu_flatten_relation"

# Marks the text cast ``TO_JSON`` is rewritten to. The text-cast rewrite would otherwise read it
# as a VARIANT -> VARCHAR coercion and unquote a string variant on a second pass.
TO_JSON_RESULT_META = "sfemu_to_json_result"

# Snowflake's ``CURRENT_VERSION()`` reports the server's own release version. DuckDB's
# ``VERSION()`` returns ``v1.5.5`` (with a leading ``v``), which breaks clients such as
# snowflake-sqlalchemy that parse each dot-separated component as an ``int``.
EMULATED_SNOWFLAKE_VERSION = "8.23.1"
