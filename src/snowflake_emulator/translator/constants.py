"""Dialect names and parser markers shared by the translator modules."""

READ_DIALECT = "snowflake"
WRITE_DIALECT = "duckdb"

# ``ARRAY_FLATTEN`` is renamed before parsing: sqlglot's node for it accepts one argument,
# so the documented two-argument Snowflake form fails to parse outright.
_FLATTEN_MARKER = "SFEMU_ARRAY_FLATTEN"
_ARRAY_FLATTEN_NAME = "ARRAY_FLATTEN"
