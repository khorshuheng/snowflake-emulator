"""Constants shared by catalog management and SQL translation."""

# DuckDB reserves the ``information_schema`` schema name, so Snowflake's
# ``INFORMATION_SCHEMA`` is exposed under a non-reserved name and references are
# rewritten at the AST level (see ``translator._rewrite_information_schema``). The name
# is kept uppercase so it matches the emulator's Snowflake-style identifier casing.
INFORMATION_SCHEMA_SCHEMA = "SF_INFORMATION_SCHEMA"

# The account name ``SHOW GRANTS ON ACCOUNT`` reports. Snowflake uses the account locator;
# the emulator has no real account, so the rows name this constant.
EMULATED_ACCOUNT_NAME = "SFEMU_ACCOUNT"
