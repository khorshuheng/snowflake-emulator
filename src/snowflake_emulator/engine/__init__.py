"""Statement-handler implementations.

The execution loop in :mod:`snowflake_emulator.executor` owns parsing, transpilation
and the default DuckDB execution path. Handlers for statement families that need
Snowflake-shaped results live here.
"""
