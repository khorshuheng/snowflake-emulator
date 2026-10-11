"""FastAPI application entrypoint for the Snowflake emulator."""

from __future__ import annotations

import argparse

from fastapi import FastAPI

from snowflake_emulator.middleware import GzipRequestMiddleware
from snowflake_emulator.routers import auth, queries, statements
from snowflake_emulator.settings import settings

app = FastAPI(
    title="Snowflake Emulator",
    description=(
        "A local emulation of the Snowflake SQL API v2, backed by DuckDB. "
        "Snowflake SQL is translated to DuckDB SQL via sqlglot before execution."
    ),
    version="0.1.0",
)

app.add_middleware(GzipRequestMiddleware)

app.include_router(auth.router)
app.include_router(statements.router)
app.include_router(queries.router)
app.include_router(queries.monitoring_router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser for the ``snowflake-emulator`` console script."""
    parser = argparse.ArgumentParser(
        prog="snowflake-emulator",
        description="Run the Snowflake emulator (FastAPI + DuckDB).",
    )
    parser.add_argument("--host", default=None, help="Bind address (default: config).")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: config).")
    parser.add_argument(
        "--db-path",
        dest="db_path",
        default=None,
        help="DuckDB database file (default: config, or :memory:).",
    )
    parser.add_argument(
        "--stage-root",
        dest="stage_root",
        default=None,
        help="Root directory for staged files (default: config, or a temp dir).",
    )
    parser.add_argument(
        "--isolation",
        choices=["shared", "isolated"],
        default=None,
        help="Database isolation mode (default: config).",
    )
    return parser


def apply_args(args: argparse.Namespace) -> None:
    """Apply parsed CLI flags onto the runtime settings (only when provided)."""
    if args.host is not None:
        settings.server.host = args.host
    if args.port is not None:
        settings.server.port = args.port
    if args.db_path is not None:
        settings.persistence.database_path = args.db_path
    if args.stage_root is not None:
        settings.staging.stage_root = args.stage_root
    if args.isolation is not None:
        settings.persistence.isolation = args.isolation


def main(argv: list[str] | None = None) -> None:
    """Run the development server (used by the `snowflake-emulator` console script)."""
    import uvicorn

    args = build_parser().parse_args(argv)
    apply_args(args)
    uvicorn.run(
        "snowflake_emulator.main:app",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
