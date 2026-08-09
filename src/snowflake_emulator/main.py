"""FastAPI application entrypoint for the Snowflake emulator."""

from __future__ import annotations

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


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


def main() -> None:
    """Run the development server (used by the `snowflake-emulator` console script)."""
    import uvicorn

    uvicorn.run(
        "snowflake_emulator.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
