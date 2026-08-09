"""ASGI middleware that transparently decompresses gzip-encoded request bodies.

The official ``snowflake-connector-python`` driver gzip-compresses every request
body by default. Starlette/FastAPI only decompress *response* bodies out of the
box (via ``GZipMiddleware``), so incoming gzip payloads need to be handled here
before route handlers try to JSON-decode them.
"""

from __future__ import annotations

import gzip

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class GzipRequestMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        if headers.get(b"content-encoding", b"").lower() != b"gzip":
            await self.app(scope, receive, send)
            return

        body = b""
        more_body = True
        while more_body:
            message: Message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        decompressed = gzip.decompress(body) if body else b""

        new_headers = [
            (k, v) for k, v in scope["headers"] if k.lower() not in (b"content-encoding", b"content-length")
        ]
        new_headers.append((b"content-length", str(len(decompressed)).encode()))
        new_scope = dict(scope)
        new_scope["headers"] = new_headers

        sent = False

        async def receive_decompressed() -> Message:
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": decompressed, "more_body": False}

        await self.app(new_scope, receive_decompressed, send)
