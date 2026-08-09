"""ASGI middleware for budgeted OpenAI-compatible JSON request bodies."""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from .request_adapter import compress_request_payload


class EntrolyASGIMiddleware:
    """Compress JSON chat inputs before an ASGI gateway handles them.

    The middleware is bounded and fail-open. It buffers only JSON POST bodies
    up to ``max_body_bytes`` and replays the exact original bytes on any error.
    """

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        budget: int = 50_000,
        preserve_last_n: int = 4,
        max_body_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.app = app
        self.budget = budget
        self.preserve_last_n = preserve_last_n
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        if (
            scope.get("type") != "http"
            or scope.get("method") not in {"POST", "PUT"}
            or b"application/json" not in headers.get(b"content-type", b"").lower()
        ):
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        total = 0
        more = True
        while more:
            message = await receive()
            chunk = message.get("body", b"")
            chunks.append(chunk)
            total += len(chunk)
            more = bool(message.get("more_body", False))
        original = b"".join(chunks)
        rendered = original
        if total <= self.max_body_bytes:
            try:
                payload = json.loads(original)
                if isinstance(payload, dict):
                    result = compress_request_payload(
                        payload,
                        budget=self.budget,
                        preserve_last_n=self.preserve_last_n,
                    )
                    if result.changed:
                        rendered = json.dumps(
                            result.payload, ensure_ascii=False, separators=(",", ":")
                        ).encode("utf-8")
            except Exception:
                rendered = original

        delivered = False

        async def replay() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {"type": "http.request", "body": b"", "more_body": False}
            delivered = True
            return {"type": "http.request", "body": rendered, "more_body": False}

        new_scope = dict(scope)
        new_scope["headers"] = [
            (key, str(len(rendered)).encode("ascii")) if key.lower() == b"content-length" else (key, value)
            for key, value in scope.get("headers", [])
        ]
        await self.app(new_scope, replay, send)


__all__ = ["EntrolyASGIMiddleware"]
