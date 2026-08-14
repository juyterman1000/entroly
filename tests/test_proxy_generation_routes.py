from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute

from entroly.proxy_generation_routes import install_generation_routes


class _FakeProxy:
    def __init__(self):
        self.paths = []

    async def handle_proxy(self, request):
        self.paths.append(request.url.path)
        return JSONResponse({"path": request.url.path})


async def _exercise(path: str):
    proxy = _FakeProxy()

    async def catch_all(_request):
        return JSONResponse({"catch_all": True})

    app = Starlette(
        routes=[Route("/{path:path}", catch_all, methods=["POST"])]
    )
    app.state.proxy = proxy
    install_generation_routes(app)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(path, json={"model": "test", "input": "hi"})
    return response, proxy, app


def test_unprefixed_chat_completions_uses_full_canonical_handler():
    response, proxy, _ = asyncio.run(_exercise("/chat/completions"))
    assert response.status_code == 200
    assert proxy.paths == ["/v1/chat/completions"]
    assert response.headers["X-Entroly-Route-Normalized"] == "/v1/chat/completions"


def test_unprefixed_responses_uses_full_canonical_handler():
    response, proxy, _ = asyncio.run(_exercise("/responses"))
    assert response.status_code == 200
    assert proxy.paths == ["/v1/responses"]
    assert response.headers["X-Entroly-Route-Normalized"] == "/v1/responses"


def test_http_aliases_do_not_register_websocket_protocols():
    _, _, app = asyncio.run(_exercise("/responses"))
    assert not any(isinstance(route, WebSocketRoute) for route in app.router.routes)
