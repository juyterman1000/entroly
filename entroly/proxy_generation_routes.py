"""Normalize compatible HTTP generation routes into Entroly's full pipeline.

Only HTTP POST generation routes are aliased. WebSocket protocols are not
conflated with HTTP Responses semantics.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.routing import Route

from . import proxy as _proxy

_HTTP_GENERATION_ALIASES = {
    "/chat/completions": "/v1/chat/completions",
    "/responses": "/v1/responses",
}


async def _normalized_generation_request(request: Request, canonical_path: str):
    scope = dict(request.scope)
    scope["path"] = canonical_path
    scope["raw_path"] = canonical_path.encode("ascii")
    normalized = Request(scope, receive=request.receive)
    response = await request.app.state.proxy.handle_proxy(normalized)
    response.headers["X-Entroly-Route-Normalized"] = canonical_path
    return response


def _alias_endpoint(canonical_path: str):
    async def endpoint(request: Request):
        return await _normalized_generation_request(request, canonical_path)

    endpoint.__name__ = "entroly_alias_" + canonical_path.strip("/").replace("/", "_")
    return endpoint


def install_generation_routes(app: Any) -> None:
    routes = getattr(getattr(app, "router", None), "routes", None)
    if not isinstance(routes, list):
        return
    existing = {getattr(route, "path", None) for route in routes}
    catch_all_index = next(
        (
            index
            for index, route in enumerate(routes)
            if getattr(route, "path", None) == "/{path:path}"
        ),
        len(routes),
    )
    additions: list[Route] = []
    for alias, canonical in _HTTP_GENERATION_ALIASES.items():
        if alias in existing:
            continue
        additions.append(
            Route(
                alias,
                endpoint=_alias_endpoint(canonical),
                methods=["POST"],
                name=f"entroly-normalized-{canonical.strip('/').replace('/', '-')}",
            )
        )
    routes[catch_all_index:catch_all_index] = additions


def install_proxy_generation_routes() -> None:
    current = _proxy.create_proxy_app
    if hasattr(current, "__entroly_generation_routes_original__"):
        return

    def create_proxy_app(*args: Any, **kwargs: Any):
        app = current(*args, **kwargs)
        install_generation_routes(app)
        return app

    create_proxy_app.__entroly_generation_routes_original__ = current
    _proxy.create_proxy_app = create_proxy_app


install_proxy_generation_routes()


__all__ = [
    "_HTTP_GENERATION_ALIASES",
    "install_generation_routes",
    "install_proxy_generation_routes",
]
