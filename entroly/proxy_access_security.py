"""Capability boundary for explicitly remote Entroly proxy listeners.

Loopback proxy use remains zero-configuration. Binding the proxy to a wildcard,
LAN, container-network, or DNS address is a materially different trust model:
an unauthenticated caller can submit its own provider credential and receive a
model answer augmented with the local repository's selected context.

Remote mode therefore requires three independent operator decisions:

* ``ENTROLY_ALLOW_REMOTE_PROXY=1`` — remote binding is intentional;
* ``ENTROLY_PROXY_ACCESS_TOKEN`` — an unpredictable URL-safe capability;
* ``ENTROLY_REMOTE_PROXY_TRUSTED_TRANSPORT=1`` — the operator acknowledges that
  TLS, a private tunnel, or an equivalently trusted network protects the token.

Every HTTP route, including health and provider-compatible catch-all routes,
requires ``X-Entroly-Access-Token``. The header is stripped before downstream
provider-header construction so it can never be forwarded upstream.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import re
from typing import Any, Awaitable, Callable

from . import proxy as _proxy
from .proxy_config import ProxyConfig

_ACCESS_HEADER = b"x-entroly-access-token"
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~-]{32,512}$")
_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)"
    r"(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*\.?$"
)

ASGIReceive = Callable[..., Awaitable[Any]]
ASGISend = Callable[..., Awaitable[Any]]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _validated_access_token(raw: object) -> str | None:
    return raw if isinstance(raw, str) and _TOKEN_RE.fullmatch(raw) else None


def _classify_bind_host(host: object) -> tuple[str, bool]:
    """Return a normalized bind host and whether it is literal loopback."""
    if not isinstance(host, str):
        raise ValueError("proxy host must be a safe bind address")
    value = host.strip()
    if (
        not value
        or len(value) > 253
        or any(character.isspace() or ord(character) < 32 for character in value)
        or "://" in value
        or "/" in value
        or "\\" in value
        or value.startswith("[")
        or value.endswith("]")
    ):
        raise ValueError("proxy host must be a safe bind address")
    if value.casefold().rstrip(".") == "localhost":
        return "127.0.0.1", True
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        if not _HOSTNAME_RE.fullmatch(value):
            raise ValueError("proxy host must be a valid IP address or DNS name") from exc
        return value.rstrip(".").casefold(), False
    return address.compressed, address.is_loopback


def _remote_access_contract(config: ProxyConfig) -> tuple[bool, str | None]:
    host, loopback = _classify_bind_host(config.host)
    config.host = host
    if loopback:
        return False, None
    if not _env_enabled("ENTROLY_ALLOW_REMOTE_PROXY"):
        raise ValueError(
            "remote proxy binding is disabled; set ENTROLY_ALLOW_REMOTE_PROXY=1 "
            "only when remote access is intentional"
        )
    token = _validated_access_token(os.environ.get("ENTROLY_PROXY_ACCESS_TOKEN"))
    if token is None:
        raise ValueError(
            "remote proxy binding requires ENTROLY_PROXY_ACCESS_TOKEN with "
            "32-512 URL-safe characters"
        )
    if not _env_enabled("ENTROLY_REMOTE_PROXY_TRUSTED_TRANSPORT"):
        raise ValueError(
            "remote proxy binding requires "
            "ENTROLY_REMOTE_PROXY_TRUSTED_TRANSPORT=1 after configuring TLS, "
            "a private tunnel, or an equivalently trusted network"
        )
    return True, token


async def _unauthorized(send: ASGISend) -> None:
    body = json.dumps(
        {
            "error": "entroly_access_denied",
            "detail": "A valid X-Entroly-Access-Token capability is required.",
        },
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store, max-age=0"),
                (b"www-authenticate", b'EntrolyToken realm="entroly-proxy"'),
                (b"x-content-type-options", b"nosniff"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class RemoteProxyAccessMiddleware:
    """Require one exact access capability and remove it before app dispatch."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        validated = _validated_access_token(token)
        if validated is None:
            raise ValueError("remote proxy access token is invalid")
        self.app = app
        self._token = validated.encode("ascii")

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        raw_headers = scope.get("headers")
        if not isinstance(raw_headers, list):
            await _unauthorized(send)
            return
        supplied: list[bytes] = []
        filtered: list[tuple[bytes, bytes]] = []
        for pair in raw_headers:
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or not isinstance(pair[0], bytes)
                or not isinstance(pair[1], bytes)
            ):
                await _unauthorized(send)
                return
            name, value = bytes(pair[0]), bytes(pair[1])
            if name.lower() == _ACCESS_HEADER:
                supplied.append(value)
            else:
                filtered.append((name, value))
        if (
            len(supplied) != 1
            or len(supplied[0]) > 512
            or not hmac.compare_digest(supplied[0], self._token)
        ):
            await _unauthorized(send)
            return
        protected_scope = dict(scope)
        protected_scope["headers"] = filtered
        await self.app(protected_scope, receive, send)


_ORIGINAL_CREATE_PROXY_APP = _proxy.create_proxy_app


def create_proxy_app(
    engine: Any,
    config: ProxyConfig | None = None,
    start_dashboard: bool = True,
    start_autotune: bool | None = None,
):
    """Create a proxy app with mandatory capability auth for remote binds."""
    selected = config if config is not None else ProxyConfig()
    remote, token = _remote_access_contract(selected)
    app = _ORIGINAL_CREATE_PROXY_APP(
        engine,
        selected,
        start_dashboard=start_dashboard,
        start_autotune=start_autotune,
    )
    app.state.remote_access_required = remote
    app.state.remote_bind_host = selected.host
    if remote:
        assert token is not None
        app.add_middleware(RemoteProxyAccessMiddleware, token=token)
    return app


_proxy.create_proxy_app = create_proxy_app

__all__ = [
    "RemoteProxyAccessMiddleware",
    "create_proxy_app",
]
