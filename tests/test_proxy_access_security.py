from __future__ import annotations

import asyncio

import pytest

import entroly.proxy as proxy_module
import entroly.proxy_access_security as security
from entroly.proxy_config import ProxyConfig


def _clear_remote_env(monkeypatch) -> None:
    for name in (
        "ENTROLY_ALLOW_REMOTE_PROXY",
        "ENTROLY_PROXY_ACCESS_TOKEN",
        "ENTROLY_REMOTE_PROXY_TRUSTED_TRANSPORT",
    ):
        monkeypatch.delenv(name, raising=False)


def _enable_remote(monkeypatch, token: str) -> None:
    monkeypatch.setenv("ENTROLY_ALLOW_REMOTE_PROXY", "1")
    monkeypatch.setenv("ENTROLY_PROXY_ACCESS_TOKEN", token)
    monkeypatch.setenv("ENTROLY_REMOTE_PROXY_TRUSTED_TRANSPORT", "1")


def test_proxy_public_factory_is_access_hardened() -> None:
    assert proxy_module.create_proxy_app is security.create_proxy_app


@pytest.mark.parametrize(
    ("host", "expected", "loopback"),
    [
        ("localhost", "127.0.0.1", True),
        ("LOCALHOST.", "127.0.0.1", True),
        ("127.0.0.1", "127.0.0.1", True),
        ("127.9.8.7", "127.9.8.7", True),
        ("::1", "::1", True),
        ("0.0.0.0", "0.0.0.0", False),
        ("::", "::", False),
        ("10.0.0.8", "10.0.0.8", False),
        ("Proxy.Internal.", "proxy.internal", False),
    ],
)
def test_proxy_bind_hosts_are_classified_without_dns_resolution(
    host: str, expected: str, loopback: bool
) -> None:
    assert security._classify_bind_host(host) == (expected, loopback)


@pytest.mark.parametrize(
    "host",
    [
        "",
        None,
        "http://127.0.0.1",
        "127.0.0.1/path",
        "[::1]",
        "bad host",
        "*.example.com",
        "example..com",
        "host\nname",
    ],
)
def test_malformed_proxy_bind_hosts_are_rejected(host) -> None:
    with pytest.raises(ValueError, match="host"):
        security._classify_bind_host(host)


def test_loopback_proxy_needs_no_remote_capability(monkeypatch) -> None:
    _clear_remote_env(monkeypatch)
    config = ProxyConfig(host="localhost")

    remote, token = security._remote_access_contract(config)

    assert remote is False
    assert token is None
    assert config.host == "127.0.0.1"


def test_remote_proxy_requires_explicit_allow_flag(monkeypatch) -> None:
    _clear_remote_env(monkeypatch)

    with pytest.raises(ValueError, match="disabled"):
        security._remote_access_contract(ProxyConfig(host="0.0.0.0"))


def test_remote_proxy_requires_valid_separate_token(monkeypatch) -> None:
    _clear_remote_env(monkeypatch)
    monkeypatch.setenv("ENTROLY_ALLOW_REMOTE_PROXY", "1")
    monkeypatch.setenv("ENTROLY_REMOTE_PROXY_TRUSTED_TRANSPORT", "1")

    for token in (None, "short", "x" * 513, "x" * 31 + "\n", "<script>" + "x" * 32):
        if token is None:
            monkeypatch.delenv("ENTROLY_PROXY_ACCESS_TOKEN", raising=False)
        else:
            monkeypatch.setenv("ENTROLY_PROXY_ACCESS_TOKEN", token)
        with pytest.raises(ValueError, match="ACCESS_TOKEN"):
            security._remote_access_contract(ProxyConfig(host="0.0.0.0"))


def test_remote_proxy_requires_trusted_transport_acknowledgement(monkeypatch) -> None:
    _clear_remote_env(monkeypatch)
    monkeypatch.setenv("ENTROLY_ALLOW_REMOTE_PROXY", "1")
    monkeypatch.setenv("ENTROLY_PROXY_ACCESS_TOKEN", "a" * 48)

    with pytest.raises(ValueError, match="TRUSTED_TRANSPORT"):
        security._remote_access_contract(ProxyConfig(host="0.0.0.0"))


class _FakeEngine:
    def stats(self):
        return {}


def test_loopback_app_health_remains_zero_configuration(monkeypatch) -> None:
    from httpx import ASGITransport, AsyncClient

    _clear_remote_env(monkeypatch)
    app = security.create_proxy_app(
        _FakeEngine(),
        ProxyConfig(host="127.0.0.1"),
        start_dashboard=False,
        start_autotune=False,
    )

    async def run():
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://127.0.0.1:9377",
        ) as client:
            return await client.get("/health")

    response = asyncio.run(run())

    assert response.status_code == 200
    assert app.state.remote_access_required is False


def test_remote_app_rejects_missing_wrong_duplicate_and_query_tokens(monkeypatch) -> None:
    from httpx import ASGITransport, AsyncClient

    token = "remote_capability_" + "x" * 32
    _enable_remote(monkeypatch, token)
    app = security.create_proxy_app(
        _FakeEngine(),
        ProxyConfig(host="0.0.0.0"),
        start_dashboard=False,
        start_autotune=False,
    )

    async def run():
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://proxy.internal:9377",
        ) as client:
            missing = await client.get("/health")
            wrong = await client.get(
                "/health",
                headers={"X-Entroly-Access-Token": "wrong" * 10},
            )
            query = await client.get(f"/health?access_token={token}")
            duplicate = await client.get(
                "/health",
                headers=[
                    ("X-Entroly-Access-Token", token),
                    ("X-Entroly-Access-Token", token),
                ],
            )
            allowed = await client.get(
                "/health",
                headers={"X-Entroly-Access-Token": token},
            )
            return missing, wrong, query, duplicate, allowed

    missing, wrong, query, duplicate, allowed = asyncio.run(run())

    for denied in (missing, wrong, query, duplicate):
        assert denied.status_code == 401
        assert denied.json()["error"] == "entroly_access_denied"
        assert token not in denied.text
        assert denied.headers["cache-control"].startswith("no-store")
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "ok"
    assert app.state.remote_access_required is True
    assert app.state.remote_bind_host == "0.0.0.0"
    assert token not in repr(app.user_middleware)


def test_access_header_is_removed_before_downstream_dispatch() -> None:
    token = "t" * 48
    observed: dict[str, object] = {}
    events: list[dict] = []

    async def downstream(scope, _receive, send) -> None:
        observed["headers"] = scope["headers"]
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(event):
        events.append(event)

    middleware = security.RemoteProxyAccessMiddleware(
        downstream,
        token_digest=security._access_digest(token),
    )
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "headers": [
            (b"x-entroly-access-token", token.encode("ascii")),
            (b"authorization", b"Bearer provider-key"),
        ],
    }

    asyncio.run(middleware(scope, receive, send))

    headers = observed["headers"]
    assert (b"authorization", b"Bearer provider-key") in headers
    assert not any(name.lower() == b"x-entroly-access-token" for name, _value in headers)
    assert events[0]["status"] == 204


def test_invalid_asgi_header_shape_fails_closed_without_dispatch() -> None:
    token = "z" * 48
    called = False
    events: list[dict] = []

    async def downstream(_scope, _receive, _send) -> None:
        nonlocal called
        called = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(event):
        events.append(event)

    middleware = security.RemoteProxyAccessMiddleware(
        downstream,
        token_digest=security._access_digest(token),
    )
    scope = {
        "type": "http",
        "headers": [("x-entroly-access-token", token.encode("ascii"))],
    }

    asyncio.run(middleware(scope, receive, send))

    assert not called
    assert events[0]["status"] == 401


def test_websocket_scope_fails_closed_and_lifespan_still_reaches_app() -> None:
    token = "w" * 48
    calls: list[str] = []
    events: list[dict] = []

    async def downstream(scope, _receive, send) -> None:
        calls.append(scope["type"])
        await send({"type": "lifespan.startup.complete"})

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(event):
        events.append(event)

    middleware = security.RemoteProxyAccessMiddleware(
        downstream,
        token_digest=security._access_digest(token),
    )

    asyncio.run(middleware({"type": "websocket"}, receive, send))
    assert events[-1]["type"] == "websocket.close"
    assert events[-1]["code"] == 4401
    assert calls == []

    asyncio.run(middleware({"type": "lifespan"}, receive, send))
    assert calls == ["lifespan"]
