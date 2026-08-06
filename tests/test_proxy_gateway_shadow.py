from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from entroly.cache_routing import CacheAwareRouter
from entroly.gateway_control_plane import GatewayControlPlane
from entroly.provider_policy import GatewayRedactionPolicy
from entroly.proxy_gateway_shadow import (
    GatewayShadowObserver,
    _install_route,
    _run_shadow_handle_proxy,
)
from entroly.usage_ledger import UsagePricingCatalog


def _request(body: dict, *, path: str = "/v1/chat/completions") -> Request:
    payload = json.dumps(body).encode("utf-8")
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1:9377"),
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer secret-api-token"),
            (b"x-request-id", b"request-123"),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 9377),
    }
    return Request(scope, receive)


def _proxy(*, pricing: UsagePricingCatalog | None = None):
    router = CacheAwareRouter()
    observer = GatewayShadowObserver(
        control_plane=GatewayControlPlane(
            cache_router=router,
            redaction_policy=GatewayRedactionPolicy(enabled=False),
        ),
        max_records=4,
    )
    return SimpleNamespace(
        _cache_router=router,
        _pricing_catalog=pricing,
        _gateway_redaction=GatewayRedactionPolicy(enabled=False),
        _gateway_shadow_observer=observer,
        _escalation_ladder={"gpt-4o-mini": ("gpt-4o", 6.0)},
    )


def test_shadow_observation_keeps_the_legacy_proxy_authoritative() -> None:
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Keep answers grounded."},
            {"role": "user", "content": "Explain the change."},
        ],
    }
    original_body = copy.deepcopy(body)
    request = _request(body)
    proxy = _proxy()
    received: dict = {}

    async def original(_proxy, original_request):
        received.update(await original_request.json())
        return JSONResponse({"ok": True}, status_code=201)

    response = asyncio.run(_run_shadow_handle_proxy(proxy, request, original))

    assert received == original_body
    assert response.status_code == 201
    assert response.headers["x-entroly-gateway-shadow"] == "observed"
    assert response.headers["x-entroly-gateway-authoritative"] == "legacy-proxy"
    snapshot = proxy._gateway_shadow_observer.snapshot()
    assert snapshot["planned"] == 1
    assert snapshot["completed"] == 1
    assert snapshot["request_mutation"] is False
    assert snapshot["provider_switch_execution"] is False
    assert snapshot["automatic_retry_execution"] is False
    assert snapshot["recent"][0]["source_provider"] == "openai"
    assert snapshot["recent"][0]["planned_model"] == "gpt-4o-mini"
    assert snapshot["recent"][0]["boundary"] == "same-provider"
    assert response.headers["x-entroly-gateway-decision"] == "stay"


def test_shadow_does_not_invent_switch_without_auditable_pricing() -> None:
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "policy " * 2_000},
            {"role": "user", "content": "routine task"},
        ],
        "max_tokens": 2_000,
    }
    proxy = _proxy()
    observer = proxy._gateway_shadow_observer

    ticket = observer.observe_request(
        proxy,
        body=body,
        headers={},
        path="/v1/chat/completions",
        request_id="unpriced-case",
    )

    assert ticket is not None
    assert ticket.planned_model == "gpt-4o-mini"
    assert observer.snapshot()["would_switch"] == 0


def test_shadow_can_report_a_same_provider_would_switch_without_executing_it() -> None:
    catalog = UsagePricingCatalog.from_mapping(
        {
            "source": "test",
            "models": {
                "openai:gpt-4o-mini": {
                    "input_per_million": 100,
                    "output_per_million": 100,
                    "cache_read_per_million": 100,
                },
                "openai:gpt-4o": {
                    "input_per_million": 0.01,
                    "output_per_million": 0.01,
                    "cache_read_per_million": 0.01,
                },
            },
        }
    )
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "policy " * 2_000},
            {"role": "user", "content": "routine task"},
        ],
        "max_tokens": 2_000,
    }
    proxy = _proxy(pricing=catalog)
    observer = proxy._gateway_shadow_observer

    ticket = observer.observe_request(
        proxy,
        body=body,
        headers={"x-request-id": "switch-case"},
        path="/v1/chat/completions",
        request_id="switch-case",
    )

    assert ticket is not None
    assert ticket.source_provider == "openai"
    assert ticket.planned_provider == "openai"
    assert ticket.source_model == "gpt-4o-mini"
    assert ticket.planned_model == "gpt-4o"
    assert observer.snapshot()["would_switch"] == 1


def test_advanced_capability_request_keeps_unknown_alternate_out() -> None:
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Return structured data"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        },
    }
    proxy = _proxy()
    observer = proxy._gateway_shadow_observer

    ticket = observer.observe_request(
        proxy,
        body=body,
        headers={},
        path="/v1/chat/completions",
        request_id="schema-case",
    )

    assert ticket is not None
    assert ticket.planned_model == "gpt-4o-mini"
    assert ticket.excluded_targets == 0


def test_shadow_records_do_not_store_prompt_or_authentication_content() -> None:
    secret = "super-secret-prompt-value"
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": secret}],
    }
    proxy = _proxy()
    request = _request(body)

    async def original(_proxy, _request):
        return JSONResponse({"ok": True})

    asyncio.run(_run_shadow_handle_proxy(proxy, request, original))
    rendered = json.dumps(proxy._gateway_shadow_observer.snapshot(), sort_keys=True)

    assert secret not in rendered
    assert "secret-api-token" not in rendered
    assert "request-123" not in rendered
    assert proxy._gateway_shadow_observer.snapshot()[
        "records_contain_prompt_content"
    ] is False


def test_shadow_failure_never_blocks_the_authoritative_proxy() -> None:
    body = {
        "messages": [{"role": "user", "content": "hello"}],
    }
    request = _request(body, path="/custom/provider")
    proxy = _proxy()

    async def original(_proxy, original_request):
        assert await original_request.json() == body
        return JSONResponse({"ok": True}, status_code=202)

    response = asyncio.run(_run_shadow_handle_proxy(proxy, request, original))

    assert response.status_code == 202
    snapshot = proxy._gateway_shadow_observer.snapshot()
    assert snapshot["failures"] == 1
    assert snapshot["completed"] == 0


def test_shadow_installs_behind_existing_security_boundaries() -> None:
    from entroly import proxy, proxy_access_security, proxy_transport_safe

    assert (
        proxy.PromptCompilerProxy.handle_proxy
        is proxy_transport_safe._safe_handle_proxy
    )
    shadow_core = proxy_transport_safe._ORIGINAL_HANDLE_PROXY
    assert hasattr(shadow_core, "__entroly_gateway_shadow_original__")
    assert proxy.create_proxy_app is proxy_access_security.create_proxy_app
    shadow_factory = proxy_access_security._ORIGINAL_CREATE_PROXY_APP
    assert hasattr(shadow_factory, "__entroly_gateway_shadow_original__")


def test_gateway_shadow_sidecar_route_is_installed_once() -> None:
    app = Starlette()

    _install_route(app)
    _install_route(app)

    paths = [getattr(route, "path", "") for route in app.router.routes]
    assert paths.count("/gateway-shadow") == 1
