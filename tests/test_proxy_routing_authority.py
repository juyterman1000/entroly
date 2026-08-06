from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from entroly.cache_routing import CacheAwareRouter
from entroly.provider_policy import Capability, GatewayRedactionPolicy, ProviderTarget
from entroly.proxy_routing_authority import (
    RoutingAuthorityDenied,
    _coordinated_apply_target_same_provider,
    _install_route,
    _run_authority_handle_proxy,
)
from entroly.usage_ledger import UsagePricingCatalog


def _request(
    body: dict,
    *,
    path: str = "/v1/chat/completions",
    authorization: str = "Bearer secret-api-token",
) -> Request:
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
            (b"authorization", authorization.encode("utf-8")),
            (b"x-request-id", b"request-123"),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 9377),
    }
    return Request(scope, receive)


def _catalog() -> UsagePricingCatalog:
    return UsagePricingCatalog.from_mapping(
        {
            "source": "routing-authority-test",
            "models": {
                "openai:gpt-4o": {
                    "input_per_million": 10,
                    "output_per_million": 30,
                    "cache_read_per_million": 5,
                },
                "openai:gpt-4o-mini": {
                    "input_per_million": 0.15,
                    "output_per_million": 0.60,
                    "cache_read_per_million": 0.075,
                },
            },
        }
    )


def _proxy(
    *,
    pricing: UsagePricingCatalog | None = None,
    escalation_mode: str = "observe",
):
    return SimpleNamespace(
        _cache_router=CacheAwareRouter(),
        _pricing_catalog=pricing,
        _usage_ledger=None,
        _gateway_redaction=GatewayRedactionPolicy(enabled=False),
        _escalation_mode=escalation_mode,
    )


def _target(
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> ProviderTarget:
    return ProviderTarget(
        provider=provider,
        model=model,
        capabilities=frozenset({Capability.CHAT, Capability.STREAMING}),
    )


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    require_pricing: bool = True,
) -> None:
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY", "1")
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY_MODE", mode)
    monkeypatch.setenv(
        "ENTROLY_ROUTING_AUTHORITY_REQUIRE_PRICING",
        "1" if require_pricing else "0",
    )


def test_execute_mode_runs_one_same_provider_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, mode="execute")
    proxy = _proxy(pricing=_catalog())
    request = _request(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Summarize this."}],
            "max_tokens": 200,
        }
    )

    async def original(_proxy, original_request):
        body = await original_request.json()
        rewritten, rewritten_url = _coordinated_apply_target_same_provider(
            provider="openai",
            target=_target(),
            body=body,
            url="https://api.openai.com/v1/chat/completions",
        )
        assert rewritten_url == "https://api.openai.com/v1/chat/completions"
        return JSONResponse({"model": rewritten["model"]}, status_code=201)

    response = asyncio.run(
        _run_authority_handle_proxy(proxy, request, original)
    )

    assert json.loads(response.body)["model"] == "gpt-4o-mini"
    assert response.headers["x-entroly-routing-decision"] == "executed"
    snapshot = proxy._routing_authority_ledger.snapshot()
    assert snapshot["proposals"] == 1
    assert snapshot["executed"] == 1
    assert snapshot["denied"] == 0
    receipt = snapshot["recent"][0]
    assert receipt["source_provider"] == "openai"
    assert receipt["source_model"] == "gpt-4o"
    assert receipt["executed_model"] == "gpt-4o-mini"
    assert receipt["gateway_plan_id"]
    assert receipt["gateway_route_reason"] == "forced_model"
    assert receipt["mutation_count"] == 1
    assert receipt["estimated_target_micro_usd"] < receipt[
        "estimated_source_micro_usd"
    ]


def test_observe_mode_records_would_execute_but_forwards_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, mode="observe")
    proxy = _proxy(pricing=_catalog())
    request = _request(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Explain this."}],
        }
    )

    async def original(_proxy, original_request):
        body = await original_request.json()
        try:
            _coordinated_apply_target_same_provider(
                provider="openai",
                target=_target(),
                body=body,
                url="https://api.openai.com/v1/chat/completions",
            )
        except RoutingAuthorityDenied:
            pass
        return JSONResponse({"model": body["model"]})

    response = asyncio.run(
        _run_authority_handle_proxy(proxy, request, original)
    )

    assert json.loads(response.body)["model"] == "gpt-4o"
    snapshot = proxy._routing_authority_ledger.snapshot()
    assert snapshot["executed"] == 0
    assert snapshot["denied"] == 1
    receipt = snapshot["recent"][0]
    assert receipt["decision"] == "denied"
    assert receipt["reason"] == "observe_only_would_execute"
    assert receipt["gateway_plan_id"]
    assert receipt["mutation_count"] == 0


@pytest.mark.parametrize(
    ("provider", "target", "escalation_mode", "pricing", "expected_reason"),
    [
        (
            "openai",
            ProviderTarget(
                provider="anthropic",
                model="claude-3-haiku-20240307",
                capabilities=frozenset({Capability.CHAT}),
            ),
            "observe",
            True,
            "cross_provider_target_disabled",
        ),
        (
            "openai",
            _target(),
            "active",
            True,
            "competing_escalation_authority",
        ),
        (
            "openai",
            _target(),
            "observe",
            False,
            "auditable_pricing_required",
        ),
    ],
)
def test_execution_conflicts_fail_original(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    target: ProviderTarget,
    escalation_mode: str,
    pricing: bool,
    expected_reason: str,
) -> None:
    _configure(monkeypatch, mode="execute")
    proxy = _proxy(
        pricing=_catalog() if pricing else None,
        escalation_mode=escalation_mode,
    )
    request = _request(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    async def original(_proxy, original_request):
        body = await original_request.json()
        try:
            _coordinated_apply_target_same_provider(
                provider=provider,
                target=target,
                body=body,
                url="https://api.openai.com/v1/chat/completions",
            )
        except RoutingAuthorityDenied:
            pass
        return JSONResponse({"model": body["model"]})

    response = asyncio.run(
        _run_authority_handle_proxy(proxy, request, original)
    )

    assert json.loads(response.body)["model"] == "gpt-4o"
    receipt = proxy._routing_authority_ledger.snapshot()["recent"][0]
    assert receipt["decision"] == "denied"
    assert receipt["reason"] == expected_reason
    if expected_reason == "competing_escalation_authority":
        assert proxy._routing_authority_ledger.snapshot()["conflicts"] == 1


def test_unproven_advanced_capability_remains_observe_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, mode="execute")
    proxy = _proxy(pricing=_catalog())
    request = _request(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                },
            },
        }
    )

    async def original(_proxy, original_request):
        body = await original_request.json()
        try:
            _coordinated_apply_target_same_provider(
                provider="openai",
                target=_target(),
                body=body,
                url="https://api.openai.com/v1/chat/completions",
            )
        except RoutingAuthorityDenied:
            pass
        return JSONResponse({"model": body["model"]})

    asyncio.run(_run_authority_handle_proxy(proxy, request, original))

    receipt = proxy._routing_authority_ledger.snapshot()["recent"][0]
    assert receipt["decision"] == "denied"
    assert receipt["reason"] == "target_capability_unproven:json_schema"
    assert "json_schema" in receipt["required_capabilities"]


def test_second_model_mutation_is_blocked_and_recorded_as_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, mode="execute")
    proxy = _proxy(pricing=_catalog())
    request = _request(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Summarize."}],
        }
    )

    async def original(_proxy, original_request):
        body = await original_request.json()
        rewritten, url = _coordinated_apply_target_same_provider(
            provider="openai",
            target=_target(),
            body=body,
            url="https://api.openai.com/v1/chat/completions",
        )
        with pytest.raises(RoutingAuthorityDenied):
            _coordinated_apply_target_same_provider(
                provider="openai",
                target=_target(model="gpt-4o"),
                body=rewritten,
                url=url,
            )
        return JSONResponse({"model": rewritten["model"]})

    response = asyncio.run(
        _run_authority_handle_proxy(proxy, request, original)
    )

    assert json.loads(response.body)["model"] == "gpt-4o-mini"
    snapshot = proxy._routing_authority_ledger.snapshot()
    assert snapshot["executed"] == 1
    assert snapshot["conflicts"] == 1
    receipt = snapshot["recent"][0]
    assert receipt["decision"] == "executed"
    assert receipt["conflict_reason"] == "multiple_model_mutation_attempts"


def test_receipts_never_store_prompt_credentials_or_raw_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, mode="observe")
    secret_prompt = "private prompt value 9f17"
    secret_token = "Bearer top-secret-token-443"
    proxy = _proxy(pricing=_catalog())
    request = _request(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": secret_prompt}],
        },
        authorization=secret_token,
    )

    async def original(_proxy, original_request):
        body = await original_request.json()
        try:
            _coordinated_apply_target_same_provider(
                provider="openai",
                target=_target(),
                body=body,
                url="https://api.openai.com/v1/chat/completions",
            )
        except RoutingAuthorityDenied:
            pass
        return JSONResponse({"ok": True})

    asyncio.run(_run_authority_handle_proxy(proxy, request, original))
    rendered = json.dumps(
        proxy._routing_authority_ledger.snapshot(),
        sort_keys=True,
    )

    assert secret_prompt not in rendered
    assert secret_token not in rendered
    assert "request-123" not in rendered
    snapshot = proxy._routing_authority_ledger.snapshot()
    assert snapshot["records_contain_prompt_content"] is False
    assert snapshot["records_contain_credentials"] is False


def test_disabled_coordinator_preserves_existing_rewrite_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ENTROLY_ROUTING_AUTHORITY", raising=False)
    proxy = _proxy(pricing=None)
    request = _request(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    async def original(_proxy, original_request):
        body = await original_request.json()
        rewritten, _ = _coordinated_apply_target_same_provider(
            provider="openai",
            target=_target(),
            body=body,
            url="https://api.openai.com/v1/chat/completions",
        )
        return JSONResponse({"model": rewritten["model"]})

    response = asyncio.run(
        _run_authority_handle_proxy(proxy, request, original)
    )

    assert json.loads(response.body)["model"] == "gpt-4o-mini"
    assert not hasattr(proxy, "_routing_authority_ledger")


def test_routing_authority_installs_behind_existing_security_boundaries() -> None:
    from entroly import (
        proxy,
        proxy_access_security,
        proxy_transport_safe,
    )

    assert (
        proxy.PromptCompilerProxy.handle_proxy
        is proxy_transport_safe._safe_handle_proxy
    )
    authority_core = proxy_transport_safe._ORIGINAL_HANDLE_PROXY
    assert hasattr(
        authority_core,
        "__entroly_routing_authority_original__",
    )
    assert hasattr(
        authority_core,
        "__entroly_gateway_shadow_original__",
    )
    assert proxy.create_proxy_app is proxy_access_security.create_proxy_app


def test_routing_authority_sidecar_route_is_installed_once() -> None:
    app = Starlette()

    _install_route(app)
    _install_route(app)

    paths = [getattr(route, "path", "") for route in app.router.routes]
    assert paths.count("/routing-authority") == 1
