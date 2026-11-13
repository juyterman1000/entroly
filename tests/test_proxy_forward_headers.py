"""Operator-selected header forwarding (ENTROLY_FORWARD_HEADERS).

The proxy forwards a provider's own header families and drops every other
client header. A gateway *behind* Entroly (usage attribution, tracing, a
benchmark harness) keys on headers of its own, which that default strips.
The option names the ones to keep. These tests pin both halves: the default
is unchanged, and the option cannot be used to send a credential, a cookie,
a transport header or an Entroly control header upstream.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from entroly.proxy import PromptCompilerProxy, create_proxy_app
from entroly.proxy_config import ProxyConfig, forward_header_rules

CLIENT_HEADERS = {
    "x-api-key": "sk-ant-api-test",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
    "x-ccb-run-id": "run-17",
    "x-team-id": "payments",
    "x-team-cost-center": "cc-42",
    "x-internal-secret": "do-not-leak",
    "cookie": "session=abc",
    "x-entroly-bypass": "1",
    "connection": "keep-alive",
}


def _forwarded(spec: str, provider: str = "anthropic", headers=None) -> dict[str, str]:
    proxy = PromptCompilerProxy(object(), ProxyConfig(forward_headers=spec))
    out = proxy._build_headers(dict(headers or CLIENT_HEADERS), provider)
    return {key.lower(): value for key, value in out.items()}


def test_default_forwards_only_the_providers_own_headers() -> None:
    out = _forwarded("")
    assert out["x-api-key"] == "sk-ant-api-test"
    assert out["anthropic-version"] == "2023-06-01"
    for dropped in ("x-ccb-run-id", "x-team-id", "x-internal-secret", "cookie",
                    "x-entroly-bypass", "connection"):
        assert dropped not in out


def test_named_header_is_forwarded_and_nothing_else_is() -> None:
    out = _forwarded("x-ccb-run-id")
    assert out["x-ccb-run-id"] == "run-17"
    assert "x-team-id" not in out
    assert "x-internal-secret" not in out


def test_prefix_entry_forwards_the_family() -> None:
    out = _forwarded(" X-CCB-Run-Id , x-team-* ")
    assert out["x-ccb-run-id"] == "run-17"
    assert out["x-team-id"] == "payments"
    assert out["x-team-cost-center"] == "cc-42"
    assert "x-internal-secret" not in out


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_forwarding_applies_to_every_provider(provider) -> None:
    assert _forwarded("x-ccb-run-id", provider)["x-ccb-run-id"] == "run-17"


@pytest.mark.parametrize(
    "spec",
    [
        "cookie", "set-cookie", "authorization", "proxy-authorization",
        "x-api-key", "api-key", "x-goog-api-key",
        "host", "content-length", "transfer-encoding", "connection",
        "accept-encoding", "x-entroly-bypass",
        # prefixes that would reach a refused header
        "x-entroly-*", "x-ent*", "coo*", "auth*", "x-api*", "proxy-*", "sec-*",
        "content-*", "x-goog-api*",
        # too broad or malformed
        "*", "x*", "x-*"[:2] + "*", "bad header", "x-run id", "",
    ],
)
def test_refused_entries_forward_nothing_extra(spec) -> None:
    rules = forward_header_rules(spec)
    assert not rules.names and not rules.prefixes
    baseline = _forwarded("")
    assert _forwarded(spec) == baseline


def test_refused_headers_stay_out_even_under_a_valid_prefix() -> None:
    headers = {**CLIENT_HEADERS, "x-team-id": "payments"}
    rules = forward_header_rules("x-team-*")
    assert rules.allows("x-team-id")
    for name in ("cookie", "authorization", "x-api-key", "x-entroly-bypass", "host"):
        assert not rules.allows(name)
    assert "cookie" not in _forwarded("x-team-*", headers=headers)


def test_a_credential_is_never_sent_to_another_provider_through_the_option() -> None:
    # Anthropic's key header on an OpenAI-bound request is governed by the
    # provider rule; listing it must not create a second way through.
    headers = {"x-goog-api-key": "g-secret", "authorization": "Bearer sk-openai"}
    out = _forwarded("x-goog-api-key", "openai", headers)
    assert "x-goog-api-key" not in out
    assert out["authorization"] == "Bearer sk-openai"


def test_valid_and_refused_entries_in_one_list(caplog) -> None:
    forward_header_rules.cache_clear()
    with caplog.at_level(logging.INFO, logger="entroly.proxy"):
        rules = forward_header_rules("x-ccb-run-id, cookie, x-trace-*, *")
    assert rules.names == frozenset({"x-ccb-run-id"})
    assert rules.prefixes == ("x-trace-",)
    assert rules.rejected == ("cookie", "*")
    assert rules.describe() == ["x-ccb-run-id", "x-trace-*"]
    text = caplog.text
    assert "ignored cookie, *" in text
    assert "x-ccb-run-id, x-trace-*" in text


def test_option_is_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_FORWARD_HEADERS", "x-ccb-run-id")
    assert ProxyConfig.from_env().forward_headers == "x-ccb-run-id"
    monkeypatch.delenv("ENTROLY_FORWARD_HEADERS")
    assert ProxyConfig.from_env().forward_headers == ""


class _Engine:
    """Smallest engine the proxy accepts; the pipeline is not under test."""

    def __getattr__(self, name):
        raise AttributeError(name)


def _upstream_headers(monkeypatch, spec: str) -> dict[str, str]:
    """Send one request through the real proxy app; return what upstream saw."""
    monkeypatch.setenv("ENTROLY_DISABLE_TELEMETRY", "1")
    seen: dict[str, str] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.update({key.lower(): value for key, value in request.headers.items()})
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4-5", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 1},
        })

    app = create_proxy_app(_Engine(), ProxyConfig(forward_headers=spec, witness_mode="off"))

    async def go():
        app.state.proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://proxy.test") as client:
            response = await client.post(
                "/v1/messages",
                content=json.dumps({
                    "model": "claude-sonnet-4-5", "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                }),
                headers=CLIENT_HEADERS,
            )
            await response.aread()
        return response

    response = asyncio.run(go())
    assert response.status_code == 200, response.text
    return seen


def test_end_to_end_the_run_id_reaches_the_upstream_only_when_named(monkeypatch) -> None:
    assert "x-ccb-run-id" not in _upstream_headers(monkeypatch, "")
    seen = _upstream_headers(monkeypatch, "x-ccb-run-id")
    assert seen["x-ccb-run-id"] == "run-17"
    assert "x-internal-secret" not in seen
    assert "cookie" not in seen
    assert "x-entroly-bypass" not in seen
