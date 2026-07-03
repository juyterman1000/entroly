"""Proxy subscription-auth guard — the authoritative fix for proxying a Claude
Pro/Max login.

A subscription sends a first-party OAuth bearer (``sk-ant-oat…``, no
``x-api-key``) that the public Anthropic API rejects, producing a confusing
upstream 401/429 inside the client. The proxy must catch this precisely and
return actionable guidance, while NEVER false-blocking API-key or OpenAI
clients (for OpenAI, ``Authorization: Bearer <key>`` is the valid key).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from entroly.config import EntrolyConfig
from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig
from entroly.server import EntrolyEngine


@pytest.fixture
def proxy() -> PromptCompilerProxy:
    engine = EntrolyEngine(EntrolyConfig(use_persistent_index=False))
    return PromptCompilerProxy(engine, ProxyConfig())


def _payload(resp) -> dict:
    return json.loads(bytes(resp.body))


def test_blocks_anthropic_oauth_subscription(proxy):
    r = proxy._subscription_guard("anthropic", {"authorization": "Bearer sk-ant-oat01-abc"})
    assert r is not None and r.status_code == 400
    p = _payload(r)
    assert p["error"] == "subscription_not_proxyable"
    assert "claude mcp add entroly" in json.dumps(p)
    assert "entroly simulate" in json.dumps(p)


def test_allows_api_key_via_x_api_key(proxy):
    assert proxy._subscription_guard("anthropic", {"x-api-key": "sk-ant-api03-xyz"}) is None


def test_allows_api_key_sent_as_bearer(proxy):
    # Some SDKs send the API key as a bearer; that authenticates against the public API.
    assert proxy._subscription_guard("anthropic", {"authorization": "Bearer sk-ant-api03-xyz"}) is None


def test_never_blocks_openai_bearer(proxy):
    # For OpenAI, Authorization: Bearer <key> IS the valid API key — must never block,
    # even if the token superficially resembles an Anthropic OAuth token.
    assert proxy._subscription_guard("openai", {"authorization": "Bearer sk-ant-oat-lookalike"}) is None


def test_unknown_bearer_is_forwarded_fail_open(proxy):
    # Unrecognized bearer form → forward (the existing upstream handling decides).
    assert proxy._subscription_guard("anthropic", {"authorization": "Bearer mystery-token"}) is None


def test_no_subscription_bypass_switch(proxy):
    # Compliance: Entroly must NOT offer a way to proxy a first-party subscription
    # token. There is no opt-out header — the OAuth request is still blocked.
    r = proxy._subscription_guard(
        "anthropic",
        {"authorization": "Bearer sk-ant-oat01-abc", "x-entroly-allow-subscription": "1"},
    )
    assert r is not None and r.status_code == 400


def test_header_case_insensitive(proxy):
    r = proxy._subscription_guard("anthropic", {"Authorization": "Bearer sk-ant-oat01-abc"})
    assert r is not None and r.status_code == 400


def test_counter_increments_on_block(proxy):
    before = proxy._requests_subscription_blocked
    proxy._subscription_guard("anthropic", {"authorization": "Bearer sk-ant-oat01-z"})
    assert proxy._requests_subscription_blocked == before + 1


def test_no_auth_is_not_blocked(proxy):
    # Missing auth entirely → not the subscription case; let upstream 401 it.
    assert proxy._subscription_guard("anthropic", {}) is None


# ── CLI pre-flight (friendly fast-fail before launching the agent) ───────────
def test_cli_wrap_blocks_launch_without_api_key():
    """`entroly wrap claude` with no ANTHROPIC_API_KEY routes the user to MCP/
    simulate instead of launching a doomed proxy session."""
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-m", "entroly", "wrap", "claude"],
        capture_output=True, text=True, env=env, timeout=60,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "ANTHROPIC_API_KEY" in out and "claude mcp add entroly" in out
