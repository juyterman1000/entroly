from __future__ import annotations

from copy import deepcopy

from entroly.gateway_shadow import GatewayShadowObserver


def test_shadow_observer_is_read_only_and_same_provider() -> None:
    observer = GatewayShadowObserver()
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Explain the result."},
        ],
        "stream": True,
        "max_tokens": 200,
    }
    original = deepcopy(body)

    receipt = observer.observe(
        provider="openai",
        body=body,
        headers={"x-request-id": "req-123"},
        path="/v1/chat/completions",
        request_id="req-123",
    )

    assert receipt.succeeded
    assert receipt.body_unchanged
    assert body == original
    assert receipt.provider == "openai"
    assert receipt.current_model == "gpt-4o-mini"
    assert receipt.planned_model == "gpt-4o-mini"
    assert receipt.executable_targets == ("openai:gpt-4o-mini",)
    assert "streaming" in receipt.required_capabilities
    assert receipt.conversation_id


def test_shadow_receipt_contains_no_prompt_or_credentials() -> None:
    observer = GatewayShadowObserver()
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": f"token {secret}"}],
    }

    receipt = observer.observe(
        provider="openai",
        body=body,
        path="/v1/chat/completions",
        request_id="req-secret",
    )
    rendered = str(receipt.to_dict()) + str(receipt.headers())

    assert receipt.succeeded
    assert secret not in rendered
    assert "token" not in rendered
    assert receipt.headers()["X-Entroly-Gateway-Shadow-Unchanged"] == "1"


def test_shadow_observer_fails_closed_without_mutating_unknown_provider() -> None:
    observer = GatewayShadowObserver()
    body = {"model": "custom", "messages": [{"role": "user", "content": "hi"}]}
    original = deepcopy(body)

    receipt = observer.observe(
        provider="unknown-provider",
        body=body,
        path="/v1/chat/completions",
        request_id="req-unknown",
    )

    assert not receipt.succeeded
    assert receipt.route_reason == "shadow_error"
    assert receipt.body_unchanged
    assert body == original
    assert receipt.error


def test_shadow_observer_preserves_tool_capability_evidence() -> None:
    observer = GatewayShadowObserver()
    body = {
        "model": "claude-sonnet-4-20250514",
        "system": "Use tools when required.",
        "messages": [{"role": "user", "content": "Check status."}],
        "tools": [
            {
                "name": "status",
                "description": "Read status",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    }

    receipt = observer.observe(
        provider="anthropic",
        body=body,
        path="/v1/messages",
        request_id="req-tools",
    )

    assert receipt.succeeded
    assert "tools" in receipt.required_capabilities
    assert receipt.executable_targets == (
        "anthropic:claude-sonnet-4-20250514",
    )
