from __future__ import annotations

import pytest

from entroly.copilot_cli_provider_contract import (
    CopilotCLIProviderContractError,
    apply_copilot_cli_provider_contract,
)


def test_completions_uses_documented_api_key_surface_only() -> None:
    env = {
        "COPILOT_PROVIDER_BEARER_TOKEN": "stale-bearer",
        "COPILOT_PROVIDER_WIRE_API": "stale-wire",
    }
    summary = apply_copilot_cli_provider_contract(env, wire_api="completions")

    assert env["COPILOT_PROVIDER_API_KEY"] == "entroly-local-provider-route"
    assert "COPILOT_PROVIDER_BEARER_TOKEN" not in env
    assert "COPILOT_PROVIDER_WIRE_API" not in env
    assert summary == {
        "provider_auth_surface": "documented-api-key-placeholder",
        "wire_api": "completions",
        "wire_selector_experimental": False,
        "provider_secret_in_cli_env": False,
    }


def test_responses_keeps_explicit_runtime_wire_selector_but_marks_it_experimental() -> None:
    env: dict[str, str] = {}
    summary = apply_copilot_cli_provider_contract(env, wire_api="responses")

    assert env["COPILOT_PROVIDER_API_KEY"] == "entroly-local-provider-route"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "responses"
    assert "COPILOT_PROVIDER_BEARER_TOKEN" not in env
    assert summary["wire_selector_experimental"] is True


def test_provider_contract_rejects_unknown_wire_api() -> None:
    with pytest.raises(CopilotCLIProviderContractError, match="completions.*responses"):
        apply_copilot_cli_provider_contract({}, wire_api="private-api")
