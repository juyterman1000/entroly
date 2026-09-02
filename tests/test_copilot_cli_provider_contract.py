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
    assert env["ENTROLY_COPILOT_INTEGRATION_ID"] == "copilot-developer-cli"
    assert env["GITHUB_COPILOT_INTEGRATION_ID"] == "copilot-developer-cli"
    assert summary == {
        "provider_auth_surface": "documented-api-key-placeholder",
        "wire_api": "completions",
        "wire_selector_experimental": False,
        "provider_secret_in_cli_env": False,
        "integration_id": "copilot-developer-cli",
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


def test_runtime_integration_identity_is_propagated_to_entroly() -> None:
    env = {"GITHUB_COPILOT_INTEGRATION_ID": "my-product-agent"}
    summary = apply_copilot_cli_provider_contract(env, wire_api="completions")

    assert env["GITHUB_COPILOT_INTEGRATION_ID"] == "my-product-agent"
    assert env["ENTROLY_COPILOT_INTEGRATION_ID"] == "my-product-agent"
    assert summary["integration_id"] == "my-product-agent"


def test_entroly_integration_identity_is_propagated_to_runtime() -> None:
    env = {"ENTROLY_COPILOT_INTEGRATION_ID": "enterprise-agent"}
    summary = apply_copilot_cli_provider_contract(env, wire_api="completions")

    assert env["GITHUB_COPILOT_INTEGRATION_ID"] == "enterprise-agent"
    assert env["ENTROLY_COPILOT_INTEGRATION_ID"] == "enterprise-agent"
    assert summary["integration_id"] == "enterprise-agent"


def test_conflicting_integration_identities_fail_closed() -> None:
    env = {
        "ENTROLY_COPILOT_INTEGRATION_ID": "entroly-side",
        "GITHUB_COPILOT_INTEGRATION_ID": "runtime-side",
    }
    with pytest.raises(CopilotCLIProviderContractError, match="must match"):
        apply_copilot_cli_provider_contract(env, wire_api="completions")


def test_integration_identity_rejects_header_unsafe_values() -> None:
    env = {"GITHUB_COPILOT_INTEGRATION_ID": "good\r\nX-Evil: 1"}
    with pytest.raises(CopilotCLIProviderContractError, match="ASCII letters"):
        apply_copilot_cli_provider_contract(env, wire_api="completions")


def test_each_integration_id_rejection_names_its_real_cause() -> None:
    """An error that misstates its cause sends the operator to fix the wrong thing.

    The length check and the character check shared one message that named only
    the character set, so a 500-character ID composed entirely of legal
    characters was told its characters were wrong.
    """
    import pytest

    from entroly.copilot_cli_provider_contract import (
        CopilotCLIProviderContractError,
        configure_copilot_integration_identity,
    )

    with pytest.raises(CopilotCLIProviderContractError) as too_long:
        configure_copilot_integration_identity(
            {"GITHUB_COPILOT_INTEGRATION_ID": "A" * 500}
        )
    message = str(too_long.value)
    assert "500" in message and "128" in message, message
    assert "must contain only" not in message, (
        "a length rejection must not blame the character set"
    )

    with pytest.raises(CopilotCLIProviderContractError) as bad_chars:
        configure_copilot_integration_identity(
            {"GITHUB_COPILOT_INTEGRATION_ID": "not a valid id!"}
        )
    assert "must contain only" in str(bad_chars.value)

    # The boundary itself stays accepted.
    assert configure_copilot_integration_identity(
        {"GITHUB_COPILOT_INTEGRATION_ID": "A" * 128}
    ) == "A" * 128
