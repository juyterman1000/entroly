from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from entroly.copilot_capi_contract import (
    CopilotCAPIContractError,
    build_copilot_capi_headers,
)


def _base_forwarded() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer short-lived-copilot-token",
        "User-Agent": "generic-http-client/1.0",
        "Copilot-Integration-Id": "untrusted-client-id",
        "X-GitHub-Api-Version": "1900-01-01",
        "X-Interaction-Id": "not-trusted",
    }


def test_contract_synthesizes_current_required_capi_headers() -> None:
    headers = build_copilot_capi_headers(
        original={},
        forwarded=_base_forwarded(),
        integration_id="copilot-cli-chat",
        environ={},
    )

    assert headers["Authorization"] == "Bearer short-lived-copilot-token"
    assert headers["Copilot-Integration-Id"] == "copilot-cli-chat"
    assert headers["X-GitHub-Api-Version"] == "2025-04-01"
    assert "copilot" in headers["User-Agent"].casefold()
    assert "Editor-Version" not in headers
    assert "Editor-Plugin-Version" not in headers
    parsed = uuid.UUID(headers["X-Interaction-Id"])
    assert parsed.version == 4


def test_contract_preserves_valid_client_correlation_api_and_editor_metadata() -> None:
    interaction = "123e4567-e89b-12d3-a456-426614174000"
    headers = build_copilot_capi_headers(
        original={
            "User-Agent": "GitHubCopilotCLI/1.2.3",
            "x-github-api-version": "2026-06-01",
            "X-Interaction-ID": interaction,
            "X-Initiator": "AGENT",
            "OpenAI-Intent": "conversation-edits",
            "Editor-Version": "copilot-cli/1.2.3",
            "Editor-Plugin-Version": "copilot-cli/1.2.3",
            "Copilot-Vision-Request": "TRUE",
        },
        forwarded={
            "Authorization": "Bearer token",
            # A lower transport/auth layer may have its own synthetic identity;
            # client identity must come from ``original`` instead.
            "User-Agent": "GithubCopilot/1.0",
        },
        integration_id="copilot-cli-chat",
        environ={},
    )

    assert headers["X-GitHub-Api-Version"] == "2026-06-01"
    assert headers["X-Interaction-Id"] == interaction
    assert headers["X-Initiator"] == "agent"
    assert headers["OpenAI-Intent"] == "conversation-edits"
    assert headers["Editor-Version"] == "copilot-cli/1.2.3"
    assert headers["Editor-Plugin-Version"] == "copilot-cli/1.2.3"
    assert headers["Copilot-Vision-Request"] == "true"
    assert headers["User-Agent"] == "GitHubCopilotCLI/1.2.3"


def test_contract_never_promotes_lower_layer_synthetic_client_identity() -> None:
    headers = build_copilot_capi_headers(
        original={"User-Agent": "generic-client/9.0"},
        forwarded={
            "Authorization": "Bearer token",
            "User-Agent": "GithubCopilot/1.0",
            "Editor-Version": "vscode/1.0",
            "Editor-Plugin-Version": "copilot/1.0",
        },
        integration_id="copilot-cli-chat",
        environ={},
    )

    assert headers["User-Agent"].startswith("Entroly-Copilot/")
    assert headers["User-Agent"] != "GithubCopilot/1.0"
    assert "Editor-Version" not in headers
    assert "Editor-Plugin-Version" not in headers


def test_contract_does_not_trust_client_token_bound_identity_or_bad_metadata() -> None:
    headers = build_copilot_capi_headers(
        original={
            "Copilot-Integration-Id": "attacker-controlled",
            "X-Interaction-Id": "bad\r\nX-Evil: injected",
            "X-Initiator": "root",
            "Editor-Version": "bad\r\nX-Evil: injected",
            "Copilot-Vision-Request": "yes-please",
        },
        forwarded=_base_forwarded(),
        integration_id="copilot-cli-chat",
        environ={},
    )

    assert headers["Copilot-Integration-Id"] == "copilot-cli-chat"
    assert "X-Evil" not in repr(headers)
    assert "X-Initiator" not in headers
    assert "Editor-Version" not in headers
    assert "Copilot-Vision-Request" not in headers
    uuid.UUID(headers["X-Interaction-Id"])


def test_explicit_api_version_override_is_validated_fail_closed() -> None:
    with pytest.raises(CopilotCAPIContractError, match="ISO date"):
        build_copilot_capi_headers(
            original={},
            forwarded={"Authorization": "Bearer token"},
            integration_id="copilot-cli-chat",
            environ={"ENTROLY_COPILOT_API_VERSION": "2026-06-01\r\nX-Evil: 1"},
        )


def test_invalid_client_api_version_falls_back_without_failing_request() -> None:
    headers = build_copilot_capi_headers(
        original={"X-GitHub-Api-Version": "not-a-date"},
        forwarded={"Authorization": "Bearer token"},
        integration_id="copilot-cli-chat",
        environ={},
    )
    assert headers["X-GitHub-Api-Version"] == "2025-04-01"


def test_contract_never_rewrites_provider_authorization() -> None:
    headers = build_copilot_capi_headers(
        original={"Authorization": "Bearer local-dummy"},
        forwarded={"Authorization": "Bearer short-lived-copilot-token"},
        integration_id="copilot-cli-chat",
        environ={},
    )
    assert headers["Authorization"] == "Bearer short-lived-copilot-token"


def test_container_proxy_install_order_is_transport_auth_contract_access() -> None:
    source = Path("entroly/container_proxy.py").read_text(encoding="utf-8")
    final_pos = source.index("proxy_transport_final")
    auth_pos = source.index("install_copilot_subscription_transport()")
    contract_pos = source.index("install_copilot_capi_contract()")
    access_pos = source.index("proxy_access_security")
    assert final_pos < auth_pos < contract_pos < access_pos
