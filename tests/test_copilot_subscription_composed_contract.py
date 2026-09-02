from __future__ import annotations

import uuid


def test_composed_copilot_subscription_proxy_contract(monkeypatch) -> None:
    """Prove the production layer ordering, not isolated helper behavior."""
    monkeypatch.setenv("ENTROLY_COPILOT_SUBSCRIPTION", "1")
    monkeypatch.delenv("ENTROLY_COPILOT_INTEGRATION_ID", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_INTEGRATION_ID", raising=False)

    # Match container_proxy's security ordering: hardened transport first, then
    # GitHub-backed CAPI auth, exact path normalization, and trusted metadata.
    import entroly.proxy as proxy_module
    import entroly.proxy_transport_safe  # noqa: F401
    import entroly.proxy_transport_final  # noqa: F401
    from entroly.copilot_capi_contract import install_copilot_capi_contract
    from entroly.copilot_capi_routing import install_copilot_capi_routing
    from entroly.copilot_subscription_transport import (
        CopilotProviderCredential,
        CopilotTokenManager,
        install_copilot_subscription_transport,
    )
    from entroly.proxy_config import ProxyConfig

    original_create = proxy_module.create_proxy_app
    original_headers = proxy_module.PromptCompilerProxy._build_headers
    original_resolve = proxy_module.PromptCompilerProxy._resolve_target
    original_shutdown = proxy_module.PromptCompilerProxy.shutdown

    try:
        assert install_copilot_subscription_transport() is True
        assert install_copilot_capi_routing() is True
        assert install_copilot_capi_contract() is True

        # Construction is side-effect free. Seed the process-local credential to
        # exercise the exact request seam used after startup preflight has passed.
        manager = CopilotTokenManager(
            api_origin="https://api.githubcopilot.com",
            environ={},
            user_info_fetch=lambda *_args: {"chat_enabled": True},
            credential_resolver=lambda: "gho_active-user-token",
        )
        assert manager.integration_id == "copilot-developer-cli"
        manager._current = CopilotProviderCredential(
            token="gho_active-user-token",
            api_origin="https://api.githubcopilot.com",
        )
        manager._pinned_origin = "https://api.githubcopilot.com"

        proxy = object.__new__(proxy_module.PromptCompilerProxy)
        proxy.config = ProxyConfig(openai_base_url="https://api.githubcopilot.com")
        proxy._copilot_subscription_token_manager = manager

        target = proxy._resolve_target("openai", "/v1/chat/completions")
        headers = proxy._build_headers(
            {
                "Authorization": "Bearer entroly-local-provider-route",
                "Content-Type": "application/json",
                "User-Agent": "GitHubCopilotCLI/1.2.3",
                "Editor-Version": "copilot-cli/1.2.3",
                "Editor-Plugin-Version": "copilot-cli/1.2.3",
                "X-Initiator": "user",
            },
            "openai",
        )

        assert target == "https://api.githubcopilot.com/chat/completions"
        assert headers["Authorization"] == "Bearer gho_active-user-token"
        assert "entroly-local-provider-route" not in repr(headers)
        assert headers["Copilot-Integration-Id"] == "copilot-developer-cli"
        assert headers["User-Agent"] == "GitHubCopilotCLI/1.2.3"
        assert headers["Editor-Version"] == "copilot-cli/1.2.3"
        assert headers["Editor-Plugin-Version"] == "copilot-cli/1.2.3"
        assert headers["X-Initiator"] == "user"
        assert headers["X-GitHub-Api-Version"]
        uuid.UUID(headers["X-Interaction-Id"])
    finally:
        manager = locals().get("manager")
        if manager is not None:
            manager.stop()
        proxy_module.create_proxy_app = original_create
        proxy_module.PromptCompilerProxy._build_headers = original_headers
        proxy_module.PromptCompilerProxy._resolve_target = original_resolve
        proxy_module.PromptCompilerProxy.shutdown = original_shutdown
