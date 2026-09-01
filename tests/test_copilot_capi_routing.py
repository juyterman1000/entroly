from __future__ import annotations

from pathlib import Path

from entroly.copilot_capi_routing import normalize_copilot_capi_path


def test_capi_openai_compatible_paths_drop_only_exact_v1_prefixes() -> None:
    assert normalize_copilot_capi_path("/v1/chat/completions") == "/chat/completions"
    assert normalize_copilot_capi_path("/v1/responses") == "/responses"
    assert normalize_copilot_capi_path("/v1/models") == "/models"


def test_capi_path_normalizer_does_not_rewrite_unknown_or_similar_paths() -> None:
    for path in (
        "/chat/completions",
        "/responses",
        "/models",
        "/v1/embeddings",
        "/v10/chat/completions",
        "/v1/chat/completions/extra",
        "/v1beta/models/gemini",
    ):
        assert normalize_copilot_capi_path(path) == path


def test_container_proxy_installs_capi_routing_inside_existing_security_layers() -> None:
    source = Path("entroly/container_proxy.py").read_text(encoding="utf-8")
    final_pos = source.index("proxy_transport_final")
    auth_pos = source.index("install_copilot_subscription_transport()")
    routing_pos = source.index("install_copilot_capi_routing()")
    headers_pos = source.index("install_copilot_capi_contract()")
    access_pos = source.index("proxy_access_security")
    assert final_pos < auth_pos < routing_pos < headers_pos < access_pos
