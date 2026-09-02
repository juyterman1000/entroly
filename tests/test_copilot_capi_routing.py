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


def test_capi_paths_translate_despite_slash_spelling() -> None:
    """A trailing slash on the base URL is the ordinary way to get this wrong.

    Configuring the provider base URL as `.../v1/` makes the client request
    `/v1//chat/completions`. Starlette reports that path verbatim and the proxy
    does not collapse it, so the exact lookup missed, the `/v1` prefix survived
    to CAPI, and GitHub answered 404 with nothing pointing at the slash.
    """
    for path in (
        "/v1//chat/completions",
        "/v1/chat/completions/",
        "/v1///chat/completions",
    ):
        assert normalize_copilot_capi_path(path) == "/chat/completions", path
    assert normalize_copilot_capi_path("/v1/models/") == "/models"
    assert normalize_copilot_capi_path("/v1//responses") == "/responses"


def test_a_path_that_matches_nothing_is_returned_byte_for_byte() -> None:
    """Canonicalisation is for the lookup key only.

    Rewriting the outgoing path would make this a router. It translates three
    endpoints; everything else must reach the existing resolver exactly as it
    arrived, including spellings this function had to canonicalise to decide.
    """
    for path in (
        "/v1//embeddings",
        "/v1/chat/completions/../secrets",
        "/unknown//double",
        "/v1/chat/completions/extra/",
        "/",
    ):
        assert normalize_copilot_capi_path(path) == path, path


def test_container_proxy_installs_capi_routing_inside_existing_security_layers() -> None:
    source = Path("entroly/container_proxy.py").read_text(encoding="utf-8")
    final_pos = source.index("proxy_transport_final")
    auth_pos = source.index("install_copilot_subscription_transport()")
    routing_pos = source.index("install_copilot_capi_routing()")
    headers_pos = source.index("install_copilot_capi_contract()")
    access_pos = source.index("proxy_access_security")
    assert final_pos < auth_pos < routing_pos < headers_pos < access_pos
