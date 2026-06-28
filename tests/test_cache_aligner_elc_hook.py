from __future__ import annotations

import importlib


def test_cache_aligner_auto_installs_elc_when_env_enabled(monkeypatch) -> None:
    from entroly import proxy_transform
    from entroly.compression_proxy_live import reset_live_compression_proxy

    reset_live_compression_proxy()
    original = proxy_transform.compress_tool_messages

    monkeypatch.setenv("ENTROLY_COMPRESSION_PROXY_MODE", "elc")
    import entroly.cache_aligner as cache_aligner

    importlib.reload(cache_aligner)

    try:
        assert proxy_transform.compress_tool_messages is not original
    finally:
        reset_live_compression_proxy()

    assert proxy_transform.compress_tool_messages is original


def test_cache_aligner_import_is_side_effect_free_when_env_disabled(monkeypatch) -> None:
    from entroly import proxy_transform
    from entroly.compression_proxy_live import reset_live_compression_proxy

    reset_live_compression_proxy()
    original = proxy_transform.compress_tool_messages

    monkeypatch.delenv("ENTROLY_COMPRESSION_PROXY_MODE", raising=False)
    import entroly.cache_aligner as cache_aligner

    importlib.reload(cache_aligner)

    assert proxy_transform.compress_tool_messages is original
