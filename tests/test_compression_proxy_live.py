from __future__ import annotations

from entroly.compression_proxy_live import (
    install_live_compression_proxy,
    reset_live_compression_proxy,
)
from entroly import proxy_transform


def test_live_proxy_installer_is_inert_without_env(monkeypatch) -> None:
    reset_live_compression_proxy()
    monkeypatch.delenv("ENTROLY_COMPRESSION_PROXY_MODE", raising=False)

    assert install_live_compression_proxy() is False


def test_live_proxy_installer_patches_tool_compression_when_enabled(tmp_path, monkeypatch) -> None:
    reset_live_compression_proxy()
    monkeypatch.setenv("ENTROLY_COMPRESSION_PROXY_MODE", "elc")
    monkeypatch.setenv("ENTROLY_ELC_BUDGET_TOKENS", "120")
    monkeypatch.setenv("ENTROLY_COMPRESSION_STORE", str(tmp_path / "store.json"))

    try:
        assert install_live_compression_proxy() is True
        heavy = "\n".join(["compile ok" for _ in range(400)] + ["ERROR live proxy failure"])
        messages = [
            {"role": "user", "content": "why did live proxy fail?"},
            {"role": "tool", "content": heavy},
        ]

        compressed, saved = proxy_transform.compress_tool_messages(messages, policy="compress")

        assert saved > 0
        assert "entroly-elc" in compressed[1]["content"]
        assert "ERROR live proxy failure" in compressed[1]["content"]
    finally:
        reset_live_compression_proxy()


def test_live_proxy_installer_respects_preserve_policy(tmp_path, monkeypatch) -> None:
    reset_live_compression_proxy()
    monkeypatch.setenv("ENTROLY_COMPRESSION_PROXY_MODE", "elc")
    monkeypatch.setenv("ENTROLY_COMPRESSION_STORE", str(tmp_path / "store.json"))

    try:
        assert install_live_compression_proxy() is True
        heavy = "\n".join(["compile ok" for _ in range(400)] + ["ERROR preserve me"])
        messages = [{"role": "tool", "content": heavy}]

        compressed, saved = proxy_transform.compress_tool_messages(messages, policy="preserve")

        assert saved == 0
        assert compressed == messages
    finally:
        reset_live_compression_proxy()
