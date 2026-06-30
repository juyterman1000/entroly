from __future__ import annotations

from entroly.compression_proxy_direct import apply_elc_to_proxy_body


def test_direct_proxy_helper_respects_off_mode(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_COMPRESSION_PROXY_MODE", raising=False)
    body = {"messages": [{"role": "tool", "content": "x" * 5000}]}

    result = apply_elc_to_proxy_body(body, provider="openai")

    assert result.changed is False
    assert result.body == body


def test_direct_proxy_helper_compresses_when_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_COMPRESSION_PROXY_MODE", "elc")
    monkeypatch.setenv("ENTROLY_ELC_BUDGET_TOKENS", "120")
    monkeypatch.setenv("ENTROLY_COMPRESSION_STORE", str(tmp_path / "store.json"))
    heavy = "\n".join(["compile ok" for _ in range(300)] + ["ERROR direct proxy issue"])
    body = {
        "messages": [
            {"role": "user", "content": "what happened in direct proxy?"},
            {"role": "tool", "content": heavy},
        ]
    }

    result = apply_elc_to_proxy_body(body, provider="openai")

    assert result.changed
    assert result.receipt.tokens_saved > 0
    assert "ERROR direct proxy issue" in result.body["messages"][1]["content"]
