from __future__ import annotations

import sys
import types

from entroly.native_elc import compress_evidence_locked_fast, native_elc_available


def test_native_elc_accepts_dedicated_json_symbol(monkeypatch) -> None:
    fake = types.SimpleNamespace(
        elc_compress=lambda text, query, budget: '{"compressed":"ERROR keep me","changed":true,"receipt":{"original_tokens":100,"compressed_tokens":5,"savings_ratio":0.95,"compression_level":3,"content_type":"native_elc","anchors_preserved":{"anchor":1},"omitted_spans":[],"recoverable":true}}'
    )
    monkeypatch.setitem(sys.modules, "entroly_core", fake)

    assert native_elc_available()
    result = compress_evidence_locked_fast("ERROR keep me\nnoise", query="keep", budget_tokens=50)

    assert result.changed
    assert result.receipt.content_type == "native_elc"
    assert "ERROR keep me" in result.compressed


def test_pyo3_bridge_accepts_evidence_safe_block(monkeypatch) -> None:
    def py_compress_block(role, content, token_count, resolution, tool_name):
        assert role == "tool"
        assert resolution == "skeleton"
        return "src/auth/session.rs:184\nERROR refresh timeout\nhint increase token refresh slack"

    fake = types.SimpleNamespace(py_compress_block=py_compress_block)
    monkeypatch.setitem(sys.modules, "entroly_core", fake)
    text = "\n".join(
        ["compile ok" for _ in range(300)]
        + ["src/auth/session.rs:184", "ERROR refresh timeout", "hint increase token refresh slack"]
    )

    result = compress_evidence_locked_fast(text, query="auth refresh timeout", budget_tokens=120)

    assert result.changed
    assert result.receipt.content_type == "native_pyo3_block"
    assert "src/auth/session.rs:184" in result.compressed
    assert "ERROR refresh timeout" in result.compressed


def test_pyo3_bridge_rejects_evidence_loss(monkeypatch) -> None:
    fake = types.SimpleNamespace(
        py_compress_block=lambda role, content, token_count, resolution, tool_name: "compile ok\n...[omitted]"
    )
    monkeypatch.setitem(sys.modules, "entroly_core", fake)
    text = "\n".join(["compile ok" for _ in range(300)] + ["ERROR refresh timeout"])

    result = compress_evidence_locked_fast(text, query="refresh timeout", budget_tokens=120)

    assert result.changed
    assert result.receipt.content_type != "native_pyo3_block"
    assert "ERROR refresh timeout" in result.compressed
