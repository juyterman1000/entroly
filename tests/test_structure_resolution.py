"""Public structure reads use the shared native skeleton engine honestly."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from entroly.semantic_resolution import Resolution, resolve


RUST_SOURCE = """\
use std::collections::HashMap;

pub struct Ledger {
    entries: HashMap<String, u64>,
}

impl Ledger {
    pub fn record(&mut self, key: String, value: u64) {
        let previous = self.entries.get(&key).copied().unwrap_or_default();
        let updated = previous.saturating_add(value);
        self.entries.insert(key.clone(), updated);
        if updated > 10_000 {
            audit_the_sensitive_body(key);
        }
    }

    pub fn total(&self) -> u64 {
        self.entries
            .values()
            .copied()
            .fold(0_u64, u64::saturating_add)
    }
}
"""


def test_compiled_native_export_extracts_real_rust_structure() -> None:
    core = pytest.importorskip("entroly_core")
    if not hasattr(core, "extract_skeleton"):
        pytest.skip("installed native wheel predates the structure export")

    outline = core.extract_skeleton(RUST_SOURCE, "ledger.rs")

    assert outline is not None
    assert "pub struct Ledger" in outline
    assert "pub fn record" in outline
    assert "audit_the_sensitive_body" not in outline


def test_structure_uses_native_export_and_elides_bodies(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def extract_skeleton(content: str, source: str) -> str:
        calls.append((content, source))
        return "use std::collections::HashMap;\n\npub struct Ledger { ... }"

    monkeypatch.setitem(
        sys.modules,
        "entroly_core",
        SimpleNamespace(extract_skeleton=extract_skeleton),
    )
    result = resolve(
        RUST_SOURCE,
        file_path="ledger.rs",
        resolution=Resolution.STRUCTURE,
    )

    assert calls == [(RUST_SOURCE, "ledger.rs")]
    assert result.forced_resolution == "structure"
    assert result.structure_backend == "native-skeleton"
    assert result.resolution_counts == {"structure": 1}
    assert "pub struct Ledger" in result.output
    assert "audit_the_sensitive_body" not in result.output


def test_structure_fails_open_when_native_capability_is_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "entroly_core", SimpleNamespace())

    result = resolve(
        RUST_SOURCE,
        file_path="ledger.rs",
        resolution=Resolution.STRUCTURE,
    )

    assert result.output == RUST_SOURCE
    assert result.forced_resolution == "structure"
    assert result.structure_backend == "full-fallback"
    assert result.resolution_counts == {"full": 1}


def test_smart_read_reports_structure_backend(tmp_path, monkeypatch) -> None:
    from entroly.server import create_mcp_server

    source_path = tmp_path / "ledger.rs"
    source_path.write_text(RUST_SOURCE, encoding="utf-8")
    monkeypatch.setenv("ENTROLY_SOURCE", str(tmp_path))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setitem(
        sys.modules,
        "entroly_core",
        SimpleNamespace(
            extract_skeleton=lambda _content, _source: "pub struct Ledger { ... }"
        ),
    )

    mcp, _ = create_mcp_server(allowed_tools={"smart_read"})
    smart_read = mcp._tool_manager._tools["smart_read"].fn
    response = json.loads(
        smart_read(
            str(source_path),
            SimpleNamespace(session=object(), client_id="primary"),
            resolution="structure",
        )
    )

    assert response["output"] == "pub struct Ledger { ... }"
    assert response["forced_resolution"] == "structure"
    assert response["structure_backend"] == "native-skeleton"
