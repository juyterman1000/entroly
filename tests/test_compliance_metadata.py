"""Regression checks for local-first privacy and package metadata."""
from __future__ import annotations

from pathlib import Path

from entroly.controls_html import CONTROLS_HTML
from entroly.dashboard import DASHBOARD_HTML, DashboardHandler


ROOT = Path(__file__).resolve().parent.parent


def test_local_ui_sources_do_not_import_remote_fonts():
    assert "fonts.googleapis.com" not in DASHBOARD_HTML
    assert "fonts.googleapis.com" not in CONTROLS_HTML

    cli_source = (ROOT / "entroly" / "cli.py").read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in cli_source
    assert "fonts.gstatic.com" not in cli_source


def test_dashboard_csp_does_not_allow_remote_font_hosts():
    headers: list[tuple[str, str]] = []

    class HeaderRecorder:
        def send_header(self, name: str, value: str) -> None:
            headers.append((name, value))

    DashboardHandler._send_security_headers(HeaderRecorder())
    csp = dict(headers)["Content-Security-Policy"]
    assert "fonts.googleapis.com" not in csp
    assert "fonts.gstatic.com" not in csp


def test_dashboard_html_is_utf8_and_not_mojibake():
    assert '<meta charset="utf-8">' in DASHBOARD_HTML
    assert "⚡ Entroly" in DASHBOARD_HTML
    assert "💡 Point your AI tool" in DASHBOARD_HTML
    assert "Live · 3s refresh" in DASHBOARD_HTML

    mojibake_markers = ("â", "Â", "ð", "ï", "Î", "Ã", "\ufffd")
    for marker in mojibake_markers:
        assert marker not in DASHBOARD_HTML


def test_controls_html_is_utf8_and_surfaces_write_failures():
    assert '<meta charset="utf-8">' in CONTROLS_HTML
    assert "Entroly — Control Panel" in CONTROLS_HTML
    assert "Bypass mode ON — requests forwarded raw" in CONTROLS_HTML
    assert "Autotune triggered — weights will update" in CONTROLS_HTML
    assert 'id="controlError"' in CONTROLS_HTML
    assert "HTTP '+r.status+' from control API" in CONTROLS_HTML

    mojibake_markers = ("â", "Â", "ð", "ï", "Î", "Ã", "\ufffd")
    for marker in mojibake_markers:
        assert marker not in CONTROLS_HTML


def test_rust_packages_declare_apache_license_metadata():
    for rel in ("entroly-core/Cargo.toml", "entroly-wasm/Cargo.toml"):
        manifest = (ROOT / rel).read_text(encoding="utf-8")
        assert 'license = "Apache-2.0"' in manifest
        assert 'repository = "https://github.com/juyterman1000/entroly"' in manifest


def test_root_license_is_entroly_apache_2_only():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "Copyright 2026 Entroly" in license_text

    # Prevent accidentally restoring the MCP project's transitional license text.
    assert "The MCP project is undergoing a licensing transition" not in license_text
    assert "MIT License" not in license_text
    assert "CC-BY-4.0" not in license_text
