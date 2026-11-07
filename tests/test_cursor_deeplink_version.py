"""The Cursor one-click install link must install the current release.

README's "Add Entroly to Cursor" link carries its MCP config base64-encoded:
``{"command":"npx","args":["-y","entroly-mcp@1.0.84","serve"],...}``. Every
version check reads text, so the pin inside it was invisible: it stayed at
1.0.84 through four releases while the plain JSON snippet a few lines further
down moved with each bump. Cursor users who clicked it installed a release
four versions behind the one the page described.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load("check_version_staleness")
bump_version = _load("bump_version")

CURRENT = re.search(
    r'__version__\s*=\s*"([^"]+)"', (ROOT / "entroly" / "__init__.py").read_text("utf-8")
).group(1)


def _link(version: str, urlsafe: bool = True) -> str:
    config = json.dumps(
        {"command": "npx", "args": ["-y", f"entroly-mcp@{version}", "serve"]},
        separators=(",", ":"),
    ).encode()
    encoded = (base64.urlsafe_b64encode if urlsafe else base64.b64encode)(config)
    encoded = encoded.decode().rstrip("=")
    return f"[Add](cursor://anysphere.cursor-deeplink/mcp/install?name=entroly&config={encoded})"


def _pins(text: str) -> list[str]:
    found = []
    for match in checker.CURSOR_DEEPLINK.finditer(text):
        decoded = checker.decode_deeplink_config(match.group(1))
        assert decoded is not None
        found += re.findall(r"entroly-mcp@(\d+\.\d+\.\d+)", decoded)
    return found


def test_the_readme_link_installs_the_current_release() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    pins = _pins(readme)

    assert pins, "the README should still carry the Cursor install link"
    assert set(pins) == {CURRENT}


def test_the_checker_sees_a_stale_pin_inside_the_encoded_config() -> None:
    stale, _allowed = checker.scan_text("README.md", f"Cursor: {_link('1.0.1')}\n")

    assert len(stale) == 1
    assert "1.0.1" in stale[0] and "Cursor link" in stale[0]


def test_the_checker_accepts_a_current_pin() -> None:
    stale, _allowed = checker.scan_text("README.md", f"Cursor: {_link(CURRENT)}\n")

    assert stale == []


def test_bump_rewrites_the_pin_and_keeps_the_encoding(tmp_path, monkeypatch) -> None:
    for urlsafe in (True, False):
        readme = tmp_path / "README.md"
        readme.write_text(f"see {_link('1.0.1', urlsafe)} and entroly-mcp@1.0.1\n", "utf-8")
        monkeypatch.setattr(bump_version, "ROOT", tmp_path)
        monkeypatch.setattr(bump_version, "TARGETS", [
            ("README.md", r"entroly-mcp@[0-9]+\.[0-9]+\.[0-9]+", "entroly-mcp@{v}"),
        ])

        assert bump_version.main(["bump_version.py", "1.0.2"]) == 0

        text = readme.read_text(encoding="utf-8")
        assert _pins(text) == ["1.0.2"]
        assert text.endswith(" and entroly-mcp@1.0.2\n")
        encoded = checker.CURSOR_DEEPLINK.search(text).group(1)
        assert "=" not in encoded, "padding style kept"
