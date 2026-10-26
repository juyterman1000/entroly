"""Every shipped host manifest that launches `entroly serve` must stay native.

`packaging/mcpb/manifest.json` (the desktop MCP bundle) launched
`uvx --from entroly entroly serve` with an empty env, so the public launcher
took its Docker path: with Docker installed but stopped the server exited 1 at
startup, and with Docker running it pulled an image and served the image's own
source tree. The Gemini extensions and `.mcp.json` already pin
ENTROLY_NO_DOCKER=1; this keeps every serve-launching manifest consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _serve_launches(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    servers = dict(data.get("mcpServers") or {})
    mcp_config = (data.get("server") or {}).get("mcp_config")
    if mcp_config:
        servers["mcpb"] = mcp_config
    for name, server in servers.items():
        argv = [server.get("command", ""), *server.get("args", [])]
        if "serve" in argv and any("entroly" in part for part in argv):
            yield name, server


_SKIP_DIRS = {"node_modules", ".git", "target", ".venv", "venv", "dist", "build"}

MANIFESTS = sorted(
    path
    for path in ROOT.rglob("*.json")
    if not _SKIP_DIRS.intersection(path.relative_to(ROOT).parts)
    and '"serve"' in path.read_text(encoding="utf-8", errors="replace")
)


def test_the_desktop_bundle_is_among_the_checked_manifests():
    assert ROOT / "packaging" / "mcpb" / "manifest.json" in MANIFESTS


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_serve_manifests_opt_out_of_the_docker_launcher(manifest: Path):
    for name, server in _serve_launches(manifest):
        env = server.get("env") or {}
        assert env.get("ENTROLY_NO_DOCKER") == "1", (
            f"{manifest.relative_to(ROOT)} server {name!r} launches `serve` "
            "without ENTROLY_NO_DOCKER=1"
        )
