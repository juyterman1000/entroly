from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/juyterman1000/entroly"
HOMEPAGE = "https://juyterman1000.github.io/entroly/docs/index.html"
CATEGORY = "context engineering"
PACKAGE_KEYWORDS = {
    "ai-agents",
    "context-engineering",
    "context-compression",
    "context-management",
    "context-optimization",
    "mcp",
    "model-context-protocol",
}


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _json(path: str) -> dict:
    return json.loads(_text(path))


def _toml_section(path: str, section: str) -> str:
    match = re.search(
        rf"(?ms)^\[{re.escape(section)}\]\s*(.*?)(?=^\[|\Z)",
        _text(path),
    )
    assert match is not None, f"{path} is missing [{section}]"
    return match.group(1)


def test_readme_first_fold_names_the_category_and_supported_clients() -> None:
    for path in ("README.md", "PYPI_README.md"):
        first_fold = _text(path)[:5_000].casefold()
        assert "entroly — auditable context engineering for ai agents" in first_fold
        assert "context compression" in first_fold or "compress" in first_fold
        for client in ("claude code", "codex", "openclaw", "github copilot", "mcp"):
            assert client in first_fold, f"{path} does not identify {client} above the fold"


def test_python_package_metadata_is_searchable_and_connected() -> None:
    for path in ("pyproject.toml", "entroly/pyproject.toml"):
        project = _toml_section(path, "project").casefold()
        urls = _toml_section(path, "project.urls")

        assert CATEGORY in project
        assert "ai agents" in project
        for keyword in PACKAGE_KEYWORDS:
            assert f'"{keyword}"' in project, f"{path} is missing keyword {keyword}"
        assert f'Homepage = "{HOMEPAGE}"' in urls
        assert f'Source = "{REPOSITORY}"' in urls
        assert f'Issues = "{REPOSITORY}/issues"' in urls
        assert f'"Release Notes" = "{REPOSITORY}/releases"' in urls


def test_npm_packages_share_discovery_terms_and_trust_links() -> None:
    for path in (
        "entroly/npm/package.json",
        "entroly/npm-alias/package.json",
        "entroly-wasm/package.json",
    ):
        package = _json(path)
        description = package["description"].casefold()
        keywords = set(package["keywords"])

        assert CATEGORY in description
        assert "ai agent" in description
        assert PACKAGE_KEYWORDS <= keywords
        assert package["repository"]["url"] == REPOSITORY
        assert package["homepage"] == HOMEPAGE
        assert package["bugs"]["url"] == f"{REPOSITORY}/issues"


def test_mcp_and_docs_metadata_use_the_same_verified_positioning() -> None:
    manifest = _json("server.json")
    description = manifest["description"]
    homepage = _text("docs/index.html")

    assert len(description) <= 100
    assert CATEGORY in description.casefold()
    assert "recovery" in description.casefold()
    assert "receipts" in description.casefold()
    assert "verification" in description.casefold()
    assert "<title>Entroly — Auditable Context Engineering for AI Agents</title>" in homepage
    assert f'"codeRepository": "{REPOSITORY}"' in homepage
    assert 'content="Auditable context engineering for AI agents:' in homepage


def test_openclaw_listing_names_its_category_without_provider_overclaims() -> None:
    package = _json("integrations/openclaw/package.json")
    manifest = _json("integrations/openclaw/openclaw.plugin.json")

    assert CATEGORY in package["description"].casefold()
    assert CATEGORY in manifest["description"].casefold()
    assert "provider-independent" in package["description"].casefold()
    assert {
        "openclaw",
        "openclaw-plugin",
        "ai-agents",
        "context-engineering",
        "context-compression",
        "context-receipts",
    } <= set(package["keywords"])
