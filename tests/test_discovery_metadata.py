from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/juyterman1000/entroly"
HOMEPAGE = "https://juyterman1000.github.io/entroly/docs/index.html"
CATEGORY = "context engineering"
PRODUCT_IDENTITY = "context assurance"
PACKAGE_KEYWORDS = {
    "ai-agents",
    "ai-cost-optimization",
    "context-assurance",
    "context-engineering",
    "context-compression",
    "context-management",
    "context-optimization",
    "llm-cost-optimization",
    "mcp",
    "model-context-protocol",
    "reduce-ai-costs",
    "token-reduction",
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


def test_readme_first_folds_explain_their_supported_product_profiles() -> None:
    readme_first_fold = _text("README.md")[:7_500].casefold()
    for phrase in (
        "entroly — the open-source context os for ai agents",
        "keep your agent. give it a context os.",
        "recoverable compression",
        "local-first",
    ):
        assert phrase in readme_first_fold
    for client in (
        "claude code",
        "codex",
        "openclaw",
        "github copilot",
        "mcp",
    ):
        assert client in readme_first_fold, f"README.md does not identify {client} above the fold"

    pypi_first_fold = _text("PYPI_README.md")[:7_500].casefold()
    for phrase in (
        "entroly — context assurance that helps lower ai costs",
        PRODUCT_IDENTITY,
        "unnecessary ai context",
        "recoverable",
        "no agent-architecture rewrite",
        "one-time setup",
    ):
        assert phrase in pypi_first_fold
    for client in (
        "claude code",
        "codex",
        "openclaw",
        "hermes agent",
        "opencode",
        "github copilot",
        "local models",
        "mcp",
    ):
        assert client in pypi_first_fold, f"PYPI_README.md does not identify {client} above the fold"


def test_readmes_keep_cost_and_quality_claims_bounded() -> None:
    combined = (_text("README.md") + _text("PYPI_README.md")).casefold()
    required = (
        "does not promise a universal compression percentage",
        "fixed-price subscription",
        "subscription price may not change",
        "not a provider invoice",
        "does not establish universal truth",
        "not shipped yet",
    )
    for phrase in required:
        assert phrase in combined

    forbidden = (
        "guarantees bill reduction",
        "always lowers ai bills",
        "zero setup",
        "works with every ai app",
        "your files never leave your machine",
    )
    for phrase in forbidden:
        assert phrase not in combined


def test_python_package_metadata_is_searchable_and_connected() -> None:
    for path in ("pyproject.toml", "entroly/pyproject.toml"):
        project = _toml_section(path, "project").casefold()
        urls = _toml_section(path, "project.urls")

        assert PRODUCT_IDENTITY in project
        assert "ai agents" in project
        assert "reduce avoidable token usage" in project
        for keyword in PACKAGE_KEYWORDS:
            assert f'"{keyword}"' in project, f"{path} is missing keyword {keyword}"
        assert f'Homepage = "{HOMEPAGE}"' in urls
        assert f'Source = "{REPOSITORY}"' in urls
        assert f'Issues = "{REPOSITORY}/issues"' in urls
        assert f'"Release Notes" = "{REPOSITORY}/releases"' in urls


def test_npm_packages_share_cost_discovery_terms_and_trust_links() -> None:
    for path in (
        "entroly/npm/package.json",
        "entroly/npm-alias/package.json",
        "entroly-wasm/package.json",
    ):
        package = _json(path)
        description = package["description"].casefold()
        keywords = set(package["keywords"])

        assert PRODUCT_IDENTITY in description
        assert "ai agent" in description
        assert "token usage" in description
        assert PACKAGE_KEYWORDS <= keywords
        assert package["repository"]["url"] == REPOSITORY
        assert package["homepage"] == HOMEPAGE
        assert package["bugs"]["url"] == f"{REPOSITORY}/issues"


def test_mcp_and_docs_metadata_keep_the_verified_context_os_boundary() -> None:
    manifest = _json("server.json")
    description = manifest["description"]
    homepage = _text("docs/index.html")

    assert len(description) <= 100
    assert CATEGORY in description.casefold()
    assert "context os" in description.casefold()
    assert "recovery" in description.casefold()
    assert "receipts" in description.casefold()
    assert "verification" in description.casefold()
    assert "<title>Entroly — The Open-Source Context OS for AI Agents</title>" in homepage
    assert f'"codeRepository": "{REPOSITORY}"' in homepage
    assert 'content="Open-source Context OS for AI agents:' in homepage


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
