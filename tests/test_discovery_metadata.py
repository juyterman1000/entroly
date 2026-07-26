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
        "entroly — drop-in context assurance to lower ai operational cost",
        "reduce unnecessary context without losing control of critical evidence.",
        "content-addressed evidence",
        "recoverable compression",
        "context os",
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
        "we guarantee savings",
        "guaranteed savings",
        "zero setup required",
        "works with every ai app",
    )
    for phrase in forbidden:
        assert phrase not in combined


def test_python_package_metadata_is_searchable_and_connected() -> None:
    pyproject = _text("pyproject.toml")
    project = _toml_section("pyproject.toml", "project")
    urls = _toml_section("pyproject.toml", "project.urls")
    assert "Context Assurance" in project
    assert "AI costs" in project or "AI cost" in project
    assert REPOSITORY in urls
    assert HOMEPAGE in urls
    for keyword in PACKAGE_KEYWORDS:
        assert f'"{keyword}"' in pyproject


def test_npm_packages_share_cost_discovery_terms_and_trust_links() -> None:
    for package_path in (
        "package.json",
        "entroly-mcp/package.json",
        "entroly-wasm/package.json",
    ):
        package = _json(package_path)
        keywords = set(package.get("keywords", []))
        assert {
            "ai-cost-optimization",
            "context-assurance",
            "context-engineering",
        } <= keywords
        assert package["homepage"] == HOMEPAGE
        assert package["repository"]["url"].endswith("juyterman1000/entroly.git")


def test_mcp_and_docs_metadata_keep_the_verified_context_os_boundary() -> None:
    server = _json("server.json")
    mcp = _json("mcp.json")
    for payload in (server, mcp):
        serialized = json.dumps(payload).casefold()
        assert CATEGORY in serialized
        assert PRODUCT_IDENTITY in serialized
        assert "ai cost" in serialized or "cost optimization" in serialized
        assert REPOSITORY in json.dumps(payload)


def test_openclaw_listing_names_its_category_without_provider_overclaims() -> None:
    listing = _json("integrations/openclaw/clawhub.json")
    serialized = json.dumps(listing).casefold()
    assert CATEGORY in serialized
    assert PRODUCT_IDENTITY in serialized
    assert "provider authentication" in serialized
    assert "guaranteed" not in serialized
