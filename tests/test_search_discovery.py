from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://juyterman1000.github.io/entroly"
PAGES = {
    "docs/ai-cost-optimization.html": {
        "canonical": f"{SITE}/ai-cost-optimization.html",
        "lastmod": "2026-08-09",
        "title_terms": ("AI Cost Optimization", "Entroly"),
        "body_terms": (
            "Context Assurance",
            "lower LLM API input costs",
            "Fixed ChatGPT or Claude subscription",
        ),
    },
    "docs/agent-integrations.html": {
        "canonical": f"{SITE}/agent-integrations.html",
        "lastmod": "2026-07-25",
        "title_terms": ("Entroly", "OpenClaw", "Hermes", "OpenCode"),
        "body_terms": ("context assurance", "exact-recovery contract"),
    },
    "docs/openclaw-context-engine.html": {
        "canonical": f"{SITE}/openclaw-context-engine.html",
        "lastmod": "2026-07-25",
        "title_terms": ("Entroly", "OpenClaw", "Context Engine"),
        "body_terms": ("OpenClaw", "context assurance engine", "Context Receipts"),
    },
    "docs/hermes-context-engine.html": {
        "canonical": f"{SITE}/hermes-context-engine.html",
        "lastmod": "2026-07-25",
        "title_terms": ("Entroly", "Hermes Agent", "Context Engine"),
        "body_terms": ("Hermes Agent", "context engine", "hash-only"),
    },
    "docs/opencode-context-assurance.html": {
        "canonical": f"{SITE}/opencode-context-assurance.html",
        "lastmod": "2026-07-25",
        "title_terms": ("Entroly", "OpenCode", "Context Assurance"),
        "body_terms": ("OpenCode", "MCP", "verification status"),
    },
}


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    assert match is not None, pattern
    return match.group(1).strip()


def test_docs_crawler_files_exist() -> None:
    assert (ROOT / "docs" / "robots.txt").is_file()
    assert (ROOT / "docs" / "sitemap.xml").is_file()


def test_legacy_docs_urls_redirect_to_pages_root() -> None:
    """GitHub Pages serves docs/ as the site root; keep old /docs/ URLs alive."""
    redirects = {
        "docs/docs/index.html": "../",
        "docs/docs/ai-cost-optimization.html": "../ai-cost-optimization.html",
        "docs/docs/agent-integrations.html": "../agent-integrations.html",
    }
    for path, target in redirects.items():
        page = _text(path)
        assert 'content="noindex,follow"' in page
        assert f'url={target}"' in page
        assert f'href="{target}"' in page


def test_search_and_answer_crawlers_are_explicitly_allowed() -> None:
    robots = _text("docs/robots.txt")
    for user_agent in (
        "Googlebot",
        "Bingbot",
        "OAI-SearchBot",
        "ChatGPT-User",
        "Claude-SearchBot",
        "Claude-User",
        "PerplexityBot",
        "Perplexity-User",
    ):
        assert f"User-agent: {user_agent}\nAllow: /" in robots
    assert f"Sitemap: {SITE}/sitemap.xml" in robots


def test_discovery_sitemap_is_current_and_complete() -> None:
    root = ET.fromstring(_text("docs/sitemap.xml"))
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    entries = {
        item.findtext("s:loc", namespaces=namespace): item.findtext(
            "s:lastmod", namespaces=namespace
        )
        for item in root.findall("s:url", namespace)
    }
    for spec in PAGES.values():
        assert entries[spec["canonical"]] == spec["lastmod"]
    assert entries[f"{SITE}/"] == "2026-08-31"


@pytest.mark.skipif(
    not all((ROOT / p).is_file() for p in PAGES),
    reason="intent pages not yet generated",
)
def test_intent_pages_have_unique_search_metadata_and_valid_json_ld() -> None:
    titles: set[str] = set()
    canonicals: set[str] = set()
    for path, spec in PAGES.items():
        page = _text(path)
        title = _match(r"<title>(.*?)</title>", page)
        description = _match(
            r'<meta\s+name="description"\s+content="([^"]+)"', page
        )
        canonical = _match(r'<link\s+rel="canonical"\s+href="([^"]+)"', page)
        robots = _match(r'<meta\s+name="robots"\s+content="([^"]+)"', page)
        heading = _match(r"<h1>(.*?)</h1>", page)

        assert 20 <= len(title) <= 70
        assert 90 <= len(description) <= 180
        assert canonical == spec["canonical"]
        assert "index" in robots and "follow" in robots
        assert title not in titles
        assert canonical not in canonicals
        titles.add(title)
        canonicals.add(canonical)

        visible = re.sub(r"<[^>]+>", " ", page)
        for term in spec["title_terms"]:
            assert term.casefold() in title.casefold()
        for term in spec["body_terms"]:
            assert term.casefold() in visible.casefold()
        assert "Entroly" in heading
        assert "Direct answer:" in page or path.endswith("agent-integrations.html")

        blocks = re.findall(
            r'<script\s+type="application/ld\+json">\s*(.*?)\s*</script>',
            page,
            flags=re.DOTALL,
        )
        assert blocks, f"{path} has no JSON-LD"
        documents = [json.loads(block) for block in blocks]
        encoded = json.dumps(documents)
        assert "SoftwareApplication" in encoded
        assert "BreadcrumbList" in encoded
        assert spec["canonical"] in encoded
        assert "aggregateRating" not in encoded
        assert '"review"' not in encoded


@pytest.mark.skipif(
    not (ROOT / "docs" / "agent-integrations.html").is_file(),
    reason="agent-integrations page not yet generated",
)
def test_integration_hub_links_cost_and_every_agent_intent_page() -> None:
    hub = _text("docs/agent-integrations.html")
    for path in (
        "ai-cost-optimization.html",
        "openclaw-context-engine.html",
        "hermes-context-engine.html",
        "opencode-context-assurance.html",
    ):
        assert f'href="{path}"' in hub


def test_llms_index_names_cost_and_integrations_with_bounded_answers() -> None:
    canonical = _text("llms.txt")
    mirror = _text("docs/llms.txt")
    assert canonical == mirror
    for path in (
        "ai-cost-optimization.html",
        "agent-integrations.html",
        "openclaw-context-engine.html",
        "hermes-context-engine.html",
        "opencode-context-assurance.html",
    ):
        assert f"{SITE}/{path}" in canonical
    assert "hash-only lookup" in canonical
    assert "does not accept a query" in canonical
    assert "How can Entroly reduce AI costs?" in canonical
    assert "Does Entroly lower ChatGPT Plus or Claude subscription prices?" in canonical
    assert "No fixed percentage is guaranteed" in canonical
    assert "not yet shipped" in canonical
