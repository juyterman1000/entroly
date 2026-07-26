from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = "docs/gpt-5-6-sol-terra-luna.html"
CANONICAL = "https://juyterman1000.github.io/entroly/docs/gpt-5-6-sol-terra-luna.html"


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    assert match is not None, pattern
    return match.group(1).strip()


def test_gpt_5_6_page_has_bounded_search_metadata_and_json_ld() -> None:
    page = _text(PAGE)

    title = _match(r"<title>(.*?)</title>", page)
    description = _match(
        r'<meta\s+name="description"\s+content="([^"]+)"', page
    )
    canonical = _match(r'<link\s+rel="canonical"\s+href="([^"]+)"', page)
    robots = _match(r'<meta\s+name="robots"\s+content="([^"]+)"', page)
    heading = _match(r"<h1>(.*?)</h1>", page)

    assert 20 <= len(title) <= 70
    assert 90 <= len(description) <= 180
    assert canonical == CANONICAL
    assert "index" in robots and "follow" in robots
    assert "Entroly" in title
    assert "GPT-5.6" in title
    assert "Sol" in heading and "Terra" in heading and "Luna" in heading
    assert "Direct answer:" in page

    blocks = re.findall(
        r'<script\s+type="application/ld\+json">\s*(.*?)\s*</script>',
        page,
        flags=re.DOTALL,
    )
    assert blocks
    documents = [json.loads(block) for block in blocks]
    encoded = json.dumps(documents)
    assert "SoftwareApplication" in encoded
    assert "TechArticle" in encoded
    assert "BreadcrumbList" in encoded
    assert "FAQPage" in encoded
    assert CANONICAL in encoded
    assert "aggregateRating" not in encoded
    assert '"review"' not in encoded


def test_gpt_5_6_page_lists_verified_entroly_contract() -> None:
    page = _text(PAGE)

    for model_id in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert model_id in page
    for term in (
        "1,050,000",
        "128,000",
        "Context Receipt",
        "exact recovery",
        "provider-bound",
        "Savings vary by workload",
    ):
        assert term.casefold() in page.casefold()


def test_gpt_5_6_discovery_assets_are_mirrored_and_indexed() -> None:
    llms = _text("llms.txt")
    assert llms == _text("docs/llms.txt")
    assert CANONICAL in llms
    assert "Does Entroly support GPT-5.6 Sol, Terra, and Luna?" in llms
    assert "Can Entroly reduce GPT-5.6 API costs?" in llms

    sitemap = _text("sitemap.xml")
    assert sitemap == _text("docs/sitemap.xml")
    root = ET.fromstring(sitemap)
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    entries = {
        item.findtext("s:loc", namespaces=namespace): item.findtext(
            "s:lastmod", namespaces=namespace
        )
        for item in root.findall("s:url", namespace)
    }
    assert entries[CANONICAL] == "2026-07-26"


def test_serena_profile_matches_entroly_polyglot_architecture() -> None:
    profile = _text(".serena/project.yml")

    assert 'project_name: "entroly"' in profile
    for language in ("python", "rust", "typescript"):
        assert f"  - {language}" in profile
    assert '  - "entroly-wasm"' in profile
    assert '  - "integrations/openclaw"' in profile
    assert "Never count a retrieval miss" in profile
    assert "pure-Python fallback" in profile
