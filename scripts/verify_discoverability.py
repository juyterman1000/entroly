#!/usr/bin/env python3
"""Verify Entroly's public discovery surfaces without asserting rankings.

This verifier protects the repository-controlled prerequisites for discovery:
one canonical homepage, consistent entity identifiers, crawlable intent pages,
evidence links, and explicit measurement boundaries. Search position and answer-
engine citations remain external observations and are deliberately out of scope.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SITE = "https://juyterman1000.github.io/entroly"
HOME = f"{SITE}/docs/index.html"
ENTITY_IDS = {
    f"{SITE}/#organization",
    f"{SITE}/#website",
    f"{SITE}/#software",
}
AUTHORITY_PAGES = {
    "docs/token-economics.html": f"{SITE}/docs/token-economics.html",
    "docs/token-compression-tools.html": f"{SITE}/docs/token-compression-tools.html",
    "docs/best-context-compression-tools.html": (
        f"{SITE}/docs/best-context-compression-tools.html"
    ),
}
REQUIRED_DIMENSIONS = {
    "active_tokens",
    "recovered_tokens",
    "effective_tokens",
    "task_success",
    "evidence_retention",
    "exact_recovery",
    "latency",
    "provider_observed_cost",
}
ALLOWED_OBSERVATION_STATUSES = {
    "requires_site_owner_connection",
    "baseline_pending",
}


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _json_ld_documents(page: str) -> list[Any]:
    blocks = re.findall(
        r'<script\s+type="application/ld\+json">\s*(.*?)\s*</script>',
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return [json.loads(block) for block in blocks]


def _sitemap_entries() -> dict[str, str | None]:
    root = ET.fromstring(_read("sitemap.xml"))
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return {
        item.findtext("s:loc", namespaces=namespace): item.findtext(
            "s:lastmod", namespaces=namespace
        )
        for item in root.findall("s:url", namespace)
        if item.findtext("s:loc", namespaces=namespace)
    }


def _local_page_for_url(url: str) -> Path | None:
    prefix = f"{SITE}/"
    if not url.startswith(prefix):
        return None
    return ROOT / url.removeprefix(prefix)


def collect_failures() -> list[str]:
    failures: list[str] = []

    root_page = _read("index.html")
    canonical_match = re.search(
        r'<link\s+rel="canonical"\s+href="([^"]+)"', root_page, re.IGNORECASE
    )
    if canonical_match is None or canonical_match.group(1) != HOME:
        failures.append("root index must canonicalize to the docs homepage")
    if f"url={HOME}" not in root_page or f'window.location.replace("{HOME}")' not in root_page:
        failures.append("root index must hand crawlers and users to the canonical homepage")

    try:
        sitemap = _sitemap_entries()
    except (ET.ParseError, OSError) as exc:
        failures.append(f"invalid sitemap: {exc}")
        sitemap = {}
    if f"{SITE}/" in sitemap:
        failures.append("sitemap must not publish the root redirect as a second homepage")
    if sitemap.get(HOME) != "2026-08-31":
        failures.append("canonical homepage is missing its current sitemap date")

    homepage = _read("docs/index.html")
    for relative, canonical in AUTHORITY_PAGES.items():
        local_link = relative.removeprefix("docs/")
        if f'href="{local_link}"' not in homepage:
            failures.append(f"homepage does not link to authority page {relative}")
        if sitemap.get(canonical) != "2026-08-31":
            failures.append(f"authority page has stale or missing sitemap entry: {relative}")

        page = _read(relative)
        robots = re.search(
            r'<meta\s+name="robots"\s+content="([^"]+)"', page, re.IGNORECASE
        )
        if robots is None or not {"index", "follow"}.issubset(
            {part.strip().casefold() for part in robots.group(1).split(",")}
        ):
            failures.append(f"authority page is not explicitly crawlable: {relative}")
        canonical_tag = re.search(
            r'<link\s+rel="canonical"\s+href="([^"]+)"', page, re.IGNORECASE
        )
        if canonical_tag is None or canonical_tag.group(1) != canonical:
            failures.append(f"authority page canonical is inconsistent: {relative}")
        try:
            documents = _json_ld_documents(page)
        except json.JSONDecodeError as exc:
            failures.append(f"invalid JSON-LD in {relative}: {exc}")
            documents = []
        encoded = json.dumps(documents, sort_keys=True)
        for entity_id in ENTITY_IDS:
            if entity_id not in encoded:
                failures.append(f"{relative} is missing stable entity id {entity_id}")
        for schema_type in ("SoftwareApplication", "WebPage", "BreadcrumbList"):
            if schema_type not in encoded:
                failures.append(f"{relative} is missing {schema_type} structured data")
        if canonical not in encoded:
            failures.append(f"{relative} structured data omits its canonical URL")
        if '"aggregateRating"' in encoded or '"review"' in encoded:
            failures.append(f"{relative} contains unsupported review or rating markup")

    registry_path = ROOT / "docs/discoverability-registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        failures.append(f"invalid discoverability registry: {exc}")
        registry = {}

    entity = registry.get("entity", {})
    if entity.get("canonical_homepage") != HOME:
        failures.append("registry canonical homepage is inconsistent")
    if entity.get("software_id") != f"{SITE}/#software":
        failures.append("registry software id is inconsistent")

    contract = registry.get("measurement_contract", {})
    if contract.get("primary_outcome") != "cost per successful, evidence-supported task":
        failures.append("registry primary outcome is not the evidence-supported task metric")
    dimensions = set(contract.get("required_dimensions", []))
    if not REQUIRED_DIMENSIONS.issubset(dimensions):
        failures.append("registry omits one or more required measurement dimensions")
    normalized_boundaries = " ".join(contract.get("boundaries", [])).casefold()
    for boundary in ("no universal", "task success", "first-party", "engine, locale"):
        if boundary not in normalized_boundaries:
            failures.append(f"registry omits discoverability boundary {boundary!r}")

    intents = registry.get("intent_targets", [])
    intent_ids = [intent.get("id") for intent in intents]
    if not intent_ids or len(intent_ids) != len(set(intent_ids)):
        failures.append("registry intent ids must be present and unique")
    for intent in intents:
        intent_id = intent.get("id", "<missing>")
        canonical = intent.get("canonical_url", "")
        if canonical not in sitemap:
            failures.append(f"registry intent is absent from sitemap: {intent_id}")
        local_page = _local_page_for_url(canonical)
        if local_page is None or not local_page.is_file():
            failures.append(f"registry intent lacks a local canonical page: {intent_id}")
        if not intent.get("queries") or not intent.get("evidence"):
            failures.append(f"registry intent lacks queries or evidence: {intent_id}")
        if not intent.get("answer_boundary"):
            failures.append(f"registry intent lacks an answer boundary: {intent_id}")

    channels = registry.get("observation_channels", [])
    if not channels:
        failures.append("registry must declare external observation channels")
    for channel in channels:
        if channel.get("status") not in ALLOWED_OBSERVATION_STATUSES:
            failures.append(
                f"observation channel overstates activation: {channel.get('name', '<missing>')}"
            )

    llms = _read("llms.txt")
    if llms != _read("docs/llms.txt"):
        failures.append("root and docs llms.txt mirrors differ")
    if f"{SITE}/docs/discoverability-registry.json" not in llms:
        failures.append("llms.txt does not expose the discoverability registry")
    if 'href="discoverability-registry.json"' not in homepage:
        failures.append("homepage does not expose the discoverability registry")

    public_evidence = " ".join(_read("docs/public-evidence.md").casefold().split())
    for phrase in (
        "establish a google ranking",
        "private transcript",
        "first-party entroly page",
        "nondeterministic observations",
    ):
        if phrase not in public_evidence:
            failures.append(f"public evidence omits discovery boundary {phrase!r}")

    return failures


def main() -> int:
    failures = collect_failures()
    if failures:
        print("Discoverability verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Discoverability verification passed.")
    print("Repository-controlled discovery prerequisites are consistent; rankings remain external.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
