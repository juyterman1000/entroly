from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
TOKENOMICS = (ROOT / "docs" / "live-tokenomics.md").read_text(encoding="utf-8")
INTEGRATIONS = (ROOT / "docs" / "integration-hub.md").read_text(encoding="utf-8")
ADOPTION = json.loads(
    (ROOT / "docs" / "adoption-evidence.json").read_text(encoding="utf-8")
)


def test_live_tokenomics_is_front_loaded_and_uses_real_commands() -> None:
    assert README.index("## ⚡ Live Tokenomics") < README.index(
        "## What is Entroly? (in plain English)"
    )
    assert "entroly value --json" in README
    assert "entroly dashboard" in README
    assert "Estimated cost avoided" in README
    assert "entroly_proxy_tokens_saved_total" in README


def test_public_counter_contract_refuses_unmeasured_worldwide_claims() -> None:
    combined = README + TOKENOMICS
    normalized = " ".join(combined.split()).casefold()
    required_boundaries = (
        "not a fabricated",
        "rounded down to whole 1,000-token units",
        "not an exact worldwide total",
        "Downloads are package fetches, not users",
        "fails closed to the checked-in proof",
    )
    for boundary in required_boundaries:
        assert boundary.casefold() in normalized

    assert "downloads ×" not in normalized
    assert "guarantees realized savings" not in normalized


def test_download_milestone_is_reconciled_and_not_called_users() -> None:
    components = ADOPTION["components"]
    total = sum(int(item["events"]) for item in components.values())
    assert total == ADOPTION["headline"]["events"] == 100_438
    assert ADOPTION["headline"]["unique_users"] is False
    assert ADOPTION["headline"]["successful_activations"] is False
    assert "100,438 downloads" in README
    assert "Measured across different distribution sources" in README


def test_requested_integration_names_are_discoverable_from_readme() -> None:
    expected = (
        "Vercel AI SDK",
        "OpenAI SDK",
        "Anthropic SDK",
        "LangChain",
        "Agno",
        "Strands Agents",
        "CrewAI",
        "AutoGen",
        "LiteLLM",
        "Claude Code on Vertex AI",
        "Claude Code on Azure AI Foundry",
        "Claude Code in VS Code",
        "VS Code Copilot",
        "OpenClaw",
        "OpenCode",
        "Grok",
        "MCP",
    )
    for name in expected:
        assert name in README
        assert name in INTEGRATIONS


def test_integration_statuses_match_code_backed_scope() -> None:
    direct_evidence = (
        "entroly-wasm/js/app_sdk.js",
        "entroly-wasm/test_app_sdk.js",
        "entroly/integrations/langchain.py",
        "tests/test_langchain_deep_integration.py",
        "entroly/integrations/litellm.py",
        "tests/test_framework_request_adapters.py",
    )
    for relative in direct_evidence:
        assert relative in INTEGRATIONS
        assert (ROOT / relative).is_file()

    for heading in ("Agno", "Strands Agents", "CrewAI", "AutoGen"):
        section = re.search(
            rf"### {re.escape(heading)}\n(?P<body>.*?)(?=\n### |\n## |\Z)",
            INTEGRATIONS,
            flags=re.DOTALL,
        )
        assert section is not None
        assert "no dedicated" in section.group("body").casefold()


def test_local_markdown_links_in_new_docs_resolve() -> None:
    for document in (ROOT / "docs" / "live-tokenomics.md", ROOT / "docs" / "integration-hub.md"):
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            assert (document.parent / relative).resolve().exists(), (
                f"broken local link in {document.relative_to(ROOT)}: {target}"
            )
