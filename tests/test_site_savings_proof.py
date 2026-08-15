from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
PROOF = json.loads(
    (ROOT / "docs" / "site-savings-proof.json").read_text(encoding="utf-8")
)


def test_site_savings_proof_reconciles_tokens_and_default_usd() -> None:
    assert PROOF["tokens_saved"] == PROOF["source_tokens"] - PROOF["selected_tokens"]
    assert PROOF["checks"] == {"passed": 12, "failed": 0}
    assert PROOF["exact_recovery_verified"] is True

    usd = PROOF["usd_model"]
    expected = (
        PROOF["tokens_saved"]
        / 1_000_000
        * usd["default_input_usd_per_million_tokens"]
    )
    assert usd["default_modeled_usd_saved"] == expected


def test_why_section_front_loads_savings_with_proof_and_local_totals() -> None:
    why = HTML.index('<section id="why">')
    savings = HTML.index('id="savings-proof"', why)
    comparison = HTML.index('<div class="comp-grid">', why)
    assert why < savings < comparison

    assert f'data-tokens-saved="{PROOF["tokens_saved"]}"' in HTML
    assert "709,927" in HTML
    assert "Modeled input cost avoided for this proof run" in HTML
    assert "Proof calculator: USD per 1M input tokens" in HTML
    assert "entroly value --json" in HTML
    assert "entroly dashboard" in HTML
    assert 'href="site-savings-proof.json"' in HTML


def test_site_does_not_present_the_proof_as_worldwide_savings() -> None:
    normalized = " ".join(HTML.split())
    assert "Conservative totals reported by opted-in Entroly proxy installations" in normalized
    assert "modeled, not a provider invoice" in normalized
    assert "modeled_from_visitor_supplied_rate" == PROOF["usd_model"]["status"]
    assert any("not a worldwide" in item for item in PROOF["limitations"])


def test_site_live_counter_is_configurable_polled_and_fails_closed() -> None:
    assert (
        'name="entroly-community-savings-endpoint" '
        'content="https://entroly-community-savings.'
        'entroly-community-savings-worker.workers.dev/v1/public-savings"'
        in HTML
    )
    assert "entroly.community-savings.v1" in HTML
    assert "reported_provider_tokens_saved" in HTML
    assert "reported_modeled_input_cost_avoided_usd" in HTML
    assert "credentials: 'omit'" in HTML
    assert "window.setInterval(pollCommunitySavings, 60_000)" in HTML
    assert "Verified proof fallback · live aggregate unavailable" in HTML
