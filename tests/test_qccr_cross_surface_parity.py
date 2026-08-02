"""Cross-surface contract for QCCR ranking.

Python (pip/MCP/SDK, via PyO3) and npm/WASM (via wasm-bindgen) wrap the SAME
`entroly-qccr` crate, so the ranking logic cannot drift by construction. What
CAN drift is everything around it: which weights each wrapper passes, how each
tokenizes before the call, feature flags (`regex-full` for native,
`regex-lite` for WASM), and the shape each returns.

`tests/fixtures/qccr_parity.json` freezes inputs and the expected file order so
both surfaces can be checked against one artifact rather than against each
other's current behaviour. This file enforces the Python side; the npm side
should load the same fixture.

The `regex-lite` split is the concrete reason this is not paranoia: the two
builds compile different regex engines, and the tokenizer's identifier and
sentence regexes run through them.

Regenerating
------------
The fixture encodes deliberate outcomes, not just whatever the code did:

* prose_query_meets_identifier -- "credit card charged" must reach
  `charge_card`/`StripeGateway`, a prose-to-identifier match.
* expansion_must_not_outvote_query -- 59 expansion terms fire on this query,
  and the file answering it must still rank first. This is the case that
  regressed: the answer ranked 23rd of 66 before expansion terms were
  down-weighted and saturated.
* stem_bridges_surface_forms -- "queries"/"cards" must meet
  `match_query`/`card` through the stemmer.

If a change reorders any of these, that is a product decision. Re-record only
with a reason in the commit message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "qccr_parity.json"


def _cases():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"] == "entroly.qccr-parity.v1"
    assert data["cases"], "fixture has no cases -- the guard would pass vacuously"
    return data["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_python_surface_matches_the_frozen_rank_order(case):
    qccr = pytest.importorskip(
        "entroly.qccr", reason="QCCR requires the native engine"
    )
    if not getattr(qccr, "_HAS_RUST", False):
        pytest.skip("native entroly_core not installed; QCCR has no pure-Python path")

    sources = list(case["files"].keys())
    texts = [case["files"][s] for s in sources]
    ranked = qccr._rust_rank_files(sources, texts, case["query"], {})
    order = [sources[i] for i, _ in sorted(ranked, key=lambda t: -t[1])]

    assert order == case["expected_rank_order"], (
        f"[{case['id']}] rank order drifted.\n"
        f"  expected: {case['expected_rank_order']}\n"
        f"  actual:   {order}\n"
        f"Either a ranking change was unintended, or the fixture needs "
        f"re-recording with a stated reason (see this module's docstring)."
    )


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_expansion_size_is_recorded_not_assumed(case):
    """Expansion size drives the dilution this ranking has to survive.

    Recorded so a change in the intent clusters shows up here rather than as an
    unexplained reordering: the retry case ranks correctly *because* the
    selector tolerates 59 injected terms, not because none fire.
    """
    qccr = pytest.importorskip("entroly.qccr")
    if not getattr(qccr, "_HAS_RUST", False):
        pytest.skip("native entroly_core not installed")

    actual = len(qccr._rust_expand_query(case["query"]))
    assert actual == case["expansion_terms"], (
        f"[{case['id']}] expansion went from {case['expansion_terms']} to "
        f"{actual} terms. That changes what the ranker must survive; confirm "
        f"the rank order above is still right, then re-record."
    )
