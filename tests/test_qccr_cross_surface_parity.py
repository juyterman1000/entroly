"""Cross-surface contract for QCCR ranking.

Python/PyO3 and npm/WASM wrap the same ``Entroly-qccr`` crate, but feature
flags and marshaling can still drift. The fixture includes an explicit source
order so both wrappers execute the same input vector independent of JSON map
ordering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "qccr_parity.json"


def _cases():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"] == "entroly.qccr-parity.v2"
    assert data["cases"], "fixture has no cases -- the guard would pass vacuously"
    for case in data["cases"]:
        assert set(case["source_order"]) == set(case["files"])
        assert len(case["source_order"]) == len(case["files"])
    return data["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["id"])
def test_python_surface_matches_the_frozen_rank_order(case):
    qccr = pytest.importorskip(
        "entroly.qccr", reason="QCCR requires the native engine"
    )
    if not getattr(qccr, "_HAS_RUST", False):
        pytest.skip("native entroly_core not installed; QCCR has no pure-Python path")

    sources = list(case["source_order"])
    texts = [case["files"][source] for source in sources]
    ranked = qccr._rust_rank_files(sources, texts, case["query"], {})
    order = [
        sources[index]
        for index, _score in sorted(ranked, key=lambda item: (-item[1], item[0]))
    ]

    assert order == case["expected_rank_order"], (
        f"[{case['id']}] rank order drifted.\n"
        f"  expected: {case['expected_rank_order']}\n"
        f"  actual:   {order}\n"
        "Either the ranking change was unintended, or the fixture must be "
        "re-recorded with a stated product reason."
    )


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["id"])
def test_expansion_size_is_recorded_not_assumed(case):
    qccr = pytest.importorskip("entroly.qccr")
    if not getattr(qccr, "_HAS_RUST", False):
        pytest.skip("native entroly_core not installed")

    actual = len(qccr._rust_expand_query(case["query"]))
    assert actual == case["expansion_terms"], (
        f"[{case['id']}] expansion went from {case['expansion_terms']} to "
        f"{actual} terms. Confirm the rank order, then re-record intentionally."
    )
