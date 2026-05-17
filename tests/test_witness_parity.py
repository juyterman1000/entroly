"""WITNESS ships ONE behaviour — this is the gate that keeps it that way.

Background
----------
WITNESS had two implementations (Python here; Rust in entroly-core).
The proxy/MCP hot path silently preferred Rust when the native engine
was installed, and the Rust path diverged badly from the calibrated,
benchmarked Python behaviour — so native-engine users got a different,
broken WITNESS than every report measured. The product fix was not a
better divergence ledger: it was to ship **one** behaviour. The
validated Python implementation is now the single shipped path for
every surface (proxy, MCP, SDK, CLI); the Rust fast-path is OFF unless
`ENTROLY_WITNESS_RUST=1` is set.

This file has two roles:

1. **Shipped-behaviour tests (always run).** Assert that the default
   analyzer — exactly what every user gets — produces the validated
   decisions. This is the end-user guarantee.

2. **Rust conformance gate (only when ENTROLY_WITNESS_RUST=1).** If
   anyone opts the Rust path back on, it must reproduce the validated
   decisions on *every* fixture, with **no tolerated divergence**. This
   converts "two paths drift forever" into "the second path cannot ship
   until it is identical." There is deliberately no xfail: a divergent
   Rust path must fail the build, not be tracked.
"""

from __future__ import annotations

import os

import pytest

from entroly.witness import WitnessAnalyzer

RUST_OPT_IN = os.environ.get("ENTROLY_WITNESS_RUST", "0") == "1"


# (name, knowledge, question, answer, expect_suppressed)
# Single-sentence knowledge → no best-sentence-selection ambiguity, so
# each case is an unambiguous behavioural anchor.
FIXTURES = [
    (
        "recombination_hallucination",
        "Cadmium chloride is highly soluble in water and only slightly "
        "soluble in pure ethanol alcohol solvent.",
        "Cadmium chloride is only slightly soluble in which solvent?",
        "It is slightly soluble in water mixed with a faint hint of "
        "alcohol vapour.",
        True,
    ),
    (
        "wrong_slot_filler_hallucination",
        "Warren Bryant served as the chief executive officer of the "
        "Longs Drugs pharmacy chain headquartered in California.",
        "In which US state is the pharmacy chain whose chief executive "
        "was Warren Bryant headquartered?",
        "The pharmacy chain led by chief executive Warren Bryant is "
        "headquartered in Utah.",
        True,
    ),
    (
        "grounded_extractive_retained",
        "Warren Bryant served as the chief executive officer of the "
        "Longs Drugs pharmacy chain headquartered in California.",
        "In which US state is the pharmacy chain whose chief executive "
        "was Warren Bryant headquartered?",
        "Warren Bryant served as the chief executive officer of the "
        "pharmacy chain headquartered in California.",
        False,
    ),
]


def _suppressed(analyzer: WitnessAnalyzer, k: str, q: str, a: str) -> bool:
    _r, rw = analyzer.analyze_and_rewrite(
        f"{k}\n\nQuestion: {q}", a, mode="strict"
    )
    return rw.suppressed_count > 0


@pytest.mark.parametrize("name,k,q,a,expect", FIXTURES,
                         ids=[c[0] for c in FIXTURES])
def test_shipped_behaviour(name, k, q, a, expect):
    """The behaviour every user gets — default analyzer, no flags. This
    is the product guarantee: hallucinations suppressed, grounded
    answers retained, identically for everyone."""
    analyzer = WitnessAnalyzer(profile="benchmark_qa")  # exactly as shipped
    assert _suppressed(analyzer, k, q, a) == expect, (
        f"[{name}] shipped WITNESS decision != validated behaviour"
    )


@pytest.mark.skipif(
    not RUST_OPT_IN,
    reason="Rust path is off by default; conformance gate only runs "
    "when ENTROLY_WITNESS_RUST=1 (the explicit re-enable path).",
)
@pytest.mark.parametrize("name,k,q,a,expect", FIXTURES,
                         ids=[c[0] for c in FIXTURES])
def test_rust_conformance_gate(name, k, q, a, expect):
    """HARD GATE to re-enable Rust: it must reproduce the validated
    decision on every fixture. No xfail, no tolerance — a divergent
    Rust path fails the build instead of silently shipping."""
    py = _suppressed(WitnessAnalyzer(force_python=True, profile="benchmark_qa"),
                     k, q, a)
    rs = _suppressed(WitnessAnalyzer(force_python=False, profile="benchmark_qa"),
                     k, q, a)
    assert py == expect, f"[{name}] python (validated) != expected"
    assert rs == py, (
        f"[{name}] Rust path diverges (rust={rs} python={py}). Rust must "
        f"be identical before it may ship. Fix entroly-core/src/"
        f"witness.rs, do not relax this gate."
    )
