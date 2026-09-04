"""Reported-metric honesty: a receipt must not report a number it did not measure.

``omitted_context`` is a bounded, score-descending *listing*. Counting entries
in that listing yields a value that saturates at the listing cap, so a receipt
that withheld 299 relevant chunks used to be indistinguishable from one that
withheld 20 -- and the ``omitted_evidence_pressure`` control derived from that
saturated count could read ``low`` while a clear majority of relevant evidence
was withheld.
"""

from __future__ import annotations

import pytest

from entroly.context_receipts import run_receipt_pipeline
from entroly.context_receipts.selection import MAX_OMITTED_LISTED

QUERY = "aardvark telemetry checkpoint"

# Every document gets unique filler. Repeated content is collapsed by the
# redundancy/dedup paths, which would silently shrink the corpus and make the
# saturation impossible to observe.
_FILLER = ("zebra", "mango", "quartz", "lantern", "gravel", "wombat", "syrup")


def _relevant_doc(i: int) -> tuple[str, str]:
    words = " ".join(f"{w}{i}_{j}" for j, w in enumerate(_FILLER))
    return (
        f"rel_{i:04d}.md",
        f"# S{i} aardvark telemetry checkpoint\n"
        f"aardvark telemetry checkpoint entry {i}\n{words}\n"
        + " ".join(f"t{i}q{k}" for k in range(12)),
    )


def _irrelevant_doc(i: int) -> tuple[str, str]:
    words = " ".join(f"{w}{i}_{j}x" for j, w in enumerate(_FILLER))
    return (
        f"irr_{i:04d}.md",
        f"# U{i} unrelated bookkeeping\n"
        f"nothing in record {i} concerns the query\n{words}\n"
        + " ".join(f"u{i}q{k}" for k in range(12)),
    )


def _corpus(relevant: int, irrelevant: int = 0) -> list[tuple[str, str]]:
    return [_relevant_doc(i) for i in range(relevant)] + [
        _irrelevant_doc(i) for i in range(irrelevant)
    ]


def _band(pressure: float) -> str:
    if pressure > 0.4:
        return "high"
    if pressure > 0.15:
        return "medium"
    return "low"


def _receipt(documents, budget, prefer_rust):
    return run_receipt_pipeline(
        documents, query=QUERY, token_budget=budget, prefer_rust=prefer_rust
    )


def _omitted_warning(receipt) -> str:
    matches = [w for w in receipt["warnings"] if "relevant chunk(s) were omitted" in w]
    assert matches, f"no omitted-evidence warning in {receipt['warnings']}"
    return matches[0]


@pytest.mark.parametrize("prefer_rust", [True, False], ids=["rust", "python"])
@pytest.mark.parametrize("documents", [60, 300])
def test_omitted_evidence_count_does_not_saturate_at_the_listing_cap(
    documents, prefer_rust
):
    receipt = _receipt(_corpus(documents), budget=90, prefer_rust=prefer_rust)
    risk = receipt["risk_summary"]

    # Mechanism guard: the fixture must actually overflow the listing, and no
    # document may have been collapsed away before selection ran.
    assert risk["total_chunks"] == documents
    assert len(receipt["omitted_context"]) == MAX_OMITTED_LISTED
    assert risk["omitted_context_listed"] == MAX_OMITTED_LISTED
    assert risk["omitted_context_listing_truncated"] is True

    # The count taken from the listing is a lower bound and must say so.
    assert risk["omitted_relevant_chunks_exact"] is False
    assert risk["omitted_chunks_total"] == documents - risk["selected_chunks"]
    assert risk["omitted_relevant_chunks_upper_bound"] == risk["omitted_chunks_total"]

    warning = _omitted_warning(receipt)
    assert warning.startswith("at least "), warning
    assert str(risk["omitted_chunks_total"]) in warning


@pytest.mark.parametrize("prefer_rust", [True, False], ids=["rust", "python"])
def test_omitted_evidence_scale_is_visible_across_two_corpus_sizes(prefer_rust):
    """Two corpora that differ by 240 withheld chunks must not report the same."""
    small = _receipt(_corpus(60), budget=90, prefer_rust=prefer_rust)["risk_summary"]
    large = _receipt(_corpus(300), budget=90, prefer_rust=prefer_rust)["risk_summary"]

    # Pre-fix, this was the *only* omitted-evidence quantity in the receipt and
    # it read 20 for both corpora.
    assert small["omitted_relevant_chunks"] == large["omitted_relevant_chunks"]
    assert small["omitted_chunks_total"] != large["omitted_chunks_total"]
    assert large["omitted_chunks_total"] - small["omitted_chunks_total"] == 240


@pytest.mark.parametrize("prefer_rust", [True, False], ids=["rust", "python"])
def test_omitted_evidence_pressure_fails_closed_when_the_listing_truncates(
    prefer_rust,
):
    """The control must not read ``low`` while most relevant evidence is withheld."""
    receipt = _receipt(_corpus(300), budget=3920, prefer_rust=prefer_rust)
    risk = receipt["risk_summary"]
    selected = risk["selected_chunks"]
    listed_relevant = risk["omitted_relevant_chunks"]
    true_omitted = risk["total_chunks"] - selected

    # Mechanism guard: enough context fits that the *saturated* count would have
    # produced a reassuring band, while the true withholding is severe.
    assert risk["omitted_context_listing_truncated"] is True
    assert risk["omitted_relevant_chunks_exact"] is False
    saturated_band = _band(listed_relevant / (selected + listed_relevant))
    true_band = _band(true_omitted / (selected + true_omitted))
    assert saturated_band == "low"
    assert true_band == "high"

    controls = risk["controls"]
    assert controls["omitted_evidence_pressure"] == "high"
    assert (
        controls["omitted_evidence_pressure_basis"]
        == "upper_bound_truncated_omitted_listing"
    )


@pytest.mark.parametrize("prefer_rust", [True, False], ids=["rust", "python"])
def test_relevant_count_stays_exact_when_the_listing_reaches_the_zero_score_tail(
    prefer_rust,
):
    """Negative control: no fail-closed escalation when the count is provably complete.

    The listing is score-descending, so a truncated listing that still contains
    a zero-score entry has already passed every relevant omission.
    """
    receipt = _receipt(_corpus(10, irrelevant=60), budget=60, prefer_rust=prefer_rust)
    risk = receipt["risk_summary"]

    assert risk["total_chunks"] == 70
    assert risk["omitted_context_listing_truncated"] is True
    assert any(item["score"] <= 0 for item in receipt["omitted_context"])
    assert risk["omitted_relevant_chunks_exact"] is True
    assert (
        risk["controls"]["omitted_evidence_pressure_basis"]
        == "exact_relevant_omitted_count"
    )
    assert (
        risk["omitted_relevant_chunks_upper_bound"] == risk["omitted_relevant_chunks"]
    )
    assert not _omitted_warning(receipt).startswith("at least ")


@pytest.mark.parametrize("prefer_rust", [True, False], ids=["rust", "python"])
def test_untruncated_listing_reports_an_exact_count(prefer_rust):
    receipt = _receipt(_corpus(5), budget=90, prefer_rust=prefer_rust)
    risk = receipt["risk_summary"]

    assert risk["total_chunks"] == 5
    assert len(receipt["omitted_context"]) < MAX_OMITTED_LISTED
    assert risk["omitted_context_listing_truncated"] is False
    assert risk["omitted_relevant_chunks_exact"] is True
    assert risk["omitted_relevant_chunks"] == risk["omitted_chunks_total"]
    assert _omitted_warning(receipt) == (
        f"{risk['omitted_relevant_chunks']} relevant chunk(s) were omitted; "
        "inspect omitted_context."
    )
