"""Fail-closed certificate adapter for the QCCR Python binding."""

from __future__ import annotations

from typing import Any


def _logical_source(source: str) -> str:
    head, separator, tail = str(source or "").rpartition("#")
    return head if separator and tail.isdigit() else str(source or "")


def attach_sufficiency(
    selected: list[dict],
    *,
    candidate_utility: dict[str, float],
    chunk_utility: dict | None = None,
    by_file: dict[str, list[dict]],
    query: str,
    token_budget: int,
) -> None:
    """Attach an honest file-level, explicitly uncalibrated certificate.

    QCCR exposes a utility for every candidate file, so captured mass and shadow
    price are measurable here. It does not expose which original chunks or
    sentences produced each synthetic selected fragment, so boundary exposure
    is marked unmeasured rather than incorrectly reported as zero.
    """

    if not candidate_utility:
        return
    del chunk_utility

    from .sufficiency import Candidate, _idf, _lexical_terms, _query_terms, certify

    chosen = {
        _logical_source(str(fragment.get("source") or ""))
        for fragment in selected
        if isinstance(fragment, dict)
    }
    terms = set(_query_terms(query))
    candidates: list[Candidate] = []
    corpus: list[str] = []
    attainable_terms: set[str] = set()

    for source, utility in candidate_utility.items():
        text = "\n".join(
            str(fragment.get("content") or "")
            for fragment in by_file.get(source, [])
        )
        corpus.append(text)
        lexical = _lexical_terms(text)
        matched = tuple(sorted(term for term in terms if term in lexical))
        attainable_terms.update(matched)
        candidates.append(
            Candidate(
                unit_id=source,
                utility=float(utility),
                cost=max(1, len(text) // 4),
                selected=source in chosen,
                anchors=matched,
                neighbourhood=(source,),
            )
        )

    # Retention must be measured from what was actually delivered, not from the
    # full source file merely because some synthetic output from that file won.
    delivered_text = "\n".join(
        str(fragment.get("content") or "")
        for fragment in selected
        if isinstance(fragment, dict)
    )
    delivered_lexical = _lexical_terms(delivered_text)
    retained = {term for term in terms if term in delivered_lexical}
    delivered_tokens = max(1, len(delivered_text) // 4) if delivered_text else 0

    certificate = certify(
        candidates,
        query_term_idf={term: _idf(term, corpus) for term in terms},
        retained_terms=retained,
        unattainable_terms=terms - attainable_terms,
        budget_exhausted=delivered_tokens >= token_budget * 0.95,
        boundary_exposure_measured=False,
    )
    payload: dict[str, Any] = certificate.to_dict()
    for fragment in selected:
        if isinstance(fragment, dict):
            fragment["sufficiency"] = payload
