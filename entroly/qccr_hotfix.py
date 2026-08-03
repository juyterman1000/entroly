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
    """Attach a token-boundary-safe, explicitly uncalibrated certificate."""
    if not candidate_utility:
        return

    from .sufficiency import (
        Candidate,
        _idf,
        _lexical_terms,
        _query_terms,
        certify,
        stem,
    )

    chosen = {_logical_source(str(fragment.get("source") or "")) for fragment in selected}
    terms = set(_query_terms(query))
    term_forms = [(term, stem(term)) for term in sorted(terms)]
    candidates: list[Candidate] = []
    corpus: list[str] = []
    attainable_terms: set[str] = set()

    def hits(text: str) -> tuple[str, ...]:
        lexical = _lexical_terms(text)
        return tuple(
            term
            for term, reduced in term_forms
            if term in lexical or (reduced is not None and reduced in lexical)
        )

    if chunk_utility:
        for key, utility in chunk_utility.items():
            source, part = key
            group = by_file.get(source, [])
            if not isinstance(part, int) or part < 0 or part >= len(group):
                continue
            text = str(group[part].get("content") or "")
            corpus.append(text)
            matched = hits(text)
            attainable_terms.update(matched)
            neighbours = tuple(
                f"{source}#{index}"
                for index in range(max(0, part - 1), min(len(group), part + 2))
            )
            candidates.append(
                Candidate(
                    unit_id=f"{source}#{part}",
                    utility=float(utility),
                    cost=max(1, len(text) // 4),
                    selected=source in chosen,
                    anchors=matched,
                    neighbourhood=neighbours,
                )
            )
    else:
        for source, utility in candidate_utility.items():
            text = "\n".join(
                str(fragment.get("content") or "")
                for fragment in by_file.get(source, [])
            )
            corpus.append(text)
            matched = hits(text)
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

    retained = {
        term
        for candidate in candidates
        if candidate.selected
        for term in candidate.anchors
    }
    delivered = sum(candidate.cost for candidate in candidates if candidate.selected)
    certificate = certify(
        candidates,
        query_term_idf={term: _idf(term, corpus) for term in terms},
        retained_terms=retained,
        unattainable_terms=terms - attainable_terms,
        budget_exhausted=delivered >= token_budget * 0.95,
        calibrated=False,
    )
    payload: dict[str, Any] = certificate.to_dict()
    for fragment in selected:
        if isinstance(fragment, dict):
            fragment["sufficiency"] = payload
