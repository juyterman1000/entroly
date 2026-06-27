"""Receipt and Markdown report generation."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
from dataclasses import asdict
from typing import Any

from .dependencies import detect_dependencies
from .models import (
    SCHEMA_VERSION,
    CompressionRatio,
    ContextIndex,
    ContextReceipt,
    stable_hash,
)
from .novelty import novelty_frontier_assessment, novelty_summary
from .retrieval import rank_chunks
from .selection import select_context


def _int_or_default(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _compression_ratio(source_tokens: int, selected_tokens: int) -> CompressionRatio:
    source = max(0, _int_or_default(source_tokens))
    selected = max(0, _int_or_default(selected_tokens))
    if source == 0:
        return CompressionRatio(0, selected, 0, 1.0, 1.0, 0.0)
    selected_ratio = selected / source
    source_ratio = source / max(1, selected)
    return CompressionRatio(
        source_tokens=source,
        selected_tokens=selected,
        tokens_saved=max(0, source - selected),
        selected_to_source_ratio=round(selected_ratio, 6),
        source_to_selected_ratio=round(source_ratio, 6),
        reduction_pct=round((1.0 - selected_ratio) * 100.0, 3),
    )


_REVIEW_ORDER = {"low": 0, "medium": 1, "high": 2}


def _max_review_level(current: str, candidate: str) -> str:
    return (
        candidate
        if _REVIEW_ORDER.get(candidate, 0) > _REVIEW_ORDER.get(current, 0)
        else current
    )


def _apply_novelty_risk(
    risk_summary: dict[str, Any], novelty: dict[str, Any], warnings: list[str]
) -> None:
    assessment = novelty_frontier_assessment(novelty)
    pressure = str(assessment["pressure"])
    controls = risk_summary.get("controls")
    if not isinstance(controls, dict):
        controls = {}
        risk_summary["controls"] = controls
    controls["novelty_frontier"] = pressure
    risk_summary["novelty_summary"] = novelty
    risk_summary["novelty_frontier_assessment"] = assessment
    if pressure == "high":
        risk_summary["review_level"] = _max_review_level(
            str(risk_summary.get("review_level", "low")), "high"
        )
        warnings.append(
            "Novelty frontier is high: omitted context contains distinctive concepts absent from selected context."
        )
    elif pressure == "medium":
        risk_summary["review_level"] = _max_review_level(
            str(risk_summary.get("review_level", "low")), "medium"
        )
        warnings.append(
            "Novelty frontier is medium: review omitted frontier terms before relying on this receipt."
        )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float_metric(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        metric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return metric if math.isfinite(metric) else default


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _warning_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _risk_summary(
    index: ContextIndex,
    *,
    selected_count: int,
    selected_tokens: int,
    omitted_relevant: int,
    dependencies: list,
    warnings: list[str],
) -> dict:
    total_chunks = len(index.chunks)
    unresolved = sum(1 for dep in dependencies if not dep.resolved)
    missing_dependency_warnings = sum(
        1 for warning in warnings if "Dependency not included" in warning
    )
    token_coverage = selected_tokens / max(
        1, sum(chunk.token_count for chunk in index.chunks)
    )
    chunk_coverage = selected_count / max(1, total_chunks)
    omission_pressure = min(
        1.0, omitted_relevant / max(1, selected_count + omitted_relevant)
    )
    dependency_pressure = min(
        1.0, (unresolved + missing_dependency_warnings) / max(1, len(dependencies))
    )
    coverage_score = max(
        0.0,
        min(
            1.0,
            0.45 * token_coverage
            + 0.35 * chunk_coverage
            + 0.20 * (1.0 - dependency_pressure),
        ),
    )
    review_level = "low"
    if coverage_score < 0.45 or dependency_pressure > 0.4:
        review_level = "high"
    elif coverage_score < 0.70 or omission_pressure > 0.25:
        review_level = "medium"
    return {
        "coverage_score": round(coverage_score, 6),
        "review_level": review_level,
        "selected_chunks": selected_count,
        "total_chunks": total_chunks,
        "chunk_coverage": round(chunk_coverage, 6),
        "token_coverage": round(token_coverage, 6),
        "omitted_relevant_chunks": omitted_relevant,
        "unresolved_dependency_count": unresolved,
        "missing_dependency_warning_count": missing_dependency_warnings,
        "controls": {
            "dependency_closure": "complete"
            if unresolved == 0 and missing_dependency_warnings == 0
            else "partial",
            "omitted_evidence_pressure": "high"
            if omission_pressure > 0.4
            else "medium"
            if omission_pressure > 0.15
            else "low",
            "replayable_fingerprints": True,
            "local_no_llm_judgment": True,
        },
    }


def attach_novelty_frontier(
    receipt: dict[str, Any],
    index: ContextIndex | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a receipt dict with Python novelty-frontier fields attached.

    The Python SDK normally prefers the Rust engine for speed. Until the native
    engine grows the same novelty-frontier implementation, this bridge keeps the
    public Python receipt contract consistent by post-processing Rust receipts
    with the deterministic Python novelty audit and then recomputing the receipt
    id/hash over the enriched payload.
    """
    enriched = copy.deepcopy(receipt)
    risk_summary = enriched.setdefault("risk_summary", {})
    if not isinstance(risk_summary, dict):
        risk_summary = {}
        enriched["risk_summary"] = risk_summary

    py_receipt = ContextReceipt.from_dict(enriched)
    py_index = ContextIndex.from_dict(index) if isinstance(index, dict) else index
    omitted_text_by_chunk_id = (
        {chunk.chunk_id: chunk.text for chunk in py_index.chunks}
        if py_index is not None
        else None
    )
    novelty = novelty_summary(
        py_receipt.query,
        py_receipt.selected_context,
        py_receipt.omitted_context,
        omitted_text_by_chunk_id=omitted_text_by_chunk_id,
    )
    warnings = _warning_list(enriched.get("warnings", []))
    _apply_novelty_risk(risk_summary, novelty, warnings)
    enriched["warnings"] = list(dict.fromkeys(warnings))

    payload = {
        key: value
        for key, value in enriched.items()
        if key not in {"receipt_id", "reproducibility_hash"}
    }
    reproducibility_hash = stable_hash(payload)
    enriched["receipt_id"] = "cr_" + reproducibility_hash[:12]
    enriched["reproducibility_hash"] = reproducibility_hash
    return enriched


def build_receipt(
    index: ContextIndex, *, query: str, token_budget: int
) -> ContextReceipt:
    safe_token_budget = max(0, _int_or_default(token_budget))
    safe_query = "" if query is None else str(query)
    ranked = rank_chunks(index, safe_query)
    dependencies = detect_dependencies(index)
    selection = select_context(
        index, ranked, dependencies, token_budget=safe_token_budget
    )
    source_tokens = sum(chunk.token_count for chunk in index.chunks)
    selected_tokens = sum(item.token_count for item in selection.selected)
    ratio = _compression_ratio(source_tokens, selected_tokens)
    source_fingerprints = {
        "documents": {doc.source_path: doc.fingerprint for doc in index.documents},
        "chunks": {chunk.chunk_id: chunk.fingerprint for chunk in index.chunks},
    }
    ranking_reasons = {rank.chunk_id: rank.reasons for rank in ranked}
    warning_candidates = list(selection.warnings)
    high_omitted = [item for item in selection.omitted if item.score > 0]
    if high_omitted:
        warning_candidates.append(
            f"{len(high_omitted)} relevant chunk(s) were omitted; inspect omitted_context."
        )
    warnings = list(dict.fromkeys(warning_candidates))
    risk_summary = _risk_summary(
        index,
        selected_count=len(selection.selected),
        selected_tokens=selected_tokens,
        omitted_relevant=len(high_omitted),
        dependencies=dependencies,
        warnings=warnings,
    )
    omitted_text_by_chunk_id = {chunk.chunk_id: chunk.text for chunk in index.chunks}
    novelty = novelty_summary(
        safe_query,
        selection.selected,
        selection.omitted,
        omitted_text_by_chunk_id=omitted_text_by_chunk_id,
    )
    _apply_novelty_risk(risk_summary, novelty, warnings)
    warnings = list(dict.fromkeys(warnings))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "query": safe_query,
        "token_budget": safe_token_budget,
        "selected_context": [asdict(item) for item in selection.selected],
        "omitted_context": [asdict(item) for item in selection.omitted],
        "dependency_links": [asdict(link) for link in dependencies],
        "ranking_reasons": ranking_reasons,
        "compression_ratio": asdict(ratio),
        "source_fingerprints": source_fingerprints,
        "risk_summary": risk_summary,
        "warnings": warnings,
        "outcome_links": [],
    }
    reproducibility_hash = stable_hash(payload)
    return ContextReceipt(
        receipt_id="cr_" + reproducibility_hash[:12],
        schema_version=SCHEMA_VERSION,
        query=safe_query,
        token_budget=safe_token_budget,
        selected_context=selection.selected,
        omitted_context=selection.omitted,
        dependency_links=dependencies,
        ranking_reasons=ranking_reasons,
        compression_ratio=ratio,
        source_fingerprints=source_fingerprints,
        risk_summary=risk_summary,
        warnings=warnings,
        reproducibility_hash=reproducibility_hash,
        outcome_links=[],
    )


def markdown_report(receipt: ContextReceipt) -> str:
    ratio = receipt.compression_ratio
    risk_summary = _mapping(receipt.risk_summary)
    controls = _mapping(risk_summary.get("controls"))
    novelty = _mapping(risk_summary.get("novelty_summary"))
    novelty_assessment = _mapping(risk_summary.get("novelty_frontier_assessment"))
    lines = [
        f"# Context Receipt {receipt.receipt_id}",
        "",
        f"Query: `{receipt.query}`",
        "",
        "## Token Budget",
        "",
        f"- Budget: {receipt.token_budget}",
        f"- Source tokens: {ratio.source_tokens}",
        f"- Selected tokens: {ratio.selected_tokens}",
        f"- Reduction: {ratio.reduction_pct:.1f}%",
        f"- Source-to-selected ratio: {ratio.source_to_selected_ratio:.2f}:1",
        "",
        "## Coverage And Risk Controls",
        "",
        f"- Coverage score: {_float_metric(risk_summary.get('coverage_score')):.3f}",
        f"- Review level: {risk_summary.get('review_level', 'unknown')}",
        f"- Dependency closure: {controls.get('dependency_closure', 'unknown')}",
        f"- Omitted evidence pressure: {controls.get('omitted_evidence_pressure', 'unknown')}",
        f"- Novelty frontier pressure: {controls.get('novelty_frontier', 'unknown')}",
        f"- Novelty frontier action: {novelty_assessment.get('action', 'unknown')}",
        f"- Replayable fingerprints: {controls.get('replayable_fingerprints', False)}",
        "",
        "## Novelty Frontier",
        "",
        f"- Control: {novelty.get('control', 'unknown')}",
        f"- Selected novel terms: {', '.join(_string_list(novelty.get('selected_novel_terms'))) or 'none'}",
        f"- Omitted frontier terms: {', '.join(_string_list(novelty.get('omitted_frontier_terms'))) or 'none'}",
        f"- Frontier gap ratio: {_float_metric(novelty.get('frontier_gap_ratio')):.3f}",
        f"- Frontier score ratio: {_float_metric(novelty.get('frontier_gap_score_ratio')):.3f}",
        f"- Assessment rationale: {novelty_assessment.get('rationale', 'unknown')}",
        "",
        "## Included Context",
        "",
    ]
    if receipt.selected_context:
        for item in receipt.selected_context:
            heading = f" - {item.section_heading}" if item.section_heading else ""
            lines.extend(
                [
                    f"### {item.chunk_id}",
                    f"- Source: `{item.source_path}`{heading}",
                    f"- Tokens: {item.token_count}; score: {item.score:.4f}",
                    f"- Why: {'; '.join(item.reasons)}",
                    f"- Fingerprint: `{item.fingerprint}`",
                ]
            )
            if item.dependencies_included:
                lines.append(
                    f"- Dependencies included: {', '.join(item.dependencies_included)}"
                )
            if item.dependencies_missing:
                lines.append(
                    f"- Dependencies missing or unresolved: {', '.join(item.dependencies_missing)}"
                )
            lines.append("")
    else:
        lines.extend(["No context chunks were selected.", ""])

    lines.extend(["## Omitted Context", ""])
    if receipt.omitted_context:
        for item in receipt.omitted_context:
            heading = f" - {item.section_heading}" if item.section_heading else ""
            lines.extend(
                [
                    f"### {item.chunk_id}",
                    f"- Source: `{item.source_path}`{heading}",
                    f"- Tokens: {item.token_count}; score: {item.score:.4f}",
                    f"- Why omitted: {item.omission_reason}",
                    f"- Ranking reason: {'; '.join(item.reasons)}",
                    f"- Preview: {item.text_preview}",
                    "",
                ]
            )
    else:
        lines.extend(["No relevant omitted chunks were tracked.", ""])

    lines.extend(["## Dependency Graph Summary", ""])
    if receipt.dependency_links:
        for link in receipt.dependency_links:
            target = link.target_chunk_id or "UNRESOLVED"
            status = "resolved" if link.resolved else "unresolved"
            lines.append(
                f"- `{link.source_chunk_id}` -> `{target}` ({link.relation_type}, {status}): {link.evidence}"
            )
    else:
        lines.append("No explicit dependency links were detected.")
    lines.append("")

    lines.extend(["## Risks And Warnings", ""])
    if receipt.warnings:
        lines.extend(f"- {warning}" for warning in receipt.warnings)
    else:
        lines.append("- No warnings were emitted by the local heuristics.")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Reproducibility hash: `{receipt.reproducibility_hash}`",
            f"- Schema: `{receipt.schema_version}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def explain_omitted(receipt: ContextReceipt, chunk_id: str) -> str:
    for item in receipt.omitted_context:
        if item.chunk_id == chunk_id:
            return (
                f"{chunk_id} was omitted from {item.source_path}: {item.omission_reason}. "
                f"Score={item.score:.4f}. Ranking reasons: {'; '.join(item.reasons)}. "
                f"Preview: {item.text_preview}"
            )
    selected_ids = {item.chunk_id for item in receipt.selected_context}
    if chunk_id in selected_ids:
        return f"{chunk_id} was not omitted; it is present in selected_context."
    return f"{chunk_id} is not present in this receipt's selected or omitted context."
