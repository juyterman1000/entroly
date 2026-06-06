"""Budgeted context selection for Context Receipts."""

from __future__ import annotations

from collections import defaultdict

from .models import (
    ContextIndex,
    DependencyLink,
    DocumentChunk,
    OmittedContextItem,
    RankedChunk,
    SelectedContextItem,
)
from .retrieval import tokenize


class SelectionResult:
    def __init__(
        self,
        selected: list[SelectedContextItem],
        omitted: list[OmittedContextItem],
        warnings: list[str],
    ) -> None:
        self.selected = selected
        self.omitted = omitted
        self.warnings = warnings


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def select_context(
    index: ContextIndex,
    ranked: list[RankedChunk],
    dependency_links: list[DependencyLink],
    *,
    token_budget: int,
    max_omitted: int = 20,
) -> SelectionResult:
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    ranks = {rank.chunk_id: rank for rank in ranked}
    deps_by_source: dict[str, list[DependencyLink]] = defaultdict(list)
    for link in dependency_links:
        deps_by_source[link.source_chunk_id].append(link)

    selected_ids: list[str] = []
    selected_set: set[str] = set()
    selected_tokens = 0
    token_sets: dict[str, set[str]] = {chunk.chunk_id: set(tokenize(chunk.text)) for chunk in index.chunks}
    warnings: list[str] = []

    def can_add(chunk: DocumentChunk) -> bool:
        return selected_tokens + chunk.token_count <= token_budget

    def add(chunk_id: str) -> bool:
        nonlocal selected_tokens
        chunk = chunks[chunk_id]
        if chunk_id in selected_set or not can_add(chunk):
            return False
        selected_set.add(chunk_id)
        selected_ids.append(chunk_id)
        selected_tokens += chunk.token_count
        return True

    for rank in ranked:
        if rank.final_score <= 0 and selected_ids:
            continue
        chunk = chunks[rank.chunk_id]
        redundant = any(_jaccard(token_sets[rank.chunk_id], token_sets[sid]) >= 0.82 for sid in selected_ids)
        if redundant:
            continue
        if not add(rank.chunk_id):
            continue
        for dep in deps_by_source.get(rank.chunk_id, []):
            if dep.target_chunk_id and dep.target_chunk_id in chunks and dep.target_chunk_id not in selected_set:
                if not add(dep.target_chunk_id):
                    warnings.append(
                        f"Dependency not included due to budget: {rank.chunk_id} -> {dep.target_chunk_id} ({dep.relation_type})"
                    )
            elif not dep.resolved and dep.warning:
                warnings.append(dep.warning)

    selected_items: list[SelectedContextItem] = []
    for chunk_id in selected_ids:
        chunk = chunks[chunk_id]
        rank = ranks.get(chunk_id, RankedChunk(chunk_id, 0.0, 0.0, 0.0, 0.0, ["included as dependency"]))
        deps = deps_by_source.get(chunk_id, [])
        selected_items.append(
            SelectedContextItem(
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                section_heading=chunk.section_heading,
                page_number=chunk.page_number,
                byte_start=chunk.byte_start,
                byte_end=chunk.byte_end,
                token_start=chunk.token_start,
                token_end=chunk.token_end,
                token_count=chunk.token_count,
                score=rank.final_score,
                reasons=rank.reasons,
                dependencies_included=[
                    d.target_chunk_id for d in deps if d.target_chunk_id and d.target_chunk_id in selected_set
                ],
                dependencies_missing=[
                    d.target_chunk_id or d.evidence for d in deps if not d.target_chunk_id or d.target_chunk_id not in selected_set
                ],
                fingerprint=chunk.fingerprint,
                text=chunk.text,
            )
        )

    omitted: list[OmittedContextItem] = []
    for rank in ranked:
        if rank.chunk_id in selected_set:
            continue
        chunk = chunks[rank.chunk_id]
        reason = "lower ranked than selected context under token budget"
        if selected_tokens + chunk.token_count > token_budget:
            reason = "budget_limit"
        if any(_jaccard(token_sets[rank.chunk_id], token_sets[sid]) >= 0.82 for sid in selected_ids):
            reason = "redundant_with_selected_context"
        if any(
            chunk.document_id == chunks[sid].document_id and abs(chunk.chunk_index - chunks[sid].chunk_index) == 1
            for sid in selected_ids
        ):
            reason = "nearby_relevant_context_omitted_due_to_budget"
            warnings.append(f"Nearby relevant chunk omitted: {chunk.chunk_id} from {chunk.source_path}")
        if any(d.target_chunk_id == rank.chunk_id and d.source_chunk_id in selected_set for d in dependency_links):
            reason = "dependency_not_included_due_to_budget"
        omitted.append(
            OmittedContextItem(
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                section_heading=chunk.section_heading,
                page_number=chunk.page_number,
                token_count=chunk.token_count,
                score=rank.final_score,
                reasons=rank.reasons,
                omission_reason=reason,
                fingerprint=chunk.fingerprint,
                text_preview=_preview(chunk.text),
            )
        )
        if len(omitted) >= max_omitted:
            break

    if not selected_items:
        warnings.append("No chunks fit inside the token budget.")
    unresolved = [d for d in dependency_links if not d.resolved]
    if unresolved:
        warnings.append(f"{len(unresolved)} dependency reference(s) could not be resolved to an ingested chunk.")
    return SelectionResult(selected_items, omitted, list(dict.fromkeys(warnings)))
