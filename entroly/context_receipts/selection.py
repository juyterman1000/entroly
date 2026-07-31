"""Budgeted context selection for Context Receipts."""

from __future__ import annotations

from collections import defaultdict, deque

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


def _dependency_closure(
    root_chunk_id: str,
    chunks: dict[str, DocumentChunk],
    deps_by_source: dict[str, list[DependencyLink]],
) -> tuple[list[str], list[str]]:
    """Return a deterministic, cycle-safe resolved dependency closure.

    Context sent to an agent must not contain a referencing fragment while
    silently dropping a dependency merely because the remaining budget became
    too small.  The selector therefore treats each root plus all transitively
    reachable, resolved dependencies as one atomic bundle.

    Unresolved references cannot be closed locally.  They are returned as
    warnings so the receipt fails visibly rather than pretending the bundle is
    complete.
    """

    ordered = [root_chunk_id]
    seen = {root_chunk_id}
    pending = deque([root_chunk_id])
    warnings: list[str] = []

    while pending:
        source_id = pending.popleft()
        dependencies = sorted(
            deps_by_source.get(source_id, []),
            key=lambda item: (
                item.relation_type,
                item.target_chunk_id or "",
                item.evidence,
            ),
        )
        for dependency in dependencies:
            target_id = dependency.target_chunk_id
            if target_id and target_id in chunks:
                if target_id not in seen:
                    seen.add(target_id)
                    ordered.append(target_id)
                    pending.append(target_id)
                continue

            warning = dependency.warning
            if not warning:
                warning = (
                    f"Unresolved dependency from {source_id}: "
                    f"{target_id or dependency.evidence}"
                )
            warnings.append(warning)

    return ordered, list(dict.fromkeys(warnings))


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
    closure_cache: dict[str, tuple[list[str], list[str]]] = {}

    def dependency_closure(chunk_id: str) -> tuple[list[str], list[str]]:
        if chunk_id not in closure_cache:
            closure_cache[chunk_id] = _dependency_closure(
                chunk_id, chunks, deps_by_source
            )
        return closure_cache[chunk_id]

    selected_ids: list[str] = []
    selected_set: set[str] = set()
    selected_tokens = 0
    token_sets: dict[str, set[str]] = {
        chunk.chunk_id: set(tokenize(chunk.text)) for chunk in index.chunks
    }
    warnings: list[str] = []

    for rank in ranked:
        if rank.chunk_id not in chunks:
            continue
        if rank.final_score <= 0:
            continue
        redundant = any(
            _jaccard(token_sets[rank.chunk_id], token_sets[sid]) >= 0.82
            for sid in selected_ids
        )
        if redundant:
            continue

        bundle_ids, bundle_warnings = dependency_closure(rank.chunk_id)
        new_bundle_ids = [
            chunk_id for chunk_id in bundle_ids if chunk_id not in selected_set
        ]
        bundle_tokens = sum(chunks[chunk_id].token_count for chunk_id in new_bundle_ids)
        remaining_tokens = max(0, token_budget - selected_tokens)
        if bundle_tokens > remaining_tokens:
            dependency_count = max(0, len(bundle_ids) - 1)
            if dependency_count:
                warnings.append(
                    "Dependency bundle omitted atomically due to budget: "
                    f"{rank.chunk_id} requires "
                    f"{bundle_tokens} token(s) including {dependency_count} "
                    f"resolved dependency chunk(s), {remaining_tokens} remain."
                )
            continue
        for chunk_id in new_bundle_ids:
            chunk = chunks[chunk_id]
            selected_set.add(chunk_id)
            selected_ids.append(chunk_id)
            selected_tokens += chunk.token_count
        warnings.extend(bundle_warnings)

    selected_items: list[SelectedContextItem] = []
    for chunk_id in selected_ids:
        chunk = chunks[chunk_id]
        rank = ranks.get(
            chunk_id,
            RankedChunk(chunk_id, 0.0, 0.0, 0.0, 0.0, ["included as dependency"]),
        )
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
                    d.target_chunk_id
                    for d in deps
                    if d.target_chunk_id and d.target_chunk_id in selected_set
                ],
                dependencies_missing=[
                    d.target_chunk_id or d.evidence
                    for d in deps
                    if not d.target_chunk_id or d.target_chunk_id not in selected_set
                ],
                fingerprint=chunk.fingerprint,
                text=chunk.text,
                fragment_sha256=chunk.fragment_sha256,
                source_sha256=chunk.source_sha256,
            )
        )

    omitted: list[OmittedContextItem] = []
    for rank in ranked:
        if rank.chunk_id in selected_set or rank.chunk_id not in chunks:
            continue
        chunk = chunks[rank.chunk_id]
        reason = "lower ranked than selected context under token budget"
        if selected_tokens + chunk.token_count > token_budget:
            reason = "budget_limit"
        if any(
            _jaccard(token_sets[rank.chunk_id], token_sets[sid]) >= 0.82
            for sid in selected_ids
        ):
            reason = "redundant_with_selected_context"
        if any(
            chunk.document_id == chunks[sid].document_id
            and abs(chunk.chunk_index - chunks[sid].chunk_index) == 1
            for sid in selected_ids
        ):
            reason = "nearby_relevant_context_omitted_due_to_budget"
            warnings.append(
                f"Nearby relevant chunk omitted: {chunk.chunk_id} from {chunk.source_path}"
            )
        bundle_ids, _ = dependency_closure(rank.chunk_id)
        missing_bundle_tokens = sum(
            chunks[chunk_id].token_count
            for chunk_id in bundle_ids
            if chunk_id not in selected_set
        )
        if (
            len(bundle_ids) > 1
            and selected_tokens + missing_bundle_tokens > token_budget
        ):
            reason = "dependency_bundle_exceeds_budget"
        elif any(
            d.target_chunk_id == rank.chunk_id and d.source_chunk_id in selected_set
            for d in dependency_links
        ):
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
                byte_start=chunk.byte_start,
                byte_end=chunk.byte_end,
                fragment_sha256=chunk.fragment_sha256,
                source_sha256=chunk.source_sha256,
            )
        )
        if len(omitted) >= max_omitted:
            break

    if not selected_items:
        if any(rank.final_score > 0 for rank in ranked):
            warnings.append("No chunks fit inside the token budget.")
            warnings.append(
                "Relevant chunks were found, but none fit inside the token budget."
            )
        else:
            warnings.append(
                "No relevant chunks matched the query; selection failed closed."
            )
    unresolved = [d for d in dependency_links if not d.resolved]
    if unresolved:
        warnings.append(
            f"{len(unresolved)} dependency reference(s) could not be resolved to an ingested chunk."
        )
    return SelectionResult(selected_items, omitted, list(dict.fromkeys(warnings)))
