"""Budgeted context selection for Context Receipts."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable

from .models import (
    ContextIndex,
    DependencyLink,
    DocumentChunk,
    OmittedContextItem,
    RankedChunk,
    SelectedContextItem,
    stable_hash,
)
from .retrieval import tokenize


EXACT_CLOSED_SET_ROOT_LIMIT = 14
_SCORE_EPSILON = 1e-9


class SelectionResult:
    def __init__(
        self,
        selected: list[SelectedContextItem],
        omitted: list[OmittedContextItem],
        warnings: list[str],
        certificate: dict,
    ) -> None:
        self.selected = selected
        self.omitted = omitted
        self.warnings = warnings
        self.certificate = certificate


@dataclass(frozen=True)
class _SelectionPlan:
    selected_ids: tuple[str, ...]
    selected_root_ids: tuple[str, ...]
    token_count: int
    objective_score: float
    optimizer: str
    exact: bool


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


def _rounded_score(value: float) -> float:
    return round(max(0.0, value), 6)


def _is_better_plan(candidate: _SelectionPlan, incumbent: _SelectionPlan) -> bool:
    if candidate.objective_score > incumbent.objective_score + _SCORE_EPSILON:
        return True
    if incumbent.objective_score > candidate.objective_score + _SCORE_EPSILON:
        return False
    if candidate.token_count != incumbent.token_count:
        return candidate.token_count < incumbent.token_count
    return candidate.selected_root_ids < incumbent.selected_root_ids


def _fractional_relaxed_upper_bound(
    chunks: dict[str, DocumentChunk],
    ranked: list[RankedChunk],
    token_budget: int,
) -> float:
    """Upper-bound the closed-set objective after relaxing every dependency.

    This is deliberately optimistic: it ignores dependency costs, redundancy,
    and indivisibility.  The resulting fractional-knapsack value can therefore
    certify a conservative regret ceiling for heuristic plans without claiming
    that answer quality itself has been optimized.
    """

    remaining = max(0, token_budget)
    total = 0.0
    candidates: list[tuple[float, float, str, int]] = []
    for rank in ranked:
        chunk = chunks.get(rank.chunk_id)
        score = _rounded_score(rank.final_score)
        if chunk is None or score <= 0:
            continue
        cost = max(0, chunk.token_count)
        if cost == 0:
            total += score
            continue
        candidates.append((score / cost, score, rank.chunk_id, cost))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    for _, score, _, cost in candidates:
        if remaining <= 0:
            break
        used = min(remaining, cost)
        total += score * (used / cost)
        remaining -= used
    return _rounded_score(total)


def select_context(
    index: ContextIndex,
    ranked: list[RankedChunk],
    dependency_links: list[DependencyLink],
    *,
    token_budget: int,
    max_omitted: int = 20,
) -> SelectionResult:
    token_budget = max(0, token_budget)
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

    token_sets: dict[str, set[str]] = {
        chunk.chunk_id: set(tokenize(chunk.text)) for chunk in index.chunks
    }
    positive_roots = [
        rank
        for rank in ranked
        if rank.chunk_id in chunks and rank.final_score > 0
    ]
    positive_score_by_id = {
        rank.chunk_id: _rounded_score(rank.final_score) for rank in positive_roots
    }
    rank_position = {
        rank.chunk_id: position for position, rank in enumerate(positive_roots)
    }

    def materialize_root_ids(
        root_ids: set[str],
        *,
        optimizer: str,
        exact: bool,
    ) -> _SelectionPlan | None:
        """Materialize a candidate root set using deterministic rank order."""

        selected: list[str] = []
        selected_set: set[str] = set()
        selected_roots: list[str] = []
        used_tokens = 0
        for rank in positive_roots:
            root_id = rank.chunk_id
            if root_id not in root_ids:
                continue
            if root_id not in selected_set and any(
                _jaccard(token_sets[root_id], token_sets[selected_id]) >= 0.82
                for selected_id in selected
            ):
                return None
            bundle_ids, _ = dependency_closure(root_id)
            new_ids = [
                chunk_id for chunk_id in bundle_ids if chunk_id not in selected_set
            ]
            new_tokens = sum(chunks[chunk_id].token_count for chunk_id in new_ids)
            if used_tokens + new_tokens > token_budget:
                return None
            selected_roots.append(root_id)
            for chunk_id in new_ids:
                selected_set.add(chunk_id)
                selected.append(chunk_id)
                used_tokens += chunks[chunk_id].token_count
        objective = sum(
            positive_score_by_id.get(chunk_id, 0.0) for chunk_id in selected
        )
        return _SelectionPlan(
            selected_ids=tuple(selected),
            selected_root_ids=tuple(selected_roots),
            token_count=used_tokens,
            objective_score=_rounded_score(objective),
            optimizer=optimizer,
            exact=exact,
        )

    def greedy_plan(
        chooser: Callable[
            [list[RankedChunk], list[str], set[str], int], RankedChunk | None
        ],
        *,
        optimizer: str,
    ) -> _SelectionPlan:
        selected: list[str] = []
        selected_set: set[str] = set()
        selected_roots: list[str] = []
        remaining_roots = list(positive_roots)
        used_tokens = 0
        while remaining_roots:
            rank = chooser(remaining_roots, selected, selected_set, used_tokens)
            if rank is None:
                break
            remaining_roots.remove(rank)
            root_id = rank.chunk_id
            if root_id in selected_set:
                selected_roots.append(root_id)
                continue
            if any(
                _jaccard(token_sets[root_id], token_sets[selected_id]) >= 0.82
                for selected_id in selected
            ):
                continue
            bundle_ids, _ = dependency_closure(root_id)
            new_ids = [
                chunk_id for chunk_id in bundle_ids if chunk_id not in selected_set
            ]
            new_tokens = sum(chunks[chunk_id].token_count for chunk_id in new_ids)
            if used_tokens + new_tokens > token_budget:
                continue
            selected_roots.append(root_id)
            for chunk_id in new_ids:
                selected_set.add(chunk_id)
                selected.append(chunk_id)
                used_tokens += chunks[chunk_id].token_count

        # Delivery order is always rank-stable, even when density chose the set.
        plan = materialize_root_ids(
            set(selected_roots), optimizer=optimizer, exact=False
        )
        if plan is None:
            raise AssertionError("greedy selector produced an infeasible plan")
        return plan

    def choose_rank(
        candidates: list[RankedChunk],
        _selected: list[str],
        _selected_set: set[str],
        _used_tokens: int,
    ) -> RankedChunk | None:
        return candidates[0] if candidates else None

    def choose_density(
        candidates: list[RankedChunk],
        selected: list[str],
        selected_set: set[str],
        used_tokens: int,
    ) -> RankedChunk | None:
        choices: list[tuple[float, float, int, str, RankedChunk]] = []
        for rank in candidates:
            root_id = rank.chunk_id
            if root_id not in selected_set and any(
                _jaccard(token_sets[root_id], token_sets[selected_id]) >= 0.82
                for selected_id in selected
            ):
                continue
            bundle_ids, _ = dependency_closure(root_id)
            new_ids = [
                chunk_id for chunk_id in bundle_ids if chunk_id not in selected_set
            ]
            new_tokens = sum(chunks[chunk_id].token_count for chunk_id in new_ids)
            if used_tokens + new_tokens > token_budget:
                continue
            marginal_score = sum(
                positive_score_by_id.get(chunk_id, 0.0) for chunk_id in new_ids
            )
            density = (
                marginal_score / new_tokens
                if new_tokens > 0
                else float("inf")
                if marginal_score > 0
                else 0.0
            )
            choices.append(
                (
                    density,
                    marginal_score,
                    -rank_position[root_id],
                    root_id,
                    rank,
                )
            )
        if not choices:
            return None
        choices.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        return choices[0][4]

    rank_plan = greedy_plan(choose_rank, optimizer="rank_order_atomic_greedy")
    if len(positive_roots) <= EXACT_CLOSED_SET_ROOT_LIMIT:
        best_plan = _SelectionPlan(
            selected_ids=(),
            selected_root_ids=(),
            token_count=0,
            objective_score=0.0,
            optimizer="exact_dependency_closed_enumeration",
            exact=True,
        )
        root_ids = [rank.chunk_id for rank in positive_roots]
        for mask in range(1, 1 << len(root_ids)):
            candidate_roots = {
                root_id
                for position, root_id in enumerate(root_ids)
                if mask & (1 << position)
            }
            candidate = materialize_root_ids(
                candidate_roots,
                optimizer="exact_dependency_closed_enumeration",
                exact=True,
            )
            if candidate is not None and _is_better_plan(candidate, best_plan):
                best_plan = candidate
        plan = best_plan
    else:
        density_plan = greedy_plan(
            choose_density, optimizer="marginal_density_atomic_greedy"
        )
        if _is_better_plan(density_plan, rank_plan):
            plan = _SelectionPlan(
                selected_ids=density_plan.selected_ids,
                selected_root_ids=density_plan.selected_root_ids,
                token_count=density_plan.token_count,
                objective_score=density_plan.objective_score,
                optimizer="best_of_rank_and_marginal_density",
                exact=False,
            )
        else:
            plan = _SelectionPlan(
                selected_ids=rank_plan.selected_ids,
                selected_root_ids=rank_plan.selected_root_ids,
                token_count=rank_plan.token_count,
                objective_score=rank_plan.objective_score,
                optimizer="best_of_rank_and_marginal_density",
                exact=False,
            )

    selected_ids = list(plan.selected_ids)
    selected_set = set(selected_ids)
    selected_tokens = plan.token_count
    warnings: list[str] = []
    for root_id in plan.selected_root_ids:
        _, bundle_warnings = dependency_closure(root_id)
        warnings.extend(bundle_warnings)
    for rank in positive_roots:
        if rank.chunk_id in selected_set:
            continue
        bundle_ids, _ = dependency_closure(rank.chunk_id)
        missing_ids = [
            chunk_id for chunk_id in bundle_ids if chunk_id not in selected_set
        ]
        missing_tokens = sum(chunks[chunk_id].token_count for chunk_id in missing_ids)
        if len(bundle_ids) > 1 and selected_tokens + missing_tokens > token_budget:
            warnings.append(
                "Dependency bundle omitted atomically due to budget: "
                f"{rank.chunk_id} requires {missing_tokens} token(s) including "
                f"{max(0, len(bundle_ids) - 1)} resolved dependency chunk(s), "
                f"{max(0, token_budget - selected_tokens)} remain."
            )

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

    remaining_budget = max(0, token_budget - selected_tokens)
    recovery_frontier: list[dict] = []
    for rank in positive_roots:
        if rank.chunk_id in selected_set:
            continue
        bundle_ids, _ = dependency_closure(rank.chunk_id)
        missing_ids = [
            chunk_id for chunk_id in bundle_ids if chunk_id not in selected_set
        ]
        missing_tokens = sum(chunks[chunk_id].token_count for chunk_id in missing_ids)
        marginal_score = _rounded_score(
            sum(positive_score_by_id.get(chunk_id, 0.0) for chunk_id in missing_ids)
        )
        density = (
            marginal_score / missing_tokens
            if missing_tokens > 0
            else float("inf")
            if marginal_score > 0
            else 0.0
        )
        digest_bound = all(
            bool(chunks[chunk_id].fragment_sha256 and chunks[chunk_id].source_sha256)
            for chunk_id in bundle_ids
        )
        bundle_payload = {
            "chunk_ids": bundle_ids,
            "missing_chunk_ids": missing_ids,
            "root_chunk_id": rank.chunk_id,
        }
        recovery_frontier.append(
            {
                "root_chunk_id": rank.chunk_id,
                "chunk_ids": bundle_ids,
                "missing_chunk_ids": missing_ids,
                "missing_tokens": missing_tokens,
                "marginal_score": marginal_score,
                "marginal_density": (
                    "infinite" if density == float("inf") else round(density, 6)
                ),
                "fits_remaining_budget": missing_tokens <= remaining_budget,
                "source_spans_digest_bound": digest_bound,
                "bundle_sha256": stable_hash(bundle_payload),
            }
        )
    recovery_frontier.sort(
        key=lambda item: (
            0 if item["marginal_density"] == "infinite" else 1,
            0.0
            if item["marginal_density"] == "infinite"
            else -float(item["marginal_density"]),
            -float(item["marginal_score"]),
            int(item["missing_tokens"]),
            str(item["root_chunk_id"]),
        )
    )

    closure_satisfied = all(
        set(dependency_closure(chunk_id)[0]).issubset(selected_set)
        for chunk_id in selected_ids
    )
    selected_spans_digest_bound = all(
        bool(chunks[chunk_id].fragment_sha256 and chunks[chunk_id].source_sha256)
        for chunk_id in selected_ids
    )
    relaxed_upper_bound = _fractional_relaxed_upper_bound(
        chunks, positive_roots, token_budget
    )
    exact_optimality = plan.exact
    certified_upper_bound = (
        plan.objective_score if exact_optimality else relaxed_upper_bound
    )
    certificate = {
        "schema": "entroly.selection-certificate.v1",
        "optimizer": plan.optimizer,
        "optimality": (
            "exact_for_internal_relevance_objective"
            if exact_optimality
            else "heuristic_with_relaxed_upper_bound"
        ),
        "objective": {
            "name": "sum_positive_final_score",
            "scope": "retrieval score only; not answer quality",
            "constraints": [
                "hard_token_budget",
                "transitive_resolved_dependency_closure",
                "redundant_root_exclusion",
            ],
            "positive_candidate_roots": len(positive_roots),
            "exact_root_limit": EXACT_CLOSED_SET_ROOT_LIMIT,
            "selected_score": plan.objective_score,
            "rank_order_control_score": rank_plan.objective_score,
            "no_regression_vs_rank_order": (
                plan.objective_score + _SCORE_EPSILON >= rank_plan.objective_score
            ),
            "certified_upper_bound_score": certified_upper_bound,
            "relaxed_fractional_upper_bound_score": relaxed_upper_bound,
            "certified_regret_upper_bound": _rounded_score(
                max(0.0, certified_upper_bound - plan.objective_score)
            ),
        },
        "feasibility": {
            "hard_budget_satisfied": selected_tokens <= token_budget,
            "resolved_dependency_closure_satisfied": closure_satisfied,
            "selected_tokens": selected_tokens,
            "token_budget": token_budget,
            "selected_source_spans_digest_bound": (
                bool(selected_ids) and selected_spans_digest_bound
            ),
        },
        "recovery_frontier": recovery_frontier[:8],
    }
    certificate["certificate_sha256"] = stable_hash(certificate)
    return SelectionResult(
        selected_items,
        omitted,
        list(dict.fromkeys(warnings)),
        certificate,
    )
