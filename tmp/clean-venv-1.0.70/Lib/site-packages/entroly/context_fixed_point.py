"""Proof-guided fixed-point control for context-efficient model calls.

The controller is deliberately provider-neutral: callers provide the model
callable, so Entroly never makes a surprise remote request. Each retry preserves
the original committed context as a byte-identical prefix and appends only exact,
fingerprint-verified omitted chunks selected under a bounded token budget.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from .context_receipts.models import text_fingerprint
from .verified_efficiency import (
    AuditArtifact,
    PreparedContext,
    VerifiedEfficiencyLayer,
    VerifiedOutput,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

_SCHEMA_VERSION = "entroly.proof-guided-fixed-point.v1"
_RECOVERED_SEPARATOR = "\n\n[ENTROLY VERIFIED RECOVERED EVIDENCE]\n"
_MAX_PLANNER_CANDIDATES = 128
_MAX_RECOVERY_TOKEN_BUDGET = 100_000
_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}")
_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "and",
        "are",
        "because",
        "been",
        "before",
        "being",
        "but",
        "can",
        "could",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "into",
        "its",
        "may",
        "not",
        "only",
        "should",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "will",
        "with",
        "would",
    }
)


class FixedPointError(RuntimeError):
    """Base class for proof-guided fixed-point failures."""


class FixedPointModelError(FixedPointError):
    """The caller-supplied model failed and the failed round was audited."""

    def __init__(self, message: str, *, round_index: int, audit: AuditArtifact) -> None:
        super().__init__(message)
        self.round_index = round_index
        self.audit = audit


class FixedPointVerificationError(FixedPointError):
    """A fixed-point output failed verification and the failure was audited."""

    def __init__(self, message: str, *, round_index: int, audit: AuditArtifact) -> None:
        super().__init__(message)
        self.round_index = round_index
        self.audit = audit


class FixedPointRecoveryError(FixedPointError):
    """Selected evidence failed exact recovery and the partial state was audited."""

    def __init__(self, message: str, *, round_index: int, audit: AuditArtifact) -> None:
        super().__init__(message)
        self.round_index = round_index
        self.audit = audit


@dataclass(frozen=True)
class EvidenceObligation:
    """One unsupported claim that needs additional grounding evidence."""

    claim_id: str
    claim_text: str
    claim_hash: str
    decision: str
    phi: float
    weight: float
    terms: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceCandidate:
    """One omitted chunk scored by expected proof gain and token cost."""

    chunk_id: str
    source_path: str
    fingerprint: str
    token_count: int
    expected_error_reduction: float
    utility_per_token: float
    supported_claim_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidencePlan:
    """A deterministic bounded-knapsack recovery decision."""

    obligations: tuple[EvidenceObligation, ...]
    candidates: tuple[EvidenceCandidate, ...]
    selected: tuple[EvidenceCandidate, ...]
    eligible_candidate_count: int
    candidate_limit_applied: bool
    token_budget: int
    selected_tokens: int
    expected_error_reduction: float


@dataclass(frozen=True)
class FixedPointModelRequest:
    """Provider-neutral request passed to the caller's explicit model callback."""

    schema_version: str
    round_index: int
    query: str
    stable_context_prefix: str
    appended_evidence: str
    full_context: str
    previous_output: str | None
    recovered_chunk_ids: tuple[str, ...]
    remaining_recovery_tokens: int
    previous_round_artifact_id: str | None


@dataclass(frozen=True)
class FixedPointRound:
    """One generate, verify, and optionally recover decision."""

    round_index: int
    request: FixedPointModelRequest
    verified_output: VerifiedOutput
    obligations: tuple[EvidenceObligation, ...]
    recovery_plan: EvidencePlan | None
    recovered_for_next_round: tuple[dict[str, Any], ...]
    decision: str
    audit: AuditArtifact


@dataclass(frozen=True)
class FixedPointResult:
    """Bounded result with the safest verified output produced by the loop."""

    schema_version: str
    status: str
    converged: bool
    final_output: VerifiedOutput
    rounds: tuple[FixedPointRound, ...]
    recovered_chunk_ids: tuple[str, ...]
    recovery_tokens_used: int
    recovery_token_budget: int
    audit: AuditArtifact


@dataclass(frozen=True)
class FixedPointSession:
    """Resumable local state for hosts that own the model transport.

    A host calls :func:`start_proof_guided_fixed_point`, sends ``next_request``
    to its already-configured model route, and supplies the returned text to
    :func:`advance_proof_guided_fixed_point`.  Entroly never acquires provider
    credentials or creates an undeclared billable request.
    """

    prepared: PreparedContext
    max_rounds: int
    recovery_token_budget: int
    max_chunks_per_round: int
    profile: str
    round_index: int = 0
    recovered_chunks: tuple[dict[str, Any], ...] = ()
    recovered_chunk_ids: tuple[str, ...] = ()
    recovery_tokens_used: int = 0
    previous_output: str | None = None
    previous_round_artifact_id: str | None = None
    rounds: tuple[FixedPointRound, ...] = ()


@dataclass(frozen=True)
class FixedPointStep:
    """One verified host-driven round and its optional continuation request."""

    round: FixedPointRound
    session: FixedPointSession
    next_request: FixedPointModelRequest | None
    result: FixedPointResult | None


ModelCall = Callable[[FixedPointModelRequest], str]


def _stem(token: str) -> str:
    word = token.lower().strip("_-")
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> frozenset[str]:
    return frozenset(
        stemmed
        for token in _TOKEN_RE.findall(text)
        if (stemmed := _stem(token)) and stemmed not in _STOP_WORDS
    )


def _bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return min(1.0, max(0.0, parsed))


def _evidence_obligations(verified: VerifiedOutput) -> tuple[EvidenceObligation, ...]:
    certificates = verified.suppression.get("certificates", [])
    obligations: list[EvidenceObligation] = []
    if not isinstance(certificates, list):
        return ()
    for index, item in enumerate(certificates):
        if not isinstance(item, Mapping):
            continue
        decision = str(item.get("decision", "")).lower()
        action = str(item.get("action", "")).lower()
        if decision == "supported" and action == "pass":
            continue
        claim_text = str(item.get("claim_text", ""))
        claim_id = str(item.get("claim_id", "") or f"claim-{index}")
        phi = _bounded_float(item.get("phi"))
        severity = 1.0 if decision == "hallucinated" else 0.65
        obligations.append(
            EvidenceObligation(
                claim_id=claim_id,
                claim_text=claim_text,
                claim_hash=text_fingerprint(claim_text),
                decision=decision or "unsupported",
                phi=phi,
                weight=round(severity + (1.0 - phi), 8),
                terms=tuple(sorted(_terms(claim_text))),
            )
        )
    return tuple(obligations)


def _candidate_is_better(
    candidate: tuple[float, tuple[str, ...]],
    incumbent: tuple[float, tuple[str, ...]] | None,
) -> bool:
    if incumbent is None:
        return True
    candidate_value, candidate_ids = candidate
    incumbent_value, incumbent_ids = incumbent
    if not math.isclose(candidate_value, incumbent_value, abs_tol=1e-12):
        return candidate_value > incumbent_value
    return candidate_ids < incumbent_ids


def _select_knapsack(
    candidates: Sequence[EvidenceCandidate],
    *,
    token_budget: int,
    max_chunks: int,
) -> tuple[EvidenceCandidate, ...]:
    by_id = {candidate.chunk_id: candidate for candidate in candidates}
    states: dict[tuple[int, int], tuple[float, tuple[str, ...]]] = {(0, 0): (0.0, ())}
    for candidate in sorted(candidates, key=lambda item: item.chunk_id):
        updated = dict(states)
        for (used_tokens, used_chunks), (value, ids) in states.items():
            new_tokens = used_tokens + candidate.token_count
            new_chunks = used_chunks + 1
            if new_tokens > token_budget or new_chunks > max_chunks:
                continue
            key = (new_tokens, new_chunks)
            proposal = (
                value + candidate.expected_error_reduction,
                (*ids, candidate.chunk_id),
            )
            if _candidate_is_better(proposal, updated.get(key)):
                updated[key] = proposal
        states = updated

    _best_key, best = min(
        states.items(),
        key=lambda item: (
            -item[1][0],
            item[0][0],
            item[0][1],
            item[1][1],
        ),
    )
    selected = [by_id[chunk_id] for chunk_id in best[1]]
    return tuple(
        sorted(selected, key=lambda item: (-item.utility_per_token, item.chunk_id))
    )


class ProofGuidedRecoveryPlanner:
    """Turn unsupported claims into an auditable value-of-evidence plan."""

    def plan(
        self,
        verified: VerifiedOutput,
        commit: Mapping[str, Any],
        *,
        excluded_chunk_ids: Iterable[str] = (),
        token_budget: int,
        max_chunks: int,
    ) -> EvidencePlan:
        if token_budget < 0:
            raise ValueError("token_budget cannot be negative")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        obligations = _evidence_obligations(verified)
        if not obligations:
            return EvidencePlan((), (), (), 0, False, token_budget, 0, 0.0)

        excluded = {str(item) for item in excluded_chunk_ids}
        receipt = commit.get("receipt", {})
        omitted = (
            receipt.get("omitted_context", []) if isinstance(receipt, Mapping) else []
        )
        recovery = commit.get("recovery_bundle", {})
        chunks = recovery.get("chunks", {}) if isinstance(recovery, Mapping) else {}
        query_terms = (
            _terms(str(receipt.get("query", "")))
            if isinstance(receipt, Mapping)
            else frozenset()
        )
        candidates: list[EvidenceCandidate] = []

        for item in omitted if isinstance(omitted, list) else []:
            if not isinstance(item, Mapping):
                continue
            chunk_id = str(item.get("chunk_id", ""))
            if not chunk_id or chunk_id in excluded:
                continue
            entry = chunks.get(chunk_id) if isinstance(chunks, Mapping) else None
            if not isinstance(entry, Mapping):
                continue
            text = str(entry.get("text", ""))
            if text_fingerprint(text) != str(entry.get("content_sha", "")):
                continue
            chunk_terms = _terms(text)
            expected_reduction = 0.0
            supported_claim_ids: list[str] = []
            for obligation in obligations:
                obligation_terms = frozenset(obligation.terms)
                overlap = obligation_terms & chunk_terms
                if not overlap:
                    continue
                coverage = len(overlap) / max(1, len(obligation_terms))
                expected_reduction += obligation.weight * coverage
                supported_claim_ids.append(obligation.claim_id)
            if not supported_claim_ids:
                continue
            query_overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
            expected_reduction += 0.15 * query_overlap
            try:
                token_count = max(1, int(item.get("token_count", 0) or 0))
            except (TypeError, ValueError, OverflowError):
                continue
            expected_reduction = round(expected_reduction, 12)
            candidates.append(
                EvidenceCandidate(
                    chunk_id=chunk_id,
                    source_path=str(item.get("source_path", "")),
                    fingerprint=str(item.get("fingerprint", "")),
                    token_count=token_count,
                    expected_error_reduction=expected_reduction,
                    utility_per_token=round(expected_reduction / token_count, 12),
                    supported_claim_ids=tuple(sorted(supported_claim_ids)),
                )
            )

        ordered_all = tuple(
            sorted(
                candidates, key=lambda item: (-item.utility_per_token, item.chunk_id)
            )
        )
        fitting = tuple(
            item for item in ordered_all if item.token_count <= token_budget
        )
        planner_candidates = fitting[:_MAX_PLANNER_CANDIDATES]
        visible_candidates = (
            planner_candidates
            if planner_candidates
            else ordered_all[:_MAX_PLANNER_CANDIDATES]
        )
        selected = _select_knapsack(
            planner_candidates,
            token_budget=token_budget,
            max_chunks=max_chunks,
        )
        return EvidencePlan(
            obligations=obligations,
            candidates=visible_candidates,
            selected=selected,
            eligible_candidate_count=len(fitting),
            candidate_limit_applied=len(fitting) > _MAX_PLANNER_CANDIDATES,
            token_budget=token_budget,
            selected_tokens=sum(item.token_count for item in selected),
            expected_error_reduction=round(
                sum(item.expected_error_reduction for item in selected), 12
            ),
        )


def _format_appended_evidence(chunks: Sequence[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    blocks = [
        f"[Verified recovered chunk {item['chunk_id']}]\n{item['text']}"
        for item in chunks
    ]
    return _RECOVERED_SEPARATOR + "\n\n".join(blocks)


def _obligation_commitments(
    obligations: Sequence[EvidenceObligation],
) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": item.claim_id,
            "claim_hash": item.claim_hash,
            "decision": item.decision,
            "phi": item.phi,
            "weight": item.weight,
        }
        for item in obligations
    ]


def _plan_commitment(plan: EvidencePlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "token_budget": plan.token_budget,
        "selected_tokens": plan.selected_tokens,
        "expected_error_reduction": plan.expected_error_reduction,
        "eligible_candidate_count": plan.eligible_candidate_count,
        "candidate_limit_applied": plan.candidate_limit_applied,
        "candidates": [
            {
                "chunk_id": item.chunk_id,
                "fingerprint": item.fingerprint,
                "token_count": item.token_count,
                "expected_error_reduction": item.expected_error_reduction,
                "utility_per_token": item.utility_per_token,
                "supported_claim_ids": item.supported_claim_ids,
            }
            for item in plan.candidates
        ],
        "selected": [
            {
                "chunk_id": item.chunk_id,
                "fingerprint": item.fingerprint,
                "token_count": item.token_count,
                "expected_error_reduction": item.expected_error_reduction,
                "utility_per_token": item.utility_per_token,
                "supported_claim_ids": item.supported_claim_ids,
            }
            for item in plan.selected
        ],
    }


def _validate_fixed_point_config(
    *,
    max_rounds: int,
    recovery_token_budget: int,
    max_chunks_per_round: int,
) -> None:
    if not isinstance(max_rounds, int) or isinstance(max_rounds, bool):
        raise ValueError("max_rounds must be an integer within [1, 8]")
    if not 1 <= max_rounds <= 8:
        raise ValueError("max_rounds must be an integer within [1, 8]")
    if (
        not isinstance(recovery_token_budget, int)
        or isinstance(recovery_token_budget, bool)
        or recovery_token_budget < 0
        or recovery_token_budget > _MAX_RECOVERY_TOKEN_BUDGET
    ):
        raise ValueError("recovery_token_budget must be an integer within [0, 100000]")
    if (
        not isinstance(max_chunks_per_round, int)
        or isinstance(max_chunks_per_round, bool)
        or not 1 <= max_chunks_per_round <= 16
    ):
        raise ValueError("max_chunks_per_round must be an integer within [1, 16]")


def _request_for_session(session: FixedPointSession) -> FixedPointModelRequest:
    appended_evidence = _format_appended_evidence(session.recovered_chunks)
    return FixedPointModelRequest(
        schema_version=_SCHEMA_VERSION,
        round_index=session.round_index,
        query=str(session.prepared.receipt.get("query", "")),
        stable_context_prefix=session.prepared.context,
        appended_evidence=appended_evidence,
        full_context=session.prepared.context + appended_evidence,
        previous_output=session.previous_output,
        recovered_chunk_ids=session.recovered_chunk_ids,
        remaining_recovery_tokens=(
            session.recovery_token_budget - session.recovery_tokens_used
        ),
        previous_round_artifact_id=session.previous_round_artifact_id,
    )


def start_proof_guided_fixed_point(
    layer: VerifiedEfficiencyLayer,
    prepared: PreparedContext,
    *,
    max_rounds: int = 3,
    recovery_token_budget: int = 1_200,
    max_chunks_per_round: int = 3,
    profile: str = "rag",
) -> tuple[FixedPointSession, FixedPointModelRequest]:
    """Start a resumable fixed-point session without making a model call."""
    _validate_fixed_point_config(
        max_rounds=max_rounds,
        recovery_token_budget=recovery_token_budget,
        max_chunks_per_round=max_chunks_per_round,
    )
    layer._load_prepared_commit(prepared)
    session = FixedPointSession(
        prepared=prepared,
        max_rounds=max_rounds,
        recovery_token_budget=recovery_token_budget,
        max_chunks_per_round=max_chunks_per_round,
        profile=str(profile),
    )
    return session, _request_for_session(session)


def _audit_model_error(
    layer: VerifiedEfficiencyLayer,
    session: FixedPointSession,
    request: FixedPointModelRequest,
    exc: Exception,
) -> FixedPointModelError:
    error_audit = layer._persist_audit(
        {
            "artifact_type": "fixed_point_model_error",
            "parent_artifact_id": session.prepared.audit.artifact_id,
            "previous_round_artifact_id": session.previous_round_artifact_id,
            "commit_id": session.prepared.commit_id,
            "round_index": session.round_index,
            "grounding_context_hash": text_fingerprint(request.full_context),
            "error_type": type(exc).__name__,
            "status": "model_error",
        }
    )
    return FixedPointModelError(
        f"model callback failed at fixed-point round {session.round_index}: {exc}",
        round_index=session.round_index,
        audit=error_audit,
    )


def advance_proof_guided_fixed_point(
    layer: VerifiedEfficiencyLayer,
    session: FixedPointSession,
    model_output: str,
) -> FixedPointStep:
    """Verify one host-provided output and return a bounded continuation.

    The supplied output must correspond to the session's current request.  All
    persisted commits and prior signed round artifacts are revalidated before
    the state can advance, making a resumed or cross-process flow fail closed.
    """
    _validate_fixed_point_config(
        max_rounds=session.max_rounds,
        recovery_token_budget=session.recovery_token_budget,
        max_chunks_per_round=session.max_chunks_per_round,
    )
    if session.round_index != len(session.rounds):
        raise FixedPointVerificationError(
            "fixed-point session round history is inconsistent",
            round_index=session.round_index,
            audit=layer._persist_audit(
                {
                    "artifact_type": "fixed_point_session_error",
                    "parent_artifact_id": session.prepared.audit.artifact_id,
                    "commit_id": session.prepared.commit_id,
                    "round_index": session.round_index,
                    "status": "invalid_round_history",
                }
            ),
        )
    if session.round_index >= session.max_rounds:
        raise ValueError("fixed-point session has already reached its round bound")
    commit = layer._load_prepared_commit(session.prepared)
    for prior_round in session.rounds:
        if not layer.verify_audit_artifact(prior_round.audit).valid:
            raise FixedPointVerificationError(
                "fixed-point session contains an invalid prior round artifact",
                round_index=session.round_index,
                audit=prior_round.audit,
            )
    request = _request_for_session(session)
    if not isinstance(model_output, str):
        error = TypeError("model_call must return a string")
        raise _audit_model_error(layer, session, request, error) from error

    planner = ProofGuidedRecoveryPlanner()
    recovered_chunks = [dict(item) for item in session.recovered_chunks]
    recovered_ids = list(session.recovered_chunk_ids)
    recovery_tokens_used = session.recovery_tokens_used
    try:
        verified = layer._verify_output_with_context(
            session.prepared,
            model_output,
            grounding_context=request.full_context,
            profile=session.profile,
            mode="strict",
            recovered_chunk_ids=session.recovered_chunk_ids,
            fixed_point_metadata={
                "schema_version": _SCHEMA_VERSION,
                "round_index": session.round_index,
                "previous_round_artifact_id": session.previous_round_artifact_id,
                "stable_prefix_hash": text_fingerprint(session.prepared.context),
                "appended_evidence_hash": text_fingerprint(request.appended_evidence),
                "recovery_tokens_used": recovery_tokens_used,
            },
        )
    except Exception as exc:
        error_audit = layer._persist_audit(
            {
                "artifact_type": "fixed_point_verification_error",
                "parent_artifact_id": session.prepared.audit.artifact_id,
                "previous_round_artifact_id": session.previous_round_artifact_id,
                "commit_id": session.prepared.commit_id,
                "round_index": session.round_index,
                "grounding_context_hash": text_fingerprint(request.full_context),
                "raw_output_hash": text_fingerprint(model_output),
                "error_type": type(exc).__name__,
                "status": "verification_error",
            }
        )
        raise FixedPointVerificationError(
            f"output verification failed at fixed-point round {session.round_index}: {exc}",
            round_index=session.round_index,
            audit=error_audit,
        ) from exc

    obligations = _evidence_obligations(verified)
    remaining_tokens = session.recovery_token_budget - recovery_tokens_used
    plan: EvidencePlan | None = None
    recovered_for_next: list[dict[str, Any]] = []

    if int(verified.suppression.get("n_claims", 0) or 0) == 0:
        status = "no_verifiable_claims"
        decision = "stop_no_verifiable_claims"
    elif not obligations:
        status = "supported"
        decision = "stop_supported"
    elif session.round_index + 1 >= session.max_rounds:
        status = "max_rounds_reached"
        decision = "stop_max_rounds"
    elif remaining_tokens <= 0:
        status = "recovery_budget_exhausted"
        decision = "stop_recovery_budget"
    else:
        plan = planner.plan(
            verified,
            commit,
            excluded_chunk_ids=recovered_ids,
            token_budget=remaining_tokens,
            max_chunks=session.max_chunks_per_round,
        )
        if not plan.candidates:
            omitted = session.prepared.receipt.get("omitted_context", [])
            unseen_omitted = [
                item
                for item in omitted
                if isinstance(item, Mapping)
                and str(item.get("chunk_id", "")) not in recovered_ids
            ]
            status = (
                "no_supporting_omitted_evidence"
                if unseen_omitted
                else "no_omitted_evidence"
            )
            decision = f"stop_{status}"
        elif not plan.selected:
            status = "recovery_budget_exhausted"
            decision = "stop_no_candidate_fits_budget"
        else:
            try:
                for candidate in plan.selected:
                    recovery = layer.recover(session.prepared, candidate.chunk_id)
                    if len(recovery.chunks) != 1:
                        raise FixedPointError(
                            "selected evidence did not recover exactly one chunk"
                        )
                    recovered = dict(recovery.chunks[0])
                    recovered["recovery_artifact_id"] = recovery.audit.artifact_id
                    recovered_for_next.append(recovered)
                    recovered_chunks.append(recovered)
                    recovered_ids.append(candidate.chunk_id)
                    recovery_tokens_used += candidate.token_count
            except Exception as exc:
                error_audit = layer._persist_audit(
                    {
                        "artifact_type": "fixed_point_recovery_error",
                        "parent_artifact_id": session.prepared.audit.artifact_id,
                        "previous_round_artifact_id": (
                            session.previous_round_artifact_id
                        ),
                        "commit_id": session.prepared.commit_id,
                        "round_index": session.round_index,
                        "output_artifact_id": verified.audit.artifact_id,
                        "selected_chunk_ids": [item.chunk_id for item in plan.selected],
                        "recovered_before_error": [
                            item["chunk_id"] for item in recovered_for_next
                        ],
                        "error_type": type(exc).__name__,
                        "status": "recovery_error",
                    }
                )
                raise FixedPointRecoveryError(
                    "exact evidence recovery failed at fixed-point round "
                    f"{session.round_index}: {exc}",
                    round_index=session.round_index,
                    audit=error_audit,
                ) from exc
            status = "continue"
            decision = "recover_and_retry"

    round_audit = layer._persist_audit(
        {
            "artifact_type": "proof_guided_fixed_point_round",
            "parent_artifact_id": session.prepared.audit.artifact_id,
            "previous_round_artifact_id": session.previous_round_artifact_id,
            "commit_id": session.prepared.commit_id,
            "round_index": session.round_index,
            "grounding_context_hash": text_fingerprint(request.full_context),
            "output_artifact_id": verified.audit.artifact_id,
            "obligations": _obligation_commitments(obligations),
            "recovery_plan": _plan_commitment(plan),
            "recovery_artifact_ids": [
                item["recovery_artifact_id"] for item in recovered_for_next
            ],
            "cumulative_recovered_chunk_ids": recovered_ids,
            "recovery_tokens_used": recovery_tokens_used,
            "recovery_token_budget": session.recovery_token_budget,
            "decision": decision,
            "status": status,
        }
    )
    round_result = FixedPointRound(
        round_index=session.round_index,
        request=request,
        verified_output=verified,
        obligations=obligations,
        recovery_plan=plan,
        recovered_for_next_round=tuple(recovered_for_next),
        decision=decision,
        audit=round_audit,
    )
    next_session = FixedPointSession(
        prepared=session.prepared,
        max_rounds=session.max_rounds,
        recovery_token_budget=session.recovery_token_budget,
        max_chunks_per_round=session.max_chunks_per_round,
        profile=session.profile,
        round_index=session.round_index + 1,
        recovered_chunks=tuple(recovered_chunks),
        recovered_chunk_ids=tuple(recovered_ids),
        recovery_tokens_used=recovery_tokens_used,
        previous_output=model_output,
        previous_round_artifact_id=round_audit.artifact_id,
        rounds=(*session.rounds, round_result),
    )
    if status == "continue":
        return FixedPointStep(
            round=round_result,
            session=next_session,
            next_request=_request_for_session(next_session),
            result=None,
        )
    result = FixedPointResult(
        schema_version=_SCHEMA_VERSION,
        status=status,
        converged=status == "supported",
        final_output=verified,
        rounds=next_session.rounds,
        recovered_chunk_ids=next_session.recovered_chunk_ids,
        recovery_tokens_used=recovery_tokens_used,
        recovery_token_budget=session.recovery_token_budget,
        audit=round_audit,
    )
    return FixedPointStep(
        round=round_result,
        session=next_session,
        next_request=None,
        result=result,
    )


def run_proof_guided_fixed_point(
    layer: VerifiedEfficiencyLayer,
    prepared: PreparedContext,
    *,
    model_call: ModelCall,
    max_rounds: int = 3,
    recovery_token_budget: int = 1_200,
    max_chunks_per_round: int = 3,
    profile: str = "rag",
) -> FixedPointResult:
    """Generate, verify, recover exact evidence, and retry under hard bounds.

    ``model_call`` is invoked exactly once per recorded round. Entroly does not
    choose a provider or perform a network operation itself.
    """
    if not callable(model_call):
        raise TypeError("model_call must be callable")
    session, request = start_proof_guided_fixed_point(
        layer,
        prepared,
        max_rounds=max_rounds,
        recovery_token_budget=recovery_token_budget,
        max_chunks_per_round=max_chunks_per_round,
        profile=profile,
    )
    while True:
        try:
            model_output = model_call(request)
        except Exception as exc:
            raise _audit_model_error(layer, session, request, exc) from exc
        step = advance_proof_guided_fixed_point(layer, session, model_output)
        if step.result is not None:
            return step.result
        assert step.next_request is not None
        session = step.session
        request = step.next_request



__all__ = [
    "EvidenceCandidate",
    "EvidenceObligation",
    "EvidencePlan",
    "FixedPointError",
    "FixedPointModelError",
    "FixedPointModelRequest",
    "FixedPointRecoveryError",
    "FixedPointResult",
    "FixedPointRound",
    "FixedPointSession",
    "FixedPointStep",
    "FixedPointVerificationError",
    "ModelCall",
    "ProofGuidedRecoveryPlanner",
    "advance_proof_guided_fixed_point",
    "run_proof_guided_fixed_point",
    "start_proof_guided_fixed_point",
]
