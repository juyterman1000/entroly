"""Local structural context assurance; downstream decision risk is unmeasured.

Commitments authenticate against the caller's declared index, not an external
truth oracle or the current filesystem. RELATE diagnostics are heuristic vetoes,
not proofs of semantic preservation. No provider is called by this module.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .context_receipts.models import byte_digest, stable_hash
from .tokens import _encoding, count_tokens

ASSURANCE_SCHEMA = "entroly.context-assurance.v1"


@dataclass(frozen=True)
class AssuranceScope:
    project: str
    session: str
    agent: str

    def __post_init__(self) -> None:
        if any(not isinstance(x, str) or not x.strip() for x in asdict(self).values()):
            raise ValueError("project, session and agent must be nonempty strings")


@dataclass(frozen=True)
class HardObligation:
    """At least `minimum` of the declared exact evidence carriers must survive.

    Two carriers can each be omitted independently while their joint omission
    fails. This contract asserts caller-declared coverage, not inferred truth.
    """

    obligation_id: str
    carriers: tuple[str, ...]
    minimum: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.obligation_id, str) or not self.obligation_id.strip():
            raise ValueError("obligation_id must be nonempty")
        if not self.carriers or any(
            not isinstance(x, str) or not x for x in self.carriers
        ):
            raise ValueError("carriers must identify evidence")
        if len(set(self.carriers)) != len(self.carriers):
            raise ValueError("carriers must be unique")
        if type(self.minimum) is not int or not 1 <= self.minimum <= len(self.carriers):
            raise ValueError("minimum must be an integer within the carrier count")
        object.__setattr__(self, "carriers", tuple(sorted(self.carriers)))


class ContextAssuranceError(RuntimeError):
    def __init__(self, certificate: Mapping[str, Any]) -> None:
        super().__init__(f"context assurance failed closed: {certificate['verdict']}")
        self.certificate = copy.deepcopy(dict(certificate))


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "certificate_id": "ca_" + stable_hash(payload)}


def _base(query: str) -> dict[str, Any]:
    return {
        "schema": ASSURANCE_SCHEMA,
        "execution": "LOCAL_DETERMINISTIC",
        "query_commitment": byte_digest(query),
        "decision_risk": {
            "status": "unmeasured",
            "upper_risk_bound": None,
            "calibration_id": None,
            "decision_divergence_regret": None,
            "task_context_regret": None,
        },
        "scope": {"authority": "declared_evidence", "model": None},
    }


def _budget(texts: Sequence[str], budget: int, recorded: int) -> dict[str, Any]:
    rendered = "\n\n".join(texts)
    measured = count_tokens(rendered)
    exact = _encoding() is not None
    return {
        "status": "passed"
        if exact and measured <= budget
        else "failed"
        if measured > budget
        else "unavailable",
        "tokenizer": "o200k_base" if exact else "character_heuristic",
        "rendering": "double_newline_join_of_selected_text",
        "selected_tokens": measured,
        "recorded_unit_tokens": recorded,
        "budget": budget,
        "scope": "selected_text_only; transport wrappers require separate accounting",
    }


def audit_receipt(
    index: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    mandatory_ids: Sequence[str] = (),
    obligations: Sequence[HardObligation] = (),
    joint_limit: int = 64,
) -> dict[str, Any]:
    """Audit the complete index, including omissions outside the preview list.

    Hard obligations are checked against the final retained set on every call.
    No union of cached individual witnesses can discharge a joint obligation.
    The bounded RELATE check is explicitly unavailable above `joint_limit`.
    """
    if type(joint_limit) is not int or joint_limit < 0:
        raise ValueError("joint_limit must be a nonnegative integer")
    if len({o.obligation_id for o in obligations}) != len(obligations):
        raise ValueError("obligation ids must be unique")
    chunks = list(index.get("chunks", []))
    selected = list(receipt.get("selected_context", []))
    by_id = {c["chunk_id"]: c for c in chunks}
    ids = [c["chunk_id"] for c in selected]
    retained = set(ids)
    omitted = sorted(set(by_id) - retained)
    unique = len(by_id) == len(chunks) and len(retained) == len(ids)
    known = retained <= set(by_id)
    source_digests = {
        d["source_path"]: d.get("source_sha256") for d in index.get("documents", [])
    }
    identity_keys = (
        "source_path",
        "byte_start",
        "byte_end",
        "fragment_sha256",
        "source_sha256",
        "text",
    )
    bytes_valid = all(
        c.get("fragment_sha256") == byte_digest(c["text"])
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(c.get("source_sha256", "")))
        and c["source_sha256"] == source_digests.get(c.get("source_path"))
        and type(c.get("byte_start")) is int
        and type(c.get("byte_end")) is int
        and c["byte_start"] >= 0
        and c["byte_end"] - c["byte_start"] == len(c["text"].encode("utf-8"))
        for c in chunks
    )
    selected_valid = known and all(
        all(item.get(k) == by_id[item["chunk_id"]].get(k) for k in identity_keys)
        for item in selected
    )
    fingerprints = receipt.get("source_fingerprints", {})
    census_valid = fingerprints.get("fragment_bytes") == {
        c["chunk_id"]: c.get("fragment_sha256") for c in chunks
    }
    commitments = {
        cid: {k: by_id[cid].get(k) for k in identity_keys if k != "text"}
        for cid in sorted(by_id)
    }
    mandatory = set(mandatory_ids)
    coverage = [
        {
            "obligation_id": o.obligation_id,
            "status": "passed"
            if set(o.carriers) <= set(by_id)
            and len(retained & set(o.carriers)) >= o.minimum
            else "failed",
            "retained_carriers": len(retained & set(o.carriers)),
            "minimum": o.minimum,
        }
        for o in sorted(obligations, key=lambda o: o.obligation_id)
    ]
    links = receipt.get("dependency_links")
    dependency_status = "unavailable" if not isinstance(links, list) else "passed"
    for link in links or []:
        if link["source_chunk_id"] in retained and (
            not link.get("resolved") or link.get("target_chunk_id") not in retained
        ):
            dependency_status = "failed"
    checks = {
        "evidence_available": "passed" if chunks and selected else "unavailable",
        "partition": "passed" if unique and known else "failed",
        "selected_bytes_match_index": "passed" if selected_valid else "failed",
        "exact_byte_commitments": "passed"
        if bytes_valid and census_valid
        else "failed",
        "mandatory_evidence": "passed"
        if mandatory <= retained and mandatory <= set(by_id)
        else "failed",
        "declared_joint_obligations": "passed"
        if all(o["status"] == "passed" for o in coverage)
        else "failed",
        "declared_dependency_closure": dependency_status,
    }
    joint: dict[str, Any] = {
        "status": "unavailable",
        "hard_signal_count": None,
        "soft_signal_count": None,
        "scope": "RELATE heuristics",
    }
    if (
        unique
        and known
        and bytes_valid
        and len(chunks) <= joint_limit
        and sum(len(c["text"].encode("utf-8")) for c in chunks) <= 256_000
    ):
        from .relate.constraints import compile_query_contract
        from .relate.joint_omission import verify_joint_omission_safety
        from .relate.types import EvidenceCandidate

        candidates = [
            EvidenceCandidate(
                cid,
                by_id[cid]["text"],
                by_id[cid]["token_count"],
                recoverable_ref=by_id[cid]["fragment_sha256"],
            )
            for cid in sorted(by_id)
        ]
        witness = verify_joint_omission_safety(
            [c for c in candidates if c.candidate_id in omitted],
            candidates,
            compile_query_contract(str(receipt.get("query", ""))),
        )
        # Lexical/topic coverage estimates remain separate from hard vetoes.
        soft = (
            "obligation_support_may_depend_on_omitted",
            "joint_coverage_insufficient",
        )
        joint.update(
            status="evaluated",
            hard_signal_count=sum(not r.startswith(soft) for r in witness.reasons),
            soft_signal_count=sum(r.startswith(soft) for r in witness.reasons),
        )
    if not omitted:
        joint.update(status="evaluated", hard_signal_count=0, soft_signal_count=0)
    budget = _budget(
        [item["text"] for item in selected],
        receipt["token_budget"],
        sum(item["token_count"] for item in selected),
    )
    statuses = [*checks.values(), budget["status"]]
    verdict = (
        "rejected"
        if "failed" in statuses
        else "expansion_required"
        if joint["hard_signal_count"]
        else "uncertain"
        if "unavailable" in statuses
        or joint["status"] == "unavailable"
        or joint["soft_signal_count"]
        else "structurally_valid_risk_unmeasured"
    )
    objective = (
        receipt.get("risk_summary", {})
        .get("selection_certificate", {})
        .get("objective", {})
    )
    payload = {
        **_base(str(receipt.get("query", ""))),
        "input_commitment": stable_hash(commitments),
        "selected_commitment": stable_hash([commitments.get(cid) for cid in ids]),
        "omission_commitment": stable_hash({cid: commitments[cid] for cid in omitted}),
        "policy_commitment": stable_hash(
            {
                "mandatory": sorted(mandatory),
                "obligations": [
                    asdict(o)
                    for o in sorted(obligations, key=lambda o: o.obligation_id)
                ],
                "joint_limit": joint_limit,
            }
        ),
        "structural_assurance": {
            "checks": checks,
            "budget": budget,
            "joint_obligations": coverage,
        },
        "joint_omission_diagnostics": joint,
        "optimizer_assurance": {
            "scope": "internal_selection_objective",
            "internal_selection_regret_upper_bound": objective.get(
                "certified_regret_upper_bound"
            ),
        },
        "recovery": {
            "status": "available_in_declared_index" if bytes_valid else "unavailable",
            "omitted_count": len(omitted),
            "listed_omitted_count": len(receipt.get("omitted_context", [])),
            "persistence": "not_checked",
            "authorization": "not_checked",
        },
        "verdict": verdict,
    }
    return _seal(payload)


def attach_receipt_assurance(
    index: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Preserve the receipt v1 shape and hash its versioned nested extension."""
    enriched = copy.deepcopy(dict(receipt))
    enriched.setdefault("risk_summary", {})["context_assurance"] = audit_receipt(
        index, receipt
    )
    payload = {
        k: v
        for k, v in enriched.items()
        if k not in {"receipt_id", "reproducibility_hash"}
    }
    digest = stable_hash(payload)
    return {
        **enriched,
        "receipt_id": "cr_" + digest[:12],
        "reproducibility_hash": digest,
    }


def audit_engine_selection(
    selected: Sequence[Mapping[str, Any]], *, query: str, token_budget: int
) -> dict[str, Any]:
    """Same risk boundary for QCCR/native/fallback results without a span census.

    Engine excerpts and beliefs cannot earn source-evidence assurance from their
    retrieval score. Transport integrations must audit their final text separately.
    """
    texts = [str(item.get("content", "")) for item in selected]
    budget = _budget(
        texts, token_budget, sum(int(item.get("token_count") or 0) for item in selected)
    )
    return _seal(
        {
            **_base(query),
            "selected_commitment": stable_hash(
                [
                    {"source": item.get("source"), "bytes": byte_digest(text)}
                    for item, text in zip(selected, texts)
                ]
            ),
            "structural_assurance": {
                "budget": budget,
                "checks": {
                    "source_span_census": "unavailable",
                    "declared_joint_obligations": "unavailable",
                    "declared_dependency_closure": "unavailable",
                },
            },
            "optimizer_assurance": {"scope": "internal_selection_objective"},
            "recovery": {"status": "not_checked"},
            "verdict": "rejected" if budget["status"] == "failed" else "uncertain",
        }
    )


def require_assurance(
    certificate: Mapping[str, Any], *, decision_risk: bool = False
) -> None:
    """Fail closed; this version has no calibrated downstream risk authority."""
    if (
        decision_risk
        or certificate.get("verdict") != "structurally_valid_risk_unmeasured"
    ):
        raise ContextAssuranceError(certificate)


def bind_assurance_scope(
    certificate: Mapping[str, Any], scope: AssuranceScope
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(certificate))
    payload.pop("certificate_id", None)
    payload["scope"].update(asdict(scope))
    return _seal(payload)


def recover_context_evidence(
    commit: Mapping[str, Any], chunk_id: str, *, scope: AssuranceScope
) -> dict[str, Any]:
    """Recover any omitted committed chunk, including beyond the preview cap.

    The caller must trust the supplied commit (or verify its existing signed
    audit). Declared scope binding is not principal authentication. No source
    path is opened: recovery uses only the supplied committed local snapshot.
    """
    from .context_commit import verify_context_commit
    from .context_receipts.recover import recover_omitted

    if commit.get("assurance_scope") != asdict(scope):
        raise ValueError("context recovery scope mismatch")
    if not verify_context_commit(commit).valid:
        raise ValueError("context commit integrity verification failed")
    receipt = commit["receipt"]
    if chunk_id in {item["chunk_id"] for item in receipt["selected_context"]}:
        raise ValueError("requested evidence was not omitted")
    entry = commit["recovery_bundle"]["chunks"].get(chunk_id)
    if not entry:
        raise ValueError("requested evidence is not in the committed snapshot")
    # This descriptor is authenticated by the whole commit, rather than by the
    # bounded receipt preview. Reuse the exact-byte recovery implementation.
    descriptor = {**entry, "chunk_id": chunk_id, "omission_reason": "not_selected"}
    expanded = {**receipt, "omitted_context": [descriptor]}
    recovered = recover_omitted(expanded, chunk_id, bundle=commit["recovery_bundle"])
    if len(recovered) != 1 or recovered[0]["verification_level"] != "exact_utf8_bytes":
        raise ValueError("exact committed evidence recovery failed")
    return recovered[0]
