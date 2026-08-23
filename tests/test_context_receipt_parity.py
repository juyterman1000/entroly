"""One receipt contract, identical in every runtime.

The audit's sharpest architectural finding was that "any agent can continue" is
not true cross-runtime. Work Graph continuity reaches Python and npm alike, but
receipts existed only in Python — so a Node caller could join a workstream and
still not prove what evidence it received. Worse, the Python receipt's
``reproducibility_hash`` is computed over the whole enriched record *including
selected text*, so it could never be reproduced elsewhere. That is not a defect
in the host receipt; it is proof the host receipt cannot be the shared contract.

``ContextReceiptEnvelope`` is the shared contract: identity, commitments,
references, budget, policy — and nothing a host happens to render. These tests
hold the Python half of it.

The golden vector is the anchor. Both bindings call the same engine function, so
"byte-equal across runtimes" reduces to one value each runtime asserts
independently. Drift in either breaks its own test rather than quietly
diverging from the other. The same constant is pinned in
``engine_contracts::tests::GOLDEN_RECEIPT_COMMITMENT`` and must be asserted by
the Node suite for the parity claim to be complete.
"""

from __future__ import annotations

import hashlib
import json

import pytest

entroly_core = pytest.importorskip("entroly_core")

pytestmark = pytest.mark.skipif(
    not hasattr(entroly_core, "context_receipt_build_json"),
    reason="native entroly_core does not expose the Context Receipt contract",
)

# Must equal engine_contracts::tests::GOLDEN_RECEIPT_COMMITMENT.
GOLDEN_COMMITMENT = "672457349ba403bc885ea2104162fe212fb8e9bddf51a884df27d33c37a77c84"
GOLDEN_RECEIPT_ID = "cr_672457349ba403bc"


def _golden() -> str:
    return entroly_core.context_receipt_build_json(
        "repo:golden",
        "sha256:repo-golden",
        "sha256:graph-golden",
        "workstream:golden",
        "sha256:source-golden",
        ["ref:alpha", "ref:beta"],
        ["ref:omitted"],
        ["ref:pinned"],
        ["ref:recoverable"],
        ["handle:alpha"],
        ["evidence:alpha"],
        4096,
        "knapsack/v1",
        "exec:golden",
        1_700_000_000_000,
    )


def test_python_reproduces_the_golden_commitment() -> None:
    """The parity anchor, asserted from the Python side."""
    receipt = json.loads(_golden())

    assert receipt["receipt_commitment"] == GOLDEN_COMMITMENT
    assert receipt["receipt_id"] == GOLDEN_RECEIPT_ID
    assert receipt["schema_version"] == entroly_core.context_receipt_schema_version()


def test_the_commitment_is_stable_across_calls() -> None:
    assert json.loads(_golden())["receipt_commitment"] == json.loads(_golden())[
        "receipt_commitment"
    ]


def test_reference_order_and_duplicates_do_not_change_the_commitment() -> None:
    """Canonicalisation is what lets two runtimes enumerate differently.

    The commitment attests to *which* evidence was involved. Ranking order is
    presentation and stays in the host receipt.
    """
    shuffled = entroly_core.context_receipt_build_json(
        "repo:golden",
        "sha256:repo-golden",
        "sha256:graph-golden",
        "workstream:golden",
        "sha256:source-golden",
        ["ref:beta", "ref:alpha", "ref:beta"],
        ["ref:omitted"],
        ["ref:pinned"],
        ["ref:recoverable"],
        ["handle:alpha"],
        ["evidence:alpha"],
        4096,
        "knapsack/v1",
        "exec:golden",
        1_700_000_000_000,
    )

    assert json.loads(shuffled)["receipt_commitment"] == GOLDEN_COMMITMENT
    assert json.loads(shuffled)["selected_refs"] == ["ref:alpha", "ref:beta"]


def test_a_changed_field_changes_the_commitment() -> None:
    """The other half of determinism: equivalence must not be too generous."""
    different = entroly_core.context_receipt_build_json(
        "repo:golden",
        "sha256:repo-golden",
        "sha256:graph-golden",
        "workstream:golden",
        "sha256:source-golden",
        ["ref:alpha", "ref:beta"],
        ["ref:omitted"],
        ["ref:pinned"],
        ["ref:recoverable"],
        ["handle:alpha"],
        ["evidence:alpha"],
        8192,  # budget differs
        "knapsack/v1",
        "exec:golden",
        1_700_000_000_000,
    )

    assert json.loads(different)["receipt_commitment"] != GOLDEN_COMMITMENT


def test_verification_round_trips() -> None:
    receipt = _golden()

    assert entroly_core.context_receipt_verify_json(receipt) == receipt
    assert entroly_core.context_receipt_commitment(receipt) == GOLDEN_COMMITMENT


def test_a_tampered_receipt_is_refused_not_returned() -> None:
    """Fail closed: verification must raise, not hand back an unverified value."""
    tampered = _golden().replace('"budget_tokens":4096', '"budget_tokens":999999')

    with pytest.raises(Exception) as exc_info:
        entroly_core.context_receipt_verify_json(tampered)

    assert "receipt_commitment" in str(exc_info.value)


def test_an_unknown_schema_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newer receipt cannot be interpreted under today's rules."""
    future = _golden().replace('"schema_version":1', '"schema_version":99')

    with pytest.raises(Exception) as exc_info:
        entroly_core.context_receipt_verify_json(future)

    assert "schema_version" in str(exc_info.value)


def test_the_graph_reference_carries_no_receipt_body() -> None:
    """Section 8's rule, enforced at the boundary rather than trusted to callers."""
    graph_ref = json.loads(
        entroly_core.context_receipt_graph_ref_json(
            _golden(), "workstream:golden", "agent:claude", "session:1"
        )
    )

    assert graph_ref["receipt_id"] == GOLDEN_RECEIPT_ID
    assert graph_ref["reproducibility_hash"] == GOLDEN_COMMITMENT
    for body_field in ("selected_refs", "omitted_refs", "selection_policy", "budget_tokens"):
        assert body_field not in graph_ref


def test_invalid_input_is_rejected_rather_than_committed() -> None:
    """An empty repository id has no valid commitment, so there must be none."""
    with pytest.raises(Exception) as exc_info:
        entroly_core.context_receipt_build_json(
            "", "sha256:repo", "sha256:graph", "workstream:1"
        )

    assert "repository_id" in str(exc_info.value)


# ── Recovery handles ────────────────────────────────────────────────────────
#
# Section 9: "never call destructive omission recoverable", and "a recovery
# operation must verify expected commitment before silently returning material".
# Both are enforced in the engine, so both must behave identically here and in
# Node. The fixture hashes itself rather than carrying a literal digest, so the
# two runtimes agree by computing rather than by copying.

GOLDEN_HANDLE_ID = "rh_61e976bc425ad0de"
FIXTURE_BODY = b"recoverable bytes"


def _fixture_commitment() -> str:
    return hashlib.sha256(FIXTURE_BODY).hexdigest()


def _recoverable_handle() -> str:
    return entroly_core.recovery_handle_build_json(
        "repo:demo",
        "cr_672457349ba403bc",
        "omitted_but_recoverable",
        "src/auth.py",
        "sha256:source",
        _fixture_commitment(),
        0,
        17,
        "commit:abc123",
        "",
        1_700_000_000_000,
    )


def test_python_reproduces_the_golden_handle_id() -> None:
    assert json.loads(_recoverable_handle())["handle_id"] == GOLDEN_HANDLE_ID


def test_a_recovery_promise_without_a_commitment_is_refused() -> None:
    """The refusal is the contract, not a validation nicety."""
    with pytest.raises(Exception) as exc_info:
        entroly_core.recovery_handle_build_json(
            "repo:demo", "cr_x", "omitted_but_recoverable", "src/auth.py", "sha256:source"
        )

    assert "recover" in str(exc_info.value).lower()


def test_a_recovery_promise_without_a_way_back_is_refused() -> None:
    """A commitment says what the bytes were, not where to find them again."""
    with pytest.raises(Exception) as exc_info:
        entroly_core.recovery_handle_build_json(
            "repo:demo", "cr_x", "omitted_but_recoverable",
            "", "", _fixture_commitment(),
        )

    assert "recover" in str(exc_info.value).lower()


def test_destructive_omission_is_expressible_without_evidence() -> None:
    """If the honest state were hard to express, callers would overclaim."""
    handle = entroly_core.recovery_handle_build_json(
        "repo:demo", "cr_x", "omitted_and_unavailable"
    )

    assert json.loads(handle)["disposition"] == "omitted_and_unavailable"


def test_recovered_bytes_are_verified_against_the_commitment() -> None:
    handle = _recoverable_handle()

    assert entroly_core.recovery_handle_verify_bytes(handle, FIXTURE_BODY) == "verified"
    assert (
        entroly_core.recovery_handle_verify_bytes(handle, b"different bytes")
        == "commitment_mismatch"
    )


def test_an_unavailable_handle_has_nothing_to_verify() -> None:
    handle = entroly_core.recovery_handle_build_json(
        "repo:demo", "cr_x", "omitted_and_unavailable"
    )

    assert entroly_core.recovery_handle_verify_bytes(handle, b"anything") == "not_recoverable"


def test_an_edited_handle_fails_closed() -> None:
    edited = _recoverable_handle().replace("src/auth.py", "src/other.py")

    with pytest.raises(Exception) as exc_info:
        entroly_core.recovery_handle_verify_json(edited)

    assert "handle_id" in str(exc_info.value)


def test_an_unknown_disposition_is_rejected() -> None:
    with pytest.raises(Exception) as exc_info:
        entroly_core.recovery_handle_build_json("repo:demo", "cr_x", "probably_fine")

    assert "disposition" in str(exc_info.value)


# ── Provenance-bearing memory ────────────────────────────────────────────────

GOLDEN_MEMORY_ID = "mem_a3b337c53411d1a5"


def _golden_memory(*, trust: str = "observed", evidence: list[str] | None = None) -> str:
    return entroly_core.memory_record_build_json(
        "repo:demo",
        "vault/beliefs/auth.md",
        trust,
        task_id="task:auth",
        workstream_id="workstream:1",
        source_agent="agent:claude",
        source_session="session:1",
        source_execution="exec:1",
        content_commitment="sha256:content",
        evidence_ids=["evidence:1"] if evidence is None else evidence,
        created_at_ms=1_700_000_000_000,
        observed_at_ms=1_700_000_000_000,
    )


def test_python_reproduces_the_golden_memory_id() -> None:
    memory = json.loads(_golden_memory())

    assert memory["memory_id"] == GOLDEN_MEMORY_ID
    assert memory["schema_version"] == entroly_core.memory_record_schema_version()
    assert entroly_core.memory_record_verify_json(_golden_memory()) == _golden_memory()


def test_memory_admissibility_matches_the_engine_contract() -> None:
    assert (
        entroly_core.memory_record_admissibility(
            _golden_memory(), 1_700_000_100_000
        )
        == "admissible"
    )
    unsupported = _golden_memory(trust="verified", evidence=[])
    assert (
        entroly_core.memory_record_admissibility(
            unsupported, 1_700_000_100_000
        )
        == "unsupported"
    )
    assert entroly_core.memory_record_admissibility(_golden_memory(), -1) == "unsupported"


def test_memory_requires_producer_provenance() -> None:
    with pytest.raises(Exception) as exc_info:
        entroly_core.memory_record_build_json(
            "repo:demo",
            "vault/beliefs/auth.md",
            "observed",
            source_session="session:1",
            source_execution="exec:1",
            content_commitment="sha256:content",
            evidence_ids=["evidence:1"],
        )

    assert "source_agent" in str(exc_info.value)


def test_an_edited_memory_fails_closed() -> None:
    edited = _golden_memory().replace("agent:claude", "agent:someone")

    with pytest.raises(Exception) as exc_info:
        entroly_core.memory_record_verify_json(edited)

    assert "record_commitment" in str(exc_info.value)


# ── Routing, execution, freshness and continuation ────────────────────────

GOLDEN_ROUTING_ID = "route_66d4c04a18b4e70f"
GOLDEN_OUTCOME_ID = "outcome_a130681ddd63dc84"
GOLDEN_VERIFICATION_ID = "verify_4e1487e3d6e73b36"
GOLDEN_CONTINUATION_ID = "continuation_53eba6ee3a52be48"


def _route() -> str:
    return entroly_core.routing_decision_build_json(
        "repo:demo",
        "task:auth",
        "workstream:1",
        "openai",
        "gpt-5",
        "responses-api",
        8192,
        "policy:v1",
        ["capability_match", "lowest_verified_cost"],
        ["sha256:features"],
        [],
        "cr_672457349ba403bc",
        ["evidence:benchmark"],
        1_700_000_000_000,
    )


def _outcome() -> str:
    route = json.loads(_route())
    return entroly_core.model_execution_outcome_build_json(
        route["routing_id"],
        route["repository_id"],
        route["task_id"],
        route["workstream_id"],
        route["provider"],
        route["model"],
        route["runtime"],
        route["receipt_id"],
        "sha256:request",
        "sha256:response",
        "succeeded",
        "passed",
        420,
        1200,
        240,
        17_500,
        "",
        ["evidence:test"],
        1_700_000_000_500,
    )


def _verification() -> str:
    outcome = json.loads(_outcome())
    return entroly_core.verification_record_build_json(
        "repo:demo",
        outcome["outcome_id"],
        outcome["outcome_commitment"],
        "sha256:head-a",
        "passed",
        ["evidence:test"],
        ["sha256:source-a", "sha256:config-a"],
        1_700_000_000_600,
        1_700_000_001_000,
    )


def _continuation() -> str:
    route = json.loads(_route())
    outcome = json.loads(_outcome())
    verification = json.loads(_verification())
    return entroly_core.work_continuation_proof_build_json(
        "repo:demo",
        7,
        "sha256:graph",
        "workstream:1",
        "agent:claude",
        "agent:codex",
        "sha256:handoff",
        ["sha256:receipt"],
        [route["decision_commitment"]],
        [outcome["outcome_commitment"]],
        [verification["record_commitment"]],
        ["sha256:memory"],
        ["run Linux CI"],
        ["rh_61e976bc425ad0de"],
        1_700_000_000_700,
    )


def test_routing_execution_and_continuation_contracts_round_trip() -> None:
    assert json.loads(_route())["routing_id"] == GOLDEN_ROUTING_ID
    assert json.loads(_outcome())["outcome_id"] == GOLDEN_OUTCOME_ID
    assert json.loads(_verification())["verification_id"] == GOLDEN_VERIFICATION_ID
    assert json.loads(_continuation())["proof_id"] == GOLDEN_CONTINUATION_ID
    assert entroly_core.routing_decision_verify_json(_route()) == _route()
    assert (
        entroly_core.model_execution_outcome_verify_json(_outcome()) == _outcome()
    )
    assert (
        entroly_core.verification_record_verify_json(_verification())
        == _verification()
    )
    assert (
        entroly_core.work_continuation_proof_verify_json(_continuation())
        == _continuation()
    )


def test_temporal_verification_fails_closed_on_head_or_dependency_change() -> None:
    assert (
        entroly_core.verification_record_freshness(
            _verification(), "sha256:head-a", 1_700_000_000_700
        )
        == "current"
    )
    assert (
        entroly_core.verification_record_freshness(
            _verification(), "sha256:head-b", 1_700_000_000_700
        )
        == "stale"
    )
    assert (
        entroly_core.verification_record_freshness(
            _verification(),
            "sha256:head-a",
            1_700_000_000_700,
            ["sha256:config-a"],
        )
        == "invalidated"
    )


def test_continuation_proof_is_graph_bound_and_tamper_evident() -> None:
    assert (
        entroly_core.work_continuation_proof_state(
            _continuation(), "repo:demo", 7, "sha256:graph"
        )
        == "valid"
    )
    assert (
        entroly_core.work_continuation_proof_state(
            _continuation(), "repo:demo", 8, "sha256:new-graph"
        )
        == "stale"
    )
    edited = _continuation().replace("agent:codex", "agent:other")
    with pytest.raises(Exception, match="proof_commitment"):
        entroly_core.work_continuation_proof_verify_json(edited)
