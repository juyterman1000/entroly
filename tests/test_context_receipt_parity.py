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
