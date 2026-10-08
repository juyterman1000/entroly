"""Counterexamples and bounded oracles for declared structural assurance."""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import asdict

import pytest

from entroly import (
    AssuranceScope,
    ContextAssuranceError,
    HardObligation,
    assure_context,
)
from entroly.context_assurance import (
    audit_engine_selection,
    audit_receipt,
    recover_context_evidence,
    require_assurance,
)
from entroly.context_commit import create_context_commit, verify_context_commit
from entroly.context_receipts import (
    ingest_documents,
    run_receipt_pipeline,
    select_from_index,
)
from entroly.context_receipts.models import stable_hash
from entroly.context_receipts.recover import recovery_path

SCOPE = AssuranceScope("repo:test", "session:test", "agent:test")
DOCS = [
    ("a.txt", "alpha apple evidence"),
    ("b.txt", "beta berry evidence"),
    ("c.txt", "gamma grape evidence"),
]


def selection_fixture():
    index = ingest_documents(DOCS, prefer_rust=False)
    receipt = select_from_index(
        index, query="evidence", token_budget=1000, prefer_rust=False
    )
    return index, receipt, [c["chunk_id"] for c in index["chunks"]]


def retain(index, receipt, ids):
    result = copy.deepcopy(receipt)
    result["selected_context"] = [c for c in index["chunks"] if c["chunk_id"] in ids]
    return result


def test_individual_omissions_do_not_compose():
    index, receipt, (a, b, c) = selection_fixture()
    obligation = HardObligation("critical_value", (a, b))
    for kept in ({a, c}, {b, c}):
        cert = audit_receipt(
            index, retain(index, receipt, kept), obligations=[obligation]
        )
        assert (
            cert["structural_assurance"]["checks"]["declared_joint_obligations"]
            == "passed"
        )
    joint = audit_receipt(index, retain(index, receipt, {c}), obligations=[obligation])
    assert (
        joint["structural_assurance"]["checks"]["declared_joint_obligations"]
        == "failed"
    )
    assert joint["verdict"] == "rejected"


def test_joint_obligation_and_dependency_closure_bounded_oracle():
    index, receipt, ids = selection_fixture()
    for bits in itertools.product((False, True), repeat=len(ids)):
        kept = {cid for cid, keep in zip(ids, bits) if keep}
        candidate = retain(index, receipt, kept)
        candidate["dependency_links"] = [
            {"source_chunk_id": ids[0], "target_chunk_id": ids[1], "resolved": True},
            {"source_chunk_id": ids[1], "target_chunk_id": ids[2], "resolved": True},
        ]
        for minimum in (1, 2, 3):
            obligation = HardObligation("support", tuple(ids), minimum)
            cert = audit_receipt(index, candidate, obligations=[obligation])
            checks = cert["structural_assurance"]["checks"]
            assert (checks["declared_joint_obligations"] == "passed") == (
                len(kept) >= minimum
            )
            # An independent oracle: every reachable descendant must survive.
            valid = all(
                set(ids[pos:]) <= kept for pos, cid in enumerate(ids) if cid in kept
            )
            assert (checks["declared_dependency_closure"] == "passed") == valid


def test_certificate_is_deterministic_and_unordered_census_permutation_invariant():
    index, receipt, ids = selection_fixture()
    first = audit_receipt(index, receipt, mandatory_ids=ids)
    assert first == audit_receipt(index, receipt, mandatory_ids=list(reversed(ids)))
    permuted = copy.deepcopy(index)
    permuted["chunks"].reverse()
    assert first == audit_receipt(permuted, receipt, mandatory_ids=ids)
    assert first["decision_risk"]["upper_risk_bound"] is None
    assert first["verdict"] == "structurally_valid_risk_unmeasured"


@pytest.mark.parametrize(
    "mutation", ["text", "byte_end", "source_sha256", "duplicate", "unknown"]
)
def test_changed_selection_cannot_keep_a_structural_verdict(mutation):
    index, receipt, _ = selection_fixture()
    changed = copy.deepcopy(receipt)
    item = changed["selected_context"][0]
    if mutation == "text":
        item["text"] += " corrupt"
    elif mutation == "byte_end":
        item["byte_end"] += 1
    elif mutation == "source_sha256":
        item["source_sha256"] = "sha256:" + "0" * 64
    elif mutation == "unknown":
        item["chunk_id"] = "missing"
    else:
        changed["selected_context"].append(copy.deepcopy(item))
    cert = audit_receipt(index, changed)
    assert cert["verdict"] == "rejected"
    with pytest.raises(ContextAssuranceError):
        require_assurance(cert)


def test_source_mutation_invalidates_old_receipt():
    index, receipt, _ = selection_fixture()
    changed_index = ingest_documents(
        [(p, t + " changed") for p, t in DOCS], prefer_rust=False
    )
    assert audit_receipt(changed_index, receipt)["verdict"] == "rejected"


def test_missing_metadata_never_strengthens_the_verdict():
    index, receipt, _ = selection_fixture()
    for field in ("dependency_links", "source_fingerprints"):
        degraded = copy.deepcopy(receipt)
        degraded.pop(field)
        assert (
            audit_receipt(index, degraded)["verdict"]
            != "structurally_valid_risk_unmeasured"
        )
    degraded_index = copy.deepcopy(index)
    degraded_index["chunks"][0].pop("fragment_sha256")
    assert audit_receipt(degraded_index, receipt)["verdict"] == "rejected"


def test_budget_uses_rendered_text_and_declares_tokenizer_scope():
    index, receipt, _ = selection_fixture()
    receipt["token_budget"] = 1
    for item in receipt["selected_context"]:
        item["token_count"] = 0  # Metadata cannot conceal a rendered overrun.
    cert = audit_receipt(index, receipt)
    assert cert["structural_assurance"]["budget"]["status"] == "failed"
    assert cert["verdict"] == "rejected"


def test_unavailable_tokenizer_cannot_earn_hard_budget_assurance(monkeypatch):
    import entroly.context_assurance as assurance

    index, receipt, _ = selection_fixture()
    monkeypatch.setattr(assurance, "_encoding", lambda: None)
    cert = audit_receipt(index, receipt)
    assert cert["structural_assurance"]["budget"]["status"] == "unavailable"
    assert cert["verdict"] == "uncertain"


def test_joint_work_limit_is_visible_and_not_a_pass():
    index, receipt, ids = selection_fixture()
    cert = audit_receipt(index, retain(index, receipt, {ids[0]}), joint_limit=0)
    assert cert["joint_omission_diagnostics"]["status"] == "unavailable"
    assert cert["verdict"] == "uncertain"


def test_hard_relate_constraint_loss_cannot_be_overridden_by_summary_coverage():
    docs = [
        ("a.txt", "Service must never erase production records."),
        ("b.txt", "Service runtime logging monitoring metrics overview."),
        ("c.txt", "Service runtime tracing latency monitoring details."),
    ]
    index = ingest_documents(docs, prefer_rust=False)
    receipt = select_from_index(
        index, query="summarize service runtime", token_budget=1000, prefer_rust=False
    )
    kept = {c["chunk_id"] for c in index["chunks"] if c["source_path"] != "a.txt"}
    cert = audit_receipt(index, retain(index, receipt, kept))
    assert cert["joint_omission_diagnostics"]["hard_signal_count"] > 0
    assert cert["verdict"] == "expansion_required"


@pytest.mark.parametrize("prefer_rust", [False, True])
def test_public_pipeline_binds_nested_certificate_to_receipt_hash(prefer_rust):
    receipt = run_receipt_pipeline(
        DOCS, query="evidence", token_budget=1000, prefer_rust=prefer_rust
    )
    assert (
        receipt["risk_summary"]["context_assurance"]["schema"]
        == "entroly.context-assurance.v1"
    )
    payload = {
        k: v
        for k, v in receipt.items()
        if k not in {"receipt_id", "reproducibility_hash"}
    }
    assert receipt["reproducibility_hash"] == stable_hash(payload)


def test_scoped_sdk_contract_and_risk_requirement():
    result = assure_context(
        DOCS, query="evidence", budget=1000, scope=SCOPE, prefer_rust=False
    )
    assert result["certificate"]["scope"]["session"] == SCOPE.session
    assert verify_context_commit(result["context_commit"]).valid
    with pytest.raises(ContextAssuranceError):
        assure_context(
            DOCS,
            query="evidence",
            budget=1000,
            scope=SCOPE,
            require_decision_risk=True,
            prefer_rust=False,
        )
    with pytest.raises(ContextAssuranceError):
        assure_context(
            DOCS,
            query="evidence",
            budget=1000,
            scope=SCOPE,
            mandatory_ids=["not_in_corpus"],
            prefer_rust=False,
        )


def test_every_omission_is_committed_and_recoverable_after_restart(tmp_path):
    docs = [(f"item{i}.txt", f"evidence value {i}\r\n") for i in range(45)]
    commit = create_context_commit(
        docs, query="evidence", token_budget=5, assurance_scope=SCOPE, prefer_rust=False
    )
    path = tmp_path / "commit.json"
    path.write_text(json.dumps(commit), encoding="utf-8")
    restored = json.loads(path.read_text(encoding="utf-8"))
    selected_ids = {x["chunk_id"] for x in restored["receipt"]["selected_context"]}
    omitted_ids = set(restored["recovery_bundle"]["chunks"]) - selected_ids
    assert len(omitted_ids) > len(restored["receipt"]["omitted_context"])
    assert restored["receipt"]["risk_summary"]["context_assurance"]["recovery"][
        "omitted_count"
    ] == len(omitted_ids)
    for cid in omitted_ids:
        recovered = recover_context_evidence(restored, cid, scope=SCOPE)
        assert recovered["text"].encode("utf-8") == restored["recovery_bundle"][
            "chunks"
        ][cid]["text"].encode("utf-8")
        assert recovered["verification_level"] == "exact_utf8_bytes"


def test_recovery_rejects_cross_scope_and_corruption():
    commit = create_context_commit(
        DOCS, query="evidence", token_budget=1, assurance_scope=SCOPE, prefer_rust=False
    )
    cid = next(iter(commit["recovery_bundle"]["chunks"]))
    for key in asdict(SCOPE):
        other = AssuranceScope(**{**asdict(SCOPE), key: "other"})
        with pytest.raises(ValueError, match="scope"):
            recover_context_evidence(commit, cid, scope=other)
    broken = copy.deepcopy(commit)
    broken["recovery_bundle"]["chunks"][cid]["text"] += " tampered"
    with pytest.raises(ValueError, match="integrity"):
        recover_context_evidence(broken, cid, scope=SCOPE)


@pytest.mark.parametrize(
    "identity",
    [
        "../../secret",
        "cr_../x",
        "cr_foo/bar",
        "cr_foo\\bar",
        "C:\\secret",
        "cr_x:stream",
        "",
        "cr_x\x00",
    ],
)
def test_recovery_identity_cannot_traverse_the_store(tmp_path, identity):
    with pytest.raises(ValueError):
        recovery_path(identity, tmp_path)


def test_recovery_refuses_a_symlinked_artifact(tmp_path):
    outside = tmp_path.parent / "external-recovery.json"
    outside.write_text("{}", encoding="utf-8")
    link = tmp_path / "cr_test.recovery.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation privilege unavailable")
    with pytest.raises(ValueError, match="declared store"):
        recovery_path("cr_test", tmp_path)


def test_engine_optimizer_proxy_is_not_semantic_assurance():
    selected = [
        {
            "source": "a",
            "content": "alpha",
            "token_count": 1,
            "sufficiency": {"scope": "semantic", "verdict": "sufficient"},
        }
    ]
    cert = audit_engine_selection(selected, query="alpha", token_budget=10)
    assert cert["verdict"] == "uncertain"
    assert cert["decision_risk"]["status"] == "unmeasured"
    with pytest.raises(ContextAssuranceError):
        require_assurance(cert)


def test_proxy_certificate_covers_final_rendered_and_aligned_text(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import entroly.proxy as proxy_module
    from entroly.proxy import PromptCompilerProxy
    from entroly.proxy_config import ProxyConfig
    from entroly.tokens import count_tokens

    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "0")
    monkeypatch.setenv("ENTROLY_VAULT_COUPLING", "0")
    fragment = {"id": "a", "source": "a.txt", "content": "alpha", "token_count": 1}
    engine = SimpleNamespace(
        _turn_counter=0,
        advance_turn=lambda: None,
        optimize_context=lambda *args: {"selected_fragments": [fragment]},
    )
    config = ProxyConfig(
        enable_adaptive_budget=False,
        enable_dynamic_budget=False,
        enable_hierarchical_compression=False,
        enable_security_scan=False,
        enable_ltm=False,
        enable_context_scaffold=False,
        enable_prompt_directives=False,
    )
    proxy = PromptCompilerProxy(engine, config)
    monkeypatch.setattr(
        proxy_module, "format_context_block", lambda *args, **kw: "rendered alpha"
    )
    proxy._cache_aligner = SimpleNamespace(
        align=lambda *args: ("aligned final beta", False)
    )
    body = {"model": "test-model", "messages": [{"role": "user", "content": "alpha"}]}
    result = proxy._run_pipeline("alpha", body)
    cert = result["context_assurance"]
    assert result["context"] == "aligned final beta"
    assert cert["structural_assurance"]["budget"]["selected_tokens"] == count_tokens(
        result["context"]
    )
    expected = audit_engine_selection(
        [{"source": "proxy:rendered-context", "content": result["context"]}],
        query="alpha",
        token_budget=cert["structural_assurance"]["budget"]["budget"],
    )
    assert cert == expected
    assert cert["verdict"] == "uncertain"
