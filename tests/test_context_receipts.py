import copy
import json
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import entroly.context_receipts as context_receipts_module
import entroly.server as entroly_server
from entroly import (
    create_context_receipt,
    explain_receipt_omission,
    render_context_receipt,
)
from entroly.context_receipts import (
    NoveltyAssessmentPolicy,
    assess_novelty_frontier,
    explain_omitted,
    ingest_documents,
    markdown_report,
    run_receipt_pipeline,
    select_from_index,
)
from entroly.context_receipts.ingest import read_documents_from_path
from entroly.context_receipts.models import RankedChunk
from entroly.context_receipts.selection import select_context
from entroly.context_receipts.receipts import attach_novelty_frontier
from entroly.context_receipts.novelty import novelty_frontier_assessment
from entroly.context_receipts.retrieval import rank_chunks


def _contract_docs() -> list[tuple[str, str]]:
    return [
        (
            "master.md",
            "# Section 1 Definitions\n\n"
            '"Change of Control" means a merger, sale of substantially all assets, or replacement of control.\n\n'
            "# Section 2 Assignment\n\n"
            "Neither party may assign this Agreement without prior written consent.\n",
        ),
        (
            "addendum.md",
            "# Addendum A\n\n"
            "Pursuant to Section 1, the Change of Control provision is modified for public offerings. "
            "See Schedule A.\n",
        ),
        (
            "schedule-a.md",
            "# Schedule A\n\n"
            "The reporting covenant applies after a Change of Control notice is delivered.\n",
        ),
    ]


def test_ingest_preserves_cross_document_metadata_and_fingerprints():
    first = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    second = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)

    assert first["schema_version"] == "context-receipt.v1"
    assert len(first["documents"]) == 3
    assert len(first["chunks"]) >= 3
    assert first["documents"][0]["fingerprint"] == second["documents"][0]["fingerprint"]
    assert first["chunks"][0]["chunk_id"] == second["chunks"][0]["chunk_id"]
    assert all(chunk["byte_end"] >= chunk["byte_start"] for chunk in first["chunks"])
    assert any(
        (chunk["section_heading"] or "").startswith("Section 1")
        for chunk in first["chunks"]
    )


def test_public_pipeline_sanitizes_malformed_numeric_options():
    receipt = run_receipt_pipeline(
        _contract_docs(),
        query="Which addendum modifies change of control?",
        token_budget=float("inf"),
        chunk_tokens=float("inf"),
        overlap_tokens=float("nan"),
        prefer_rust=False,
    )

    assert receipt["token_budget"] == 0
    assert "No chunks fit inside the token budget." in receipt["warnings"]
    assert receipt["compression_ratio"]["source_tokens"] > 0
    assert receipt["compression_ratio"]["selected_tokens"] == 0


def test_rust_pipeline_receives_sanitized_numeric_options(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}

    class FakeCore:
        @staticmethod
        def context_receipts_run(
            docs, query, token_budget, chunk_tokens, overlap_tokens
        ):
            calls["run"] = (token_budget, chunk_tokens, overlap_tokens)
            return json.dumps(
                {
                    "schema_version": "context-receipt.v1",
                    "query": query,
                    "token_budget": token_budget,
                    "selected_context": [],
                    "omitted_context": [],
                    "dependency_links": [],
                    "ranking_reasons": {},
                    "compression_ratio": {},
                    "source_fingerprints": {},
                    "risk_summary": {},
                    "warnings": [],
                    "outcome_links": [],
                    "receipt_id": "cr_fake",
                    "reproducibility_hash": "fake",
                }
            )

        @staticmethod
        def context_receipts_ingest(docs, chunk_tokens, overlap_tokens):
            calls["ingest"] = (chunk_tokens, overlap_tokens)
            return json.dumps(
                {
                    "schema_version": "context-receipt.v1",
                    "documents": [],
                    "chunks": [],
                    "chunk_token_limit": chunk_tokens,
                    "chunk_overlap": overlap_tokens,
                    "source_fingerprints": {},
                }
            )

    monkeypatch.setitem(sys.modules, "entroly_core", FakeCore)

    receipt = run_receipt_pipeline(
        _contract_docs(),
        query="change control",
        token_budget=float("inf"),
        chunk_tokens=float("inf"),
        overlap_tokens=float("nan"),
    )

    assert calls["run"] == (0, 360, 32)
    assert calls["ingest"] == (360, 32)
    assert receipt["token_budget"] == 0


def test_public_pipeline_clamps_negative_chunk_tokens_for_rust_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}

    class FakeCore:
        @staticmethod
        def context_receipts_run(
            docs, query, token_budget, chunk_tokens, overlap_tokens
        ):
            calls["run"] = (query, token_budget, chunk_tokens, overlap_tokens)
            return json.dumps(
                {
                    "schema_version": "context-receipt.v1",
                    "query": query,
                    "token_budget": token_budget,
                    "selected_context": [],
                    "omitted_context": [],
                    "dependency_links": [],
                    "ranking_reasons": {},
                    "compression_ratio": {},
                    "source_fingerprints": {},
                    "risk_summary": {},
                    "warnings": [],
                    "outcome_links": [],
                    "receipt_id": "cr_fake",
                    "reproducibility_hash": "fake",
                }
            )

        @staticmethod
        def context_receipts_ingest(docs, chunk_tokens, overlap_tokens):
            calls["ingest"] = (chunk_tokens, overlap_tokens)
            return json.dumps(
                {
                    "schema_version": "context-receipt.v1",
                    "documents": [],
                    "chunks": [],
                    "chunk_token_limit": chunk_tokens,
                    "chunk_overlap": overlap_tokens,
                    "source_fingerprints": {},
                }
            )

    monkeypatch.setitem(sys.modules, "entroly_core", FakeCore)

    receipt = run_receipt_pipeline(
        _contract_docs(),
        query=None,
        token_budget=-1,
        chunk_tokens=-100,
        overlap_tokens=-5,
    )

    assert calls["run"] == ("", 0, 40, 0)
    assert calls["ingest"] == (40, 0)
    assert receipt["query"] == ""


def test_public_pipeline_clamps_overlap_below_chunk_tokens_for_rust_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}

    class FakeCore:
        @staticmethod
        def context_receipts_run(
            docs, query, token_budget, chunk_tokens, overlap_tokens
        ):
            calls["run"] = (chunk_tokens, overlap_tokens)
            return json.dumps(
                {
                    "schema_version": "context-receipt.v1",
                    "query": query,
                    "token_budget": token_budget,
                    "selected_context": [],
                    "omitted_context": [],
                    "dependency_links": [],
                    "ranking_reasons": {},
                    "compression_ratio": {},
                    "source_fingerprints": {},
                    "risk_summary": {},
                    "warnings": [],
                    "outcome_links": [],
                    "receipt_id": "cr_fake",
                    "reproducibility_hash": "fake",
                }
            )

        @staticmethod
        def context_receipts_ingest(docs, chunk_tokens, overlap_tokens):
            calls["ingest"] = (chunk_tokens, overlap_tokens)
            return json.dumps(
                {
                    "schema_version": "context-receipt.v1",
                    "documents": [],
                    "chunks": [],
                    "chunk_token_limit": chunk_tokens,
                    "chunk_overlap": overlap_tokens,
                    "source_fingerprints": {},
                }
            )

    monkeypatch.setitem(sys.modules, "entroly_core", FakeCore)

    run_receipt_pipeline(
        _contract_docs(),
        query="change control",
        token_budget=25,
        chunk_tokens=9,
        overlap_tokens=99,
    )

    assert calls["run"] == (40, 39)
    assert calls["ingest"] == (40, 39)


def test_python_ingest_clamps_overlap_below_internal_chunk_floor():
    index = ingest_documents(
        _contract_docs(),
        chunk_tokens=8,
        overlap_tokens=999,
        prefer_rust=False,
    )

    assert index["chunk_token_limit"] == 40
    assert index["chunk_overlap"] == 39


def test_select_from_index_sanitizes_non_string_query():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)

    receipt = select_from_index(
        index,
        query=None,
        token_budget=40,
        prefer_rust=False,
    )

    assert receipt["query"] == ""


def test_receipt_schema_defined_terms_dependencies_and_compression_ratio():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Does this contract have a change-of-control provision?",
        token_budget=180,
        prefer_rust=False,
    )

    required = {
        "receipt_id",
        "query",
        "selected_context",
        "omitted_context",
        "dependency_links",
        "ranking_reasons",
        "compression_ratio",
        "source_fingerprints",
        "risk_summary",
        "token_budget",
        "warnings",
        "reproducibility_hash",
    }
    assert required <= set(receipt)
    assert receipt["selected_context"]
    assert (
        receipt["compression_ratio"]["source_tokens"]
        >= receipt["compression_ratio"]["selected_tokens"]
    )
    assert receipt["compression_ratio"]["selected_to_source_ratio"] <= 1.0
    assert receipt["source_fingerprints"]["documents"]
    assert receipt["risk_summary"]["controls"]["replayable_fingerprints"] is True
    assert receipt["risk_summary"]["review_level"] in {"low", "medium", "high"}
    relation_types = {link["relation_type"] for link in receipt["dependency_links"]}
    assert "defined_term" in relation_types
    assert "pursuant_to" in relation_types or "structural_reference" in relation_types
    assert any(link["resolved"] for link in receipt["dependency_links"])


def test_reproducibility_hash_is_stable_for_same_inputs():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    first = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    second = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )

    assert first["receipt_id"] == second["receipt_id"]
    assert first["reproducibility_hash"] == second["reproducibility_hash"]


def test_dependency_warning_when_budget_excludes_required_reference():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies the Change of Control provision pursuant to Section 1?",
        token_budget=24,
        prefer_rust=False,
    )

    assert receipt["selected_context"]
    assert any(
        "Dependency not included due to budget" in warning
        or "dependency reference" in warning
        for warning in receipt["warnings"]
    ) or any(item["dependencies_missing"] for item in receipt["selected_context"])


def test_omitted_nearby_context_and_explain():
    docs = [
        (
            "policy.md",
            "# Section 1 Definitions\n\nA covered user is an employee or contractor.\n\n"
            "# Section 2 Access Review\n\nAccess review evidence must include the manager approval, system owner approval, and ticket id.\n\n"
            "# Section 3 Retention\n\nReview records are retained for seven years and may be archived after audit closure.\n",
        )
    ]
    index = ingest_documents(docs, chunk_tokens=30, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="What evidence is required for access review?",
        token_budget=35,
        prefer_rust=False,
    )

    assert receipt["omitted_context"]
    assert any(
        item["omission_reason"] == "nearby_relevant_context_omitted_due_to_budget"
        for item in receipt["omitted_context"]
    )
    chunk_id = receipt["omitted_context"][0]["chunk_id"]
    explanation = explain_omitted(receipt, chunk_id, prefer_rust=False)
    assert chunk_id in explanation
    assert "omitted" in explanation


def test_markdown_report_contains_receipt_sections():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which policy depends on this definition?",
        token_budget=80,
        prefer_rust=False,
    )
    report = markdown_report(receipt, prefer_rust=False)

    assert "# Context Receipt" in report
    assert "## Included Context" in report
    assert "## Omitted Context" in report
    assert "## Dependency Graph Summary" in report
    assert "## Risks And Warnings" in report


def test_python_sdk_exports_context_receipt_api():
    receipt = create_context_receipt(
        _contract_docs(),
        query="Which addendum modifies change of control?",
        budget=80,
        chunk_tokens=55,
        prefer_rust=False,
    )
    report = render_context_receipt(receipt, prefer_rust=False)
    chunk_id = (
        receipt["omitted_context"][0]["chunk_id"]
        if receipt["omitted_context"]
        else receipt["selected_context"][0]["chunk_id"]
    )

    assert receipt["receipt_id"].startswith("cr_")
    assert "# Context Receipt" in report
    assert chunk_id in explain_receipt_omission(receipt, chunk_id, prefer_rust=False)


def test_mcp_server_registers_context_receipt_tools():
    source = inspect.getsource(entroly_server.create_mcp_server)

    assert "def create_context_receipt(" in source
    assert "def create_context_receipt_from_path(" in source
    assert "def render_context_receipt(" in source
    assert "def explain_receipt_omission(" in source


def test_npm_wasm_package_exposes_context_receipt_api():
    root = Path(__file__).resolve().parents[1]
    package = json.loads(
        (root / "entroly-wasm" / "package.json").read_text(encoding="utf-8")
    )
    index_js = (root / "entroly-wasm" / "index.js").read_text(encoding="utf-8")
    index_dts = (root / "entroly-wasm" / "index.d.ts").read_text(encoding="utf-8")

    assert "js/context_receipts.js" in package["files"]
    assert "createContextReceipt" in index_js
    assert "renderContextReceipt" in index_js
    assert "export function createContextReceipt" in index_dts
    assert "export function explainReceiptOmission" in index_dts


def test_npm_bridge_documents_context_receipt_cli():
    root = Path(__file__).resolve().parents[1]
    package = json.loads(
        (root / "entroly" / "npm" / "package.json").read_text(encoding="utf-8")
    )
    readme = (root / "entroly" / "npm" / "README.md").read_text(encoding="utf-8")

    assert "context-receipts" in package["keywords"]
    assert "npx -y entroly-mcp ingest ./docs" in readme
    assert "npx -y entroly-mcp select --query" in readme


def test_wasm_context_receipts_js_smoke():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    script = r"""
const cr = require('./entroly-wasm/js/context_receipts');
const receipt = cr.createContextReceipt({
  'master.md': '# Section 1 Definitions\n\n"Change of Control" means a sale of substantially all assets.\n',
  'addendum.md': '# Addendum A\n\nPursuant to Section 1, the Change of Control provision is modified.\n'
}, { query: 'Which addendum modifies change of control?', budget: 80, chunkTokens: 55 });
const report = cr.renderContextReceipt(receipt);
if (!receipt.receipt_id.startsWith('cr_')) throw new Error('missing receipt id');
if (!receipt.selected_context.length) throw new Error('no selected context');
if (!report.includes('# Context Receipt')) throw new Error('missing report header');
console.log(receipt.receipt_id);
"""
    result = subprocess.run(
        [node, "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip().startswith("cr_")


def test_cli_ingest_select_receipt_explain_smoke(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for name, text in _contract_docs():
        (docs_dir / name).write_text(text, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    ingest = subprocess.run(
        [sys.executable, "-m", "entroly.cli", "ingest", str(docs_dir), "--python"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert ingest.returncode == 0, ingest.stderr + ingest.stdout

    receipt_path = tmp_path / "receipt.json"
    report_path = tmp_path / "receipt.md"
    select = subprocess.run(
        [
            sys.executable,
            "-m",
            "entroly.cli",
            "select",
            "--query",
            "Does this contract have a change-of-control provision?",
            "--budget",
            "45",
            "--receipt",
            str(receipt_path),
            "--report",
            str(report_path),
            "--python",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert select.returncode == 0, select.stderr + select.stdout
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["receipt_id"].startswith("cr_")
    assert report_path.exists()

    rendered = subprocess.run(
        [sys.executable, "-m", "entroly.cli", "receipt", str(receipt_path), "--python"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert rendered.returncode == 0, rendered.stderr + rendered.stdout
    assert "# Context Receipt" in rendered.stdout

    omitted_id = (
        receipt["omitted_context"][0]["chunk_id"]
        if receipt["omitted_context"]
        else receipt["selected_context"][0]["chunk_id"]
    )
    explained = subprocess.run(
        [
            sys.executable,
            "-m",
            "entroly.cli",
            "explain",
            "--receipt",
            str(receipt_path),
            "--why-omitted",
            omitted_id,
            "--python",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert explained.returncode == 0, explained.stderr + explained.stdout
    assert omitted_id in explained.stdout


def test_read_documents_from_path_filters_supported_files(tmp_path: Path):
    (tmp_path / "a.md").write_text("# A\n\nalpha", encoding="utf-8")
    (tmp_path / "app.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (tmp_path / "Dockerfile.prod").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (tmp_path / "Containerfile.dev").write_text("FROM fedora:latest\n", encoding="utf-8")
    (tmp_path / "Justfile").write_text("test:\n    pytest\n", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01")

    docs = read_documents_from_path(tmp_path)
    sources = {Path(source).name for source, _ in docs}

    assert sources == {
        "Containerfile.dev",
        "Dockerfile",
        "Dockerfile.prod",
        "Justfile",
        "a.md",
        "app.py",
    }


def test_read_documents_from_path_skips_generated_dependency_dirs(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    node_modules = tmp_path / "node_modules" / "dep"
    node_modules.mkdir(parents=True)
    (node_modules / "README.md").write_text("# Dependency copy\n", encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config.toml").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}\n", encoding="utf-8")
    egg_info = tmp_path / "sample.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("Metadata-Version: 2.1\n", encoding="utf-8")

    docs = read_documents_from_path(tmp_path)

    assert [Path(source).name for source, _ in docs] == ["README.md"]


def test_receipt_tracks_local_novelty_frontier():
    docs = [
        (
            "alpha.md",
            "# Core Policy\n\nAccess review requires manager approval, ticket evidence, and quarterly attestation.\n",
        ),
        (
            "beta.md",
            "# Exception Policy\n\nAccess review exceptions require compensating controls and executive signoff.\n",
        ),
    ]
    index = ingest_documents(docs, chunk_tokens=16, prefer_rust=False)
    receipt = select_from_index(
        index, query="access review", token_budget=18, prefer_rust=False
    )

    novelty = receipt["risk_summary"]["novelty_summary"]
    assert novelty["control"] == "local_counterfactual_concept_frontier.v3"
    assert "access" in novelty["query_terms"]
    assert "review" in novelty["query_terms"]
    assert "compensating" in novelty["selected_novel_terms"]
    assert "manager" in novelty["omitted_frontier_terms"]
    assert novelty["selected_novel_term_count"] >= len(novelty["selected_novel_terms"])
    assert (
        novelty["selected_novel_term_occurrences"]
        >= novelty["selected_novel_term_count"]
    )
    assert novelty["frontier_gap_score_ratio"] > 0
    assert novelty["omitted_frontier_profiles"]["manager"]["chunk_count"] == 1
    assert novelty["omitted_frontier_term_count"] >= len(
        novelty["omitted_frontier_terms"]
    )
    assert novelty["frontier_gap_ratio"] > 0
    assert novelty["omitted_full_text_chunks"] == len(receipt["omitted_context"])
    assert novelty["omitted_truncated_chunks"] == 0
    assert novelty["per_selected_chunk"]
    assert receipt["risk_summary"]["controls"]["novelty_frontier"] == "high"
    assessment = receipt["risk_summary"]["novelty_frontier_assessment"]
    assert assessment["pressure"] == "high"
    assert (
        assessment["action"]
        == "Review omitted frontier terms before relying on this receipt."
    )
    assert assessment["thresholds"]["high_score_ratio"] == 0.45
    assert receipt["risk_summary"]["review_level"] == "high"
    assert any("Novelty frontier is high" in warning for warning in receipt["warnings"])


def test_novelty_frontier_uses_full_omitted_text_not_preview_only():
    long_tail = (
        " ".join(["filler"] * 90)
        + " ultradeepnovelty ultradeepnovelty ultradeepnovelty"
    )
    docs = [
        (
            "selected.md",
            "# Access Review\n\nAccess review requires manager approval and ticket evidence.\n",
        ),
        (
            "omitted.md",
            f"# Access Review Exception\n\nAccess review exception details {long_tail}.\n",
        ),
    ]
    index = ingest_documents(docs, chunk_tokens=110, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="manager approval ticket evidence",
        token_budget=12,
        prefer_rust=False,
    )

    omitted = receipt["omitted_context"]
    assert omitted
    assert all("ultradeepnovelty" not in item["text_preview"] for item in omitted)
    novelty = receipt["risk_summary"]["novelty_summary"]
    assert "ultradeepnovelty" in novelty["omitted_frontier_terms"]
    assert novelty["omitted_full_text_chunks"] == len(omitted)


def test_novelty_frontier_assessment_classifies_rehydrated_summaries():
    low = novelty_frontier_assessment(
        {"omitted_frontier_term_count": 0, "selected_novel_term_count": 4}
    )
    assert low["pressure"] == "low"
    assert low["action"] == "No novelty-specific review is required."

    medium = novelty_frontier_assessment(
        {
            "frontier_gap_score_ratio": 0.2,
            "omitted_frontier_term_count": 1,
            "selected_novel_term_count": 8,
        }
    )
    assert medium["pressure"] == "medium"

    high = novelty_frontier_assessment(
        {
            "frontier_gap_score_ratio": 0.46,
            "omitted_frontier_term_count": 2,
            "selected_novel_term_count": 8,
        }
    )
    assert high["pressure"] == "high"
    assert high["thresholds"]["high_score_ratio"] == 0.45

    stricter = NoveltyAssessmentPolicy(
        medium_score_ratio=0.3, high_score_ratio=0.9, minimum_high_terms=10
    )
    reassessed = novelty_frontier_assessment(
        {
            "frontier_gap_score_ratio": 0.46,
            "omitted_frontier_term_count": 2,
            "selected_novel_term_count": 8,
        },
        policy=stricter,
    )
    assert reassessed["pressure"] == "medium"
    assert reassessed["thresholds"]["high_score_ratio"] == 0.9


def test_novelty_assessment_policy_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="medium_score_ratio"):
        NoveltyAssessmentPolicy(medium_score_ratio=1.2)

    with pytest.raises(ValueError, match="less than or equal"):
        NoveltyAssessmentPolicy(medium_score_ratio=0.7, high_score_ratio=0.4)

    with pytest.raises(ValueError, match="minimum_high_terms"):
        NoveltyAssessmentPolicy(minimum_high_terms=-1)


def test_novelty_frontier_assessment_sanitizes_rehydrated_metrics():
    assessed = novelty_frontier_assessment(
        {
            "frontier_gap_score_ratio": float("inf"),
            "frontier_gap_ratio": "not-a-number",
            "omitted_frontier_term_count": -8,
            "selected_novel_term_count": True,
        }
    )

    assert assessed["pressure"] == "low"
    assert assessed["frontier_gap_score_ratio"] == 0.0
    assert assessed["omitted_frontier_term_count"] == 0
    assert assessed["selected_novel_term_count"] == 0

    fallback_assessed = novelty_frontier_assessment(
        {
            "frontier_gap_score_ratio": float("nan"),
            "frontier_gap_ratio": 0.2,
            "omitted_frontier_term_count": 1,
            "selected_novel_term_count": 8,
        }
    )
    assert fallback_assessed["pressure"] == "medium"
    assert fallback_assessed["frontier_gap_score_ratio"] == 0.2


def test_public_novelty_frontier_assessment_accepts_receipt_or_summary():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    novelty = receipt["risk_summary"]["novelty_summary"]

    from_receipt = assess_novelty_frontier(receipt)
    from_summary = assess_novelty_frontier(novelty)

    assert from_receipt == from_summary
    assert from_receipt["pressure"] in {"low", "medium", "high"}
    assert assess_novelty_frontier({"risk_summary": "corrupt"})["pressure"] == "low"
    assert assess_novelty_frontier("corrupt")["pressure"] == "low"


def test_markdown_report_contains_novelty_frontier():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    report = markdown_report(receipt, prefer_rust=False)

    assert "## Novelty Frontier" in report
    assert "Novelty frontier pressure:" in report
    assert "Novelty frontier action:" in report
    assert "Frontier score ratio:" in report
    assert "Assessment rationale:" in report
    assert "local_counterfactual_concept_frontier.v3" in report


def test_markdown_report_tolerates_malformed_rehydrated_risk_summary():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    receipt["risk_summary"] = {
        "coverage_score": "not-a-number",
        "controls": "corrupt",
        "novelty_summary": {
            "selected_novel_terms": "corrupt",
            "omitted_frontier_terms": ["valid", 123],
            "frontier_gap_ratio": "bad",
            "frontier_gap_score_ratio": float("inf"),
        },
        "novelty_frontier_assessment": "corrupt",
    }

    report = markdown_report(receipt, prefer_rust=False)

    assert "Coverage score: 0.000" in report
    assert "Dependency closure: unknown" in report
    assert "Selected novel terms: none" in report
    assert "Omitted frontier terms: valid, 123" in report
    assert "Frontier score ratio: 0.000" in report


def test_markdown_report_tolerates_non_finite_metrics():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    receipt["risk_summary"]["coverage_score"] = float("nan")
    receipt["risk_summary"]["novelty_summary"]["frontier_gap_score_ratio"] = float(
        "inf"
    )

    report = markdown_report(receipt, prefer_rust=False)

    assert "Coverage score: 0.000" in report
    assert "Frontier score ratio: 0.000" in report


def test_attach_novelty_frontier_normalizes_corrupt_controls_and_warnings():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    receipt["risk_summary"]["controls"] = "corrupt"
    receipt["warnings"] = "legacy warning"

    enriched = attach_novelty_frontier(receipt, index)

    assert isinstance(enriched["risk_summary"]["controls"], dict)
    assert enriched["risk_summary"]["controls"]["novelty_frontier"] in {
        "low",
        "medium",
        "high",
    }
    assert "legacy warning" in enriched["warnings"]
    assert all(isinstance(warning, str) for warning in enriched["warnings"])
    assert enriched["receipt_id"].startswith("cr_")


def test_attach_novelty_frontier_does_not_mutate_input_receipt():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    original = copy.deepcopy(receipt)

    enriched = attach_novelty_frontier(receipt, index)

    assert enriched is not receipt
    assert enriched["risk_summary"] is not receipt["risk_summary"]
    assert receipt == original


def test_markdown_report_normalizes_scalar_rehydrated_warnings_and_risk_summary():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    receipt["risk_summary"] = "corrupt"
    receipt["warnings"] = "legacy warning"
    receipt["ranking_reasons"] = "corrupt"

    report = markdown_report(receipt, prefer_rust=False)

    assert "Coverage score: 0.000" in report
    risks_section = report.split("## Risks And Warnings", maxsplit=1)[1].split(
        "## Reproducibility", maxsplit=1
    )[0]
    warning_lines = [
        line for line in risks_section.splitlines() if line.startswith("- ")
    ]
    assert warning_lines == ["- legacy warning"]


def test_markdown_report_tolerates_malformed_rehydrated_context_items():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    receipt["selected_context"] = [
        "corrupt",
        {
            "chunk_id": 123,
            "source_path": None,
            "token_count": "bad",
            "score": float("nan"),
            "reasons": "legacy reason",
            "dependencies_included": "corrupt",
            "dependencies_missing": ["dep"],
            "fingerprint": None,
            "text": None,
        },
    ]
    receipt["omitted_context"] = [
        "corrupt",
        {
            "chunk_id": "omitted",
            "score": float("inf"),
            "reasons": "legacy omitted reason",
            "text_preview": None,
        },
    ]
    receipt["dependency_links"] = [
        "corrupt",
        {"source_chunk_id": "source", "target_chunk_id": 123, "resolved": "yes"},
    ]
    receipt["compression_ratio"] = "corrupt"

    report = markdown_report(receipt, prefer_rust=False)

    assert "### 123" in report
    assert "score: 0.0000" in report
    assert "### omitted" in report
    assert "`source` -> `123` (, resolved):" in report
    assert "Source tokens: 0" in report


def test_default_markdown_report_bypasses_rust_for_malformed_context_items(
    monkeypatch: pytest.MonkeyPatch,
):
    class ExplodingRustCore:
        def context_receipts_report(self, _payload: str) -> str:
            raise AssertionError(
                "malformed nested receipt items should render in Python"
            )

    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    receipt["selected_context"] = ["corrupt"]
    monkeypatch.setattr(
        context_receipts_module, "_rust_core", lambda: ExplodingRustCore()
    )

    report = markdown_report(receipt)

    assert "No context chunks were selected." in report


def test_default_markdown_report_bypasses_rust_for_malformed_rehydrated_receipts(
    monkeypatch: pytest.MonkeyPatch,
):
    class ExplodingRustCore:
        def context_receipts_report(self, _payload: str) -> str:
            raise AssertionError("malformed receipt should render in Python")

    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    receipt["risk_summary"] = "corrupt"
    receipt["warnings"] = "legacy warning"
    receipt["ranking_reasons"] = "corrupt"
    monkeypatch.setattr(
        context_receipts_module, "_rust_core", lambda: ExplodingRustCore()
    )

    report = markdown_report(receipt)

    assert "Coverage score: 0.000" in report
    assert "- legacy warning" in report


def test_default_rust_sdk_path_enriches_context_receipts_with_novelty():
    pytest.importorskip("entroly_core")
    docs = _contract_docs()
    index = ingest_documents(docs, chunk_tokens=55, prefer_rust=True)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=True,
    )

    novelty = receipt["risk_summary"]["novelty_summary"]
    assert novelty["control"] == "local_counterfactual_concept_frontier.v3"
    assert receipt["risk_summary"]["controls"]["novelty_frontier"] in {
        "low",
        "medium",
        "high",
    }
    assert receipt["risk_summary"]["novelty_frontier_assessment"]["pressure"] in {
        "low",
        "medium",
        "high",
    }
    assert novelty["omitted_full_text_chunks"] == len(receipt["omitted_context"])
    assert receipt["receipt_id"].startswith("cr_")


def test_default_markdown_report_renders_python_novelty_section_for_enriched_rust_receipts():
    pytest.importorskip("entroly_core")
    receipt = run_receipt_pipeline(
        _contract_docs(),
        query="Which addendum modifies change of control?",
        token_budget=80,
        chunk_tokens=55,
        prefer_rust=True,
    )

    report = markdown_report(receipt)

    assert "## Novelty Frontier" in report
    assert "Novelty frontier pressure:" in report
    assert "local_counterfactual_concept_frontier.v3" in report


def test_rust_capability_checks_are_per_entrypoint(monkeypatch: pytest.MonkeyPatch):
    class IngestOnlyRustCore:
        def context_receipts_ingest(self, docs, chunk_tokens, overlap_tokens):
            assert docs == [("doc.md", "hello world")]
            return json.dumps(
                {
                    "schema_version": "context-receipt.v1",
                    "documents": [],
                    "chunks": [],
                    "chunk_token_limit": chunk_tokens,
                    "chunk_overlap": overlap_tokens,
                    "source_fingerprints": {},
                    "engine": "partial-rust",
                }
            )

    core = IngestOnlyRustCore()

    def capability_core(*required: str):
        return core if all(hasattr(core, name) for name in required) else None

    monkeypatch.setattr(context_receipts_module, "_rust_core", capability_core)

    index = ingest_documents(
        [("doc.md", "hello world")],
        chunk_tokens=9,
        overlap_tokens=1,
        prefer_rust=True,
    )

    assert index["engine"] == "partial-rust"
    assert index["chunk_token_limit"] == 40


def test_explain_omitted_bypasses_rust_for_malformed_rehydrated_receipts(
    monkeypatch: pytest.MonkeyPatch,
):
    class ExplodingRustCore:
        def context_receipts_explain_omitted(
            self, _payload: str, _chunk_id: str
        ) -> str:
            raise AssertionError("malformed receipt should explain in Python")

    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(
        index,
        query="Which addendum modifies change of control?",
        token_budget=80,
        prefer_rust=False,
    )
    omitted_id = "manual-omitted"
    receipt["omitted_context"] = [
        {
            "chunk_id": omitted_id,
            "source_path": "manual.md",
            "token_count": 3,
            "score": 0.4,
            "reasons": ["manual regression"],
            "omission_reason": "outside budget",
            "fingerprint": "sha256:test",
            "text_preview": "manual omitted preview",
        }
    ]
    receipt["warnings"] = "legacy warning"
    receipt["ranking_reasons"] = "corrupt"
    monkeypatch.setattr(
        context_receipts_module,
        "_rust_core",
        lambda *required: (
            ExplodingRustCore()
            if "context_receipts_explain_omitted" in required
            else None
        ),
    )

    explanation = explain_omitted(receipt, omitted_id)

    assert f"{omitted_id} was omitted" in explanation


def test_context_index_from_dict_tolerates_malformed_rehydrated_index():
    from entroly.context_receipts.models import ContextIndex

    index = ContextIndex.from_dict(
        {
            "schema_version": None,
            "documents": [
                "corrupt",
                {
                    "document_id": 123,
                    "source_path": None,
                    "token_count": "bad",
                    "byte_count": float("nan"),
                    "chunk_ids": "legacy-chunk",
                    "extra": "ignored",
                },
            ],
            "chunks": [
                "corrupt",
                {
                    "chunk_id": 456,
                    "document_id": None,
                    "source_path": None,
                    "page_number": "bad",
                    "chunk_index": "bad",
                    "score": "ignored",
                    "token_count": float("inf"),
                    "text": None,
                    "extra": "ignored",
                },
            ],
            "chunk_token_limit": "bad",
            "chunk_overlap": False,
            "source_fingerprints": "corrupt",
        }
    )

    assert index.schema_version == "context-receipt.v1"
    assert len(index.documents) == 1
    assert index.documents[0].document_id == "123"
    assert index.documents[0].token_count == 0
    assert index.documents[0].byte_count == 0
    assert index.documents[0].chunk_ids == ["legacy-chunk"]
    assert len(index.chunks) == 1
    assert index.chunks[0].chunk_id == "456"
    assert index.chunks[0].page_number == 0
    assert index.chunks[0].token_count == 0
    assert index.chunks[0].text == ""
    assert index.chunk_token_limit == 360
    assert index.chunk_overlap == 32
    assert index.source_fingerprints == {}


def test_rank_chunks_sanitizes_non_finite_semantic_scores_and_reranker_output():
    index = ingest_documents(
        [("policy.md", "Access review requires manager approval and ticket evidence.")],
        chunk_tokens=55,
        prefer_rust=False,
    )
    py_index = context_receipts_module.ContextIndex.from_dict(index)

    class BadSemanticScorer:
        def score(self, _query, chunks):
            return {chunks[0].chunk_id: float("nan")}

    semantic_rank = rank_chunks(
        py_index, "access review", semantic_scorer=BadSemanticScorer()
    )[0]
    assert semantic_rank.semantic_score == 0.0
    assert semantic_rank.final_score >= 0.0

    def bad_reranker(_query, ranks):
        ranks[0].final_score = float("inf")
        ranks[0].semantic_score = float("nan")
        ranks[0].rerank_score = True
        ranks[0].reasons = "legacy rerank reason"
        return ranks

    reranked = rank_chunks(py_index, "access review", reranker=bad_reranker)[0]

    assert reranked.final_score == 0.0
    assert reranked.semantic_score == 0.0
    assert reranked.rerank_score == 0.0
    assert reranked.reasons == ["legacy rerank reason"]


def test_rank_chunks_accepts_mapping_rows_from_legacy_rerankers():
    index = ingest_documents(
        [("policy.md", "Access review requires manager approval and ticket evidence.")],
        chunk_tokens=55,
        prefer_rust=False,
    )
    py_index = context_receipts_module.ContextIndex.from_dict(index)

    def mapping_reranker(_query, ranks):
        return [
            {
                "chunk_id": ranks[0].chunk_id,
                "lexical_score": "1.25",
                "semantic_score": float("inf"),
                "rerank_score": False,
                "final_score": "2.5",
                "reasons": ("legacy mapping reranker",),
            }
        ]

    reranked = rank_chunks(py_index, "access review", reranker=mapping_reranker)[0]

    assert reranked.chunk_id == py_index.chunks[0].chunk_id
    assert reranked.lexical_score == 1.25
    assert reranked.semantic_score == 0.0
    assert reranked.rerank_score == 0.0
    assert reranked.final_score == 2.5
    assert reranked.reasons == ["legacy mapping reranker"]


def test_assess_novelty_frontier_tolerates_overflowing_integer_metrics():
    assessment = assess_novelty_frontier(
        {
            "frontier_gap_score_ratio": 0.9,
            "omitted_frontier_term_count": float("inf"),
            "selected_novel_term_count": float("inf"),
        }
    )

    assert assessment["pressure"] == "low"
    assert assessment["omitted_frontier_term_count"] == 0
    assert assessment["selected_novel_term_count"] == 0


def test_rank_chunks_preserves_audit_coverage_after_partial_reranker_output():
    index = ingest_documents(
        [
            ("access.md", "Access review requires manager approval."),
            ("audit.md", "Audit evidence requires ticket retention."),
        ],
        chunk_tokens=8,
        prefer_rust=False,
    )
    py_index = context_receipts_module.ContextIndex.from_dict(index)

    def partial_reranker(_query, ranks):
        return [
            {"chunk_id": "unknown", "final_score": 999.0},
            {"chunk_id": ranks[-1].chunk_id, "final_score": 3.0},
            {"chunk_id": ranks[-1].chunk_id, "final_score": 2.0},
        ]

    reranked = rank_chunks(py_index, "access audit", reranker=partial_reranker)

    assert [rank.chunk_id for rank in reranked] == [
        py_index.chunks[-1].chunk_id,
        py_index.chunks[0].chunk_id,
    ]
    assert reranked[0].final_score == 3.0


def test_rank_chunks_falls_back_when_legacy_reranker_returns_non_iterable():
    index = ingest_documents(
        [
            ("access.md", "Access review requires manager approval."),
            ("audit.md", "Audit evidence requires ticket retention."),
        ],
        chunk_tokens=8,
        prefer_rust=False,
    )
    py_index = context_receipts_module.ContextIndex.from_dict(index)
    baseline = rank_chunks(py_index, "access audit")

    def scalar_reranker(_query, _ranks):
        return None

    reranked = rank_chunks(py_index, "access audit", reranker=scalar_reranker)

    assert [rank.chunk_id for rank in reranked] == [rank.chunk_id for rank in baseline]


def test_rank_chunks_accepts_mapping_container_from_legacy_reranker():
    index = ingest_documents(
        [
            ("access.md", "Access review requires manager approval."),
            ("audit.md", "Audit evidence requires ticket retention."),
        ],
        chunk_tokens=8,
        prefer_rust=False,
    )
    py_index = context_receipts_module.ContextIndex.from_dict(index)

    def mapping_container_reranker(_query, ranks):
        return {
            "first": {"chunk_id": ranks[-1].chunk_id, "final_score": 4.0},
            "ignored": {"chunk_id": "unknown", "final_score": 999.0},
        }

    reranked = rank_chunks(
        py_index, "access audit", reranker=mapping_container_reranker
    )

    assert [rank.chunk_id for rank in reranked] == [
        py_index.chunks[-1].chunk_id,
        py_index.chunks[0].chunk_id,
    ]
    assert reranked[0].final_score == 4.0


def test_rank_chunks_ignores_non_mapping_semantic_scorer_output():
    index = ingest_documents(
        [("policy.md", "Access review evidence with manager approval.")],
        chunk_tokens=20,
        prefer_rust=False,
    )
    py_index = context_receipts_module.ContextIndex.from_dict(index)

    class LegacySemanticScorer:
        def score(self, _query, _chunks):
            return None

    ranked = rank_chunks(
        py_index, "access review", semantic_scorer=LegacySemanticScorer()
    )

    assert len(ranked) == len(py_index.chunks)
    assert ranked[0].semantic_score == 0.0
    assert ranked[0].final_score >= 0.0


def test_rust_core_detects_preloaded_extension_without_import_spec(
    monkeypatch: pytest.MonkeyPatch,
):
    class PreloadedCore:
        def context_receipts_ingest(self, docs, chunk_tokens, overlap_tokens):
            return json.dumps(
                {
                    "schema_version": "context-receipt.v1",
                    "documents": [],
                    "chunks": [],
                    "chunk_token_limit": chunk_tokens,
                    "chunk_overlap": overlap_tokens,
                    "source_fingerprints": {},
                    "doc_count": len(docs),
                }
            )

    monkeypatch.setitem(sys.modules, "entroly_core", PreloadedCore())

    index = ingest_documents(
        [("policy.md", "Access review evidence.")], prefer_rust=True
    )

    assert index["doc_count"] == 1


def test_float_coercion_handles_overflowing_custom_numbers():
    class OverflowingFloat:
        def __float__(self):
            raise OverflowError("too large")

    receipt = context_receipts_module.ContextReceipt.from_dict(
        {
            "selected_context": [
                {
                    "chunk_id": "c1",
                    "score": OverflowingFloat(),
                    "reasons": ["kept"],
                }
            ],
            "omitted_context": [
                {
                    "chunk_id": "c2",
                    "score": OverflowingFloat(),
                    "reasons": ["omitted"],
                }
            ],
            "compression_ratio": {
                "selected_to_source_ratio": OverflowingFloat(),
                "source_to_selected_ratio": OverflowingFloat(),
                "reduction_pct": OverflowingFloat(),
            },
        }
    )

    assert receipt.selected_context[0].score == 0.0
    assert receipt.omitted_context[0].score == 0.0
    assert receipt.compression_ratio.selected_to_source_ratio == 0.0
    assert receipt.compression_ratio.source_to_selected_ratio == 1.0
    assert receipt.compression_ratio.reduction_pct == 0.0


def test_rank_chunks_sanitizes_overflowing_float_values():
    class OverflowingFloat:
        def __float__(self):
            raise OverflowError("too large")

    index = ingest_documents(
        [("policy.md", "Access review evidence with manager approval.")],
        chunk_tokens=20,
        prefer_rust=False,
    )
    py_index = context_receipts_module.ContextIndex.from_dict(index)

    class OverflowingSemanticScorer:
        def score(self, _query, chunks):
            return {chunks[0].chunk_id: OverflowingFloat()}

    def overflowing_reranker(_query, ranks):
        return [
            {
                "chunk_id": ranks[0].chunk_id,
                "semantic_score": OverflowingFloat(),
                "final_score": OverflowingFloat(),
            }
        ]

    ranked = rank_chunks(
        py_index,
        "access review",
        semantic_scorer=OverflowingSemanticScorer(),
        reranker=overflowing_reranker,
    )

    assert ranked[0].semantic_score == 0.0
    assert ranked[0].final_score == 0.0


def test_novelty_policy_rejects_huge_ratio_without_overflowing():
    with pytest.raises(ValueError, match="finite numeric ratio"):
        NoveltyAssessmentPolicy(medium_score_ratio=10**10000)


def test_public_ingest_normalizes_mapping_path_and_bytes_documents(monkeypatch):
    class FakeRustCore:
        def context_receipts_ingest(self, docs, chunk_tokens, overlap_tokens):
            assert docs == [
                ("docs/policy.md", "Access evidence."),
                ("bytes.md", "caf\ufffd"),
            ]
            return json.dumps(
                {
                    "schema_version": "context-receipt.v1",
                    "documents": [],
                    "chunks": [],
                    "chunk_token_limit": chunk_tokens,
                    "chunk_overlap": overlap_tokens,
                    "source_fingerprints": {},
                }
            )

    monkeypatch.setattr(context_receipts_module, "_rust_core", lambda *_: FakeRustCore())
    index = ingest_documents(
        [
            {"path": Path("docs/policy.md"), "content": "Access evidence."},
            ("bytes.md", b"caf\xff"),
            "malformed",
            ("missing-text.md",),
            {"source_path": "missing-text.md"},
        ]
    )

    assert index["chunk_token_limit"] == 360


def test_python_ingest_skips_malformed_document_rows_without_crashing():
    index = ingest_documents(
        [
            {"source_path": "valid.md", "text": "Access reviews require evidence."},
            "not-a-document-row",
            ("missing-text.md",),
            {"path": "none-text.md", "content": None},
        ],
        chunk_tokens=20,
        prefer_rust=False,
    )

    assert [doc["source_path"] for doc in index["documents"]] == ["valid.md"]
    assert index["chunks"]


def test_build_receipt_and_selection_tolerate_malformed_direct_inputs():
    index = context_receipts_module.ContextIndex.from_dict(
        ingest_documents(
            [("policy.md", "Access review evidence with manager approval.")],
            chunk_tokens=20,
            prefer_rust=False,
        )
    )
    receipt = context_receipts_module._py_build_receipt(
        index, query=None, token_budget=50
    )
    bogus_rank = RankedChunk(
        "missing", 1.0, 0.0, 0.0, 1.0, ["legacy rank row"]
    )
    valid_rank = RankedChunk(
        index.chunks[0].chunk_id, 1.0, 0.0, 0.0, 1.0, ["valid rank row"]
    )

    selection = select_context(index, [bogus_rank, valid_rank], [], token_budget=50)

    assert receipt.query == ""
    assert [item.chunk_id for item in selection.selected] == [index.chunks[0].chunk_id]
