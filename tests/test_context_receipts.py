import json
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import entroly.server as entroly_server
from entroly import create_context_receipt, explain_receipt_omission, render_context_receipt
from entroly.context_receipts import (
    explain_omitted,
    ingest_documents,
    markdown_report,
    select_from_index,
)
from entroly.context_receipts.ingest import read_documents_from_path


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
    assert any((chunk["section_heading"] or "").startswith("Section 1") for chunk in first["chunks"])


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
    assert receipt["compression_ratio"]["source_tokens"] >= receipt["compression_ratio"]["selected_tokens"]
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
    first = select_from_index(index, query="Which addendum modifies change of control?", token_budget=80, prefer_rust=False)
    second = select_from_index(index, query="Which addendum modifies change of control?", token_budget=80, prefer_rust=False)

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
        "Dependency not included due to budget" in warning or "dependency reference" in warning
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
    assert any(item["omission_reason"] == "nearby_relevant_context_omitted_due_to_budget" for item in receipt["omitted_context"])
    chunk_id = receipt["omitted_context"][0]["chunk_id"]
    explanation = explain_omitted(receipt, chunk_id, prefer_rust=False)
    assert chunk_id in explanation
    assert "omitted" in explanation


def test_markdown_report_contains_receipt_sections():
    index = ingest_documents(_contract_docs(), chunk_tokens=55, prefer_rust=False)
    receipt = select_from_index(index, query="Which policy depends on this definition?", token_budget=80, prefer_rust=False)
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
    chunk_id = receipt["omitted_context"][0]["chunk_id"] if receipt["omitted_context"] else receipt["selected_context"][0]["chunk_id"]

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
    package = json.loads((root / "entroly-wasm" / "package.json").read_text(encoding="utf-8"))
    index_js = (root / "entroly-wasm" / "index.js").read_text(encoding="utf-8")
    index_dts = (root / "entroly-wasm" / "index.d.ts").read_text(encoding="utf-8")

    assert "js/context_receipts.js" in package["files"]
    assert "createContextReceipt" in index_js
    assert "renderContextReceipt" in index_js
    assert "export function createContextReceipt" in index_dts
    assert "export function explainReceiptOmission" in index_dts


def test_npm_bridge_documents_context_receipt_cli():
    root = Path(__file__).resolve().parents[1]
    package = json.loads((root / "entroly" / "npm" / "package.json").read_text(encoding="utf-8"))
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

    omitted_id = receipt["omitted_context"][0]["chunk_id"] if receipt["omitted_context"] else receipt["selected_context"][0]["chunk_id"]
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

    assert sources == {"Containerfile.dev", "Dockerfile", "Dockerfile.prod", "Justfile", "a.md", "app.py"}


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
