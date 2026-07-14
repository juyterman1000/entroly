from __future__ import annotations

import json
import threading
import urllib.request
from http.server import HTTPServer

from entroly.context_receipts.models import stable_hash
from entroly.context_sessions import ContextSessionIndex
from entroly.dashboard import DashboardHandler
from entroly.session_intelligence import SessionReceiptChain


def _receipt(query: str = "Fix auth", *, model: str = "gpt-5.6-sol") -> dict:
    payload = {
        "schema_version": "context-receipt.v1",
        "query": query,
        "model": model,
        "token_budget": 1000,
        "selected_context": [
            {
                "chunk_id": "src-1",
                "source_path": "src/auth.py",
                "token_count": 400,
                "score": 0.9,
                "reasons": ["direct symbol match"],
                "text": "important evidence " * 80,
                "fingerprint": "sha256:selected",
            }
        ],
        "omitted_context": [
            {
                "chunk_id": "test-1",
                "source_path": "tests/test_auth.py",
                "token_count": 600,
                "score": 0.3,
                "reasons": ["budget pressure"],
                "omission_reason": "lower marginal value",
                "text_preview": "omitted test evidence",
                "fingerprint": "sha256:omitted",
                "recoverable": True,
            }
        ],
        "dependency_links": [],
        "ranking_reasons": {},
        "compression_ratio": {
            "source_tokens": 1000,
            "selected_tokens": 400,
            "tokens_saved": 600,
        },
        "source_fingerprints": {},
        "risk_summary": {"review_level": "medium"},
        "warnings": ["one relevant chunk omitted"],
        "outcome_links": [],
    }
    digest = stable_hash(payload)
    return {**payload, "receipt_id": f"cr_{digest[:12]}", "reproducibility_hash": digest}


def _write_session(root, *, standalone: bool = False):
    receipts = root / "receipts"
    receipts.mkdir(parents=True)
    receipt = _receipt()
    (receipts / f"{receipt['receipt_id']}.json").write_text(json.dumps(receipt), encoding="utf-8")
    if not standalone:
        chain = SessionReceiptChain("codex-task-127")
        chain.append(receipt, created_at=1_700_000_000)
        sessions = root / "sessions" / "task-127"
        sessions.mkdir(parents=True)
        chain.write_json(sessions / "session_chain.json")
    return receipt


def test_context_session_index_surfaces_receipts_context_ring_and_honest_cost(tmp_path):
    receipt = _write_session(tmp_path)
    index = ContextSessionIndex([tmp_path])

    listing = index.list_sessions(query="auth")
    assert listing["total"] == 1
    summary = listing["sessions"][0]
    assert summary["session_id"] == "codex-task-127"
    assert summary["selected_tokens"] == 400
    assert summary["omitted_tokens"] == 600
    assert summary["selection_ratio"] == 0.4
    assert summary["model"]["model"] == "openai/gpt-5.6-sol"
    assert summary["model"]["context_window"] == 1_050_000
    assert summary["model"]["context_input_estimate_usd"] == 0.002
    assert summary["model"]["request_estimate_usd"] is None

    detail = index.get_session(summary["key"])
    assert detail is not None
    assert detail["integrity"]["valid"] is True
    view = detail["receipts"][0]
    assert view["receipt_id"] == receipt["receipt_id"]
    assert len(view["selected"][0]["excerpt"]) <= 601
    assert view["omitted"][0]["recoverable"] is True
    assert view["integrity"]["valid"] is True


def test_standalone_and_corrupt_receipts_degrade_visibly(tmp_path):
    receipt = _write_session(tmp_path, standalone=True)
    receipt["query"] = "tampered"
    receipt_path = tmp_path / "receipts" / f"{receipt['receipt_id']}.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (tmp_path / "receipts" / "broken-receipt.json").write_text("{bad", encoding="utf-8")

    index = ContextSessionIndex([tmp_path])
    listing = index.list_sessions()

    assert listing["total"] == 1
    assert listing["diagnostics"]
    detail = index.get_session(listing["sessions"][0]["key"])
    assert detail is not None
    assert detail["integrity"]["valid"] is False
    assert "mismatch" in " ".join(detail["integrity"]["issues"])


def test_dashboard_context_session_api(tmp_path, monkeypatch):
    _write_session(tmp_path)
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    server = HTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(f"{base}/api/context/sessions?q=auth", timeout=5) as response:
            listing = json.loads(response.read())
        assert listing["total"] == 1
        key = listing["sessions"][0]["key"]
        with urllib.request.urlopen(f"{base}/api/context/sessions/{key}", timeout=5) as response:
            detail = json.loads(response.read())
        assert detail["summary"]["session_id"] == "codex-task-127"
        assert detail["receipts"][0]["omitted_count"] == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
