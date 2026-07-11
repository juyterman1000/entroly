from __future__ import annotations

import io
import json
from pathlib import Path

from entroly.openclaw_bridge import assemble, handle_request, serve


def _long_text(prefix: str, words: int = 500) -> str:
    return prefix + " " + " ".join(f"token{i}" for i in range(words))


def test_health_contract():
    result = handle_request({"operation": "health"})
    assert result == {
        "schema_version": "entroly.openclaw.bridge.v1",
        "ok": True,
        "status": "ready",
    }


def test_assemble_passes_through_context_within_budget(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    result = assemble(
        {
            "operation": "assemble",
            "session_id": "session/one",
            "messages": messages,
            "token_budget": 100,
            "receipt_dir": str(tmp_path),
        }
    )
    assert result["messages"] == messages
    assert result["changed"] is False
    assert result["receipt_path"]
    receipt_path = tmp_path / f"session-one-{result['receipt_id']}.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["changed"] is False
    assert receipt["local_only"] is True
    assert receipt["recovery_source"] == "openclaw_transcript_unmodified"
    assert receipt["message_decisions"][0]["action"] == "preserved"


def test_assemble_compresses_old_text_and_preserves_protected_messages(tmp_path):
    system = {"role": "system", "content": _long_text("SYSTEM", 120)}
    old_user = {"role": "user", "content": _long_text("OLD", 900)}
    structured = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "call-1", "name": "read"}],
    }
    latest = {"role": "user", "content": "Fix the authentication bug."}
    messages = [system, old_user, structured, latest]
    result = assemble(
        {
            "operation": "assemble",
            "session_id": "s2",
            "messages": messages,
            "token_budget": 700,
            "preserve_last_n": 1,
            "receipt_dir": str(tmp_path),
            "distill": False,
        }
    )
    assert result["changed"] is True
    assert result["messages"][0] == system
    assert result["messages"][2] == structured
    assert result["messages"][3] == latest
    assert len(result["messages"][1]["content"]) < len(old_user["content"])
    assert result["tokens_saved"] > 0
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["message_decisions"][1]["action"] == "compressed"


def test_protected_context_over_budget_fails_open_with_warning(tmp_path):
    messages = [{"role": "system", "content": _long_text("SYSTEM", 500)}]
    result = assemble(
        {
            "operation": "assemble",
            "session_id": "s3",
            "messages": messages,
            "token_budget": 10,
            "receipt_dir": str(tmp_path),
        }
    )
    assert result["messages"] == messages
    assert result["changed"] is False
    assert any("exact original context" in warning for warning in result["warnings"])


def test_jsonl_server_correlates_success_and_error(tmp_path):
    requests = [
        {"request_id": "1", "operation": "health"},
        {"request_id": "2", "operation": "unknown"},
    ]
    input_stream = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    output_stream = io.StringIO()
    assert serve(input_stream, output_stream) == 0
    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[0]["request_id"] == "1"
    assert responses[0]["ok"] is True
    assert responses[1]["request_id"] == "2"
    assert responses[1]["ok"] is False
    assert "unsupported operation" in responses[1]["error"]
