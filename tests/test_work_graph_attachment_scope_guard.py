from __future__ import annotations


def test_primary_attachment_scope_exposes_complete_work_graph_loop():
    from entroly.session_attach import DEFAULT_SCOPES, TOOL_SCOPES

    expected = {
        "work_state",
        "work_claim",
        "work_resume",
        "work_handoff",
        "work_record_context",
        "work_record_memory",
        "work_record_execution",
    }
    assert TOOL_SCOPES["continuity"] == expected
    assert "continuity" in DEFAULT_SCOPES
