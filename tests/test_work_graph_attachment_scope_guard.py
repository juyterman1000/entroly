from __future__ import annotations


def test_work_graph_tools_are_not_in_main_attachment_scopes_until_main_mcp_exposes_them():
    from entroly.session_attach import TOOL_SCOPES

    all_tools = {tool for tools in TOOL_SCOPES.values() for tool in tools}
    assert all_tools.isdisjoint(
        {"work_state", "work_claim", "work_resume", "work_handoff"}
    )
