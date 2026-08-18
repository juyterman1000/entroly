from __future__ import annotations


def test_work_graph_attachment_tools_are_not_registered_before_main_mcp_wiring():
    # The dedicated entroly-work-graph-mcp server is production code. Do not add
    # its tool names to session_attach scopes until the monolithic main MCP
    # server registers the same tools in the same change; otherwise the existing
    # scope-parity gate correctly rejects default attachments.
    from entroly.session_attach import TOOL_SCOPES

    all_tools = {tool for tools in TOOL_SCOPES.values() for tool in tools}
    assert not ({"work_state", "work_claim", "work_resume", "work_handoff"} & all_tools)
