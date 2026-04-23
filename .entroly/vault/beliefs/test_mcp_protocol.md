---
claim_id: fb1c2928-7035-4f28-9503-b37f149305a4
entity: test_mcp_protocol
status: inferred
confidence: 0.75
sources:
  - tests\test_mcp_protocol.py:89
  - tests\test_mcp_protocol.py:120
  - tests\test_mcp_protocol.py:126
  - tests\test_mcp_protocol.py:144
last_checked: 2026-04-23T03:07:07.937156+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_mcp_protocol

**Language:** python
**Lines of code:** 159


## Functions
- `def mcp_server()` — Start the MCP server as a subprocess.
- `def test_mcp_server_starts(mcp_server)` — The MCP server process should be running.
- `def test_mcp_initialize(mcp_server)` — Send initialize request and verify the server responds.
- `def test_mcp_list_tools(mcp_server)` — Request the list of available tools.

## Dependencies
- `errors`
- `json`
- `os`
- `pytest`
- `subprocess`
- `sys`
- `threading`
- `time`
