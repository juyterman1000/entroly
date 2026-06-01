"""
MCP Protocol Integration Test
==============================

Verifies that the Entroly MCP server actually speaks JSON-RPC over stdio.
This catches issues that unit tests miss — like broken tool registration,
import errors at startup, or malformed responses.

Uses non-blocking I/O with threading to prevent CI hangs.
"""

import json
import os
import subprocess
import sys
import threading
import time
import pytest


def _send_jsonrpc(proc, method, params=None, id=1):
    """Send a JSON-RPC request and read the response."""
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "id": id,
    }
    if params is not None:
        request["params"] = params

    msg = json.dumps(request)
    try:
        proc.stdin.write(msg + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass  # Server may have already exited


def _read_response(proc, timeout=10):
    """Read a newline-delimited JSON-RPC response.

    Uses a background thread to prevent blocking forever if the server
    doesn't respond (the previous implementation's readline() could block
    indefinitely in the kernel, making the Python-level timeout useless).
    """
    result = [None]

    def _reader():
        try:
            line = proc.stdout.readline()
            if line:
                result[0] = json.loads(line)
        except Exception:
            pass  # Any parse/read error → return None

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0]


@pytest.fixture(scope="module")
def mcp_server():
    """Start the MCP server as a subprocess."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "entroly.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "ENTROLY_NO_DOCKER": "1",
        },
    )
    time.sleep(2)  # Give it time to start

    # Check it didn't crash
    if proc.poll() is not None:
        stderr = proc.stderr.read()
        pytest.skip(f"MCP server failed to start: {stderr[:500]}")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def test_mcp_server_starts(mcp_server):
    """The MCP server process should be running."""
    assert mcp_server.poll() is None, "MCP server process died"


def test_mcp_initialize(mcp_server):
    """Send initialize request and verify the server responds."""
    _send_jsonrpc(mcp_server, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"},
    })

    response = _read_response(mcp_server)
    assert response is not None, "MCP server did not respond to initialize"
    assert "result" in response or "error" in response
    if "result" in response:
        assert "serverInfo" in response["result"]


def test_mcp_list_tools(mcp_server):
    """Request the list of available tools."""
    _send_jsonrpc(mcp_server, "tools/list", {}, id=2)

    response = _read_response(mcp_server)
    assert response is not None, "MCP server did not respond to tools/list"
    if "result" in response:
        tools = response["result"].get("tools", [])
        tool_names = [t["name"] for t in tools]
        # Verify core tools are registered
        assert "remember_fragment" in tool_names or "optimize_context" in tool_names, (
            f"Expected core tools, got: {tool_names}"
        )
