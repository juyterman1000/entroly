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
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
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


def _read_response(proc, timeout=30):
    """Read a newline-delimited JSON-RPC response.

    Uses a background thread to prevent blocking forever if the server
    doesn't respond (the previous implementation's readline() could block
    indefinitely in the kernel, making the Python-level timeout useless).

    Reads lines until a JSON-RPC message arrives, skipping any non-JSON
    output (e.g. a startup banner that leaked to stdout). The longer default
    timeout absorbs slow cold-starts on loaded CI runners: the server's
    import graph plus Rust-engine load can take well over 10s there, even
    though it responds in ~3s on a warm machine.
    """
    responses = getattr(proc, "_entroly_responses", None)
    if responses is None:
        raise RuntimeError("MCP response pump was not initialized")
    try:
        return responses.get(timeout=timeout)
    except queue.Empty:
        return None


def _start_output_pumps(proc):
    """Drain stdout and stderr continuously so the server cannot block on a pipe."""
    responses = queue.Queue()
    stderr_tail = deque(maxlen=200)
    proc._entroly_responses = responses

    def _read_stdout():
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if isinstance(msg, dict) and ("result" in msg or "error" in msg):
                    responses.put(msg)
        except (OSError, ValueError):
            return

    def _drain_stderr():
        try:
            for line in proc.stderr:
                stderr_tail.append(line.rstrip())
        except (OSError, ValueError):
            return

    threading.Thread(target=_read_stdout, daemon=True).start()
    threading.Thread(target=_drain_stderr, daemon=True).start()
    return stderr_tail


@pytest.fixture(scope="module")
def mcp_server():
    """Start the MCP server as a subprocess.

    Launched in an empty scratch directory, not the repo root.
    ``create_mcp_server()`` scans its working directory at startup, so
    starting in a large tree (the repo, or a CI checkout) makes cold-start
    scale with file count — measured ~1.6s in an empty dir vs ~9s+ in this
    repo, enough to blow past the response window on a loaded CI runner.
    This is a protocol test; it must not depend on the cwd's size.
    """
    scratch = tempfile.mkdtemp(prefix="entroly-mcp-test-")
    proc = subprocess.Popen(
        [sys.executable, "-m", "entroly.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=scratch,
        env={
            **os.environ,
            # This fixture verifies the local stdio protocol. Inheriting an
            # opt-in federation setting can add unrelated startup work and
            # make requests depend on external state.
            "ENTROLY_FEDERATION": "0",
            "ENTROLY_NO_DOCKER": "1",
        },
    )
    stderr_tail = _start_output_pumps(proc)
    time.sleep(2)  # Give it time to start

    # A startup crash is a product failure, never an optional capability.
    # Skipping here used to let a completely broken MCP server produce a green
    # test suite, destroying the value of this integration test.
    if proc.poll() is not None:
        shutil.rmtree(scratch, ignore_errors=True)
        pytest.fail(
            "MCP server failed to start; this must fail closed, not skip. "
            f"stderr={chr(10).join(stderr_tail)[-2000:]!r}"
        )

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    shutil.rmtree(scratch, ignore_errors=True)


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
        assert "entroly_retrieve" in tool_names, (
            f"Expected reversible CCR tool, got: {tool_names}"
        )
        for continuity_tool in ("work_state", "work_resume", "work_handoff"):
            assert continuity_tool in tool_names, (
                f"Expected Work Graph continuity tool {continuity_tool}, got: {tool_names}"
            )
        for receipt_tool in (
            "create_context_receipt",
            "create_context_receipt_from_path",
            "render_context_receipt",
            "explain_receipt_omission",
        ):
            assert receipt_tool in tool_names, (
                f"Expected Context Receipt tool {receipt_tool}, got: {tool_names}"
            )


def test_mcp_retrieve_round_trip(mcp_server):
    """Store through MCP and lazily retrieve the exact original source."""
    original = "def login(token):\n    return validate(token)\n"
    _send_jsonrpc(mcp_server, "tools/call", {
        "name": "remember_fragment",
        "arguments": {
            "content": original,
            "source": "file:auth.py",
        },
    }, id=3)
    stored = _read_response(mcp_server)
    assert stored is not None and "result" in stored

    _send_jsonrpc(mcp_server, "tools/call", {
        "name": "entroly_retrieve",
        "arguments": {
            "source_or_handle": "file:auth.py",
        },
    }, id=4)
    retrieved = _read_response(mcp_server)
    assert retrieved is not None and "result" in retrieved
    payload = json.loads(retrieved["result"]["content"][0]["text"])
    assert payload["retrieval_handle"].startswith("ccr:")
    assert payload["original_content"] == original
