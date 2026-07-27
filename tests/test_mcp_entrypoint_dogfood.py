"""Independent dogfood tests for the installed ``entroly`` MCP entrypoint.

These tests intentionally avoid importing ``entroly.server`` directly.  Users
register the console command, so the console command is the contract we need to
exercise.  Startup failures are failures, never skips, and stdout must remain
JSON-RPC clean.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest


def _write_request(
    proc: subprocess.Popen[str],
    method: str,
    params: dict | None = None,
    *,
    request_id: int | None,
) -> None:
    message: dict[str, object] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    if request_id is not None:
        message["id"] = request_id
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def _stderr_tail(proc: subprocess.Popen[str], limit: int = 4000) -> str:
    if proc.stderr is None:
        return ""
    if proc.poll() is None:
        return ""
    try:
        return proc.stderr.read()[-limit:]
    except OSError:
        return ""


def _read_response(
    proc: subprocess.Popen[str],
    request_id: int,
    *,
    timeout: float = 45.0,
) -> dict:
    """Read one matching JSON-RPC response and reject stdout contamination."""

    assert proc.stdout is not None
    received: queue.Queue[tuple[str, object]] = queue.Queue()

    def reader() -> None:
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    received.put(("eof", None))
                    return
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    received.put(("contamination", stripped))
                    return
                if not isinstance(payload, dict):
                    received.put(("contamination", stripped))
                    return
                if payload.get("id") == request_id:
                    received.put(("response", payload))
                    return
                # Notifications and log messages are allowed only when they are
                # valid JSON-RPC objects. Continue until our response arrives.
        except BaseException as exc:  # pragma: no cover - diagnostic path
            received.put(("reader_error", repr(exc)))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        kind, value = received.get(timeout=timeout)
    except queue.Empty:
        pytest.fail(
            "timed out waiting for MCP response "
            f"id={request_id}; returncode={proc.poll()}; stderr={_stderr_tail(proc)!r}"
        )

    if kind == "response":
        assert isinstance(value, dict)
        return value
    if kind == "contamination":
        pytest.fail(f"non-JSON output leaked onto MCP stdout: {value!r}")
    if kind == "eof":
        pytest.fail(
            "MCP entrypoint closed stdout before responding; "
            f"returncode={proc.poll()}; stderr={_stderr_tail(proc)!r}"
        )
    pytest.fail(f"MCP response reader failed: {value!r}")


def _call_tool(
    proc: subprocess.Popen[str],
    request_id: int,
    name: str,
    arguments: dict,
) -> dict:
    _write_request(
        proc,
        "tools/call",
        {"name": name, "arguments": arguments},
        request_id=request_id,
    )
    return _read_response(proc, request_id)


@pytest.fixture(scope="module")
def installed_mcp() -> subprocess.Popen[str]:
    executable = shutil.which("entroly")
    assert executable, "installed console script `entroly` was not found on PATH"

    with tempfile.TemporaryDirectory(prefix="entroly-entrypoint-dogfood-") as raw:
        scratch = Path(raw)
        home = scratch / "home"
        state = scratch / "state"
        repo = scratch / "repo"
        home.mkdir()
        state.mkdir()
        repo.mkdir()

        # Build enough real files to exercise startup indexing and the response
        # compaction boundary that previously overflowed MCP clients.
        for index in range(420):
            (repo / f"module_{index:03d}.py").write_text(
                "def root_wiring_probe():\n"
                f"    return 'needle-root-wiring-{index:03d}'\n",
                encoding="utf-8",
            )

        env = {
            **os.environ,
            "HOME": str(home),
            "ENTROLY_DIR": str(state),
            "ENTROLY_NO_DOCKER": "1",
            "ENTROLY_DISABLE_UPDATE_CHECK": "1",
            "ENTROLY_TASK_DREAM": "0",
            "ENTROLY_EXECUTE_PROMOTED_SKILLS": "0",
        }
        proc = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=repo,
            env=env,
        )

        _write_request(
            proc,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "entroly-deep-dogfood", "version": "1"},
            },
            request_id=1,
        )
        initialized = _read_response(proc, 1)
        assert "result" in initialized, initialized
        assert initialized["result"].get("serverInfo"), initialized
        _write_request(proc, "notifications/initialized", {}, request_id=None)

        yield proc

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def test_documented_entrypoint_lists_required_tools(
    installed_mcp: subprocess.Popen[str],
) -> None:
    _write_request(installed_mcp, "tools/list", {}, request_id=2)
    response = _read_response(installed_mcp, 2)
    assert "result" in response, response
    tools = response["result"].get("tools", [])
    names = {tool.get("name") for tool in tools}
    assert {
        "remember_fragment",
        "entroly_retrieve",
        "optimize_context",
        "get_stats",
        "analyze_codebase_health",
        "smart_read",
    } <= names


def test_documented_entrypoint_exactly_recovers_large_unicode_payload(
    installed_mcp: subprocess.Popen[str],
) -> None:
    original = (
        "# exact recovery probe π 🚀\n"
        "def validate(token):\n    return token == 'needle'\n"
        + ("payload-line-αβγ\n" * 4096)
    )
    stored = _call_tool(
        installed_mcp,
        3,
        "remember_fragment",
        {"content": original, "source": "dogfood:unicode-large.py"},
    )
    assert "result" in stored, stored

    retrieved = _call_tool(
        installed_mcp,
        4,
        "entroly_retrieve",
        {"source_or_handle": "dogfood:unicode-large.py"},
    )
    assert "result" in retrieved, retrieved
    text = retrieved["result"]["content"][0]["text"]
    payload = json.loads(text)
    assert payload["original_content"] == original
    assert payload["retrieval_handle"].startswith("ccr:")


def test_optimize_context_is_bounded_and_not_alias_duplicated(
    installed_mcp: subprocess.Popen[str],
) -> None:
    optimized = _call_tool(
        installed_mcp,
        5,
        "optimize_context",
        {"query": "needle root wiring probe", "token_budget": 8000},
    )
    assert "result" in optimized, optimized
    text = optimized["result"]["content"][0]["text"]
    assert len(text.encode("utf-8")) < 1_000_000
    payload = json.loads(text)
    assert not ("selected" in payload and "selected_fragments" in payload)
    assert "provenance" in payload
    assert payload.get("selected_count", 0) > 0


def test_unknown_tool_fails_without_killing_server(
    installed_mcp: subprocess.Popen[str],
) -> None:
    unknown = _call_tool(installed_mcp, 6, "definitely_not_an_entroly_tool", {})
    assert "error" in unknown or unknown.get("result", {}).get("isError") is True
    assert installed_mcp.poll() is None

    _write_request(installed_mcp, "tools/list", {}, request_id=7)
    follow_up = _read_response(installed_mcp, 7)
    assert "result" in follow_up
