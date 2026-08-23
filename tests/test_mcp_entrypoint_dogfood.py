"""Independent dogfood tests for the installed ``entroly`` MCP entrypoint.

These tests intentionally avoid importing ``entroly.server`` directly. Users
register the console command, so the console command is the contract we need to
exercise. Startup failures are failures, never skips, and stdout must remain
JSON-RPC clean.
"""

from __future__ import annotations

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
from pathlib import Path

import pytest

from entroly import __version__


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


def _start_transport_pumps(proc: subprocess.Popen[str]) -> None:
    """Continuously drain both pipes for the lifetime of the MCP process.

    A fresh ``readline`` thread per request is unsafe: after a timeout the old
    thread remains alive and can steal a later response. Leaving stderr unread
    is also unsafe because a sufficiently chatty startup can fill the OS pipe
    and block the server before it writes its JSON-RPC response.
    """
    assert proc.stdout is not None
    assert proc.stderr is not None
    received: queue.Queue[tuple[str, object]] = queue.Queue()
    stderr_lines: deque[str] = deque(maxlen=400)
    setattr(proc, "_entroly_received", received)
    setattr(proc, "_entroly_stderr_lines", stderr_lines)

    def stdout_reader() -> None:
        try:
            for line in proc.stdout:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    received.put(("contamination", stripped))
                    continue
                if not isinstance(payload, dict):
                    received.put(("contamination", stripped))
                    continue
                received.put(("json", payload))
        except BaseException as exc:  # pragma: no cover - diagnostic path
            received.put(("reader_error", repr(exc)))
        finally:
            received.put(("eof", None))

    def stderr_reader() -> None:
        try:
            for line in proc.stderr:
                stderr_lines.append(line)
        except (OSError, ValueError):  # pragma: no cover - shutdown race
            return

    threading.Thread(target=stdout_reader, daemon=True).start()
    threading.Thread(target=stderr_reader, daemon=True).start()


def _stderr_tail(proc: subprocess.Popen[str], limit: int = 4000) -> str:
    lines = getattr(proc, "_entroly_stderr_lines", ())
    return "".join(lines)[-limit:]


def _read_response(
    proc: subprocess.Popen[str],
    request_id: int,
    *,
    timeout: float = 45.0,
) -> dict:
    """Read one matching JSON-RPC response and reject stdout contamination."""

    received = getattr(proc, "_entroly_received", None)
    assert isinstance(received, queue.Queue), "transport pumps were not started"
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(
                "timed out waiting for MCP response "
                f"id={request_id}; returncode={proc.poll()}; "
                f"stderr={_stderr_tail(proc)!r}"
            )
        try:
            kind, value = received.get(timeout=remaining)
        except queue.Empty:
            continue

        if kind == "json":
            assert isinstance(value, dict)
            if value.get("id") == request_id:
                return value
            # Notifications and valid log messages can appear between replies.
            continue
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


def _tool_payload(response: dict) -> dict:
    assert "result" in response, response
    result = response["result"]
    assert result.get("isError") is not True, result
    content = result.get("content") or []
    assert content and content[0].get("type") == "text", result
    return json.loads(content[0]["text"])


@pytest.fixture(scope="module")
def installed_mcp() -> subprocess.Popen[str]:
    script_name = "entroly.exe" if os.name == "nt" else "entroly"
    adjacent = Path(sys.executable).with_name(script_name)
    executable = str(adjacent) if adjacent.is_file() else shutil.which("entroly")
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
        _start_transport_pumps(proc)

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
        setattr(proc, "_entroly_initialized", initialized)
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
    initialized = getattr(installed_mcp, "_entroly_initialized")
    assert initialized["result"]["serverInfo"]["version"] == __version__
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
        "work_state",
        "work_resume",
        "work_handoff",
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


def test_optimize_context_payload_is_bounded_and_internally_consistent(
    installed_mcp: subprocess.Popen[str],
) -> None:
    """Serialization safety must hold across every selector backend."""
    optimized = _call_tool(
        installed_mcp,
        5,
        "optimize_context",
        {"query": "root_wiring_probe needle-root-wiring", "token_budget": 8000},
    )
    payload = _tool_payload(optimized)
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert len(encoded) < 1_000_000
    assert not ("selected" in payload and "selected_fragments" in payload)
    assert "provenance" in payload

    selected = payload.get("selected_fragments") or payload.get("selected") or []
    assert payload.get("selected_count") == len(selected)
    stats = payload.get("optimization_stats") or {}
    assert stats.get("selected_count") == len(selected)

    # ``selected_count`` is the normalized cross-backend contract. Some
    # selectors additionally expose the full corpus count; when present, it
    # must be consistent, but a fallback backend must not fail solely because
    # it omits optional corpus-size diagnostics.
    total_fragments = payload.get("total_fragments")
    if total_fragments is not None:
        assert int(total_fragments) >= len(selected)


def test_explicitly_pinned_evidence_is_selected_on_every_backend(
    installed_mcp: subprocess.Popen[str],
) -> None:
    """Retrieval success gets its own non-vacuous cross-backend contract."""
    source = "dogfood:pinned-native-selection.py"
    content = (
        "def pinned_native_selection_anchor():\n"
        "    return 'cross-backend-pinned-evidence-7f93'\n"
    )
    stored = _tool_payload(
        _call_tool(
            installed_mcp,
            8,
            "remember_fragment",
            {
                "content": content,
                "source": source,
                "is_pinned": True,
            },
        )
    )
    assert stored.get("status") == "ingested", stored

    payload = _tool_payload(
        _call_tool(
            installed_mcp,
            9,
            "optimize_context",
            {
                "query": "pinned_native_selection_anchor cross-backend-pinned-evidence-7f93",
                "token_budget": 1024,
            },
        )
    )
    selected = payload.get("selected_fragments") or payload.get("selected") or []
    assert payload.get("selected_count") == len(selected)
    assert len(selected) > 0
    assert any(fragment.get("source") == source for fragment in selected), selected
    assert any(
        "cross-backend-pinned-evidence-7f93" in fragment.get("content", "")
        for fragment in selected
    )


def test_unknown_tool_fails_without_killing_server(
    installed_mcp: subprocess.Popen[str],
) -> None:
    unknown = _call_tool(installed_mcp, 6, "definitely_not_an_entroly_tool", {})
    assert "error" in unknown or unknown.get("result", {}).get("isError") is True
    assert installed_mcp.poll() is None

    _write_request(installed_mcp, "tools/list", {}, request_id=7)
    follow_up = _read_response(installed_mcp, 7)
    assert "result" in follow_up
