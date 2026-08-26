"""Adversarial black-box tests for the installed Entroly entry point.

These tests deliberately avoid ``create_mcp_server`` and other internal helpers.
They launch the same ``entroly`` console command documented for Claude Code,
Codex, and generic MCP clients, then exercise the JSON-RPC wire contract.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from entroly import __version__


def _installed_entroly_executable() -> str:
    """Return the real installed console script without requiring activation.

    ``python -m venv .venv`` followed by ``.venv/bin/pytest`` is a valid way to
    run an isolated environment, but it does not add ``.venv/bin`` to the shell
    ``PATH``.  The console script is still installed beside the active Python
    interpreter, so use that exact script as the bounded fallback.  This keeps
    the dogfood test black-box and still fails when the user-facing entry point
    was not installed.
    """

    script_name = "entroly.exe" if os.name == "nt" else "entroly"
    candidate = Path(sys.executable).with_name(script_name)
    if candidate.is_file():
        return str(candidate)

    executable = shutil.which("entroly")
    if executable:
        return executable

    assert candidate.is_file(), (
        "The documented `entroly` console entry point is not installed. "
        f"PATH lookup failed and no script exists beside {sys.executable!r}."
    )
    return str(candidate)


class MCPProcess:
    """Small JSON-RPC harness with bounded reads and useful crash diagnostics.

    Stdout and stderr are drained continuously. This is important on Windows,
    where a child can block forever once an unread pipe fills. Responses are
    indexed by request id rather than removed and requeued, so a burst of
    out-of-order replies cannot starve the response currently being awaited.
    """

    _MAX_DIAGNOSTIC_LINES = 200

    def __init__(self, cwd: Path, env: dict[str, str] | None = None) -> None:
        executable = _installed_entroly_executable()

        child_env = {
            **os.environ,
            "ENTROLY_NO_DOCKER": "1",
            "ENTROLY_DISABLE_UPDATE_CHECK": "1",
            "PYTHONUNBUFFERED": "1",
        }
        if env:
            child_env.update(env)

        self.proc = subprocess.Popen(
            [executable],
            cwd=cwd,
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None

        self._condition = threading.Condition()
        self._responses: dict[int, dict[str, Any]] = {}
        self._notifications: deque[dict[str, Any]] = deque(
            maxlen=self._MAX_DIAGNOSTIC_LINES
        )
        self._stdout_noise: deque[str] = deque(maxlen=self._MAX_DIAGNOSTIC_LINES)
        self._stderr_tail: deque[str] = deque(maxlen=self._MAX_DIAGNOSTIC_LINES)
        self._stdout_reader = threading.Thread(
            target=self._read_stdout,
            daemon=True,
            name="entroly-dogfood-stdout",
        )
        self._stderr_reader = threading.Thread(
            target=self._read_stderr,
            daemon=True,
            name="entroly-dogfood-stderr",
        )
        self._stdout_reader.start()
        self._stderr_reader.start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                with self._condition:
                    self._stdout_noise.append(line)
                    self._condition.notify_all()
                continue
            if not isinstance(message, dict):
                with self._condition:
                    self._stdout_noise.append(line)
                    self._condition.notify_all()
                continue

            request_id = message.get("id")
            with self._condition:
                if isinstance(request_id, int):
                    self._responses[request_id] = message
                else:
                    self._notifications.append(message)
                self._condition.notify_all()

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            with self._condition:
                self._stderr_tail.append(line)
                self._condition.notify_all()

    def send(
        self,
        request_id: int | None,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        assert self.proc.stdin is not None
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if request_id is not None:
            payload["id"] = request_id
        if params is not None:
            payload["params"] = params
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def send_raw(self, payload: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(payload + "\n")
        self.proc.stdin.flush()

    def wait_for(self, request_id: int, timeout: float = 45.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            with self._condition:
                response = self._responses.pop(request_id, None)
                if response is not None:
                    return response
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=min(0.25, remaining))

            if self.proc.poll() is not None:
                pytest.fail(
                    self._diagnostic(
                        f"server exited while waiting for id={request_id}"
                    )
                )

        pytest.fail(self._diagnostic(f"timed out waiting for id={request_id}"))

    def initialize(self) -> dict[str, Any]:
        self.send(
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "deep-dogfood", "version": "1"},
            },
        )
        response = self.wait_for(1)
        assert "result" in response, self._diagnostic(f"initialize failed: {response}")
        self.send(None, "notifications/initialized", {})
        return response

    def assert_protocol_clean(self) -> None:
        with self._condition:
            noise = list(self._stdout_noise)
        assert not noise, "Non-JSON output polluted MCP stdout: " + repr(noise[:10])

    def _diagnostic(self, reason: str) -> str:
        with self._condition:
            stdout_noise = list(self._stdout_noise)[-10:]
            stderr = "\n".join(self._stderr_tail)[-4000:]
            pending_ids = sorted(self._responses)[:30]
            notifications = list(self._notifications)[-5:]
        return (
            f"{reason}; returncode={self.proc.poll()}; "
            f"pending_ids={pending_ids!r}; "
            f"notifications={notifications!r}; "
            f"stdout_noise={stdout_noise!r}; stderr={stderr!r}"
        )

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        self._stdout_reader.join(timeout=2)
        self._stderr_reader.join(timeout=2)

    def __enter__(self) -> "MCPProcess":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _tool_payload(response: dict[str, Any]) -> dict[str, Any]:
    assert "result" in response, response
    result = response["result"]
    assert result.get("isError") is not True, result
    content = result.get("content") or []
    assert content and content[0].get("type") == "text", result
    return json.loads(content[0]["text"])


def test_console_version_matches_imported_package() -> None:
    executable = _installed_entroly_executable()
    completed = subprocess.run(
        [executable, "--version"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
        env={**os.environ, "ENTROLY_DISABLE_UPDATE_CHECK": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    combined = completed.stdout + completed.stderr
    assert __version__ in combined, (
        f"entry point and imported package disagree: expected {__version__!r}, "
        f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
    )


def test_installed_entrypoint_survives_protocol_and_client_errors(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    with MCPProcess(tmp_path, {"ENTROLY_DIR": str(runtime)}) as mcp:
        initialized = mcp.initialize()
        assert "serverInfo" in initialized["result"]

        mcp.send(2, "tools/list", {})
        listed = mcp.wait_for(2)
        assert "result" in listed, listed
        names = {tool["name"] for tool in listed["result"].get("tools", [])}
        assert {"optimize_context", "entroly_retrieve"} <= names

        # A hostile client can send malformed JSON. The server should report or
        # log the parse failure, not die and strand every configured MCP client.
        mcp.send_raw("{this is not valid json")
        time.sleep(0.2)

        # Unknown methods are normal JSON-RPC errors and must not poison state.
        mcp.send(3, "entroly/definitely-not-a-method", {})
        unknown = mcp.wait_for(3)
        assert "error" in unknown, unknown

        mcp.send(4, "tools/list", {})
        assert "result" in mcp.wait_for(4)
        assert mcp.proc.poll() is None
        mcp.assert_protocol_clean()


def test_safe_unicode_nul_and_rtl_scripts_round_trip_exactly(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    original = (
        "Arabic: مرحبا | Hebrew: שלום | Hindi: नमस्ते | "
        "emoji: 🧪🔐 | combining: e\u0301 | nul:\u0000:end\nsecond line\n"
    )
    source = "file:unicode-rtl-🧪.txt"

    with MCPProcess(tmp_path, {"ENTROLY_DIR": str(runtime)}) as mcp:
        mcp.initialize()
        mcp.send(
            10,
            "tools/call",
            {
                "name": "remember_fragment",
                "arguments": {"content": original, "source": source},
            },
        )
        stored = _tool_payload(mcp.wait_for(10))
        assert stored.get("status") == "ingested", stored

        mcp.send(
            11,
            "tools/call",
            {
                "name": "entroly_retrieve",
                "arguments": {"source_or_handle": source},
            },
        )
        payload = _tool_payload(mcp.wait_for(11))
        assert payload["original_content"] == original
        assert payload["retrieval_handle"].startswith("ccr:")
        mcp.assert_protocol_clean()


def test_dangerous_unicode_controls_are_rejected_before_storage(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    source = "file:hostile-unicode.txt"
    hostile = "BOM:\ufeff bidi-override:\u202eabc\u202c"

    with MCPProcess(tmp_path, {"ENTROLY_DIR": str(runtime)}) as mcp:
        mcp.initialize()
        mcp.send(
            12,
            "tools/call",
            {
                "name": "remember_fragment",
                "arguments": {"content": hostile, "source": source},
            },
        )
        rejected = _tool_payload(mcp.wait_for(12))
        assert rejected.get("status") == "rejected", rejected
        assert rejected.get("stored") is False, rejected
        assert rejected.get("threats"), rejected

        mcp.send(
            13,
            "tools/call",
            {
                "name": "entroly_retrieve",
                "arguments": {"source_or_handle": source},
            },
        )
        missing = _tool_payload(mcp.wait_for(13))
        assert "error" in missing, missing
        mcp.assert_protocol_clean()


def test_rapid_pipelined_requests_are_not_lost(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    request_ids = list(range(100, 125))

    with MCPProcess(tmp_path, {"ENTROLY_DIR": str(runtime)}) as mcp:
        mcp.initialize()
        for request_id in request_ids:
            mcp.send(request_id, "tools/list", {})

        received = {request_id: mcp.wait_for(request_id) for request_id in request_ids}
        assert set(received) == set(request_ids)
        assert all("result" in response for response in received.values())
        assert mcp.proc.poll() is None
        mcp.assert_protocol_clean()


def test_unusable_home_falls_back_without_crashing(tmp_path: Path) -> None:
    fake_home = tmp_path / "home-is-a-file"
    fake_home.write_text("not a directory", encoding="utf-8")
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()

    env = {
        "HOME": str(fake_home),
        "TMPDIR": str(temp_dir),
    }
    with MCPProcess(tmp_path, env) as mcp:
        initialized = mcp.initialize()
        assert "result" in initialized
        mcp.send(30, "tools/list", {})
        assert "result" in mcp.wait_for(30)
        mcp.assert_protocol_clean()
