"""Adversarial black-box tests for the installed Entroly entry point.

These tests deliberately avoid ``create_mcp_server`` and other internal helpers.
They launch the same ``entroly`` console command documented for Claude Code,
Codex, and generic MCP clients, then exercise the JSON-RPC wire contract.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from entroly import __version__


class MCPProcess:
    """Small JSON-RPC harness with bounded reads and useful crash diagnostics."""

    def __init__(self, cwd: Path, env: dict[str, str] | None = None) -> None:
        executable = shutil.which("entroly")
        assert executable, (
            "The documented `entroly` console entry point is not installed. "
            "Black-box dogfood tests must exercise the user-facing command."
        )

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

        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stdout_noise: list[str] = []
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._stdout_noise.append(line)
                continue
            if isinstance(message, dict):
                self._messages.put(message)
            else:
                self._stdout_noise.append(line)

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
        deferred: list[dict[str, Any]] = []
        try:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    pytest.fail(self._diagnostic(f"server exited while waiting for id={request_id}"))
                remaining = max(0.01, deadline - time.monotonic())
                try:
                    message = self._messages.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    continue
                if message.get("id") == request_id:
                    return message
                deferred.append(message)
        finally:
            for message in deferred:
                self._messages.put(message)
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
        assert not self._stdout_noise, (
            "Non-JSON output polluted MCP stdout: " + repr(self._stdout_noise[:10])
        )

    def _diagnostic(self, reason: str) -> str:
        stderr = ""
        if self.proc.poll() is not None and self.proc.stderr is not None:
            try:
                stderr = self.proc.stderr.read()
            except OSError:
                pass
        return (
            f"{reason}; returncode={self.proc.poll()}; "
            f"stdout_noise={self._stdout_noise[:10]!r}; stderr={stderr[-4000:]!r}"
        )

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

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
    executable = shutil.which("entroly")
    assert executable
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


def test_unicode_nul_and_rtl_round_trip_exactly(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    original = (
        "BOM:\ufeff | Arabic: مرحبا | Hebrew: שלום | Hindi: नमस्ते | "
        "emoji: 🧪🔐 | combining: e\u0301 | bidi: \u202eabc\u202c | "
        "nul:\u0000:end\nsecond line\n"
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
        stored = mcp.wait_for(10)
        assert "result" in stored, stored

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
