"""Protocol tests for Entroly MCP prompts and read-only resources."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _send(proc, method: str, params: dict | None = None, request_id: int | None = 1) -> None:
    message: dict[str, object] = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        message["id"] = request_id
    if params is not None:
        message["params"] = params
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _read(proc, timeout: float = 30.0) -> dict | None:
    result: list[dict | None] = [None]

    def reader() -> None:
        while True:
            line = proc.stdout.readline()
            if not line:
                return
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict) and ("result" in message or "error" in message):
                result[0] = message
                return

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    return result[0]


@pytest.fixture(scope="module")
def initialized_server():
    scratch = tempfile.mkdtemp(prefix="entroly-mcp-discovery-")
    python_path = os.pathsep.join(
        value for value in (str(ROOT), os.environ.get("PYTHONPATH", "")) if value
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "entroly.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=scratch,
        env={
            **os.environ,
            "ENTROLY_NO_DOCKER": "1",
            "PYTHONPATH": python_path,
        },
    )
    time.sleep(2)
    if proc.poll() is not None:
        stderr = proc.stderr.read()
        shutil.rmtree(scratch, ignore_errors=True)
        pytest.fail(f"MCP server failed to start: {stderr[:1000]}")

    _send(
        proc,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "discovery-test", "version": "1.0"},
        },
        1,
    )
    initialized = _read(proc)
    assert initialized is not None and "result" in initialized
    _send(proc, "notifications/initialized", request_id=None)
    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    shutil.rmtree(scratch, ignore_errors=True)


def test_prompts_are_public_and_useful(initialized_server):
    _send(initialized_server, "prompts/list", {}, 2)
    response = _read(initialized_server)
    assert response is not None and "result" in response
    names = {prompt["name"] for prompt in response["result"].get("prompts", [])}
    assert {"context_optimization_workflow", "context_verification_workflow"} <= names

    _send(
        initialized_server,
        "prompts/get",
        {
            "name": "context_optimization_workflow",
            "arguments": {"task": "Fix authentication race", "token_budget": "8192"},
        },
        3,
    )
    prompt = _read(initialized_server)
    assert prompt is not None and "result" in prompt
    rendered = json.dumps(prompt["result"])
    assert "optimize_context" in rendered
    assert "Fix authentication race" in rendered


def test_resources_are_bounded_and_secret_free(initialized_server):
    _send(initialized_server, "resources/list", {}, 4)
    response = _read(initialized_server)
    assert response is not None and "result" in response
    uris = {resource["uri"] for resource in response["result"].get("resources", [])}
    assert {"entroly://health", "entroly://stats"} <= uris

    _send(initialized_server, "resources/read", {"uri": "entroly://health"}, 5)
    health_response = _read(initialized_server)
    assert health_response is not None and "result" in health_response
    contents = health_response["result"].get("contents", [])
    assert contents
    health = json.loads(contents[0]["text"])
    assert health["status"] == "ok"
    assert health["capabilities"]["resources"] is True
    encoded = json.dumps(health)
    assert "api_key" not in encoded.lower()
    assert "token=" not in encoded.lower()
    assert len(encoded.encode("utf-8")) < 16_384
