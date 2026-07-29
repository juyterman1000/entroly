from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path


def _call(
    proc: subprocess.Popen[str],
    stdout_lines: queue.Queue[str | None],
    stderr_tail: deque[str],
    request_id: int,
    method: str,
    params: dict,
) -> dict:
    assert proc.stdin is not None
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    deadline = time.monotonic() + 45
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"artifact MCP timed out; stderr={''.join(stderr_tail)[-4000:]}"
            )
        if proc.poll() is not None:
            raise AssertionError(
                f"artifact MCP exited {proc.returncode}; "
                f"stderr={''.join(stderr_tail)[-4000:]}"
            )
        try:
            line = stdout_lines.get(timeout=min(0.5, remaining))
        except queue.Empty:
            continue
        if line is None:
            raise AssertionError(
                f"artifact MCP stdout closed; stderr={''.join(stderr_tail)[-4000:]}"
            )
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"artifact MCP stdout pollution: {line!r}") from exc
        if candidate.get("id") == request_id:
            return candidate


def run(bin_dir: Path, expected_version: str) -> None:
    os.environ.setdefault("ENTROLY_NO_DOCKER", "1")
    os.environ.setdefault("ENTROLY_DISABLE_UPDATE_CHECK", "1")

    import entroly

    assert entroly.__version__ == expected_version, (
        entroly.__version__,
        expected_version,
    )
    for name in (
        "compress",
        "compress_messages",
        "optimize",
        "verify",
        "create_context_receipt",
        "render_context_receipt",
    ):
        assert callable(getattr(entroly, name, None)), name

    sample = "Unicode evidence مرحبا 🧪\n" * 200
    compressed = entroly.compress(sample, budget=100)
    assert compressed.strip()
    assert len(compressed) <= 400

    binary = bin_dir / ("entroly.exe" if os.name == "nt" else "entroly")
    assert binary.is_file(), binary
    proc = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **os.environ,
            "ENTROLY_NO_DOCKER": "1",
            "ENTROLY_DISABLE_UPDATE_CHECK": "1",
        },
    )
    assert proc.stdout is not None and proc.stderr is not None
    stdout_lines: queue.Queue[str | None] = queue.Queue()
    stderr_tail: deque[str] = deque(maxlen=200)

    def read_stdout() -> None:
        try:
            for line in proc.stdout:
                stdout_lines.put(line)
        finally:
            stdout_lines.put(None)

    def read_stderr() -> None:
        for line in proc.stderr:
            stderr_tail.append(line)

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=read_stderr, daemon=True).start()
    try:
        initialized = _call(
            proc,
            stdout_lines,
            stderr_tail,
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "artifact-dogfood", "version": "1"},
            },
        )
        assert "result" in initialized, initialized
        assert "serverInfo" in initialized["result"], initialized

        tools = _call(proc, stdout_lines, stderr_tail, 2, "tools/list", {})
        assert "result" in tools, tools
        names = {tool["name"] for tool in tools["result"].get("tools", [])}
        assert "optimize_context" in names
        assert "entroly_retrieve" in names
        assert "create_context_receipt" in names
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    run(args.bin_dir, args.expected_version)


if __name__ == "__main__":
    main()
