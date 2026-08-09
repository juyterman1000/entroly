"""Minimal framed LSP server used only by repository-orchestration tests."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break
        name, value = line.decode("ascii").split(":", 1)
        headers[name.lower().strip()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body)


def send(payload):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def main() -> int:
    if "--hang" in sys.argv:
        time.sleep(10)
        return 0
    if os.getenv("ENTROLY_TEST_SECRET"):
        return 19
    root = None
    source_uri = None
    while True:
        message = read_message()
        if message is None:
            return 0
        method = message.get("method")
        if method == "initialize":
            root_uri = message["params"]["rootUri"]
            parsed = urlparse(root_uri)
            raw_path = unquote(parsed.path)
            if os.name == "nt" and raw_path.startswith("/") and raw_path[2:3] == ":":
                raw_path = raw_path[1:]
            root = Path(raw_path)
            send({
                "jsonrpc": "2.0", "id": 900,
                "method": "workspace/configuration",
                "params": {"items": [{"section": "fake"}]},
            })
            response = read_message()
            if response is None or response.get("id") != 900:
                return 20
            if "--many-messages" in sys.argv:
                for position in range(32):
                    send({
                        "jsonrpc": "2.0",
                        "method": "window/logMessage",
                        "params": {"type": 3, "message": str(position)},
                    })
            encoding = "utf-8" if "--bad-encoding" in sys.argv else "utf-16"
            send({
                "jsonrpc": "2.0", "id": message["id"],
                "result": {"capabilities": {"positionEncoding": encoding}},
            })
            if "--stop-reading-after-init" in sys.argv:
                time.sleep(10)
        elif method == "textDocument/didOpen":
            source_uri = message["params"]["textDocument"]["uri"]
        elif method == "textDocument/references":
            assert root is not None and source_uri is not None
            if "--oversized-output" in sys.argv:
                sys.stdout.buffer.write(b"Content-Length: 4096\r\n\r\n")
                sys.stdout.buffer.flush()
                continue
            locations = [
                {
                    "uri": source_uri,
                    "range": {
                        "start": {"line": 0, "character": 4},
                        "end": {"line": 0, "character": 11},
                    },
                },
                {
                    "uri": (root / "caller.py").resolve().as_uri(),
                    "range": {
                        "start": {"line": 2, "character": 11},
                        "end": {"line": 2, "character": 18},
                    },
                },
                {
                    "uri": "https://invalid.example/outside.py",
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 7},
                    },
                },
            ]
            send({"jsonrpc": "2.0", "id": message["id"], "result": locations})
        elif method == "shutdown":
            send({"jsonrpc": "2.0", "id": message["id"], "result": None})
        elif method == "exit":
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
