from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPORT = Path("copilot-composite-e2e.json")
LOG = Path("copilot-composite-e2e.log")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_json(url: str, timeout: float = 45.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


def run(command: list[str], *, env: dict[str, str], cwd: Path, timeout: int = 150) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
            "timed_out": True,
        }


def advertised_tool_names(body: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in body.get("tools", []):
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
    return names


def has_tool_result(body: dict[str, Any]) -> bool:
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            return True
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {
                    "tool_result",
                    "function_call_output",
                }:
                    return True
    return False


class CopilotMockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    observations: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "entroly-e2e-model",
                            "object": "model",
                            "created": 1,
                            "owned_by": "entroly-e2e",
                        }
                    ],
                },
            )
            return
        self._json(404, {"error": {"message": f"unknown GET {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        body = json.loads(raw.decode("utf-8"))
        observation = {
            "path": self.path,
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "body": body,
            "advertised_tools": advertised_tool_names(body),
            "has_tool_result": has_tool_result(body),
        }
        self.observations.append(observation)
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": {"message": f"unexpected path {self.path}"}})
            return

        target = next(
            (name for name in observation["advertised_tools"] if "get_stats" in name),
            None,
        )
        if target is not None and not observation["has_tool_result"]:
            self._send_tool_call(body, target)
            return
        self._send_final(body)

    def _send_sse(self, chunks: list[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _send_tool_call(self, body: dict[str, Any], target: str) -> None:
        model = str(body.get("model") or "entroly-e2e-model")
        call = {
            "index": 0,
            "id": "call_entroly_get_stats",
            "type": "function",
            "function": {"name": target, "arguments": "{}"},
        }
        if not body.get("stream"):
            self._json(
                200,
                {
                    "id": "chatcmpl-entroly-tool",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [call],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
            return
        self._send_sse(
            [
                {
                    "id": "chatcmpl-entroly-tool",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "tool_calls": [call]},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-entroly-tool",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                },
            ]
        )

    def _send_final(self, body: dict[str, Any]) -> None:
        model = str(body.get("model") or "entroly-e2e-model")
        if not body.get("stream"):
            self._json(
                200,
                {
                    "id": "chatcmpl-entroly-final",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "E2E_OK"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 2,
                        "total_tokens": 42,
                    },
                },
            )
            return
        self._send_sse(
            [
                {
                    "id": "chatcmpl-entroly-final",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "E2E_OK"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-entroly-final",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 2,
                        "total_tokens": 42,
                    },
                },
            ]
        )


def main() -> int:
    from entroly.session_attach import AttachmentStore, install_attachment, uninstall_attachment

    report: dict[str, Any] = {
        "schema": "entroly.copilot-composite-e2e.v1",
        "passed": False,
    }
    with tempfile.TemporaryDirectory(prefix="entroly-copilot-composite-") as raw:
        temp = Path(raw)
        project = temp / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname = "entroly-copilot-e2e"\nversion = "0.0.0"\n',
            encoding="utf-8",
        )
        (project / "sample.py").write_text(
            "def answer(): return 'E2E_OK'\n",
            encoding="utf-8",
        )
        home = temp / "home"
        home.mkdir()
        copilot_home = temp / "copilot-home"
        state = temp / "attachment-state"

        env = os.environ.copy()
        env.update(
            {
                "HOME": str(home),
                "COPILOT_HOME": str(copilot_home),
                "COPILOT_MODEL": "entroly-e2e-model",
                "ENTROLY_DISABLE_UPDATE_CHECK": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )

        store = AttachmentStore(state)
        issued = store.create(
            client="copilot",
            project_root=project,
            scopes=("observe",),
            ttl_seconds=600,
        )
        server_name = f"entroly-{issued.grant.grant_id}"
        report["grant_id"] = issued.grant.grant_id
        report["server_name"] = server_name

        old_env = os.environ.copy()
        os.environ.clear()
        os.environ.update(env)
        try:
            installed = install_attachment(issued, store=store)
            report["install"] = [
                {
                    "returncode": item.returncode,
                    "stdout": item.stdout,
                    "stderr": item.stderr,
                }
                for item in installed
            ]
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        upstream_port = free_port()
        proxy_port = free_port()
        CopilotMockHandler.observations = []
        upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), CopilotMockHandler)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()

        proxy_env = env.copy()
        proxy_env.update(
            {
                "ENTROLY_DIR": str(temp / "proxy-state"),
                "ENTROLY_SOURCE": str(project),
                "ENTROLY_OPENAI_BASE": f"http://127.0.0.1:{upstream_port}",
                "ENTROLY_BYPASS": "1",
            }
        )
        with LOG.open("w", encoding="utf-8") as proxy_log:
            proxy = subprocess.Popen(
                ["entroly", "proxy", "--port", str(proxy_port)],
                cwd=str(project),
                env=proxy_env,
                text=True,
                stdout=proxy_log,
                stderr=subprocess.STDOUT,
            )
            try:
                report["health"] = wait_json(f"http://127.0.0.1:{proxy_port}/health")
                command = [
                    "entroly",
                    "wrap",
                    "copilot",
                    "--port",
                    str(proxy_port),
                    "--",
                    "-s",
                    "-p",
                    f"Use the get_stats tool from MCP server {server_name} exactly once. "
                    "After the tool succeeds, return exactly E2E_OK.",
                    f"--allow-tool={server_name}",
                    "--no-color",
                    "--no-ask-user",
                    "--no-auto-update",
                    "--no-custom-instructions",
                    "--disable-builtin-mcps",
                    "--no-remote",
                    "--no-remote-export",
                ]
                invocation = run(command, env=proxy_env, cwd=project)
                report["invocation"] = invocation
                report["stats"] = wait_json(f"http://127.0.0.1:{proxy_port}/stats", 10)
                report["observations"] = CopilotMockHandler.observations
                grant = store.get(issued.grant.grant_id)
                report["grant_use_count"] = grant.use_count
                output = f"{invocation.get('stdout', '')}\n{invocation.get('stderr', '')}"
                advertised = [
                    name
                    for observation in CopilotMockHandler.observations
                    for name in observation.get("advertised_tools", [])
                    if "get_stats" in name
                ]
                tool_results = [
                    observation
                    for observation in CopilotMockHandler.observations
                    if observation.get("has_tool_result")
                ]
                report["advertised_get_stats_tools"] = advertised
                report["tool_result_requests"] = len(tool_results)
                report["passed"] = bool(
                    invocation.get("returncode") == 0
                    and "E2E_OK" in output
                    and int(report["stats"].get("requests_total", 0)) >= 2
                    and advertised
                    and tool_results
                    and grant.use_count > 0
                )
                if not report["passed"]:
                    lowered = output.lower()
                    if "login" in lowered or "authentication" in lowered or "subscription" in lowered:
                        report["classification"] = "blocked_by_github_auth"
                    elif not advertised:
                        report["classification"] = "mcp_tools_not_loaded"
                    elif not tool_results:
                        report["classification"] = "tool_call_not_completed"
                    else:
                        report["classification"] = "proxy_or_response_failure"
            except Exception as exc:  # noqa: BLE001
                report["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                proxy.terminate()
                try:
                    proxy.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proxy.kill()
                upstream.shutdown()
                upstream.server_close()

        old_env = os.environ.copy()
        os.environ.clear()
        os.environ.update(env)
        try:
            try:
                removed = uninstall_attachment(issued.grant)
                report["remove"] = [item.returncode for item in removed]
            except Exception as exc:  # noqa: BLE001
                report["remove_error"] = f"{type(exc).__name__}: {exc}"
            try:
                store.revoke(issued.grant.grant_id)
            except Exception as exc:  # noqa: BLE001
                report["revoke_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": report.get("passed"),
        "classification": report.get("classification"),
        "requests_total": report.get("stats", {}).get("requests_total"),
        "grant_use_count": report.get("grant_use_count"),
        "advertised_get_stats_tools": report.get("advertised_get_stats_tools"),
        "tool_result_requests": report.get("tool_result_requests"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
