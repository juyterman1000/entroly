from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPORT_PATH = Path("agent-e2e-probe.json")
LOG_DIR = Path("agent-e2e-logs")
LOG_DIR.mkdir(exist_ok=True)


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: int = 90,
    name: str,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        result: dict[str, Any] = {
            "command": command,
            "returncode": completed.returncode,
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        result = {
            "command": command,
            "returncode": None,
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }
    (LOG_DIR / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_json(url: str, *, timeout: float = 45.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - probe records the final error
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


class MockOpenAIHandler(BaseHTTPRequestHandler):
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
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            body = {"_raw": raw.decode("utf-8", errors="replace")}
        self.observations.append(
            {
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "body": body,
            }
        )
        if self.path.endswith("/chat/completions"):
            self._chat_completions(body)
            return
        if self.path.endswith("/responses"):
            self._responses(body)
            return
        self._json(404, {"error": {"message": f"unknown POST {self.path}"}})

    def _chat_completions(self, body: dict[str, Any]) -> None:
        model = str(body.get("model") or "entroly-e2e-model")
        if not body.get("stream"):
            self._json(
                200,
                {
                    "id": "chatcmpl-entroly-e2e",
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
                        "prompt_tokens": 20,
                        "completion_tokens": 2,
                        "total_tokens": 22,
                    },
                },
            )
            return
        chunks = [
            {
                "id": "chatcmpl-entroly-e2e",
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
                "id": "chatcmpl-entroly-e2e",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 2,
                    "total_tokens": 22,
                },
            },
        ]
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

    def _responses(self, body: dict[str, Any]) -> None:
        model = str(body.get("model") or "entroly-e2e-model")
        if not body.get("stream"):
            self._json(
                200,
                {
                    "id": "resp_entroly_e2e",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": model,
                    "output": [
                        {
                            "id": "msg_entroly_e2e",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "E2E_OK",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22},
                },
            )
            return
        response = {
            "id": "resp_entroly_e2e",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": model,
            "output": [
                {
                    "id": "msg_entroly_e2e",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "E2E_OK",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22},
        }
        events = [
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
            {
                "type": "response.output_text.delta",
                "item_id": "msg_entroly_e2e",
                "output_index": 0,
                "content_index": 0,
                "delta": "E2E_OK",
            },
            {"type": "response.completed", "response": response},
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def version_probe(binary: str, name: str) -> dict[str, Any]:
    path = shutil.which(binary)
    if path is None:
        return {"installed": False, "binary": binary}
    result = run([binary, "--version"], timeout=30, name=f"{name}-version")
    return {
        "installed": True,
        "binary": path,
        "returncode": result["returncode"],
        "stdout": result["stdout"].strip(),
        "stderr": result["stderr"].strip(),
    }


def attachment_probe(client: str, *, env: dict[str, str], project: Path, state: Path) -> dict[str, Any]:
    from entroly.session_attach import AttachmentStore, install_attachment, uninstall_attachment

    store = AttachmentStore(state)
    issued = store.create(
        client=client,
        project_root=project,
        scopes=("observe", "context"),
        ttl_seconds=600,
    )
    name = f"entroly-{issued.grant.grant_id}"
    result: dict[str, Any] = {
        "client": client,
        "grant_id": issued.grant.grant_id,
        "server_name": name,
        "install_commands": [list(command) for command in issued.install_commands],
    }
    old_env = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        try:
            installed = install_attachment(issued, store=store)
            result["install"] = [
                {
                    "returncode": item.returncode,
                    "stdout": item.stdout,
                    "stderr": item.stderr,
                }
                for item in installed
            ]
        except Exception as exc:  # noqa: BLE001
            result["install_error"] = f"{type(exc).__name__}: {exc}"
            return result

        if client == "copilot":
            inspection = run(
                ["copilot", "mcp", "get", name, "--json"],
                env=env,
                timeout=90,
                name="copilot-mcp-get",
            )
        elif client == "kimi":
            inspection = run(
                ["kimi", "mcp", "test", name],
                env=env,
                timeout=90,
                name="kimi-mcp-test",
            )
        else:
            raise AssertionError(client)
        result["inspection"] = inspection
        combined = f"{inspection.get('stdout', '')}\n{inspection.get('stderr', '')}"
        result["tools_observed"] = [
            tool
            for tool in ("get_stats", "optimize_context", "repo_file_map", "entroly_retrieve")
            if tool in combined
        ]
        result["passed"] = bool(
            inspection.get("returncode") == 0 and result["tools_observed"]
        )
        try:
            removed = uninstall_attachment(issued.grant)
            result["remove"] = [item.returncode for item in removed]
        except Exception as exc:  # noqa: BLE001
            result["remove_error"] = f"{type(exc).__name__}: {exc}"
        try:
            store.revoke(issued.grant.grant_id)
        except Exception as exc:  # noqa: BLE001
            result["revoke_error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def copilot_proxy_probe(*, env: dict[str, str], project: Path, state: Path) -> dict[str, Any]:
    upstream_port = free_port()
    proxy_port = free_port()
    MockOpenAIHandler.observations = []
    upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), MockOpenAIHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    proxy_env = env.copy()
    proxy_env.update(
        {
            "ENTROLY_DIR": str(state),
            "ENTROLY_SOURCE": str(project),
            "ENTROLY_OPENAI_BASE": f"http://127.0.0.1:{upstream_port}",
            "ENTROLY_BYPASS": "1",
            "ENTROLY_DISABLE_UPDATE_CHECK": "1",
        }
    )
    proxy_log = (LOG_DIR / "entroly-proxy.log").open("w", encoding="utf-8")
    proxy = subprocess.Popen(
        ["entroly", "proxy", "--port", str(proxy_port)],
        cwd=str(project),
        env=proxy_env,
        text=True,
        stdout=proxy_log,
        stderr=subprocess.STDOUT,
    )
    result: dict[str, Any] = {
        "upstream_port": upstream_port,
        "proxy_port": proxy_port,
    }
    try:
        health = wait_json(f"http://127.0.0.1:{proxy_port}/health")
        result["health"] = health
        client_env = proxy_env.copy()
        client_env.update(
            {
                "COPILOT_HOME": str(state / "copilot-byok"),
                "COPILOT_MODEL": "entroly-e2e-model",
            }
        )
        command = [
            "entroly",
            "wrap",
            "copilot",
            "--port",
            str(proxy_port),
            "--",
            "-s",
            "-p",
            "Return exactly E2E_OK. Do not call tools.",
            "--no-color",
            "--no-ask-user",
            "--no-auto-update",
            "--no-custom-instructions",
            "--disable-builtin-mcps",
            "--no-remote",
            "--no-remote-export",
        ]
        invocation = run(
            command,
            env=client_env,
            cwd=project,
            timeout=120,
            name="copilot-byok-through-entroly",
        )
        result["invocation"] = invocation
        result["stats"] = wait_json(f"http://127.0.0.1:{proxy_port}/stats", timeout=10)
        result["upstream_observations"] = MockOpenAIHandler.observations
        output = f"{invocation.get('stdout', '')}\n{invocation.get('stderr', '')}"
        requests_total = int(result["stats"].get("requests_total", 0))
        result["passed"] = bool(
            invocation.get("returncode") == 0
            and "E2E_OK" in output
            and requests_total > 0
            and MockOpenAIHandler.observations
        )
        if not result["passed"]:
            lowered = output.lower()
            result["classification"] = (
                "blocked_by_github_auth"
                if "login" in lowered or "authentication" in lowered or "copilot subscription" in lowered
                else "protocol_or_wrapper_failure"
            )
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["passed"] = False
        return result
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proxy.kill()
        proxy_log.close()
        upstream.shutdown()
        upstream.server_close()


def main() -> int:
    root = Path.cwd().resolve()
    with tempfile.TemporaryDirectory(prefix="entroly-agent-e2e-") as temp_raw:
        temp = Path(temp_raw)
        project = temp / "project"
        project.mkdir()
        (project / "README.md").write_text("# Entroly agent compatibility E2E\n", encoding="utf-8")
        (project / "sample.py").write_text("def answer(): return 'E2E_OK'\n", encoding="utf-8")

        base_env = os.environ.copy()
        base_env.update(
            {
                "HOME": str(temp / "home"),
                "ENTROLY_DISABLE_UPDATE_CHECK": "1",
                "PYTHONUNBUFFERED": "1",
                "PATH": os.environ["PATH"],
            }
        )
        Path(base_env["HOME"]).mkdir(parents=True, exist_ok=True)

        report: dict[str, Any] = {
            "schema": "entroly.agent-compatibility-e2e-probe.v1",
            "repository": str(root),
            "versions": {
                "entroly": run(["entroly", "--version"], env=base_env, timeout=30, name="entroly-version"),
                "copilot": version_probe("copilot", "copilot"),
                "kimi": version_probe("kimi", "kimi"),
            },
            "probes": {},
        }

        copilot_env = base_env.copy()
        copilot_env["COPILOT_HOME"] = str(temp / "copilot-home")
        report["probes"]["copilot_mcp"] = attachment_probe(
            "copilot",
            env=copilot_env,
            project=project,
            state=temp / "state-copilot-mcp",
        )

        kimi_env = base_env.copy()
        report["probes"]["kimi_mcp"] = attachment_probe(
            "kimi",
            env=kimi_env,
            project=project,
            state=temp / "state-kimi-mcp",
        )

        report["probes"]["copilot_byok_proxy"] = copilot_proxy_probe(
            env=base_env,
            project=project,
            state=temp / "state-copilot-proxy",
        )

        report["summary"] = {
            name: {
                "passed": bool(probe.get("passed")),
                "classification": probe.get("classification"),
                "error": probe.get("error") or probe.get("install_error"),
            }
            for name, probe in report["probes"].items()
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
