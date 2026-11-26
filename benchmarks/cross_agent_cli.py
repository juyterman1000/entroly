"""Bounded, fresh-session CLI completions. Never install or repair an agent.

These are supplied-context calls, not autonomous agent benchmarks. Tool events
invalidate an answer. A temporary working directory is not a host sandbox;
the CLI's configuration and permission controls remain part of the trust base.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    provider: str
    model: str
    executable: str

    def __post_init__(self):
        if any(not isinstance(x, str) or not x.strip() for x in (
            self.agent_id, self.provider, self.model, self.executable,
        )):
            raise ValueError("agent fields must be nonempty strings")


def _executable(spec: AgentSpec) -> Path:
    if spec.provider not in {"codex", "claude"}:
        raise ValueError("no verified completion adapter for this provider")
    found = shutil.which(spec.executable)
    if found is None:
        raise FileNotFoundError("agent executable is not installed")
    executable = Path(found).resolve()
    if executable.suffix.lower() in {".cmd", ".bat", ".ps1"}:
        raise ValueError("provide the agent executable, not a shell shortcut")
    return executable


def _invoke(argv: list[str], root: Path, *, env: dict, prompt: str, timeout_s: float) -> str:
    """Bound wall time and transcript disk use, including descendant cleanup."""
    incoming, outgoing, errors = (root / name for name in ("stdin", "stdout", "stderr"))
    incoming.write_text(prompt, encoding="utf-8")
    limit = 4 * 1024 * 1024
    with incoming.open("rb") as stdin, outgoing.open("wb") as stdout, errors.open("wb") as stderr:
        job = None
        if os.name == "nt":
            from benchmarks.agent_process_job import WindowsJob
            job = WindowsJob()
            # The launcher cannot spawn the agent until assignment succeeds.
            wrapper = ("import subprocess,sys; gate=sys.stdin.buffer.read(1); "
                       "sys.exit(subprocess.call(sys.argv[2:],stdin=open(sys.argv[1],'rb')) if gate==b'1' else 1)")
            command = [sys.executable, "-I", "-S", "-c", wrapper, str(incoming), *argv]
        else:
            command = argv
        process = None
        deadline = time.monotonic() + timeout_s
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE if job else stdin,
                                       stdout=stdout, stderr=stderr, cwd=root,
                                       env=env, start_new_session=os.name != "nt",
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if job:
                job.assign(process)
                process.stdin.write(b"1")
                process.stdin.close()
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("agent completion exceeded wall-time limit")
                if any(p.exists() and p.stat().st_size > limit for p in (outgoing, errors, root / "answer")):
                    raise RuntimeError("agent transcript exceeded capture limit")
                time.sleep(0.02)
            if process.returncode:
                # Do not expose provider credentials or private stderr in receipts.
                raise RuntimeError(f"agent CLI exited with status {process.returncode}")
            if any(p.stat().st_size > limit for p in (outgoing, errors)):
                raise RuntimeError("agent transcript exceeded capture limit")
        finally:
            try:
                if job:
                    job.close()
                elif process is not None:
                    # The group can outlive its original parent on POSIX.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)
    if any(p.stat().st_size > limit for p in (outgoing, errors)):
        raise RuntimeError("agent transcript exceeded capture limit")
    return outgoing.read_text(encoding="utf-8")


def _child_env() -> dict:
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("OTEL_", "OPENAI_", "ANTHROPIC_", "CLAUDE_"))
           or key.upper() in {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"}}
    env.update(CLAUDE_CODE_ENABLE_TELEMETRY="0", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
               DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1", OTEL_SDK_DISABLED="true",
               OTEL_LOG_USER_PROMPTS="0", OTEL_LOG_ASSISTANT_RESPONSES="0", OTEL_LOG_RAW_API_BODIES="0")
    return env


def preflight(spec: AgentSpec) -> dict:
    executable = _executable(spec)
    with tempfile.TemporaryDirectory(prefix="entroly-agent-version-") as tmp:
        version = _invoke([str(executable), "--version"], Path(tmp), env=_child_env(),
                          prompt="", timeout_s=5).strip()
    if not version or len(version) > 4096:
        raise RuntimeError("agent version receipt is missing or oversized")
    return {"provider": spec.provider, "requested_model": spec.model, "version": version,
            "executable": str(executable), "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}


def _parse_codex(transcript: str, final: str) -> tuple[str, dict, list[str]]:
    usage, completed, message = {}, False, None
    for line in transcript.splitlines():
        event = json.loads(line)
        if not isinstance(event, dict):
            raise RuntimeError("unsupported Codex event")
        kind = event.get("type")
        if kind in {"error", "turn.failed"}:
            raise RuntimeError("Codex reported a failed turn")
        if kind in {"item.started", "item.updated", "item.completed"}:
            if event.get("item", {}).get("type") not in {"reasoning", "agent_message"}:
                raise RuntimeError("tool or unsupported item in completion-only Codex trial")
            if kind == "item.completed" and event["item"]["type"] == "agent_message":
                message = event["item"].get("text")
        elif kind not in {"thread.started", "turn.started", "turn.completed"}:
            raise RuntimeError("unsupported event in completion-only Codex trial")
        if kind == "turn.completed":
            if completed:
                raise RuntimeError("multiple turns in single-response trial")
            usage, completed = event.get("usage", {}), True
    if not completed or not final.strip() or not isinstance(message, str) or message.strip() != final.strip():
        raise RuntimeError("Codex did not produce a completed answer")
    return final, usage, []  # Codex events do not attest a resolved model identity.


def _parse_claude(transcript: str) -> tuple[str, dict, list[str]]:
    result = json.loads(transcript)
    if not isinstance(result, dict):
        raise RuntimeError("unsupported Claude response")
    if result.get("type") != "result" or result.get("is_error") or result.get("subtype") != "success":
        raise RuntimeError("Claude did not produce a successful completion")
    if not isinstance(result.get("result"), str) or not result["result"].strip():
        raise RuntimeError("Claude completion has no answer")
    if result.get("num_turns", 1) != 1 or result.get("permission_denials"):
        raise RuntimeError("Claude completion attempted additional tool turns")
    return result["result"], {"usage": result.get("usage", {}), "modelUsage": result.get("modelUsage", {})}, list(result.get("modelUsage", {}))


def generate(spec: AgentSpec, prompt: str, *, timeout_s: float) -> dict:
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be finite and positive")
    if not isinstance(prompt, str) or len(prompt.encode("utf-8")) > 1024 * 1024:
        raise ValueError("prompt must be text of at most one MiB")
    runtime = preflight(spec)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="entroly-agent-completion-") as tmp:
        root = Path(tmp)
        env = _child_env()
        executable = runtime["executable"]
        if spec.provider == "codex":
            original_home = Path(env.get("CODEX_HOME", str(Path.home() / ".codex")))
            home = root / "codex-home"
            home.mkdir(mode=0o700)
            auth = original_home / "auth.json"
            if auth.is_file():
                if json.loads(auth.read_text(encoding="utf-8")).get("OPENAI_API_KEY"):
                    raise RuntimeError("API-key authentication is excluded from account trials")
                shutil.copyfile(auth, home / "auth.json")
                (home / "auth.json").chmod(0o600)
            env["CODEX_HOME"] = str(home)
            # Only authentication is copied; no user MCP, hooks, or config files.
            argv = [executable, "exec", "--ignore-user-config", "--ephemeral", "--json",
                    "--sandbox", "read-only", "--skip-git-repo-check", "--model", spec.model,
                    "--disable", "shell_tool", "--disable", "hooks", "--disable", "multi_agent",
                    "--disable", "shell_snapshot", "--disable", "apps", "--disable", "plugins",
                    "--disable", "in_app_browser", "--disable", "skill_mcp_dependency_install",
                    "-c", 'web_search="disabled"', "--output-last-message", str(root / "answer"), "-"]
            transcript = _invoke(argv, root, env=env, prompt=prompt, timeout_s=timeout_s)
            answer_file = root / "answer"
            if not answer_file.is_file() or answer_file.stat().st_size > 4 * 1024 * 1024:
                raise RuntimeError("Codex answer missing or oversized")
            text, usage, resolved = _parse_codex(transcript, answer_file.read_text(encoding="utf-8"))
        else:
            argv = [executable, "--print", "--safe-mode", "--restricted", "--tools", "", "--strict-mcp-config",
                    "--mcp-config", '{"mcpServers":{}}', "--setting-sources", "",
                    "--no-session-persistence", "--output-format", "json", "--model", spec.model]
            transcript = _invoke(argv, root, env=env, prompt=prompt, timeout_s=timeout_s)
            text, usage, resolved = _parse_claude(transcript)
        return {"text": text, "usage": usage, "resolved_models_reported": resolved,
                "latency_s": time.monotonic() - started, "runtime": runtime,
                "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
                "scope": "fresh-session supplied-context completion; CLI permission controls trusted"}
