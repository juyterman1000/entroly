"""
Lightweight event capture endpoint for RAVS.

Called by PostToolUse hooks to record verifiable outcomes. Accepts
input via CLI args or stdin JSON. Writes to the RAVS AppendOnlyEventLog.

Design: must complete in <50ms so it never blocks the user's workflow.
Uses the same AppendOnlyEventLog as the MCP server for format parity.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from .hook_classifier import classify, Classification


def _resolve_log_path() -> str:
    """Resolve the RAVS event log path.

    Priority:
      1. ENTROLY_DIR env var (explicit override for CI / multi-project)
      2. ~/.entroly/ravs/events.jsonl (stable global default)

    Using a global default (not CWD) ensures the Claude Code hook
    always writes to the same log regardless of which directory
    Claude Code happened to be in when it ran the Bash tool.
    """
    if "ENTROLY_DIR" in os.environ:
        base = os.environ["ENTROLY_DIR"]
    else:
        base = os.path.join(os.path.expanduser("~"), ".entroly")
    ravs_dir = os.path.join(base, "ravs")
    os.makedirs(ravs_dir, exist_ok=True)
    return os.path.join(ravs_dir, "events.jsonl")


def _append_event(log_path: str, event: dict) -> None:
    """Append a single JSON event to the log. Atomic-ish via append mode."""
    line = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def capture_from_args(
    command: str,
    exit_code: int,
    stdout_tail: str = "",
    duration_ms: Optional[float] = None,
    source: str = "hook",
    source_strength: float = 0.7,
    log_path: Optional[str] = None,
) -> Optional[dict]:
    """Classify a command and write the outcome event.

    Returns the event dict if a verifiable outcome was detected,
    None if the command was not classifiable (normal shell usage).
    """
    classification = classify(
        command=command,
        exit_code=exit_code,
        stdout_tail=stdout_tail,
        source_strength=source_strength,
    )

    if classification is None:
        return None  # Not a verifiable command, skip silently

    log_path = log_path or _resolve_log_path()
    now = time.time()
    request_id = f"hook-{uuid.uuid4().hex[:12]}"

    # Write a request event (mirrors the format from server.py RAVS integration)
    request_event = {
        "type": "request",
        "request_id": request_id,
        "timestamp": now,
        "query": command,
        "model": "local",  # hook-captured, no model involved
        "cost": 0.0,
        "tokens": 0,
        "latency_ms": duration_ms or 0,
        "source": source,
    }
    _append_event(log_path, request_event)

    # Write the outcome event
    outcome_event = {
        "type": "outcome",
        "request_id": request_id,
        "timestamp": now + 0.001,  # ensure ordering
        "event_type": classification.category,
        "value": classification.verdict,
        "strength": "strong" if classification.confidence >= 0.8 else "weak",
        "tool": classification.tool,
        "exit_code": exit_code,
        "confidence": classification.confidence,
        "source": source,
        "source_strength": classification.source_strength,
    }
    _append_event(log_path, outcome_event)

    return outcome_event


def capture_from_stdin(log_path: Optional[str] = None) -> Optional[dict]:
    """Read a JSON object from stdin and capture it.

    Expected stdin format (Claude Code PostToolUse):
    {
        "tool_name": "Bash",
        "tool_input": {"command": "pytest tests/"},
        "tool_output": "6 passed in 0.5s",
        "exit_code": 0,
        "duration_ms": 500
    }
    """
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return None
        data = json.loads(raw)
    except (json.JSONDecodeError, IOError):
        return None

    command = ""
    if isinstance(data.get("tool_input"), dict):
        command = data["tool_input"].get("command", "")
    elif isinstance(data.get("tool_input"), str):
        command = data["tool_input"]
    elif "command" in data:
        command = data["command"]

    exit_code = data.get("exit_code", 0)
    stdout_tail = data.get("tool_output", data.get("stdout", ""))
    # Truncate stdout to last 1KB to keep events lean
    if len(stdout_tail) > 1024:
        stdout_tail = stdout_tail[-1024:]

    duration_ms = data.get("duration_ms", None)
    source = data.get("source", "hook")
    source_strength = data.get("source_strength", 0.7)

    return capture_from_args(
        command=command,
        exit_code=exit_code,
        stdout_tail=stdout_tail,
        duration_ms=duration_ms,
        source=source,
        source_strength=source_strength,
        log_path=log_path,
    )


def main() -> None:
    """CLI entry point: entroly ravs capture."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="entroly ravs capture",
        description="Capture a verifiable CLI outcome for RAVS",
    )
    parser.add_argument("--command", "-c", type=str, default=None,
                        help="The shell command that was run")
    parser.add_argument("--exit-code", "-e", type=int, default=None,
                        help="Process exit code")
    parser.add_argument("--stdout", "-o", type=str, default="",
                        help="Tail of stdout (last ~1KB)")
    parser.add_argument("--duration-ms", type=float, default=None,
                        help="Command duration in milliseconds")
    parser.add_argument("--source", type=str, default="hook",
                        help="Event source (hook, ci, manual)")
    parser.add_argument("--stdin", action="store_true",
                        help="Read JSON from stdin instead of args")
    parser.add_argument("--log", type=str, default=None,
                        help="Override event log path")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output")

    args = parser.parse_args()

    if args.stdin:
        result = capture_from_stdin(log_path=args.log)
    elif args.command is not None and args.exit_code is not None:
        result = capture_from_args(
            command=args.command,
            exit_code=args.exit_code,
            stdout_tail=args.stdout,
            duration_ms=args.duration_ms,
            source=args.source,
            log_path=args.log,
        )
    else:
        parser.error("Either --stdin or both --command and --exit-code are required")
        return

    if not args.quiet:
        if result:
            cat = result["event_type"]
            verdict = result["value"]
            tool = result["tool"]
            print(f"  ✓ Captured: {tool} → {cat}/{verdict} (conf={result['confidence']:.2f})")
        else:
            print("  · Skipped: not a verifiable command")
