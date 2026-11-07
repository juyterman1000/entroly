"""How a host hook payload says a command ended.

Claude Code puts no exit code in a Bash hook payload. It states the outcome
with the event that fires:

``PostToolUse``
    Fires only after the command exited 0. ``tool_response`` carries
    ``stdout``, ``stderr`` and ``interrupted``.
``PostToolUseFailure``
    Fires after a non-zero exit. There is no ``tool_response``; ``error``
    opens with an ``Exit code N`` line followed by the command's output, and
    ``is_interrupt`` marks a call the user stopped.

The hooks reference lists the two as "after a tool call succeeds" and "after
a tool call fails"; the payload shapes above were observed from Claude Code
2.1.29x. Readers that looked for a top-level ``exit_code`` found none in
either, so RAVS recorded no outcome from Claude Code at all.

This module imports nothing from Entroly, so the capture hook and the session
journal can share it without adding an edge to the package's import cycle.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

CLAUDE_CODE = "claude-code"

# Anchored to the start of ``error``: the same words later in the text are the
# command's own output, not the host's statement of the exit status.
_EXIT_LINE = re.compile(r"Exit code (\d+)[ \t]*(?:\r?\n|$)")


def exit_code(data: Mapping[str, Any], host: str | None = None) -> int | None:
    """The exit code the payload states, or None when it does not state one.

    An explicit ``exit_code`` field wins. Otherwise a ``PostToolUseFailure``
    for Bash yields the code on its leading ``Exit code N`` line, and a
    ``PostToolUse`` for Bash yields 0 -- but only when ``host`` is
    ``claude-code``. That flag is written by the installer that also registers
    the failure event; a registration without it sees successes and never
    failures, and a sample with every failure missing must not be recorded as
    evidence of anything.

    Interrupted calls, a failure claiming exit 0, and any shape not listed
    here are unknown, and unknown is never reported as success.
    """
    explicit = data.get("exit_code")
    if explicit is not None:
        if isinstance(explicit, bool):
            return None
        try:
            return int(explicit)
        except (TypeError, ValueError):
            return None

    if data.get("tool_name") != "Bash":
        return None
    event = data.get("hook_event_name")
    if event == "PostToolUseFailure":
        if data.get("is_interrupt") is True:
            return None
        error = data.get("error")
        if not isinstance(error, str):
            return None
        match = _EXIT_LINE.match(error)
        if match is None:
            return None
        code = int(match.group(1))
        return code if code != 0 else None
    if event == "PostToolUse" and host == CLAUDE_CODE:
        response = data.get("tool_response")
        if not isinstance(response, Mapping) or response.get("interrupted") is True:
            return None
        return 0
    return None


def output_text(data: Mapping[str, Any]) -> str:
    """The command's output as the payload carries it, without the status line."""
    for key in ("tool_output", "stdout"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    response = data.get("tool_response")
    if isinstance(response, Mapping):
        parts = (response.get("stdout"), response.get("stderr"))
        return "\n".join(part for part in parts if isinstance(part, str) and part)
    if isinstance(response, str):
        return response
    error = data.get("error")
    if isinstance(error, str):
        if _EXIT_LINE.match(error):
            return error.partition("\n")[2]
        return error
    return ""


__all__ = ["CLAUDE_CODE", "exit_code", "output_text"]
