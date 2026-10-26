"""The capture hook must not print the first-run welcome banner.

`entroly ravs capture --stdin --quiet` is the PostToolUse hook command. On the
first Entroly invocation under a HOME it printed the 25-line "Welcome to
Entroly" banner to stderr -- host-facing output on a tool-call hook, and the
marker it wrote meant a later interactive command never showed the banner.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def test_the_capture_hook_prints_no_banner_and_leaves_it_for_the_terminal(tmp_path) -> None:
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "USERPROFILE": str(tmp_path / "home"),
        "ENTROLY_DIR": str(tmp_path / "entroly"),
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
    }
    payload = {"tool_name": "Bash", "tool_input": {"command": "ls"}, "exit_code": 0}

    hook = subprocess.run(
        [sys.executable, "-m", "entroly", "ravs", "capture", "--stdin", "--quiet",
         "--log", str(tmp_path / "events.jsonl")],
        input=json.dumps(payload), env=env, capture_output=True, text=True, timeout=120,
    )

    assert hook.returncode == 0
    assert "Welcome to Entroly" not in hook.stderr
    assert hook.stdout == ""

    terminal = subprocess.run(
        [sys.executable, "-m", "entroly", "config"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert "Welcome to Entroly" in terminal.stderr
