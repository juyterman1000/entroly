"""The Docker fallback pull must not run silently.

`_pull_image` sent `docker pull` progress to DEVNULL and printed nothing, so
the fallback `entroly serve` sat silent for up to ~123s (two 60s attempts and
a 3s pause) before its first line. Progress must reach the user on stderr --
never stdout, which carries the MCP protocol in serve mode.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import entroly._docker_launcher as launcher


def test_pull_announces_itself_and_streams_progress_to_stderr(
    monkeypatch, tmp_path, capsys
):
    err_file = open(tmp_path / "stderr.txt", "w+", encoding="utf-8")  # a real fd
    monkeypatch.setattr(launcher.sys, "stderr", err_file)
    monkeypatch.setattr(launcher, "_PULL_CACHE_FILE", tmp_path / ".last_pull.json")
    monkeypatch.setattr(launcher, "_docker_image", lambda: "example/entroly:1.0")
    pulls: list[dict] = []

    def fake_run(command, **kwargs):
        if command[:2] == ["docker", "pull"]:
            pulls.append(kwargs)
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    launcher._pull_image()

    err_file.flush()
    assert "Pulling example/entroly:1.0" in (tmp_path / "stderr.txt").read_text()
    assert capsys.readouterr().out == ""  # stdout is the MCP channel
    assert len(pulls) == 1
    assert pulls[0]["stdout"] is err_file  # docker's progress goes to stderr
    err_file.close()
