from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import entroly._docker_launcher as launcher


def test_pull_cache_is_scoped_to_selected_image(monkeypatch, tmp_path: Path) -> None:
    """A recent pull for one tag must not suppress pulling a different tag.

    Reproduction: install/serve version X, upgrade to Y within the one-hour TTL,
    then serve again. A single global timestamp incorrectly treats Y as already
    pulled even though only X was fetched.
    """
    cache_file = tmp_path / ".last_pull_ts"
    cache_file.write_text(str(time.time()), encoding="utf-8")
    monkeypatch.setattr(launcher, "_PULL_CACHE_FILE", cache_file)
    monkeypatch.setenv("ENTROLY_DOCKER_IMAGE", "ghcr.io/juyterman1000/entroly:next-version")

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    launcher._pull_image()

    assert ["docker", "pull", "ghcr.io/juyterman1000/entroly:next-version"] in calls, (
        "a cache timestamp written for another image suppressed the selected "
        "version pull"
    )


def test_pull_failure_is_not_silently_ignored(monkeypatch, tmp_path: Path) -> None:
    """If the selected image cannot be pulled or found locally, fail loudly.

    Silently continuing moves the failure to ``docker run`` and can execute an
    unintended local image or produce a misleading error after the trust
    boundary has already been crossed.
    """
    monkeypatch.setattr(launcher, "_PULL_CACHE_FILE", tmp_path / ".last_pull_ts")
    monkeypatch.setenv("ENTROLY_DOCKER_IMAGE", "ghcr.io/juyterman1000/entroly:missing")
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def fake_run(command, **kwargs):
        if command[:2] == ["docker", "pull"]:
            return SimpleNamespace(returncode=1, stderr=b"manifest unknown")
        if command[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1, stderr=b"not found")
        raise AssertionError(f"unexpected command: {command!r}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="missing|pull|image"):
        launcher._pull_image()
