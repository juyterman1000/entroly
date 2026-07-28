from __future__ import annotations

import json
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
    """If the selected image cannot be pulled or found locally, fail loudly."""
    monkeypatch.setattr(launcher, "_PULL_CACHE_FILE", tmp_path / ".last_pull.json")
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


def test_corrupt_or_legacy_cache_is_a_miss(monkeypatch, tmp_path: Path) -> None:
    cache_file = tmp_path / ".last_pull.json"
    cache_file.write_text("not-json", encoding="utf-8")
    monkeypatch.setattr(launcher, "_PULL_CACHE_FILE", cache_file)
    monkeypatch.setenv("ENTROLY_DOCKER_IMAGE", "ghcr.io/juyterman1000/entroly:1.2.3")

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    launcher._pull_image()

    assert calls[0] == ["docker", "pull", "ghcr.io/juyterman1000/entroly:1.2.3"]
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["image"] == "ghcr.io/juyterman1000/entroly:1.2.3"
    assert isinstance(payload["timestamp"], float)


def test_recent_cache_only_skips_when_exact_image_still_exists(
    monkeypatch, tmp_path: Path
) -> None:
    image = "ghcr.io/juyterman1000/entroly:1.2.3"
    cache_file = tmp_path / ".last_pull.json"
    cache_file.write_text(
        json.dumps({"image": image, "timestamp": time.time()}),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "_PULL_CACHE_FILE", cache_file)
    monkeypatch.setenv("ENTROLY_DOCKER_IMAGE", image)

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=0, stderr=b"")
        raise AssertionError(f"unexpected command: {command!r}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    launcher._pull_image()

    assert calls == [["docker", "image", "inspect", image]]


def test_failed_refresh_can_use_only_the_exact_local_image(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    image = "ghcr.io/juyterman1000/entroly:1.2.3"
    monkeypatch.setattr(launcher, "_PULL_CACHE_FILE", tmp_path / ".last_pull.json")
    monkeypatch.setenv("ENTROLY_DOCKER_IMAGE", image)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def fake_run(command, **kwargs):
        if command[:2] == ["docker", "pull"]:
            return SimpleNamespace(returncode=1, stderr=b"network unavailable")
        if command[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=0, stderr=b"")
        raise AssertionError(f"unexpected command: {command!r}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    launcher._pull_image()

    assert "exact local image" in capsys.readouterr().err
    assert not launcher._PULL_CACHE_FILE.exists(), (
        "an offline fallback must not refresh the pull timestamp and mask outages"
    )


@pytest.mark.parametrize(
    "value",
    [
        "--help",
        "ghcr.io/repo:tag\n--privileged",
        "ghcr.io/repo:tag with-space",
        "",
    ],
)
def test_invalid_docker_image_override_fails_before_subprocess(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("ENTROLY_DOCKER_IMAGE", value)
    if value == "":
        # Empty is the documented way to use the safe versioned default.
        assert launcher._docker_image() == launcher.DEFAULT_DOCKER_IMAGE
    else:
        with pytest.raises(RuntimeError, match="valid Docker image"):
            launcher._docker_image()
