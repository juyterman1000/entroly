from __future__ import annotations

import json
import subprocess
import sys
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


def test_env_passthrough_keeps_secret_values_out_of_argv(monkeypatch) -> None:
    secret = "sk-super-secret-value-🧪"
    monkeypatch.setenv("ENTROLY_Z_API_KEY", secret)
    monkeypatch.setenv("ENTROLY_A_MODE", "مرحبا")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")

    args = launcher._env_passthrough()

    assert args[::2] == ["-e"] * (len(args) // 2)
    names = args[1::2]
    assert names == sorted(names)
    assert "ENTROLY_Z_API_KEY" in names
    assert "ENTROLY_A_MODE" in names
    assert "UNRELATED_SECRET" not in names
    assert secret not in args
    assert all("=" not in item for item in names)


def test_launch_never_places_entrolly_secret_in_docker_command(monkeypatch) -> None:
    secret = "token-visible-to-ps-if-regressed"
    captured: list[list[str]] = []

    monkeypatch.setenv("ENTROLY_PROVIDER_TOKEN", secret)
    monkeypatch.delenv("ENTROLY_NO_DOCKER", raising=False)
    monkeypatch.setattr(sys, "argv", ["entroly", "serve"])
    monkeypatch.setattr(launcher, "_docker_available", lambda: True)
    monkeypatch.setattr(launcher, "_pull_image", lambda: None)
    monkeypatch.setattr(launcher, "_docker_image", lambda: "example/entroly:1.0")
    monkeypatch.setattr(launcher.os.path, "exists", lambda _path: False)

    def fake_run(command, **kwargs):
        captured.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc:
        launcher.launch()

    assert exc.value.code == 0
    assert len(captured) == 1
    command = captured[0]
    assert secret not in command
    assert "ENTROLY_PROVIDER_TOKEN" in command
    assert f"ENTROLY_PROVIDER_TOKEN={secret}" not in command
