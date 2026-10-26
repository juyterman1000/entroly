"""`entroly serve` must start the user's MCP server on the installed engine.

Reproduced on a fresh HOME with the native engine installed:

* Docker CLI present, daemon stopped (Docker Desktop installed but not
  running): ``entroly serve`` exited 1 with "Docker is not running" and told
  the user to install the native engine that was already loaded.
* Docker running: the launcher pulled the image and ran
  ``docker run --rm -i IMAGE serve`` with no project mount, so the server
  indexed the image's working directory (``/app``, Entroly's own source)
  instead of the user's repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import entroly._docker_launcher as launcher
from entroly.native_status import QCCR_SYMBOLS, native_status

_REAL_EXISTS = os.path.exists


def _no_dockerenv(path):
    # Keep the test deterministic inside CI containers, where /.dockerenv
    # exists and would route every serve to the native server on its own.
    if path == "/.dockerenv":
        return False
    return _REAL_EXISTS(path)


@pytest.fixture
def serve_argv(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["entroly", "serve"])
    for name in ("ENTROLY_NO_DOCKER", "ENTROLY_DOCKER_IMAGE", "ENTROLY_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launcher.os.path, "exists", _no_dockerenv)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _record_routes(monkeypatch, *, docker_available: bool) -> list:
    observed: list = []
    monkeypatch.setattr(launcher, "_run_native", lambda: observed.append("native"))

    def probe() -> bool:
        observed.append("docker-probe")
        return docker_available

    monkeypatch.setattr(launcher, "_docker_available", probe)
    monkeypatch.setattr(launcher, "_pull_image", lambda: observed.append("pull"))

    def fake_run(command, **kwargs):
        observed.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return observed


def _native_installed() -> bool:
    return native_status(QCCR_SYMBOLS).ok


@pytest.mark.skipif(not _native_installed(), reason="needs the installed native engine")
def test_serve_uses_installed_native_engine_even_when_docker_runs(serve_argv, monkeypatch):
    observed = _record_routes(monkeypatch, docker_available=True)

    launcher.launch()

    assert observed == ["native"], (
        "a complete installed engine must not trigger a Docker pull/run: "
        f"{observed!r}"
    )


@pytest.mark.skipif(not _native_installed(), reason="needs the installed native engine")
def test_serve_without_docker_daemon_starts_native_instead_of_exiting(
    serve_argv, monkeypatch, capsys
):
    observed = _record_routes(monkeypatch, docker_available=False)

    launcher.launch()

    assert observed == ["native"]
    assert "Docker is not running" not in capsys.readouterr().err


def _engine_missing(monkeypatch) -> None:
    import entroly.native_status as native_status_module

    stale = native_status_module.NativeStatus(
        available=True, module=None, version="1.0.1", path=None,
        missing_symbols=(), version_ok=False,
    )
    monkeypatch.setattr(native_status_module, "native_status", lambda *_a, **_k: stale)


def test_docker_fallback_mounts_the_users_project_read_only(serve_argv, monkeypatch):
    _engine_missing(monkeypatch)
    observed = _record_routes(monkeypatch, docker_available=True)

    with pytest.raises(SystemExit) as exc:
        launcher.launch()

    assert exc.value.code == 0
    command = observed[-1]
    assert command[:2] == ["docker", "run"]
    project = os.path.abspath(str(serve_argv))
    # The server indexes its working directory; it must be the user's project.
    assert "-v" in command and "-w" in command, f"no project mount: {command!r}"
    assert ["-v", f"{project}:/workspace:ro"] == command[
        command.index("-v") : command.index("-v") + 2
    ]
    assert command[command.index("-w") + 1] == "/workspace"
    assert "ENTROLY_SOURCE=/workspace" in command
    # The project is now the working directory, so `python -m` must not import
    # modules from it (a repo with an `entroly/` or `json.py` would shadow).
    assert "PYTHONSAFEPATH=1" in command
    assert command.index("-v") < command.index(launcher._docker_image())


def test_docker_fallback_mounts_entroly_source_without_leaking_host_path(
    serve_argv, monkeypatch, tmp_path_factory
):
    source = tmp_path_factory.mktemp("explicit source")
    monkeypatch.setenv("ENTROLY_SOURCE", str(source))
    _engine_missing(monkeypatch)
    observed = _record_routes(monkeypatch, docker_available=True)

    with pytest.raises(SystemExit):
        launcher.launch()

    command = observed[-1]
    assert f"{os.path.abspath(str(source))}:/workspace:ro" in command, command
    # A host path is meaningless inside the container; forwarding the bare
    # name would hand the server a directory that does not exist there.
    assert "ENTROLY_SOURCE" not in command
    assert "ENTROLY_SOURCE=/workspace" in command


def test_no_engine_and_no_docker_names_real_remedies(serve_argv, monkeypatch, capsys):
    _engine_missing(monkeypatch)
    observed = _record_routes(monkeypatch, docker_available=False)

    with pytest.raises(SystemExit) as exc:
        launcher.launch()

    assert exc.value.code == 1
    assert "native" not in observed
    err = capsys.readouterr().err
    assert "python -m pip install -U entroly-core" in err
    # The cause is derived (here: a stale core), never a hardcoded "not installed".
    assert "1.0.1 is below the required" in err
    assert "not installed" not in err
    assert "ENTROLY_NO_DOCKER=1 entroly serve" in err
    # The same command must not be offered twice as two different options.
    assert err.count("ENTROLY_NO_DOCKER=1 entroly serve") == 1


@pytest.mark.skipif(not _native_installed(), reason="needs the installed native engine")
def test_explicit_image_override_still_selects_docker(serve_argv, monkeypatch):
    monkeypatch.setenv("ENTROLY_DOCKER_IMAGE", "example/entroly:1.0")
    observed = _record_routes(monkeypatch, docker_available=True)

    with pytest.raises(SystemExit):
        launcher.launch()

    assert "native" not in observed
    assert observed[-1][:2] == ["docker", "run"]
