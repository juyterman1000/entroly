"""A read-only HOME must not hand Entroly's data to whoever reached /tmp first.

MCP hosts, containers and managed IDE runtimes can give Entroly a HOME it
cannot write, so every data path falls back to the system temporary directory.
That directory is shared by every account on the machine, and the fallback was
the fixed name ``<tmp>/entroly``.

Run as an unprivileged user with a read-only HOME on a machine where another
account had already fallen back once, published 1.0.88 failed both
``entroly verify-claims`` and ``entroly simulate --query fix`` with
"Error: [Errno 13] Permission denied: '/tmp/entroly/checkpoints'", and printed
its "This message appears once" welcome on every command because the marker
could not be written. The account that created ``/tmp/entroly`` also decided
where everyone else's data went: the five resolvers accepted any existing
directory they could write to, including one another account had made
world-writable, and the work-graph reader trusted checkpoints found there.

A test running as root cannot be refused a directory, so here a regular file
stands in for another account's ``<tmp>/entroly``: either one makes the fixed
name unusable, and the old code failed on both in the same place.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import pytest

from entroly import _docker_launcher, cli, config, copilot_subscription, value_tracker
from entroly.work_graph_repo import _existing_checkpoint_dir

pytestmark = pytest.mark.skipif(
    not hasattr(os, "getuid"), reason="Windows already gives each account its own temp directory"
)

ROOT = Path(__file__).resolve().parents[1]


def _resolvers() -> dict:
    return {
        "checkpoints": config._project_checkpoint_dir,
        "cli data": cli._resolve_entroly_dir,
        "docker launcher": _docker_launcher._runtime_dir,
        "copilot": lambda: copilot_subscription._runtime_dir(os.environ),
        "value tracker": value_tracker.ValueTracker._default_dir,
    }


RESOLVERS = ["checkpoints", "cli data", "docker launcher", "copilot", "value tracker"]


def _resolve(name: str) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return Path(_resolvers()[name]())


@pytest.fixture()
def tmp(tmp_path: Path, monkeypatch) -> Path:
    """HOME cannot be written; the system temporary directory is a fresh fake."""
    home = tmp_path / "home-is-a-file"
    home.write_text("not a directory", encoding="utf-8")
    shared = tmp_path / "tmp"
    shared.mkdir()
    monkeypatch.delenv("ENTROLY_DIR", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(shared))
    return shared


def _mine(tmp: Path) -> Path:
    return tmp / f"entroly-{os.getuid()}"


@pytest.mark.parametrize("name", RESOLVERS)
def test_another_accounts_claim_on_the_shared_name_does_not_block_us(tmp: Path, name: str) -> None:
    (tmp / "entroly").write_text("another account got here first", encoding="utf-8")

    chosen = _resolve(name)

    assert chosen.is_relative_to(_mine(tmp)), chosen
    assert chosen.is_dir()


@pytest.mark.parametrize("name", RESOLVERS)
def test_a_world_writable_directory_at_the_shared_name_is_not_used(tmp: Path, name: str) -> None:
    planted = tmp / "entroly"
    planted.mkdir()
    planted.chmod(0o777)

    chosen = _resolve(name)

    assert not chosen.is_relative_to(planted), f"{name} wrote into {planted}"
    assert list(planted.iterdir()) == []


@pytest.mark.parametrize("name", RESOLVERS)
def test_the_fallback_belongs_to_this_account_alone(tmp: Path, name: str) -> None:
    _resolve(name)

    info = _mine(tmp).lstat()
    assert info.st_uid == os.getuid()
    assert info.st_mode & 0o777 == 0o700


def _world_writable(tmp: Path) -> Path:
    _mine(tmp).mkdir()
    _mine(tmp).chmod(0o777)
    return _mine(tmp)


def _symlink(tmp: Path) -> Path:
    elsewhere = tmp.parent / "chosen-by-someone-else"
    elsewhere.mkdir(mode=0o700)
    _mine(tmp).symlink_to(elsewhere, target_is_directory=True)
    return elsewhere


def _owned_by_another_account(tmp: Path, monkeypatch) -> Path:
    other = os.getuid() + 1
    theirs = tmp / f"entroly-{other}"
    theirs.mkdir(mode=0o700)  # owned by the real uid, which is not "us" below
    monkeypatch.setattr(os, "getuid", lambda: other)
    return theirs


UNSAFE = {
    "world-writable": lambda tmp, monkeypatch: _world_writable(tmp),
    "a symlink": lambda tmp, monkeypatch: _symlink(tmp),
    "another account's": _owned_by_another_account,
}


@pytest.mark.parametrize("unsafe", sorted(UNSAFE))
@pytest.mark.parametrize("name", RESOLVERS[1:])
def test_an_unsafe_per_account_directory_is_never_used(tmp: Path, monkeypatch, unsafe: str, name: str) -> None:
    refused = UNSAFE[unsafe](tmp, monkeypatch).resolve()

    chosen = _resolve(name)

    assert not chosen.resolve().is_relative_to(refused), f"{name} used {unsafe} {refused}"


@pytest.mark.parametrize("unsafe", sorted(UNSAFE))
def test_checkpoints_refuse_an_unsafe_directory_and_say_what_to_do(tmp: Path, monkeypatch, unsafe: str) -> None:
    UNSAFE[unsafe](tmp, monkeypatch)

    with pytest.raises(OSError) as caught:
        config._project_checkpoint_dir()

    message = str(caught.value)
    assert "ENTROLY_DIR" in message
    assert str(_mine(tmp)) in message, message
    assert not message.startswith("[Errno"), message


def test_the_work_graph_reader_ignores_checkpoints_planted_at_the_shared_name(tmp: Path) -> None:
    import hashlib

    root = tmp.parent / "repo"
    root.mkdir()
    project_hash = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    planted = tmp / "entroly" / "checkpoints" / project_hash
    planted.mkdir(parents=True)

    assert _existing_checkpoint_dir(root, None) is None

    mine = _mine(tmp) / "checkpoints" / project_hash
    mine.mkdir(parents=True)
    _mine(tmp).chmod(0o700)
    assert _existing_checkpoint_dir(root, None) == mine


def _entroly(cwd: Path, home: Path, tmp: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(ROOT),
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "NO_COLOR": "1",
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "entroly", *argv],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=120,
    )


@pytest.fixture()
def room(tmp_path: Path) -> tuple[Path, Path, Path]:
    home = tmp_path / "home-is-a-file"
    home.write_text("not a directory", encoding="utf-8")
    shared = tmp_path / "tmp"
    shared.mkdir()
    (shared / "entroly").write_text("another account got here first", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("def fix():\n    return 1\n", encoding="utf-8")
    return project, home, shared


def test_simulate_works_with_a_read_only_home_after_another_account_fell_back(room) -> None:
    project, home, shared = room

    done = _entroly(project, home, shared, "simulate", "--query", "fix")

    assert done.returncode == 0, done.stdout[-800:] + done.stderr[-800:]
    assert "Errno" not in done.stdout + done.stderr


def test_the_welcome_appears_once_with_a_read_only_home(room) -> None:
    project, home, shared = room

    first = _entroly(project, home, shared, "status")
    second = _entroly(project, home, shared, "status")

    assert "This message appears once" in first.stdout + first.stderr
    assert "This message appears once" not in second.stdout + second.stderr


def test_when_nothing_private_is_writable_the_error_names_the_fix(room) -> None:
    project, home, shared = room
    (shared / f"entroly-{os.getuid()}").write_text("taken too", encoding="utf-8")

    done = _entroly(project, home, shared, "simulate", "--query", "fix")

    assert done.returncode != 0
    assert "ENTROLY_DIR" in done.stdout + done.stderr
    assert "Traceback" not in done.stdout + done.stderr
