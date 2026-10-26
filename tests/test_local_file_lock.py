from __future__ import annotations

import errno
import multiprocessing
import os
import sys
import time
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from entroly import local_file_lock
from entroly.local_file_lock import exclusive_file_lock


def _lock_worker(path: Path, connection: Connection) -> None:
    """Accept explicit commands so contention cannot depend on startup speed."""
    try:
        connection.send("ready")
        while True:
            command, timeout = connection.recv()
            if command != "acquire":
                raise AssertionError(f"unexpected worker command: {command}")
            try:
                with exclusive_file_lock(path, timeout=timeout):
                    connection.send("acquired")
                    command = connection.recv()
                    if command == "exit":
                        # Bypass context-manager/finally cleanup completely.
                        os._exit(23)
                    if command != "release":
                        raise AssertionError(f"unexpected lock command: {command}")
                connection.send("released")
            except TimeoutError:
                connection.send("timeout")
    finally:
        connection.close()


def _expect(connection: Connection, message: str) -> None:
    assert connection.poll(10), f"timed out waiting for worker: {message}"
    assert connection.recv() == message


@contextmanager
def _worker(path: Path) -> Iterator[tuple[Connection, multiprocessing.Process]]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_lock_worker, args=(path, child))
    process.start()
    child.close()
    try:
        _expect(parent, "ready")
        yield parent, process
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        parent.close()
        assert not process.is_alive(), "lock worker failed to exit"


def test_new_lock_stays_empty_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "private" / "session.lock"
    with exclusive_file_lock(path):
        # Windows denies reads of the locked byte range, including past EOF.
        assert path.stat().st_size == 0
        inode = path.stat().st_ino
        if os.name != "nt":
            assert path.parent.stat().st_mode & 0o077 == 0
    assert path.exists()
    assert path.stat().st_ino == inode
    assert path.read_bytes() == b""


def test_existing_lock_contents_and_inode_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "session.lock"
    path.write_bytes(b"existing lock contents")
    inode = path.stat().st_ino
    for _ in range(2):
        with exclusive_file_lock(path, timeout=0):
            assert path.stat().st_ino == inode
        assert path.read_bytes() == b"existing lock contents"
    assert path.stat().st_ino == inode


def test_independent_processes_exclude_and_then_handoff(tmp_path: Path) -> None:
    path = tmp_path / "session.lock"
    with _worker(path) as (holder, _), _worker(path) as (contender, _):
        holder.send(("acquire", 1.0))
        _expect(holder, "acquired")
        inode = path.stat().st_ino

        # The holder waits for our explicit release. A completed contender
        # timeout proves that it attempted acquisition during that interval.
        contender.send(("acquire", 0.1))
        _expect(contender, "timeout")
        holder.send("release")
        _expect(holder, "released")

        contender.send(("acquire", 1.0))
        _expect(contender, "acquired")
        contender.send("release")
        _expect(contender, "released")
        assert path.stat().st_ino == inode
        assert path.read_bytes() == b""


@pytest.mark.parametrize("timeout", [0.0, 0.1])
def test_timeout_fails_before_body_while_process_holds_lock(
    tmp_path: Path, timeout: float
) -> None:
    path = tmp_path / "session.lock"
    with _worker(path) as (holder, _):
        holder.send(("acquire", 1.0))
        _expect(holder, "acquired")
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="exclusive file lock"):
            with exclusive_file_lock(path, timeout=timeout):
                pytest.fail("entered a lock held by another process")
        elapsed = time.monotonic() - started
        assert timeout <= elapsed < 2.0
        holder.send("release")
        _expect(holder, "released")
    with exclusive_file_lock(path, timeout=0):
        pass


def test_os_releases_lock_after_process_exit_without_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "session.lock"
    with _worker(path) as (holder, process):
        holder.send(("acquire", 1.0))
        _expect(holder, "acquired")
        inode = path.stat().st_ino
        holder.send("exit")
        process.join(timeout=10)
        assert process.exitcode == 23

    with _worker(path) as (successor, _):
        successor.send(("acquire", 0.0))
        _expect(successor, "acquired")
        successor.send("release")
        _expect(successor, "released")
    assert path.stat().st_ino == inode
    assert path.read_bytes() == b""


def test_body_exception_releases_lock_for_another_process(tmp_path: Path) -> None:
    path = tmp_path / "session.lock"
    with pytest.raises(RuntimeError, match="body failure"):
        with exclusive_file_lock(path):
            raise RuntimeError("body failure")
    with _worker(path) as (successor, _):
        successor.send(("acquire", 0.0))
        _expect(successor, "acquired")
        successor.send("release")
        _expect(successor, "released")


@pytest.mark.parametrize("timeout", [-1.0, float("inf"), -float("inf"), float("nan")])
def test_unbounded_or_negative_timeout_is_rejected(
    tmp_path: Path, timeout: float
) -> None:
    path = tmp_path / "private" / "session.lock"
    with pytest.raises(ValueError, match="finite and non-negative"):
        with exclusive_file_lock(path, timeout=timeout):
            pytest.fail("entered with an invalid timeout")
    assert not path.parent.exists()


def _fake_backend(monkeypatch, platform: str, operation) -> tuple[int, int]:
    """Exercise both retry branches; real process tests use the host backend."""
    monkeypatch.setattr(local_file_lock, "os", SimpleNamespace(name=platform))
    if platform == "nt":
        monkeypatch.setitem(
            sys.modules,
            "msvcrt",
            SimpleNamespace(locking=operation, LK_NBLCK=1, LK_UNLCK=2),
        )
        return 1, 2
    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(flock=operation, LOCK_EX=2, LOCK_NB=4, LOCK_UN=8),
    )
    return 6, 8


@pytest.mark.parametrize("platform", ["nt", "posix"])
@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EAGAIN, errno.EIO])
def test_backend_errors_fail_closed_with_bounded_nonblocking_retries(
    tmp_path: Path, monkeypatch, platform: str, error_number: int
) -> None:
    path = tmp_path / "session.lock"
    operations = []
    now = [0.0]
    sleeps = []

    def operation(descriptor, flag, *count):
        operations.append((flag, count))
        assert path.stat().st_size == 0
        raise OSError(error_number, "simulated lock failure")

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    acquire_flag, _ = _fake_backend(monkeypatch, platform, operation)
    monkeypatch.setattr(
        local_file_lock, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
    )
    expected_error = OSError if error_number == errno.EIO else TimeoutError
    with pytest.raises(expected_error) as raised:
        with exclusive_file_lock(path, timeout=0.004):
            pytest.fail("lock failure allowed an unlocked body")
    assert all(flag == acquire_flag for flag, _ in operations)
    assert all(count == ((1,) if platform == "nt" else ()) for _, count in operations)
    if error_number == errno.EIO:
        assert len(operations) == 1
        assert raised.value.errno == errno.EIO
        assert sleeps == []
    else:
        assert len(operations) > 1
        assert sum(sleeps) == pytest.approx(0.004)
        assert isinstance(raised.value.__cause__, OSError)
    assert path.read_bytes() == b""


@pytest.mark.parametrize("platform", ["nt", "posix"])
def test_transient_contention_retries_then_unlocks(
    tmp_path: Path, monkeypatch, platform: str
) -> None:
    path = tmp_path / "session.lock"
    operations = []

    def operation(descriptor, flag, *count):
        operations.append(flag)
        if len(operations) == 1:
            raise OSError(errno.EAGAIN, "simulated contention")

    acquire_flag, release_flag = _fake_backend(monkeypatch, platform, operation)
    sleeps = []
    monkeypatch.setattr(
        local_file_lock,
        "time",
        SimpleNamespace(monotonic=lambda: 0.0, sleep=sleeps.append),
    )
    with exclusive_file_lock(path):
        assert operations == [acquire_flag, acquire_flag]
    assert operations == [acquire_flag, acquire_flag, release_flag]
    assert sleeps == [0.001]
