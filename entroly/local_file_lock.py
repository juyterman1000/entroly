"""OS-released advisory locking for cooperating local Entroly processes."""

from __future__ import annotations

import errno
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def exclusive_file_lock(path: Path, *, timeout: float = 30.0) -> Iterator[None]:
    """Hold an exclusive local file lock, or fail before entering the body.

    Acquisition uses bounded nonblocking retries on Windows and POSIX. The OS
    releases the lock when its descriptor closes, including on process exit.
    Keep the lock file in place: unlinking it can split writers across inodes.
    This is advisory coordination, not a distributed-filesystem guarantee.
    """
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("lock timeout must be finite and non-negative")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        # Windows supports locking a byte past EOF. Never initialize a sentinel
        # before acquisition: another process may already hold that byte.
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            def acquire() -> None:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def release() -> None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

        else:
            import fcntl

            def acquire() -> None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release() -> None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        deadline = time.monotonic() + timeout
        delay = 0.001
        while True:
            try:
                acquire()
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out acquiring exclusive file lock {path}"
                    ) from error
                time.sleep(min(delay, remaining))
                delay = min(0.05, delay * 1.5)
        try:
            yield
        finally:
            release()
