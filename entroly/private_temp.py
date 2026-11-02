"""This account's own directory under the system temporary directory.

MCP hosts, containers and managed IDE runtimes can give Entroly a HOME it
cannot write, so every data path falls back to the temporary directory. That
directory is shared by every account on the machine, and the fallback used to
be the fixed name ``<tmp>/entroly``: whoever created it first owned it, the
next account to fall back failed with "Permission denied:
'/tmp/entroly/checkpoints'", and an account that created it world-writable
decided where everyone else's checkpoints and telemetry were written.

The fallback is now ``<tmp>/entroly-<uid>``. It is created mode 0700 and used
only while it is a real directory, owned by this account, that no other
account can open -- the check OpenSSH and tmux make before trusting their own
per-user directories. Anything else is refused, never repaired: an account
that can plant a directory there can also have planted what is inside it.
Windows already gives each account its own temporary directory, so there the
name stays ``entroly``.

Standard library only, so any module can import it without joining an import
cycle.
"""

from __future__ import annotations

import errno
import os
import stat
import tempfile
from pathlib import Path


class TempDirNotPrivate(PermissionError):
    """The per-account temporary directory exists but is not ours alone."""


class NoPrivateDataDir(OSError):
    """Neither the preferred directory nor the per-account fallback is usable."""


def account_temp_path() -> Path:
    """Where this account's fallback directory lives. Creates nothing."""
    getuid = getattr(os, "getuid", None)
    name = "entroly" if getuid is None else f"entroly-{getuid()}"
    return Path(tempfile.gettempdir()) / name


def _not_private(path: Path, uid: int) -> str | None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        return "it is a symbolic link"
    if not stat.S_ISDIR(info.st_mode):
        return "it is not a directory"
    if info.st_uid != uid:
        return f"it belongs to uid {info.st_uid}, not this account"
    if info.st_mode & 0o077:
        return f"other accounts can open it, mode {stat.S_IMODE(info.st_mode):04o}"
    return None


def account_temp_dir() -> Path:
    """Create if needed, verify, and return this account's fallback directory.

    Raises ``TempDirNotPrivate`` when the path exists but is not a private
    directory owned by this account, and the operating system's own error
    when it cannot be created.
    """
    path = account_temp_path()
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        path.mkdir(parents=True, exist_ok=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)  # the temp root itself
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    else:
        os.chmod(path, 0o700)  # a restrictive umask must not lock us out
    reason = _not_private(path, getuid())
    if reason:
        raise TempDirNotPrivate(errno.EACCES, reason, str(path))
    return path


def existing_account_temp_dir() -> Path | None:
    """Return the fallback directory if it already exists and is private."""
    path = account_temp_path()
    getuid = getattr(os, "getuid", None)
    try:
        if getuid is None:
            return path if path.is_dir() and not path.is_symlink() else None
        return None if _not_private(path, getuid()) else path
    except OSError:
        return None


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".write_probe-"):
            pass
        return True
    except OSError:
        return False


def writable_data_dir(preferred: Path) -> Path:
    """Return ``preferred`` if it can be written, else the private fallback.

    When neither can be used, ``preferred`` comes back unchecked, so callers
    that resolve paths at import time never raise and a later write fails
    with the operating system's error instead of landing somewhere another
    account controls.
    """
    if _writable(preferred):
        return preferred
    try:
        return account_temp_dir()
    except OSError:
        return preferred


def reason(exc: BaseException) -> str:
    """The human part of an OSError, without the errno prefix."""
    return getattr(exc, "strerror", None) or str(exc)
