"""Local artifact store for Context Receipts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


_STORE_DIR = Path(".entroly") / "receipts"

#: Kept as the relative shape of the store, not as a resolved location.
#: Anything that needs a real path must call `resolve_store()` -- these are
#: relative, so binding one to a variable freezes it against whatever the cwd
#: happened to be at import time.
DEFAULT_STORE = _STORE_DIR
DEFAULT_INDEX = _STORE_DIR / "index.json"


def resolve_store(start: str | Path | None = None) -> Path:
    """Locate the receipt store for ``start``, searching upward.

    The store used to be the relative path `.entroly/receipts`, resolved
    against whatever the process cwd was. So `entroly ingest .` at a project
    root followed by `entroly select` from any subdirectory reported "No
    Context Receipt index found" and advised running ingest again -- which
    would not have found the existing index either, it would have written a
    second one. Every project tool a user already has (git, cargo, npm, ruff)
    searches upward for its project root, so this does too.

    The walk stops at a repository boundary: a store belonging to some
    unrelated parent directory is not this project's evidence, and silently
    reading it would be worse than not finding one. When no store exists yet,
    the answer is the repository root if there is one, so that a project has a
    single store rather than one per directory a command was run from.
    """
    current = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for directory in (current, *current.parents):
        candidate = directory / _STORE_DIR
        if candidate.is_dir():
            return candidate
        if (directory / ".git").exists():
            return directory / _STORE_DIR
    return current / _STORE_DIR


def default_index_path(start: str | Path | None = None) -> Path:
    return resolve_store(start) / "index.json"


def latest_pointer_path(start: str | Path | None = None) -> Path:
    return resolve_store(start) / "latest_receipt.txt"


def ensure_store(path: Path | None = None) -> Path:
    path = resolve_store() if path is None else Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    return write_text(
        path, json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> Path:
    """Atomically replace an artifact with private POSIX permissions.

    Windows files inherit the parent directory's ACL. This is neither encryption
    nor protection from another process running as the same user. Existing
    directories keep their permissions; select a trusted parent directory.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.is_symlink():
        raise ValueError("refusing to overwrite a symlinked receipt artifact")
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            if os.name == "posix":
                os.fchmod(stream.fileno(), 0o600)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def default_receipt_path(receipt_id: str) -> Path:
    return ensure_store() / f"{receipt_id}.json"


def default_report_path(receipt_id: str) -> Path:
    return ensure_store() / f"{receipt_id}.md"


def set_latest_receipt(path: str | Path) -> None:
    ensure_store()
    write_text(latest_pointer_path(), str(Path(path)))


def latest_receipt_path() -> Path | None:
    pointer = latest_pointer_path()
    if not pointer.exists():
        return None
    raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    # The pointer records the path the receipt was written to. That was
    # cwd-relative, so a pointer written from the project root did not resolve
    # from a subdirectory; anchor a relative one to the store that holds it.
    recorded = Path(raw)
    if recorded.is_absolute() or recorded.exists():
        return recorded
    return pointer.parent / recorded.name
