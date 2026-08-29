"""Seed the belief vault without the user being told to.

The dashboard shipped an empty panel reading "No beliefs yet -- run
compile_beliefs to seed the vault". A product that knows exactly what to do
next, and asks the user to do it, has chosen to be a tool rather than a
service. Indexing already walked the tree; compiling beliefs from it needs no
further decision.

Three constraints shape this:

*It must not delay startup.* Compilation runs on a background thread. The first
request is answered from the index, which is already built; beliefs enrich
later work and nothing waits on them.

*It must not redo settled work.* A marker records the tree state that was last
compiled. Re-running against an unchanged tree is skipped, so opening a
repository ten times compiles once.

*It must fail open.* A vault that cannot be written is a degraded feature, not
a broken session. Failures are recorded for the caller to report and never
raised into the indexing path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("entroly.belief_autoseed")

# Bounded so a large repository cannot turn a first run into a long one. The
# cap is on files considered, not beliefs written.
_DEFAULT_MAX_FILES = 400
_MARKER_NAME = "autoseed.json"

_started: set[str] = set()
_lock = threading.Lock()


def autoseed_enabled() -> bool:
    """On unless explicitly disabled.

    Default-on is the point: the previous behaviour was default-off in
    practice, because it required a command most users never ran.
    """
    return os.environ.get("ENTROLY_BELIEF_AUTOSEED", "1").strip() not in {
        "0", "false", "no",
    }


def _vault_base() -> Path:
    override = os.environ.get("ENTROLY_VAULT")
    if override:
        return Path(override)
    root = os.environ.get("ENTROLY_DIR") or os.path.join(os.getcwd(), ".entroly")
    return Path(root) / "vault"


def _tree_signature(directory: Path, max_files: int) -> str:
    """A cheap fingerprint of what would be compiled.

    Names, sizes and modification times rather than contents: reading every
    file to decide whether to read every file would cost what it is trying to
    avoid. A missed edit costs one stale belief until the next change, which is
    the same exposure the incremental watcher already accepts.
    """
    digest = hashlib.sha256()
    seen = 0
    # Every matching file is hashed, not just the first ``max_files``. Stopping
    # early made each file sorting after the cap invisible to the fingerprint,
    # so editing one never invalidated the marker and its beliefs went stale
    # permanently -- not "until the next change", which is the exposure this
    # was documented as accepting. The cap belongs on what gets compiled, not
    # on what gets noticed; stat() is cheap enough to run over the whole tree.
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".py", ".rs", ".ts", ".js"}:
            continue
        if any(part in {".git", "node_modules", "__pycache__", ".venv", "target",
                        ".entroly"} for part in path.parts):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        # Nanosecond mtime: int(st_mtime) could not distinguish two edits in
        # the same second that left the size unchanged.
        digest.update(str(path).encode("utf-8", "surrogatepass"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        seen += 1
    digest.update(str(seen).encode("ascii"))
    return digest.hexdigest()


def _read_all_markers(vault: Path) -> dict[str, Any]:
    try:
        data = json.loads((vault / _MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _marker_key(directory: Path) -> str:
    return str(directory)


def _read_marker(vault: Path, directory: Path) -> dict[str, Any]:
    """The marker for one project.

    Markers are keyed by project root. A single unkeyed signature meant two
    repositories sharing an ENTROLY_DIR overwrote each other: opening A then B
    then A recompiled all three times, because the marker only ever described
    whichever was opened last.
    """
    entry = _read_all_markers(vault).get(_marker_key(directory))
    return entry if isinstance(entry, dict) else {}


def _write_marker(vault: Path, directory: Path, payload: dict[str, Any]) -> None:
    try:
        vault.mkdir(parents=True, exist_ok=True)
        markers = _read_all_markers(vault)
        markers[_marker_key(directory)] = payload
        target = vault / _MARKER_NAME
        # Renamed into place: the proxy and MCP server can both be writing
        # this, and a truncating write would let one read a partial document
        # and lose every project's marker at once.
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(markers, sort_keys=True), encoding="utf-8")
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError as exc:
        logger.debug("belief autoseed marker not written: %s", exc)


def compile_now(directory: str | os.PathLike[str],
                max_files: int = _DEFAULT_MAX_FILES) -> dict[str, Any]:
    """Compile beliefs for ``directory``, skipping an unchanged tree.

    Returns a summary rather than raising. The caller is usually a startup
    path, where an exception would trade a missing panel for a failed session.
    """
    target = Path(directory).resolve()
    vault_base = _vault_base()
    signature = _tree_signature(target, max_files)

    marker = _read_marker(vault_base, target)
    if marker.get("signature") == signature:
        return {"status": "skipped", "reason": "tree unchanged since last compile",
                "signature": signature}

    try:
        from .belief_compiler import BeliefCompiler
        from .vault import VaultConfig, VaultManager

        vault = VaultManager(VaultConfig(base_path=str(vault_base)))
        vault.ensure_structure()
        result = BeliefCompiler(vault).compile_directory(str(target), max_files)
    except Exception as exc:  # noqa: BLE001 - a degraded panel beats a dead session
        logger.warning("belief autoseed failed: %s", exc)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    _write_marker(vault_base, target, {
        "signature": signature,
        "files_processed": int(getattr(result, "files_processed", 0)),
        "beliefs_written": int(getattr(result, "beliefs_written", 0)),
    })
    return {
        "status": "compiled",
        "files_processed": int(getattr(result, "files_processed", 0)),
        "entities_extracted": int(getattr(result, "entities_extracted", 0)),
        "beliefs_written": int(getattr(result, "beliefs_written", 0)),
        "errors": len(getattr(result, "errors", []) or []),
        "signature": signature,
    }


def start_autoseed(directory: str | os.PathLike[str] | None = None,
                   max_files: int = _DEFAULT_MAX_FILES) -> bool:
    """Begin compiling beliefs in the background. Returns whether it started.

    Idempotent per directory and process, so repeated indexing during one
    session does not stack compilations.
    """
    if not autoseed_enabled():
        return False
    target = str(Path(directory or os.getcwd()).resolve())
    with _lock:
        if target in _started:
            return False
        _started.add(target)

    def run() -> None:
        summary = compile_now(target, max_files)
        if summary.get("status") == "compiled":
            logger.info(
                "Seeded %s belief(s) from %s file(s)",
                summary.get("beliefs_written", 0), summary.get("files_processed", 0))

    threading.Thread(target=run, name="entroly-belief-autoseed", daemon=True).start()
    return True


def reset_for_tests() -> None:
    with _lock:
        _started.clear()


__all__ = [
    "autoseed_enabled",
    "compile_now",
    "reset_for_tests",
    "start_autoseed",
]
