"""
entroly.watch_mode — automatic re-compression when files change.

Closes competitive gap vs Repomix: `repomix --watch`

Uses Python's built-in polling (no extra dependencies). Falls back to
watchdog if installed for lower-latency inotify/FSEvents/ReadDirectoryChanges.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Simple polling watcher (zero extra deps)
# ---------------------------------------------------------------------------

class _PollingWatcher:
    """Poll file system every *interval* seconds for changes."""

    def __init__(self, root: Path, interval: float = 1.0, patterns: list[str] | None = None):
        self.root = root
        self.interval = interval
        self.patterns = patterns or ["**/*"]
        self._snapshots: dict[str, str] = {}

    def _snapshot(self) -> dict[str, str]:
        snap: dict[str, str] = {}
        for pattern in self.patterns:
            for path in self.root.glob(pattern):
                if path.is_file():
                    try:
                        stat = path.stat()
                        snap[str(path)] = f"{stat.st_size}:{stat.st_mtime}"
                    except OSError:
                        pass
        return snap

    def changed_paths(self) -> list[str]:
        new_snap = self._snapshot()
        changed = []
        # modified or added
        for path, sig in new_snap.items():
            if self._snapshots.get(path) != sig:
                changed.append(path)
        # deleted
        for path in self._snapshots:
            if path not in new_snap:
                changed.append(path)
        self._snapshots = new_snap
        return changed

    def watch(self, callback: Callable[[list[str]], None]) -> None:
        """Block and call *callback* with list of changed paths on each change."""
        # Initial snapshot (no callback)
        self._snapshots = self._snapshot()
        try:
            while True:
                time.sleep(self.interval)
                changed = self.changed_paths()
                if changed:
                    callback(changed)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# Watchdog-backed watcher (optional, faster)
# ---------------------------------------------------------------------------

def _try_watchdog_watcher(root: Path, callback: Callable[[list[str]], None]) -> bool:
    """Try to use watchdog. Returns True if succeeded, False if not installed."""
    try:
        from watchdog.observers import Observer  # type: ignore
        from watchdog.events import FileSystemEventHandler  # type: ignore
    except ImportError:
        return False

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if not event.is_directory:
                callback([event.src_path])

    observer = Observer()
    observer.schedule(_Handler(), str(root), recursive=True)
    observer.start()
    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def watch_and_repack(
    root: str | Path,
    on_change: Callable[[list[str]], None],
    *,
    interval: float = 1.0,
    patterns: list[str] | None = None,
    use_watchdog: bool = True,
) -> None:
    """
    Watch *root* for changes and call *on_change* with a list of changed paths.

    Parameters
    ----------
    root : str | Path
        Directory to watch.
    on_change : Callable[[list[str]], None]
        Called with a list of changed/added/deleted file paths.
    interval : float
        Polling interval in seconds (only used for the fallback watcher).
    patterns : list[str] | None
        Glob patterns to watch (default: all files).
    use_watchdog : bool
        Attempt to use watchdog for lower-latency events (default: True).
    """
    root = Path(root).resolve()
    if use_watchdog and _try_watchdog_watcher(root, on_change):
        return
    # Fallback to polling
    watcher = _PollingWatcher(root, interval=interval, patterns=patterns)
    watcher.watch(on_change)


def run_watch_loop(
    root: str | Path,
    pack_fn: Callable[[], str],
    output_path: str | Path | None = None,
    *,
    interval: float = 1.0,
    quiet: bool = False,
) -> None:
    """
    High-level watch loop: call *pack_fn* once initially and on every change,
    optionally writing the result to *output_path*.

    Parameters
    ----------
    root : str | Path
        Directory to watch.
    pack_fn : Callable[[], str]
        Zero-argument function that produces the packed content string.
    output_path : str | Path | None
        If given, write packed content to this file.
    interval : float
        Polling interval in seconds.
    quiet : bool
        Suppress progress messages.
    """
    root = Path(root).resolve()

    def _run_and_write(changed_files: list[str] | None = None) -> None:
        t0 = time.monotonic()
        try:
            result = pack_fn()
        except Exception as exc:
            print(f"\n  ⚠ Pack error: {exc}", file=sys.stderr)
            return
        elapsed = time.monotonic() - t0
        tokens = len(result) // 4
        if output_path:
            Path(output_path).write_text(result, encoding="utf-8")
            if not quiet:
                print(f"  ✓ Repacked → {output_path}  ({tokens:,} tokens, {elapsed:.2f}s)")
        else:
            if not quiet:
                print(f"  ✓ Repacked  ({tokens:,} tokens, {elapsed:.2f}s)")

    if not quiet:
        print(f"\n  Entroly watch mode — watching {root}")
        print("  Press Ctrl+C to stop.\n")

    # Initial pack
    _run_and_write()

    def _on_change(changed: list[str]) -> None:
        if not quiet:
            n = len(changed)
            label = changed[0] if n == 1 else f"{n} files"
            print(f"\n  Change detected: {label}")
        _run_and_write(changed)

    watch_and_repack(root, _on_change, interval=interval)
