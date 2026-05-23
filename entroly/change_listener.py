"""
Workspace Change Listener
=========================

Bridges repo changes into the CogOps data plane.

Responsibilities:
  1. Detect new, modified, and deleted source files
  2. Mark affected beliefs stale
  3. Recompile changed files into fresh belief artifacts
  4. Run verification after each sync
  5. Persist a sync summary into actions/

This is the change-driven glue between Truth, Belief, and Verification.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .belief_compiler import BeliefCompiler
from .change_pipeline import ChangePipeline
from .vault import VaultManager
from .verification_engine import VerificationEngine

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = {".py", ".rs", ".js", ".jsx", ".ts", ".tsx"}
_SKIP_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    "node_modules", "target", "dist", "build", ".tmp",
}


@dataclass
class WorkspaceSyncResult:
    status: str
    project_dir: str
    changed_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    beliefs_written: int = 0
    verification_summary: dict[str, Any] = field(default_factory=dict)
    action_path: str = ""
    refresh_result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    scanned_files: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "project_dir": self.project_dir,
            "changed_files": self.changed_files,
            "deleted_files": self.deleted_files,
            "beliefs_written": self.beliefs_written,
            "verification_summary": self.verification_summary,
            "action_path": self.action_path,
            "refresh_result": self.refresh_result,
            "errors": self.errors,
            "scanned_files": self.scanned_files,
        }


class WorkspaceChangeListener:
    """Polls a workspace and feeds file changes into the belief pipeline."""

    def __init__(
        self,
        vault: VaultManager,
        compiler: BeliefCompiler,
        verifier: VerificationEngine,
        change_pipe: ChangePipeline,
        project_dir: str,
        state_path: str | None = None,
    ):
        self._vault = vault
        self._compiler = compiler
        self._verifier = verifier
        self._change_pipe = change_pipe
        self._project_dir = Path(project_dir).resolve()
        state_root = self._vault.config.path.parent
        state_root.mkdir(parents=True, exist_ok=True)
        self._state_path = Path(state_path).resolve() if state_path else state_root / "change_listener_state.json"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def scan_once(self, force: bool = False, max_files: int = 100) -> WorkspaceSyncResult:
        current = self._snapshot()
        previous = {} if force else self._load_state()

        changed = []
        for rel_path, mtime in current.items():
            prev = previous.get(rel_path)
            if force or prev is None or mtime > prev:
                changed.append(rel_path)

        deleted = [rel_path for rel_path in previous.keys() if rel_path not in current]

        result = WorkspaceSyncResult(
            status="no_changes",
            project_dir=str(self._project_dir),
            changed_files=changed[:max_files],
            deleted_files=deleted,
            scanned_files=len(current),
        )

        if not changed and not deleted and not force:
            return result

        result.status = "synced"

        refresh_targets = result.changed_files + result.deleted_files
        if refresh_targets:
            result.refresh_result = self._change_pipe.refresh_docs(refresh_targets)

        for rel_path in result.changed_files:
            abs_path = self._project_dir / rel_path
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
                if not content.strip():
                    continue
                belief = self._compiler.compile_file(rel_path.replace('\\', '/'), content)
                if belief is None:
                    continue
                self._vault.write_belief(belief)
                result.beliefs_written += 1
            except Exception as exc:
                result.errors.append(f"{rel_path}: {exc}")

        verification = self._verifier.full_verification_pass()
        result.verification_summary = verification.to_dict()

        action = self._vault.write_action(
            title=f"Workspace Sync: {len(result.changed_files)} changed, {len(result.deleted_files)} deleted",
            content=self._render_summary(result),
            action_type="workspace_sync",
        )
        result.action_path = action.get("path", "")

        self._save_state(current)
        logger.info(
            "WorkspaceChangeListener: synced %s changed / %s deleted -> %s beliefs",
            len(result.changed_files), len(result.deleted_files), result.beliefs_written,
        )
        return result

    def start(self, interval_s: int = 120, max_files: int = 100, force_initial: bool = False) -> dict[str, Any]:
        if self._thread and self._thread.is_alive():
            return {
                "status": "already_running",
                "project_dir": str(self._project_dir),
                "interval_s": interval_s,
                "state_path": str(self._state_path),
            }

        self._stop.clear()

        def _loop() -> None:
            if force_initial:
                try:
                    self.scan_once(force=True, max_files=max_files)
                except Exception as exc:
                    logger.warning("WorkspaceChangeListener initial sync failed: %s", exc)
            while not self._stop.wait(interval_s):
                try:
                    self.scan_once(force=False, max_files=max_files)
                except Exception as exc:
                    logger.warning("WorkspaceChangeListener sync failed: %s", exc)

        self._thread = threading.Thread(target=_loop, daemon=True, name="entroly-workspace-sync")
        self._thread.start()
        return {
            "status": "started",
            "project_dir": str(self._project_dir),
            "interval_s": interval_s,
            "state_path": str(self._state_path),
        }

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        return {"status": "stopped", "project_dir": str(self._project_dir)}

    def _snapshot(self) -> dict[str, float]:
        snapshot: dict[str, float] = {}
        for path in self._discover_source_files():
            try:
                rel = path.relative_to(self._project_dir).as_posix()
                snapshot[rel] = path.stat().st_mtime
            except OSError:
                continue
        return snapshot

    def _discover_source_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self._project_dir.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in _SUPPORTED_EXTS:
                continue
            files.append(path)
        files.sort()
        return files

    def _load_state(self) -> dict[str, float]:
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, state: dict[str, float]) -> None:
        self._state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _render_summary(self, result: WorkspaceSyncResult) -> str:
        changed = ", ".join(result.changed_files[:20]) or "None"
        deleted = ", ".join(result.deleted_files[:20]) or "None"
        verification = result.verification_summary
        return (
            f"# Workspace Sync\n\n"
            f"- Project: `{result.project_dir}`\n"
            f"- Scanned files: {result.scanned_files}\n"
            f"- Changed files: {len(result.changed_files)}\n"
            f"- Deleted files: {len(result.deleted_files)}\n"
            f"- Beliefs written: {result.beliefs_written}\n"
            f"- Verification checked: {verification.get('total_beliefs_checked', 0)}\n"
            f"- Contradictions: {verification.get('contradictions', 0)}\n"
            f"- Stale beliefs: {verification.get('stale_count', 0)}\n"
            f"- Mean confidence: {verification.get('mean_confidence', 0)}\n\n"
            f"## Changed\n{changed}\n\n"
            f"## Deleted\n{deleted}\n"
        )
