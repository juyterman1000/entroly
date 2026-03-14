"""
Entroly Auto-Index — Git-aware codebase discovery and ingestion.

On first startup (or when no persistent index exists), automatically
walks all git-tracked files, ingests relevant source code, and builds
the dependency graph. Zero manual configuration needed.

Usage:
    from entroly.auto_index import auto_index
    auto_index(engine)  # Ingests all git-tracked source files
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from entroly.server import EntrolyEngine

logger = logging.getLogger("entroly")

# File extensions to index (covers 95%+ of production codebases)
SUPPORTED_EXTENSIONS = frozenset({
    # Systems
    ".rs", ".c", ".cpp", ".h", ".hpp", ".zig",
    # Web / JS / TS
    ".js", ".ts", ".jsx", ".tsx", ".vue", ".svelte",
    # Python
    ".py", ".pyi",
    # JVM
    ".java", ".kt", ".scala",
    # Go
    ".go",
    # Ruby
    ".rb",
    # Shell / Config
    ".sh", ".bash", ".zsh",
    ".toml", ".yaml", ".yml", ".json",
    # Docs that matter
    ".md", ".rst",
    # SQL
    ".sql",
    # Docker / CI
    ".dockerfile",
})

# Files to always skip regardless of extension
SKIP_PATTERNS = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock",
    "poetry.lock", "Pipfile.lock", "composer.lock", "Gemfile.lock",
    ".DS_Store", "thumbs.db",
})

# Max file size to ingest (50 KB — larger files are usually generated)
MAX_FILE_BYTES = 50 * 1024

# Max files to index in a single pass (safety limit)
MAX_FILES = 5000


def _git_ls_files(project_dir: str) -> list[str]:
    """Get all git-tracked files, respecting .gitignore."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return []


def _walk_fallback(project_dir: str) -> list[str]:
    """Fallback file discovery when git is not available."""
    files = []
    for root, dirs, filenames in os.walk(project_dir):
        # Skip hidden dirs, node_modules, __pycache__, .git, etc.
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in {"node_modules", "__pycache__", "target", "dist",
                         "build", ".git", "venv", ".venv", "env"}
        ]
        for fname in filenames:
            rel = os.path.relpath(os.path.join(root, fname), project_dir)
            files.append(rel)
            if len(files) >= MAX_FILES:
                return files
    return files


def _should_index(rel_path: str) -> bool:
    """Decide whether a file should be indexed."""
    basename = os.path.basename(rel_path)

    # Skip lock files and system files
    if basename in SKIP_PATTERNS:
        return False

    # Dockerfile special case (no extension)
    if basename.startswith("Dockerfile"):
        return True

    # Check extension
    _, ext = os.path.splitext(basename)
    return ext.lower() in SUPPORTED_EXTENSIONS


def _estimate_tokens(content: str) -> int:
    """Fast token estimation: ~4 chars per token for code."""
    return max(1, len(content) // 4)


def auto_index(
    engine: "EntrolyEngine",
    project_dir: str | None = None,
    force: bool = False,
) -> dict:
    """Auto-index a project's codebase into the Entroly engine.

    Args:
        engine: The EntrolyEngine instance to index into.
        project_dir: Root directory to scan. Defaults to cwd.
        force: If True, re-index even if fragments already exist.

    Returns:
        Summary dict with indexed file count, token count, and duration.
    """
    project_dir = project_dir or os.getcwd()
    project_dir = os.path.abspath(project_dir)

    # Skip if engine already has fragments (loaded from persistent index)
    if not force and engine._use_rust:
        existing = engine._rust.fragment_count()
        if existing > 0:
            logger.info(
                f"Auto-index skipped: {existing} fragments already loaded "
                f"from persistent index"
            )
            return {
                "status": "skipped",
                "reason": "persistent_index_loaded",
                "existing_fragments": existing,
            }

    t0 = time.perf_counter()

    # Discover files
    files = _git_ls_files(project_dir)
    if not files:
        files = _walk_fallback(project_dir)
        discovery = "walk"
    else:
        discovery = "git"

    # Filter to indexable files
    indexable = [f for f in files if _should_index(f)][:MAX_FILES]

    # Ingest each file
    indexed = 0
    total_tokens = 0
    skipped_size = 0
    skipped_read = 0

    for rel_path in indexable:
        abs_path = os.path.join(project_dir, rel_path)

        # Skip files that are too large
        try:
            size = os.path.getsize(abs_path)
            if size > MAX_FILE_BYTES:
                skipped_size += 1
                continue
            if size == 0:
                continue
        except OSError:
            continue

        # Read and ingest
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            skipped_read += 1
            continue

        if not content.strip():
            continue

        tokens = _estimate_tokens(content)
        source = f"file:{rel_path}"

        engine.ingest_fragment(
            content=content,
            source=source,
            token_count=tokens,
            is_pinned=False,
        )

        indexed += 1
        total_tokens += tokens

    elapsed = time.perf_counter() - t0

    # Build dependency graph from import analysis
    if engine._use_rust and indexed > 0:
        try:
            # Trigger a lightweight optimize to build the dep graph
            # (dep graph is built during optimize, not ingest)
            engine.optimize_context(token_budget=1, query="")
        except Exception:
            pass  # Non-critical, dep graph will build on first real optimize

    logger.info(
        f"Auto-indexed {indexed} files ({total_tokens:,} tokens) "
        f"in {elapsed:.1f}s via {discovery} "
        f"[skipped: {skipped_size} too large, {skipped_read} unreadable]"
    )

    return {
        "status": "indexed",
        "files_indexed": indexed,
        "total_tokens": total_tokens,
        "duration_s": round(elapsed, 2),
        "discovery_method": discovery,
        "skipped_too_large": skipped_size,
        "skipped_unreadable": skipped_read,
        "project_dir": project_dir,
    }
