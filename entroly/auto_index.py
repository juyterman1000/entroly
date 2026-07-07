"""
Entroly Auto-Index — Git-aware codebase discovery and ingestion.

On first startup (or when no persistent index exists), automatically
walks all git-tracked files, ingests relevant source code, and builds
the dependency graph. Zero manual configuration needed.

LPI (Lazy Progressive Index):
  Phase 1: Parallel file reading via ThreadPoolExecutor (I/O-bound)
  Phase 2: Single batch_ingest() PyO3 call — rayon inside Rust processes
           SimHash, skeleton, entropy for ALL files simultaneously.
  Result:  1 PyO3 crossing instead of N. O(N) entropy instead of O(N²).
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from entroly.server import EntrolyEngine

logger = logging.getLogger("entroly")

# File extensions to index (covers 95%+ of production codebases)
SUPPORTED_EXTENSIONS = frozenset({
    # Systems
    ".rs", ".c", ".cpp", ".h", ".hpp", ".cc", ".hxx", ".zig",
    # Web / JS / TS
    ".js", ".ts", ".jsx", ".tsx", ".mjs", ".mts", ".cjs", ".cts",
    ".vue", ".svelte",
    # Python
    ".py", ".pyi",
    # JVM
    ".java", ".kt", ".scala",
    # .NET / C#
    ".cs", ".csx", ".fs",
    # Go
    ".go",
    # Swift / iOS
    ".swift",
    # Ruby
    ".rb",
    # PHP
    ".php",
    # Dart / Flutter
    ".dart",
    # Elixir / Erlang
    ".ex", ".exs",
    # Lua
    ".lua",
    # R
    ".r",
    # Shell / Config
    ".sh", ".bash", ".zsh",
    ".toml", ".yaml", ".yml", ".json",
    # Terraform / IaC
    ".tf", ".hcl",
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

# Directories that are never useful as first-run project evidence. These show
# up most painfully in fresh Windows source checkouts where a local virtualenv
# such as `.fresh/Lib/site-packages` can outrank the user's actual source files.
SKIP_DIR_NAMES = frozenset({
    ".git", ".hg", ".svn",
    ".venv", "venv", "env", ".env",
    ".fresh", "fresh", ".venv-clean", "venv-clean",
    ".tox", ".nox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "__pycache__", "site-packages", "dist-packages",
    "node_modules", "bower_components",
    "target", "dist", "build", ".next", ".nuxt", "coverage", "htmlcov",
})

# Max file size to ingest. Default 50 KB filters out generated artifacts
# (lockfiles, minified JS, large generated specs). Override with
# ENTROLY_MAX_FILE_BYTES — useful on documentation/list-style repos where
# the single most-important file (e.g. a long README) exceeds the default
# and would otherwise be silently dropped from the index. Capped at the
# hard ceiling below so callers can't accidentally pull in multi-MB blobs.
def _resolve_max_file_bytes() -> int:
    raw = os.environ.get("ENTROLY_MAX_FILE_BYTES")
    if not raw:
        return 50 * 1024
    try:
        v = int(raw)
        return max(1024, min(v, 500 * 1024))
    except ValueError:
        return 50 * 1024


def _resolve_source_file_soft_max_bytes() -> int:
    raw = os.environ.get("ENTROLY_MAX_SOURCE_FILE_BYTES", str(192 * 1024))
    try:
        value = int(raw)
    except ValueError:
        value = 192 * 1024
    return max(50 * 1024, min(value, 500 * 1024))


def _resolve_max_files() -> int:
    raw = os.environ.get("ENTROLY_MAX_FILES", "5000")
    try:
        return max(1, int(raw))
    except ValueError:
        return 5000


MAX_FILE_BYTES = _resolve_max_file_bytes()
SOURCE_FILE_SOFT_MAX_BYTES = _resolve_source_file_soft_max_bytes()

# Hard ceiling for massive files (500 KB) — never even attempt to read
ABSOLUTE_MAX_BYTES = 500 * 1024

# Binary/media file extensions — skip without error
BINARY_EXTENSIONS = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp", ".tiff",
    # Audio/Video
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac", ".ogg", ".webm",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz",
    # Compiled/Binary
    ".wasm", ".so", ".dll", ".dylib", ".a", ".o", ".obj", ".exe", ".bin",
    ".pyc", ".pyo", ".class", ".jar",
    # Documents/Media
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Database
    ".db", ".sqlite", ".sqlite3",
    # Other binary
    ".dat", ".pak", ".map",
})

SOURCE_CODE_EXTENSIONS = frozenset({
    ".rs", ".c", ".cpp", ".h", ".hpp", ".cc", ".hxx", ".zig",
    ".js", ".ts", ".jsx", ".tsx", ".mjs", ".mts", ".cjs", ".cts",
    ".vue", ".svelte", ".py", ".pyi", ".java", ".kt", ".scala",
    ".cs", ".csx", ".fs", ".go", ".swift", ".rb", ".php", ".dart",
    ".ex", ".exs", ".lua", ".r",
})

LOW_VALUE_LARGE_PATH_MARKERS = (
    "/generated/", "/dist/", "/build/", "/coverage/", "/__snapshots__/",
    "/fixtures/", "/fixture/", "/seed/", "/seeder/", "/public/generated/",
)

# Max files to index in a single pass (configurable via ENTROLY_MAX_FILES)
MAX_FILES = _resolve_max_files()


def _resolve_project_file(project_dir: str, rel_path: str) -> str | None:
    """Resolve a project file without following links outside the project."""
    root = os.path.realpath(project_dir)
    candidate = os.path.realpath(os.path.join(root, rel_path))
    try:
        if os.path.commonpath([root, candidate]) != root:
            return None
    except ValueError:
        return None
    return candidate


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
            and d not in SKIP_DIR_NAMES
        ]
        for fname in filenames:
            rel = os.path.relpath(os.path.join(root, fname), project_dir)
            files.append(rel)
            if len(files) >= MAX_FILES:
                return files
    return files


def _load_entrolyignore(project_dir: str) -> list[str]:
    """Load .entrolyignore patterns (one glob per line, like .gitignore)."""
    ignore_path = _resolve_project_file(project_dir, ".entrolyignore")
    if ignore_path is None or not os.path.isfile(ignore_path):
        return []
    try:
        with open(ignore_path) as f:
            return [
                line.strip() for line in f
                if line.strip() and not line.startswith("#")
            ]
    except OSError:
        return []


# Module-level cache for ignore patterns (set per auto_index call)
_ignore_patterns: list[str] = []


def _matches_ignore(rel_path: str) -> bool:
    """Check if a path matches any .entrolyignore pattern."""
    import fnmatch
    for pattern in _ignore_patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        # Also match against basename for patterns like "*.generated.ts"
        if fnmatch.fnmatch(os.path.basename(rel_path), pattern):
            return True
    return False


def _path_parts(rel_path: str) -> tuple[str, ...]:
    normalized = rel_path.replace("\\", "/").strip("/")
    return tuple(part for part in normalized.split("/") if part)


def _has_skipped_dir(rel_path: str) -> bool:
    return any(part.lower() in SKIP_DIR_NAMES for part in _path_parts(rel_path))


def _should_index(rel_path: str) -> bool:
    """Decide whether a file should be indexed."""
    basename = os.path.basename(rel_path)

    if _has_skipped_dir(rel_path):
        return False

    # Skip lock files and system files
    if basename in SKIP_PATTERNS:
        return False

    # Skip binary/media files cleanly
    _, ext = os.path.splitext(basename)
    if ext.lower() in BINARY_EXTENSIONS:
        return False

    # .entrolyignore support
    if _ignore_patterns and _matches_ignore(rel_path):
        return False

    # Dockerfile special case (no extension)
    if basename.startswith("Dockerfile"):
        return True

    # Check extension
    return ext.lower() in SUPPORTED_EXTENSIONS


def _estimate_tokens(content: str) -> int:
    """Fast token estimation: ~4 chars per token for code."""
    return max(1, len(content) // 4)


def _priority_score(rel_path: str) -> int:
    """
    LPI Priority Score — determines Phase 2 batch order.

    Returns 0-100. Higher = processed first by Rust's entropy sample.
    Core source files score 80-100, migrations/generated score 0-20.

    This ensures the fixed 50-fragment entropy sample in batch_ingest
    sees real source code, giving accurate information_score() values.
    """
    p = rel_path.lower().replace("\\", "/")
    name = os.path.basename(p)
    _, ext = os.path.splitext(name)

    if _has_skipped_dir(rel_path):
        return 5

    # Dead weight — migrations, generated, CI
    dead = (
        "/migrations/" in p
        or "/prisma/migrations" in p
        or "/clickhouse/migrations" in p
        or "/__snapshots__/" in p
        or "/.github/" in p
        or "/dist/" in p
        or "/build/" in p
        or "/.next/" in p
        or "/node_modules/" in p
        or "/site-packages/" in p
        or "/dist-packages/" in p
        or name in ("yarn.lock", "package-lock.json", "pnpm-lock.yaml", "Cargo.lock")
    )
    if dead:
        return 5

    # Core source — index first (these dominate the entropy sample)
    hot = (
        "/src/" in p
        or "/lib/" in p
        or "/app/" in p
        or "/core/" in p
        or "/server/" in p
        or "/api/" in p
        or "/routes/" in p
        or "/handlers/" in p
        or "/services/" in p
        or "/models/" in p
        or "/engine/" in p
        or "/worker/" in p
        or "/pkg/" in p
        or "/internal/" in p
    )
    if hot and ext in {".ts", ".tsx", ".py", ".rs", ".go", ".java", ".kt"}:
        return 90

    # Schema / types
    if "schema" in name:
        return 75

    # Tests — useful but lower priority than implementation
    is_test = (
        "/test/" in p or "/tests/" in p or "/spec/" in p
        or name.startswith("test_")
        or name.endswith(".test.ts") or name.endswith(".spec.ts")
        or name.endswith(".test.py") or name.endswith("_test.go")
    )
    if is_test:
        return 40

    # Root-level config files
    if ext in {".toml", ".yml", ".yaml", ".json"} and "/" not in p.strip("/"):
        return 65

    return 50


def _max_bytes_for_path(rel_path: str) -> int:
    """Return the indexing size cap for one path.

    Keep the global default conservative for generated artifacts, but allow
    larger real implementation files in source/service/worker directories.
    Skipping those files creates false economy: the optimizer saves tokens but
    cannot select the code that actually answers the query.
    """
    if os.environ.get("ENTROLY_MAX_FILE_BYTES"):
        return MAX_FILE_BYTES

    p = rel_path.lower().replace("\\", "/")
    _, ext = os.path.splitext(p.rsplit("/", 1)[-1])
    if ext not in SOURCE_CODE_EXTENSIONS:
        return MAX_FILE_BYTES
    if any(marker in p for marker in LOW_VALUE_LARGE_PATH_MARKERS):
        return MAX_FILE_BYTES
    if (
        "/src/" in p
        or "/worker/" in p
        or "/services/" in p
        or "/server/" in p
        or "/packages/" in p
        or "/lib/" in p
        or "/core/" in p
    ):
        return SOURCE_FILE_SOFT_MAX_BYTES
    return MAX_FILE_BYTES


def _source_type_token_weight(rel_path: str) -> float:
    """Source-type importance weight for knapsack efficiency correction.

    The knapsack selects fragments by efficiency = entropy_score / token_count.
    Config files (YAML, JSON) get artificially high efficiency because they are
    tiny and have high character entropy. Inflating their token count corrects
    this by reducing their apparent efficiency.

    Mathematically equivalent to multiplying the fragment's knapsack *value*
    by 1/weight, without modifying the Rust engine's scoring internals.

    Calibration (measured on Kubeflow Trainer, a Go/K8s project):
      Before fix: top 10 selected = all YAML patches (19-72 tokens)
      After fix:  top 10 selected = Go controller/reconciler source code

    Returns:
        Token inflation factor. 1.0 = no inflation (source code).
        Values > 1.0 make the fragment appear more expensive to the knapsack.
    """
    _, ext = os.path.splitext(rel_path.lower())
    basename = os.path.basename(rel_path.lower())

    # Source code: no inflation — full knapsack priority
    if ext in {
        '.go', '.py', '.pyw', '.rs',
        '.ts', '.tsx', '.js', '.jsx', '.mjs', '.mts', '.cjs', '.cts',
        '.java', '.kt', '.scala',
        '.cs', '.csx', '.fs',
        '.swift',
        '.cpp', '.cc', '.c', '.h', '.hpp', '.hxx',
        '.rb', '.php',
        '.ex', '.exs',
        '.dart', '.lua', '.zig',
        '.vue', '.svelte',
    }:
        return 1.0

    # Config / declarative: 3x inflation → 3x lower knapsack efficiency
    if ext in {'.yaml', '.yml', '.json', '.toml'}:
        return 3.0

    # Documentation: 2x inflation
    if ext in {'.md', '.rst'}:
        return 2.0

    # Infrastructure scripts / Docker / CI
    if ext in {'.sh', '.bash', '.zsh', '.dockerfile'} or basename.startswith('dockerfile'):
        return 2.0

    # SQL: moderate inflation (schemas are useful)
    if ext == '.sql':
        return 1.3

    # Unknown: slight inflation
    return 1.5


def auto_index(
    engine: EntrolyEngine,
    project_dir: str | None = None,
    force: bool = False,
) -> dict:
    """Auto-index a project's codebase using the Lazy Progressive Index (LPI).

    LPI strategy for massive codebases (VSCode ~30K files, langfuse ~2.6K):
    - Phase 1: Parallel file reading (ThreadPoolExecutor, I/O-bound, 16 threads)
    - Phase 2: Single batch_ingest() PyO3 call with rayon inside Rust:
               * SimHash: all files in parallel
               * Skeleton extraction: all files in parallel
               * Entropy: O(N) fixed sample (not O(N²) growing)
               * Dedup: sequential after parallel pre-computation

    Key property: 1 PyO3 crossing instead of N. For VSCode = 30K crossings → 1.

    Args:
        engine: The EntrolyEngine instance to index into.
        project_dir: Root directory to scan. Defaults to cwd.
        force: If True, re-index even if fragments already exist.

    Returns:
        Summary dict with indexed file count, token count, and duration.
    """
    project_dir = project_dir or os.getcwd()
    project_dir = os.path.abspath(project_dir)

    # Load .entrolyignore patterns
    global _ignore_patterns
    _ignore_patterns = _load_entrolyignore(project_dir)
    if _ignore_patterns:
        logger.info(f".entrolyignore loaded: {len(_ignore_patterns)} patterns")

    # Skip if engine already has fragments (loaded from persistent index).
    # The skip path returns the SAME schema as the fresh-index path: callers
    # read files_indexed / total_tokens / duration_s without caring whether
    # work was performed this call. `status` tells them.
    if not force and engine._use_rust:
        existing = engine._rust.fragment_count()
        if existing > 0:
            try:
                frags = list(engine._rust.export_fragments())
                existing_files = len({f.get("source", "") for f in frags if f.get("source")})
                existing_tokens = sum(int(f.get("token_count", 0)) for f in frags)
            except Exception:
                # Fragment export failed; degrade honestly — caller sees
                # status=skipped and existing_fragments still gives them
                # something to work with, but file/token counts are unknown.
                existing_files = 0
                existing_tokens = 0
            logger.info(
                f"Auto-index skipped: {existing} fragments ({existing_files} files) "
                f"already loaded from persistent index"
            )
            return {
                "status": "skipped",
                "files_indexed": existing_files,
                "total_tokens": existing_tokens,
                "beliefs_attached": 0,
                "duration_s": 0.0,
                "read_s": 0.0,
                "ingest_s": 0.0,
                "discovery_method": "cache",
                "skipped_too_large": 0,
                "skipped_unreadable": 0,
                "project_dir": project_dir,
                # Skip-specific diagnostics (kept for callers that care):
                "reason": "persistent_index_loaded",
                "existing_fragments": existing,
            }

    t0 = time.perf_counter()

    # Discover files via git (respects .gitignore — <100ms even for 100K files)
    files = _git_ls_files(project_dir)
    if not files:
        files = _walk_fallback(project_dir)
        discovery = "walk"
    else:
        discovery = "git"

    # Filter to indexable files, sort by priority so hot source is first
    all_indexable = [f for f in files if _should_index(f)]
    if len(all_indexable) > MAX_FILES:
        logger.warning(
            f"Codebase has {len(all_indexable)} indexable files, capping at {MAX_FILES}. "
            f"Set ENTROLY_MAX_FILES to increase the limit."
        )

    # Priority sort: hot source first → entropy sample sees real code
    all_indexable.sort(key=_priority_score, reverse=True)
    indexable = all_indexable[:MAX_FILES]

    t_discovery = time.perf_counter()

    # ── Phase 1: Parallel file reading (I/O-bound) ─────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _read_file(rel_path: str) -> tuple | None:
        abs_path = _resolve_project_file(project_dir, rel_path)
        if abs_path is None:
            return ("skip_read", rel_path, 0)
        try:
            size = os.path.getsize(abs_path)
            max_bytes = _max_bytes_for_path(rel_path)
            if size > ABSOLUTE_MAX_BYTES:
                return ("skip_size", rel_path, size, ABSOLUTE_MAX_BYTES)
            if size > max_bytes or size == 0:
                return ("skip_size", rel_path, size, max_bytes) if size > max_bytes else None
        except OSError:
            return None
        try:
            with open(abs_path, "rb") as fb:
                if b"\x00" in fb.read(8192):
                    return ("skip_read", rel_path, 0)
        except OSError:
            return ("skip_read", rel_path, 0)
        try:
            with open(abs_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return ("skip_read", rel_path, 0)
        if not content.strip():
            return None
        return (content, rel_path, _estimate_tokens(content))

    batch: list[tuple[str, str, int]] = []
    skipped_size = 0
    skipped_read = 0
    # Track the largest skipped files by name so the operator can see
    # WHICH file got dropped, not just "1 too large". Cap at 5 to keep
    # the log line readable on chatty repos.
    skipped_size_paths: list[tuple[str, int, int]] = []

    # 16 threads: double the I/O workers vs old code — most time is disk wait
    max_workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lpi") as pool:
        futures = {pool.submit(_read_file, rp): rp for rp in indexable}
        for future in as_completed(futures):
            data = future.result()
            if data is None:
                continue
            if data[0] == "skip_size":
                skipped_size += 1
                if len(data) >= 4:
                    skipped_size_paths.append((data[1], data[2], data[3]))
                continue
            if data[0] == "skip_read":
                skipped_read += 1
                continue
            content, rel_path, tokens = data
            # Apply source-type weighting: inflate token count for config files
            # so the knapsack deprioritizes them vs source code.
            weight = _source_type_token_weight(rel_path)
            weighted_tokens = int(tokens * weight)
            batch.append((content, f"file:{rel_path}", weighted_tokens))

    t_read = time.perf_counter()

    # ── Phase 2: Single PyO3 call into Rust batch_ingest ───────────────────────
    # Rust rayon parallelises SimHash + skeleton + entropy.
    # O(N) entropy: fixed 50-fragment sample, not growing window.
    # Single GIL acquisition for the whole batch.
    indexed = 0
    total_tokens = 0

    if engine._use_rust and batch:
        try:
            r = engine._rust.batch_ingest(batch)
            indexed = int(r.get("ingested", 0))
            total_tokens = int(r.get("total_tokens", 0))
            p1 = r.get('phase1_ms', '?')
            p2 = r.get('phase2_ms', '?')
            p3 = r.get('phase3_ms', '?')
            logger.debug(
                f"batch_ingest: {indexed}/{len(batch)} in {r.get('duration_ms', 0)}ms "
                f"(P1={p1}ms P2={p2}ms P3={p3}ms, {r.get('duplicates', 0)} dups)"
            )
        except AttributeError:
            # Older entroly_core without batch_ingest — graceful fallback
            logger.debug("batch_ingest unavailable, falling back to per-file ingest")
            for content, source, tokens in batch:
                engine.ingest_fragment(
                    content=content, source=source,
                    token_count=tokens, is_pinned=False,
                )
                indexed += 1
                total_tokens += tokens
    else:
        for content, source, tokens in batch:
            engine.ingest_fragment(
                content=content, source=source,
                token_count=tokens, is_pinned=False,
            )
            indexed += 1
            total_tokens += tokens

    elapsed = time.perf_counter() - t0
    read_s = t_read - t_discovery
    ingest_s = time.perf_counter() - t_read

    logger.info(
        f"Auto-indexed {indexed} files ({total_tokens:,} tokens) "
        f"in {elapsed:.1f}s via {discovery} "
        f"[read={read_s:.1f}s ingest={ingest_s:.1f}s "
        f"skipped: {skipped_size} too large, {skipped_read} unreadable]"
    )
    # Name the largest skipped files so the user can see WHICH artifact
    # was dropped. Use WARNING level (not info) so it surfaces under
    # default logging — silent drops of the user's main file is the
    # exact issue this guards against. Limit to top 5 by size.
    if skipped_size_paths:
        top = sorted(skipped_size_paths, key=lambda x: -x[1])[:5]
        items = ", ".join(f"{p} ({s/1024:.0f} KiB > {cap/1024:.0f} KiB cap)" for p, s, cap in top)
        more = f" (+{len(skipped_size_paths) - len(top)} more)" if len(skipped_size_paths) > len(top) else ""
        logger.warning(
            f"Skipped {skipped_size} oversized file(s): "
            f"{items}{more}. "
            f"Raise the limit with ENTROLY_MAX_FILE_BYTES or ENTROLY_MAX_SOURCE_FILE_BYTES "
            f"(capped at 500 KiB)."
        )

    # ── Vault Belief Bridge: attach pre-compiled beliefs to fragments ──
    # After batch ingest, scan vault/beliefs/*.md and match to fragments
    # by source basename. This enables IOS Belief resolution: ~200-token
    # summaries that REPLACE ~800-token code (5-10× token savings).
    beliefs_attached = 0
    if engine._use_rust:
        vault_beliefs_dir = os.path.join(
            os.environ.get("ENTROLY_VAULT", os.path.join(
                os.environ.get("ENTROLY_DIR", os.path.join(project_dir, ".entroly")),
                "vault"
            )),
            "beliefs",
        )
        if os.path.isdir(vault_beliefs_dir):
            try:
                beliefs_attached = engine._rust.load_vault_beliefs(vault_beliefs_dir)
                if beliefs_attached > 0:
                    logger.info(f"Vault beliefs: attached {beliefs_attached} beliefs to fragments")
            except Exception as e:
                logger.debug(f"Vault belief loading failed: {e}")

    return {
        "status": "indexed",
        "files_indexed": indexed,
        "total_tokens": total_tokens,
        "beliefs_attached": beliefs_attached,
        "duration_s": round(elapsed, 2),
        "read_s": round(read_s, 2),
        "ingest_s": round(ingest_s, 2),
        "discovery_method": discovery,
        "skipped_too_large": skipped_size,
        "skipped_unreadable": skipped_read,
        "project_dir": project_dir,
    }


def start_incremental_watcher(
    engine: EntrolyEngine,
    project_dir: str | None = None,
    interval_s: int = 120,
) -> None:
    """Start a background thread that periodically re-scans for new/modified files.

    Addresses the stale-index problem: files created or modified during a session
    are picked up without restarting the server.
    """
    import threading

    project_dir = project_dir or os.getcwd()
    project_dir = os.path.abspath(project_dir)

    # Track what we've already indexed (by mtime)
    _indexed_mtimes: dict[str, float] = {}

    def _initial_snapshot():
        """Capture mtimes of all currently indexed files."""
        files = _git_ls_files(project_dir) or []
        for rel_path in files:
            abs_path = _resolve_project_file(project_dir, rel_path)
            if abs_path is None:
                continue
            try:
                _indexed_mtimes[rel_path] = os.path.getmtime(abs_path)
            except OSError:
                pass

    def _scan_loop():
        _initial_snapshot()
        while True:
            time.sleep(interval_s)
            try:
                _incremental_scan()
            except Exception as e:
                logger.debug(f"Incremental re-index error: {e}")

    def _incremental_scan():
        files = _git_ls_files(project_dir) or []
        new_or_modified = []

        for rel_path in files:
            if not _should_index(rel_path):
                continue
            abs_path = _resolve_project_file(project_dir, rel_path)
            if abs_path is None:
                continue
            try:
                mtime = os.path.getmtime(abs_path)
            except OSError:
                continue
            prev_mtime = _indexed_mtimes.get(rel_path)
            if prev_mtime is None or mtime > prev_mtime:
                new_or_modified.append(rel_path)
                _indexed_mtimes[rel_path] = mtime

        if not new_or_modified:
            return

        count = 0
        for rel_path in new_or_modified[:100]:  # cap per scan
            abs_path = _resolve_project_file(project_dir, rel_path)
            if abs_path is None:
                continue
            try:
                size = os.path.getsize(abs_path)
                if size > ABSOLUTE_MAX_BYTES or size > _max_bytes_for_path(rel_path) or size == 0:
                    continue
                with open(abs_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if not content.strip():
                    continue
                tokens = _estimate_tokens(content)
                engine.ingest_fragment(
                    content=content,
                    source=f"file:{rel_path}",
                    token_count=tokens,
                    is_pinned=False,
                )
                count += 1
            except (OSError, UnicodeDecodeError):
                continue

        if count > 0:
            logger.info(f"Incremental re-index: {count} new/modified files ingested")

    t = threading.Thread(target=_scan_loop, daemon=True, name="entroly-watcher")
    t.start()
    logger.info(f"File watcher started (re-scan every {interval_s}s)")
