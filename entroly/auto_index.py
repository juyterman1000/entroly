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

import json
import logging
import os
import subprocess
import tempfile
import time
from hashlib import sha256
from pathlib import Path
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

def _resolve_state_dir_prefix(engine: EntrolyEngine, project_dir: str) -> str | None:
    """Return the state directory relative to the project when it is nested."""
    checkpoint_dir = getattr(getattr(engine, "config", None), "checkpoint_dir", None)
    if not checkpoint_dir:
        return None
    try:
        rel = os.path.relpath(
            os.path.abspath(str(checkpoint_dir)),
            os.path.abspath(project_dir),
        )
    except (ValueError, OSError):
        return None
    normalized = rel.replace("\\", "/").strip("/")
    if (
        not normalized
        or normalized == ".."
        or normalized.startswith("../")
        or os.path.isabs(rel)
    ):
        return None
    return normalized


def _matches_ignore(
    rel_path: str,
    patterns: list[str] | None = None,
) -> bool:
    """Check if a path matches any .entrolyignore pattern."""
    import fnmatch

    active_patterns = _ignore_patterns if patterns is None else patterns
    for pattern in active_patterns:
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


def _should_index(
    rel_path: str,
    *,
    ignore_patterns: list[str] | None = None,
    state_dir_prefix: str | None = None,
) -> bool:
    """Decide whether a file should be indexed."""
    basename = os.path.basename(rel_path)

    if _has_skipped_dir(rel_path):
        return False

    if state_dir_prefix:
        normalized = rel_path.replace("\\", "/").strip("/")
        if normalized == state_dir_prefix or normalized.startswith(
            state_dir_prefix + "/"
        ):
            return False

    # Skip lock files and system files
    if basename in SKIP_PATTERNS:
        return False

    # Skip binary/media files cleanly
    _, ext = os.path.splitext(basename)
    if ext.lower() in BINARY_EXTENSIONS:
        return False

    # .entrolyignore support
    active_ignore_patterns = (
        _ignore_patterns if ignore_patterns is None else ignore_patterns
    )
    if active_ignore_patterns and _matches_ignore(rel_path, active_ignore_patterns):
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


def _canonical_rel_path(rel_path: str) -> str:
    """Return the stable workspace-relative path used in fragment sources."""
    normalized = rel_path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _read_index_file(project_dir: str, rel_path: str) -> tuple[str | None, int, str | None]:
    """Read one index candidate and explain every non-content outcome.

    Returns ``(content, size_bytes, reason)``.  ``reason`` is ``None`` only
    when content is safe to ingest.  Reconciliation uses the explicit reason
    to remove stale bytes when a formerly indexed file becomes unavailable.
    """
    abs_path = _resolve_project_file(project_dir, rel_path)
    if abs_path is None:
        return None, 0, "outside_project"
    try:
        size = os.path.getsize(abs_path)
        max_bytes = _max_bytes_for_path(rel_path)
        if size > ABSOLUTE_MAX_BYTES:
            return None, size, f"too_large:{ABSOLUTE_MAX_BYTES}"
        if size > max_bytes:
            return None, size, f"too_large:{max_bytes}"
        if size == 0:
            return None, 0, "empty"
    except OSError:
        return None, 0, "unreadable"
    try:
        with open(abs_path, "rb") as file_handle:
            if b"\x00" in file_handle.read(8192):
                return None, size, "binary"
    except OSError:
        return None, size, "unreadable"
    try:
        with open(abs_path, encoding="utf-8", errors="ignore") as file_handle:
            content = file_handle.read()
    except (OSError, UnicodeDecodeError):
        return None, size, "unreadable"
    if not content.strip():
        return None, size, "empty"
    return content, size, None


def _export_file_fragments(engine: EntrolyEngine) -> dict[str, list[dict]]:
    """Group live file fragments by canonical source without losing aliases."""
    if engine._use_rust:
        fragments = [dict(fragment) for fragment in engine._rust.export_fragments()]
    else:
        fragments = [
            {
                "fragment_id": fragment.fragment_id,
                "source": fragment.source,
                "content": fragment.content,
                "token_count": fragment.token_count,
            }
            for fragment in engine._fragments.values()
        ]

    grouped: dict[str, list[dict]] = {}
    for fragment in fragments:
        source = str(fragment.get("source") or "")
        if not source.startswith("file:"):
            continue
        canonical = f"file:{_canonical_rel_path(source[5:])}"
        grouped.setdefault(canonical, []).append(fragment)
    return grouped


_DUPLICATE_LEDGER_SCHEMA_VERSION = 2


def _duplicate_ledger_path(engine: EntrolyEngine) -> Path | None:
    checkpoint_dir = getattr(getattr(engine, "config", None), "checkpoint_dir", None)
    if not checkpoint_dir:
        return None
    return Path(checkpoint_dir) / "reconcile_duplicates.json"


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _load_duplicate_ledger(engine: EntrolyEngine) -> dict[str, dict[str, str]]:
    """Load duplicate entries that identify their live representative fragment."""
    path = _duplicate_ledger_path(engine)
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable duplicate ledger %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Ignoring malformed duplicate ledger %s", path)
        return {}
    if data.get("schema_version") != _DUPLICATE_LEDGER_SCHEMA_VERSION:
        # Version 1 stored only source -> digest. It could not prove that the
        # canonical fragment still existed, so trusting it could silently omit
        # the only surviving copy of a file. Fail open and rebuild it.
        logger.info(
            "Ignoring legacy duplicate ledger %s; entries will be revalidated",
            path,
        )
        return {}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        logger.warning("Ignoring malformed duplicate ledger entries in %s", path)
        return {}

    validated: dict[str, dict[str, str]] = {}
    for source, entry in entries.items():
        if not isinstance(source, str) or not source.startswith("file:"):
            continue
        if not isinstance(entry, dict):
            continue
        content_sha256 = entry.get("content_sha256")
        representative_id = entry.get("representative_fragment_id")
        if not _valid_sha256(content_sha256):
            continue
        if not isinstance(representative_id, str) or not representative_id:
            continue
        validated[source] = {
            "content_sha256": content_sha256,
            "representative_fragment_id": representative_id,
        }
    return validated


def _save_duplicate_ledger(
    engine: EntrolyEngine,
    ledger: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Persist the duplicate ledger atomically and report the outcome."""
    path = _duplicate_ledger_path(engine)
    if path is None:
        return {"status": "disabled"}
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _DUPLICATE_LEDGER_SCHEMA_VERSION,
            "entries": ledger,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(payload, tmp, sort_keys=True, separators=(",", ":"))
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        tmp_path = None
        return {
            "status": "persisted",
            "entries": len(ledger),
            "schema_version": _DUPLICATE_LEDGER_SCHEMA_VERSION,
        }
    except OSError as exc:
        logger.warning("Failed to persist duplicate ledger %s: %s", path, exc)
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _fragments_by_id(
    grouped_fragments: dict[str, list[dict]],
) -> dict[str, dict]:
    return {
        str(fragment["fragment_id"]): fragment
        for fragments in grouped_fragments.values()
        for fragment in fragments
        if fragment.get("fragment_id")
    }


def _ledger_representative_is_live(
    entry: dict[str, str],
    *,
    grouped_fragments: dict[str, list[dict]],
    fragments_by_id: dict[str, dict],
    reads: dict[str, tuple[str | None, int, str | None]],
    current_paths: set[str],
) -> bool:
    """Prove a duplicate's representative will survive this reconciliation."""
    representative_id = entry.get("representative_fragment_id", "")
    representative = fragments_by_id.get(representative_id)
    if representative is None:
        return False

    representative_source = str(representative.get("source") or "")
    if not representative_source.startswith("file:"):
        # Reconciliation only removes file sources. A live non-file fragment is
        # therefore a stable representative for this transaction.
        return True

    canonical_source = f"file:{_canonical_rel_path(representative_source[5:])}"
    representative_fragments = grouped_fragments.get(canonical_source, [])
    if len(representative_fragments) != 1:
        return False
    if (
        str(representative_fragments[0].get("fragment_id") or "")
        != representative_id
    ):
        return False

    representative_path = canonical_source[5:]
    if representative_path not in current_paths:
        return False
    content, _size, _reason = reads.get(
        representative_path,
        (None, 0, "not_scanned"),
    )
    if content is None:
        return False
    indexed_content = str(representative.get("content") or "")
    return sha256(content.encode("utf-8")).digest() == sha256(
        indexed_content.encode("utf-8")
    ).digest()


def reconcile_index(
    engine: EntrolyEngine,
    project_dir: str | None = None,
    *,
    max_changes: int | None = None,
) -> dict:
    """Serialize and reconcile the repository index against live files."""
    mutation_lock = getattr(engine, "_index_mutation_lock", None)
    if mutation_lock is None:
        return _reconcile_index(engine, project_dir, max_changes=max_changes)
    with mutation_lock:
        return _reconcile_index(engine, project_dir, max_changes=max_changes)


def _reconcile_index(
    engine: EntrolyEngine,
    project_dir: str | None = None,
    *,
    max_changes: int | None = None,
) -> dict:
    """Reconcile the live index against exact workspace bytes.

    This is deliberately content-addressed rather than mtime-only.  A changed
    file is source-atomically removed and re-ingested; a deleted, unreadable,
    or newly excluded file is removed.  The result is an inspectable receipt,
    and successful mutations are persisted before returning.
    """
    project_dir = os.path.abspath(project_dir or os.getcwd())
    started = time.perf_counter()
    engine.wait_until_warm()

    ignore_patterns = _load_entrolyignore(project_dir)
    state_dir_prefix = _resolve_state_dir_prefix(engine, project_dir)

    discovered = _git_ls_files(project_dir)
    discovery = "git"
    if not discovered:
        discovered = _walk_fallback(project_dir)
        discovery = "walk"
    indexable = [
        _canonical_rel_path(path)
        for path in discovered
        if _should_index(
            path,
            ignore_patterns=ignore_patterns,
            state_dir_prefix=state_dir_prefix,
        )
    ]
    indexable = sorted(set(indexable), key=_priority_score, reverse=True)
    candidates = indexable[:MAX_FILES]
    # The active cap is part of the index contract. A warm cache produced with
    # a larger prior cap must not retain unchecked, potentially stale sources.
    current_paths = set(candidates)
    existing = _export_file_fragments(engine)
    existing_by_id = _fragments_by_id(existing)
    existing_source_aliases = {
        canonical: {
            str(fragment.get("source") or canonical)
            for fragment in fragments
        }
        for canonical, fragments in existing.items()
    }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    reads: dict[str, tuple[str | None, int, str | None]] = {}
    max_workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="reconcile") as pool:
        futures = {
            pool.submit(_read_index_file, project_dir, rel_path): rel_path
            for rel_path in candidates
        }
        for future in as_completed(futures):
            rel_path = futures[future]
            try:
                reads[rel_path] = future.result()
            except Exception as exc:  # fail visibly; never drop live context silently
                reads[rel_path] = (None, 0, f"read_error:{type(exc).__name__}")

    duplicate_ledger = _load_duplicate_ledger(engine)
    current_digests: dict[str, str] = {}
    changed: list[tuple[str, str, int, bool]] = []
    unavailable: dict[str, str] = {}
    unchanged = 0
    ledger_skipped = 0
    for rel_path in candidates:
        source = f"file:{rel_path}"
        content, _size, reason = reads[rel_path]
        prior = existing.get(source, [])
        if content is None:
            if prior:
                unavailable[source] = reason or "unavailable"
            continue
        current_digest = sha256(content.encode("utf-8")).hexdigest()
        current_digests[source] = current_digest
        prior_digests = {
            sha256(str(fragment.get("content") or "").encode("utf-8")).hexdigest()
            for fragment in prior
        }
        # Multiple fragments for one source are themselves stale ambiguity,
        # even when one copy happens to match current disk content.
        if len(prior) == 1 and current_digest in prior_digests:
            unchanged += 1
            continue
        ledger_entry = duplicate_ledger.get(source)
        if (
            not prior
            and ledger_entry is not None
            and ledger_entry.get("content_sha256") == current_digest
            and _ledger_representative_is_live(
                ledger_entry,
                grouped_fragments=existing,
                fragments_by_id=existing_by_id,
                reads=reads,
                current_paths=current_paths,
            )
        ):
            unchanged += 1
            ledger_skipped += 1
            continue
        changed.append((source, content, _estimate_tokens(content), bool(prior)))

    deleted: list[str] = []
    for source in existing:
        rel_path = source[5:]
        if rel_path not in current_paths:
            deleted.append(source)

    removal_reasons: dict[str, str] = {
        **{source: "deleted_or_excluded" for source in deleted},
        **unavailable,
        **{source: "content_changed" for source, _, _, had_prior in changed if had_prior},
    }
    additions = [item for item in changed if not item[3]]
    replacements = [item for item in changed if item[3]]

    ordered_mutations = sorted(removal_reasons)
    if max_changes is not None:
        mutation_budget = max(0, int(max_changes))
        ordered_mutations = ordered_mutations[:mutation_budget]
        remaining_budget = max(0, mutation_budget - len(ordered_mutations))
        additions_to_apply = additions[:remaining_budget]
    else:
        additions_to_apply = additions
    removal_sources = set(ordered_mutations)
    exact_removal_sources = sorted({
        alias
        for canonical in removal_sources
        for alias in existing_source_aliases.get(canonical, {canonical})
    })

    transaction_snapshot = (
        engine.snapshot_index_state()
        if exact_removal_sources or additions_to_apply
        else None
    )
    removal_result = engine.remove_sources(exact_removal_sources) if exact_removal_sources else {
        "status": "unchanged",
        "removed_fragments": 0,
        "removed_sources": [],
        "missing_sources": [],
    }
    removed_exact = set(removal_result.get("removed_sources", []))
    removed_canonical = {
        f"file:{_canonical_rel_path(source[5:])}"
        for source in removed_exact
        if source.startswith("file:")
    }
    removal_supported = removal_result.get("status") != "unsupported"

    errors: list[str] = []
    stale_sources: list[str] = []
    if not removal_supported:
        stale_sources = sorted(removal_sources)
        errors.append(
            "native source removal is unavailable; upgrade entroly-core before "
            "reconciling modified or deleted files"
        )

    ingested_added = 0
    ingested_replaced = 0
    added_sources: list[str] = []
    replaced_sources: list[str] = []
    duplicate_sources: list[str] = []
    confirmed_duplicates: dict[str, dict[str, str]] = {}
    ingest_errors: list[str] = []
    additions_by_source = {item[0]: item for item in additions_to_apply}
    replacements_by_source = {item[0]: item for item in replacements}
    safe_items = list(additions_by_source.values())
    if removal_supported:
        safe_items.extend(
            item for source, item in replacements_by_source.items()
            if source in removal_sources
            and existing_source_aliases.get(source, {source}).issubset(removed_exact)
        )

    for source, content, tokens, had_prior in safe_items:
        weighted_tokens = int(tokens * _source_type_token_weight(source[5:]))
        try:
            result = engine.ingest_fragment(
                content=content,
                source=source,
                token_count=weighted_tokens,
                is_pinned=False,
            )
            ingest_status = result.get("status")
            if ingest_status not in {"ingested", "duplicate"}:
                ingest_errors.append(
                    f"{source}: {result.get('reason', ingest_status)}"
                )
            elif ingest_status == "duplicate":
                # The store did not change, so a duplicate is not an addition
                # or replacement. Bind the optimization ledger to the actual
                # representative fragment; without that identity, deleting the
                # representative could make this source disappear from context.
                duplicate_sources.append(source)
                representative_id = result.get("duplicate_of")
                digest = current_digests.get(source)
                if representative_id and digest:
                    confirmed_duplicates[source] = {
                        "content_sha256": digest,
                        "representative_fragment_id": str(representative_id),
                    }
            elif had_prior:
                ingested_replaced += 1
                replaced_sources.append(source)
            else:
                ingested_added += 1
                added_sources.append(source)
        except Exception as exc:
            ingest_errors.append(f"{source}: {type(exc).__name__}: {exc}")

    errors.extend(ingest_errors)
    mutated = bool(removed_exact or ingested_added or ingested_replaced)
    rolled_back = False
    dependency_refresh: dict[str, object] = {"status": "unchanged"}
    persistence: dict[str, object] = {"status": "unchanged"}

    # A replacement is a multi-step operation (remove, ingest, rebuild,
    # persist). Restore the exact pre-scan state on any failed step so callers
    # never observe or restart from a half-applied workspace mutation.
    if ingest_errors and transaction_snapshot is not None:
        engine.restore_index_state(transaction_snapshot)
        rolled_back = True
    elif mutated:
        dependency_refresh = engine.rebuild_dependencies()
        if dependency_refresh.get("status") == "unsupported":
            errors.append(
                "native dependency rebuild is unavailable; upgrade entroly-core "
                "before trusting relationships after source replacement"
            )
            if transaction_snapshot is not None:
                engine.restore_index_state(transaction_snapshot)
                rolled_back = True

    if mutated and not rolled_back:
        try:
            persistence = engine.persist_index()
        except Exception as exc:
            persistence = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(f"index persistence failed: {type(exc).__name__}: {exc}")
            if transaction_snapshot is not None:
                engine.restore_index_state(transaction_snapshot)
                rolled_back = True

    if rolled_back:
        dependency_refresh = {"status": "rolled_back"}
        if persistence.get("status") == "unchanged":
            persistence = {"status": "rolled_back"}
        removed_exact = set()
        removed_canonical = set()
        ingested_added = 0
        ingested_replaced = 0
        added_sources = []
        replaced_sources = []
        mutated = False

    duplicate_ledger_status: dict[str, object] = {
        "status": "rolled_back" if rolled_back else "current",
        "entries": len(duplicate_ledger),
        "schema_version": _DUPLICATE_LEDGER_SCHEMA_VERSION,
    }
    if not rolled_back:
        # Re-export after the transaction. A representative that existed at the
        # start may have been removed or replaced during this same pass.
        live_fragments = _export_file_fragments(engine)
        live_fragments_by_id = _fragments_by_id(live_fragments)
        present_sources = {f"file:{rel_path}" for rel_path in current_paths}
        promoted_sources = set(added_sources) | set(replaced_sources)
        refreshed_ledger: dict[str, dict[str, str]] = {}
        for source, entry in duplicate_ledger.items():
            if source not in present_sources or source in promoted_sources:
                continue
            if current_digests.get(source) != entry.get("content_sha256"):
                continue
            if not _ledger_representative_is_live(
                entry,
                grouped_fragments=live_fragments,
                fragments_by_id=live_fragments_by_id,
                reads=reads,
                current_paths=current_paths,
            ):
                continue
            refreshed_ledger[source] = entry
        for source, entry in confirmed_duplicates.items():
            if _ledger_representative_is_live(
                entry,
                grouped_fragments=live_fragments,
                fragments_by_id=live_fragments_by_id,
                reads=reads,
                current_paths=current_paths,
            ):
                refreshed_ledger[source] = entry
        if refreshed_ledger != duplicate_ledger:
            duplicate_ledger_status = _save_duplicate_ledger(
                engine,
                refreshed_ledger,
            )

    pending = max(0, len(removal_reasons) + len(additions) - len(ordered_mutations) - len(additions_to_apply))
    status = "partial" if errors or pending else "updated" if mutated else "current"
    return {
        "status": status,
        "project_dir": project_dir,
        "discovery_method": discovery,
        "files_scanned": len(candidates),
        "files_unchanged": unchanged,
        "files_added": ingested_added,
        "files_replaced": ingested_replaced,
        "files_duplicate": len(duplicate_sources),
        "files_duplicate_skipped": ledger_skipped,
        "files_removed": len(removed_canonical - set(replacements_by_source)),
        "removed_fragments": (
            0 if rolled_back else int(removal_result.get("removed_fragments", 0))
        ),
        "added_sources": added_sources,
        "replaced_sources": replaced_sources,
        "removed_sources": sorted(removed_canonical - set(replacements_by_source)),
        "stale_sources": stale_sources,
        "unavailable_sources": unavailable,
        "pending_changes": pending,
        "dependency_refresh": dependency_refresh,
        "persistence": persistence,
        "duplicate_ledger": duplicate_ledger_status,
        "rolled_back": rolled_back,
        "errors": errors,
        "duration_s": round(time.perf_counter() - started, 3),
    }


def auto_index(
    engine: EntrolyEngine,
    project_dir: str | None = None,
    force: bool = False,
) -> dict:
    """Serialize the full index lifecycle with incremental reconciliation."""
    mutation_lock = getattr(engine, "_index_mutation_lock", None)
    if mutation_lock is None:
        return _auto_index(engine, project_dir, force)
    with mutation_lock:
        return _auto_index(engine, project_dir, force)


def _auto_index(
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

    # Warm-start is lazy by design, but auto-index is itself a mutation path.
    # Load before inspecting or changing fragments so a later first request
    # cannot overwrite freshly indexed bytes with an older persisted snapshot.
    engine.wait_until_warm()

    # Resolve scan policy per invocation. Module-global policy leaks between
    # multiple projects sharing one Python process.
    ignore_patterns = _load_entrolyignore(project_dir)
    state_dir_prefix = _resolve_state_dir_prefix(engine, project_dir)
    if ignore_patterns:
        logger.info(f".entrolyignore loaded: {len(ignore_patterns)} patterns")

    # Skip if engine already has fragments (loaded from persistent index).
    # The skip path returns the SAME schema as the fresh-index path: callers
    # read files_indexed / total_tokens / duration_s without caring whether
    # work was performed this call. `status` tells them.
    if not force and engine._use_rust:
        existing = engine._rust.fragment_count()
        if existing > 0:
            # A persisted index is a cache, not authority. Reconcile it against
            # exact workspace bytes before allowing retrieval to trust it.
            reconciliation = reconcile_index(engine, project_dir)
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
                "existing_fragments": int(engine._rust.fragment_count()),
                "reconciliation": reconciliation,
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
    all_indexable = [
        file_path
        for file_path in files
        if _should_index(
            file_path,
            ignore_patterns=ignore_patterns,
            state_dir_prefix=state_dir_prefix,
        )
    ]
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
        futures = {pool.submit(_read_index_file, project_dir, rp): rp for rp in indexable}
        for future in as_completed(futures):
            rel_path = _canonical_rel_path(futures[future])
            content, size, reason = future.result()
            if content is None and reason and reason.startswith("too_large:"):
                skipped_size += 1
                cap = int(reason.split(":", 1)[1])
                skipped_size_paths.append((rel_path, size, cap))
                continue
            if content is None:
                if reason not in {None, "empty"}:
                    skipped_read += 1
                continue
            tokens = _estimate_tokens(content)
            # Apply source-type weighting: inflate token count for config files
            # so the knapsack deprioritizes them vs source code.
            weight = _source_type_token_weight(rel_path)
            weighted_tokens = int(tokens * weight)
            batch.append((content, f"file:{rel_path}", weighted_tokens))

    t_read = time.perf_counter()

    force_removal: dict = {"status": "unchanged", "removed_fragments": 0}
    force_snapshot: dict | None = None
    if force:
        existing_file_fragments = _export_file_fragments(engine)
        exact_sources = sorted({
            str(fragment.get("source") or "")
            for fragments in existing_file_fragments.values()
            for fragment in fragments
            if fragment.get("source")
        })
        if exact_sources:
            force_snapshot = engine.snapshot_index_state()
            force_removal = engine.remove_sources(exact_sources)
            if force_removal.get("status") == "unsupported":
                logger.warning(
                    "Forced re-index was not applied because the installed native "
                    "engine cannot remove stale sources; upgrade entroly-core."
                )
                batch = []

    # ── Phase 2: Single PyO3 call into Rust batch_ingest ───────────────────────
    # Rust rayon parallelises SimHash + skeleton + entropy.
    # O(N) entropy: fixed 50-fragment sample, not growing window.
    # Single GIL acquisition for the whole batch.
    indexed = 0
    total_tokens = 0

    if engine._use_rust and batch:
        try:
            r = engine._rust.batch_ingest(batch)
            engine._fragment_cache_dirty = True
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
            try:
                for content, source, tokens in batch:
                    engine.ingest_fragment(
                        content=content, source=source,
                        token_count=tokens, is_pinned=False,
                    )
                    indexed += 1
                    total_tokens += tokens
            except Exception:
                if force_snapshot is not None:
                    engine.restore_index_state(force_snapshot)
                raise
        except Exception:
            if force_snapshot is not None:
                engine.restore_index_state(force_snapshot)
            raise
    else:
        try:
            for content, source, tokens in batch:
                engine.ingest_fragment(
                    content=content, source=source,
                    token_count=tokens, is_pinned=False,
                )
                indexed += 1
                total_tokens += tokens
        except Exception:
            if force_snapshot is not None:
                engine.restore_index_state(force_snapshot)
            raise

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

    dependency_refresh: dict[str, object] = {"status": "unchanged"}
    if indexed or force_removal.get("removed_fragments"):
        dependency_refresh = engine.rebuild_dependencies()
        if dependency_refresh.get("status") == "unsupported":
            logger.warning(
                "Full dependency graph rebuild is unavailable; upgrade entroly-core "
                "to make architecture relationships immediately queryable."
            )

    persistence: dict[str, object] = {"status": "unchanged"}
    if indexed or force_removal.get("removed_fragments"):
        try:
            persistence = engine.persist_index()
        except Exception as exc:
            persistence = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            logger.warning("Fresh index persistence failed: %s", exc)
            if force_snapshot is not None:
                engine.restore_index_state(force_snapshot)
                force_removal = {
                    "status": "rolled_back",
                    "removed_fragments": 0,
                    "reason": "persistence_failed",
                }
                dependency_refresh = {"status": "rolled_back"}
                indexed = 0
                total_tokens = 0
                beliefs_attached = 0

    return {
        "status": "error" if persistence.get("status") == "error" else "indexed",
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
        "persistence": persistence,
        "force_removal": force_removal,
        "dependency_refresh": dependency_refresh,
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

    def _scan_loop():
        while True:
            time.sleep(interval_s)
            try:
                _incremental_scan()
            except Exception as e:
                logger.warning("Incremental re-index failed: %s", e)

    def _incremental_scan():
        result = reconcile_index(engine, project_dir, max_changes=100)
        changed = result["files_added"] + result["files_replaced"] + result["files_removed"]
        if changed:
            logger.info(
                "Incremental reconciliation: %s added, %s replaced, %s removed; "
                "persistence=%s",
                result["files_added"],
                result["files_replaced"],
                result["files_removed"],
                result["persistence"].get("status"),
            )
        if result["errors"]:
            logger.warning(
                "Incremental reconciliation incomplete: %s",
                "; ".join(result["errors"]),
            )

    t = threading.Thread(target=_scan_loop, daemon=True, name="entroly-watcher")
    t.start()
    logger.info(f"File watcher started (re-scan every {interval_s}s)")
