"""
Persistent cache for SymbolManifest + CharNGramModel
=====================================================

Building the manifest + training the n-gram model is O(LOC) and takes
~10s on a 30MB repo. We persist both to .entroly/verifiers_cache/
so subsequent verifications are ~30ms (just inference).

Cache key: hash of (repo_root, file_mtimes, file_paths_subset).
Invalidation: any tracked file's mtime change triggers rebuild.

Layout:
    .entroly/verifiers_cache/
        <hash>/
            manifest.json
            ngram.pkl
            meta.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

from .ngram_model import CharNGramModel, quick_train_from_paths
from .symbol_resolution import SymbolManifest, SymbolVerifier, DEFAULT_LAMBDA

logger = logging.getLogger("entroly.verifiers.cache")


CACHE_DIR_NAME = "verifiers_cache"
META_VERSION = 1


@dataclass
class CacheMeta:
    version: int
    repo_root: str
    n_files: int
    n_chars: int
    n_symbols: int
    built_at: float
    file_hash: str


def _entroly_cache_dir(repo_root: str) -> Path:
    return Path(repo_root) / ".entroly" / CACHE_DIR_NAME


def _enumerate_files(repo_root: str, extensions: tuple[str, ...] = (".py",)) -> list[Path]:
    skip = {"__pycache__", ".git", ".venv", "venv", "node_modules",
            "target", "dist", "build", ".tox", ".pytest_cache",
            ".ruff_cache", ".mypy_cache"}
    out: list[Path] = []
    root = Path(repo_root)
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix not in extensions:
            continue
        if any(part in skip for part in path.parts):
            continue
        out.append(path)
    out.sort()
    return out


def _compute_file_hash(files: list[Path]) -> str:
    """Hash file paths + mtimes. Stable, fast, mtime-sensitive."""
    h = hashlib.sha256()
    for f in files:
        try:
            mtime = f.stat().st_mtime_ns
            h.update(str(f).encode("utf-8", errors="replace"))
            h.update(b"|")
            h.update(str(mtime).encode("ascii"))
            h.update(b"\n")
        except OSError:
            continue
    return h.hexdigest()[:16]


def load_or_build_verifier(
    repo_root: str,
    lambda_calibration: float = DEFAULT_LAMBDA,
    force_rebuild: bool = False,
) -> tuple[SymbolVerifier, CacheMeta]:
    """Return a ready-to-use verifier, building or loading as needed.

    First call: ~10s (build + persist).
    Subsequent calls (no source changes): ~30ms (load).
    Subsequent calls (some source changes): ~10s (rebuild).
    """
    files = _enumerate_files(repo_root)
    file_hash = _compute_file_hash(files)

    cache_dir = _entroly_cache_dir(repo_root) / file_hash
    meta_path = cache_dir / "meta.json"
    manifest_path = cache_dir / "manifest.json"
    ngram_path = cache_dir / "ngram.pkl"

    if not force_rebuild and meta_path.exists() and manifest_path.exists() and ngram_path.exists():
        try:
            return _load_from_cache(cache_dir, lambda_calibration)
        except Exception as e:
            logger.debug("cache load failed (%s), rebuilding", e)

    return _build_and_persist(
        repo_root, files, file_hash, cache_dir, lambda_calibration
    )


def _load_from_cache(
    cache_dir: Path,
    lambda_calibration: float,
) -> tuple[SymbolVerifier, CacheMeta]:
    with open(cache_dir / "meta.json", "r") as f:
        meta_d = json.load(f)
    meta = CacheMeta(**meta_d)
    if meta.version != META_VERSION:
        raise ValueError(f"cache version mismatch: {meta.version} vs {META_VERSION}")

    with open(cache_dir / "manifest.json", "r") as f:
        m_d = json.load(f)
    manifest = SymbolManifest(
        repo=set(m_d.get("repo", [])),
        stdlib=set(m_d.get("stdlib", [])),
        installed=set(m_d.get("installed", [])),
        builtins=set(m_d.get("builtins", [])),
    )

    with open(cache_dir / "ngram.pkl", "rb") as f:
        ngram = pickle.load(f)

    verifier = SymbolVerifier(
        manifest=manifest,
        ngram_model=ngram,
        lambda_calibration=lambda_calibration,
    )
    return verifier, meta


def _build_and_persist(
    repo_root: str,
    files: list[Path],
    file_hash: str,
    cache_dir: Path,
    lambda_calibration: float,
) -> tuple[SymbolVerifier, CacheMeta]:
    t0 = time.time()
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = SymbolManifest.build_from_codebase(repo_root)
    ngram = quick_train_from_paths([str(f) for f in files], n=4)

    meta = CacheMeta(
        version=META_VERSION,
        repo_root=str(Path(repo_root).resolve()),
        n_files=len(files),
        n_chars=ngram.total_chars,
        n_symbols=manifest.size(),
        built_at=time.time(),
        file_hash=file_hash,
    )

    # Atomic-ish writes
    tmp = cache_dir / "manifest.json.tmp"
    with open(tmp, "w") as f:
        json.dump({
            "repo": sorted(manifest.repo),
            "stdlib": sorted(manifest.stdlib),
            "installed": sorted(manifest.installed),
            "builtins": sorted(manifest.builtins),
        }, f)
    os.replace(tmp, cache_dir / "manifest.json")

    tmp = cache_dir / "ngram.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ngram, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_dir / "ngram.pkl")

    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta.__dict__, f, indent=2)

    elapsed = time.time() - t0
    logger.info(
        "built verifier cache at %s in %.1fs (%d files, %d symbols)",
        cache_dir, elapsed, meta.n_files, meta.n_symbols,
    )

    verifier = SymbolVerifier(
        manifest=manifest,
        ngram_model=ngram,
        lambda_calibration=lambda_calibration,
    )
    return verifier, meta
