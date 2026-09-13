"""
entroly.remote_pack — fetch and compress any public GitHub repository URL.

Closes competitive gap vs Repomix: `repomix --remote https://github.com/org/repo`
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# GitHub URL parsing
# ---------------------------------------------------------------------------

_GITHUB_RE = re.compile(
    r"^(?:https?://)?github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/(?:tree|blob)/([^/]+))?(?:/(.*))?$"
)


def parse_github_url(url: str) -> dict[str, str | None]:
    """Parse a GitHub URL into components."""
    m = _GITHUB_RE.match(url.strip())
    if not m:
        raise ValueError(f"Not a recognised GitHub URL: {url!r}")
    owner, repo, ref, subpath = m.groups()
    return {
        "owner": owner,
        "repo": repo.rstrip("/"),
        "ref": ref or "HEAD",
        "subpath": subpath or "",
        "clone_url": f"https://github.com/{owner}/{repo}.git",
    }


# ---------------------------------------------------------------------------
# Shallow clone
# ---------------------------------------------------------------------------

def _shallow_clone(clone_url: str, ref: str, dest: Path, depth: int = 1) -> None:
    """Perform a shallow clone to *dest*."""
    cmd = [
        "git", "clone",
        "--depth", str(depth),
        "--single-branch",
        "--branch", ref if ref != "HEAD" else "HEAD",
        clone_url,
        str(dest),
    ]
    # HEAD branch needs special handling
    if ref == "HEAD":
        cmd = [
            "git", "clone",
            "--depth", str(depth),
            "--single-branch",
            clone_url,
            str(dest),
        ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed:\n{result.stderr[:2000]}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_and_pack(
    url: str,
    *,
    output_format: str = "markdown",
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_file_size: int = 512 * 1024,
    depth: int = 1,
    include_git_log: bool = False,
    include_git_diff: bool = False,
    split_output: int | None = None,
    generate_skills: bool = False,
) -> dict:
    """
    Fetch a remote GitHub repository and pack it, returning the same
    structure as a local pack operation.

    Parameters
    ----------
    url : str
        GitHub URL, e.g. ``https://github.com/yamadashy/repomix``
    output_format : str
        One of ``markdown`` | ``xml`` | ``json`` | ``plain``
    include_patterns : list[str] | None
        Glob patterns to include (e.g. ``["**/*.py"]``)
    exclude_patterns : list[str] | None
        Additional patterns to exclude on top of .gitignore
    max_file_size : int
        Skip files larger than this many bytes (default 512 KB)
    depth : int
        Git clone depth (default 1 = tip only)
    include_git_log : bool
        Append recent git log to the pack
    include_git_diff : bool
        Append ``git diff HEAD~1..HEAD`` to the pack
    split_output : int | None
        Split output into chunks of this many bytes
    generate_skills : bool
        Also emit a Claude Agent Skills file

    Returns
    -------
    dict with keys: ``text``, ``files``, ``token_estimate``,
    ``repo``, ``chunks`` (if split), ``skills_file`` (if generate_skills)
    """
    parsed = parse_github_url(url)

    with tempfile.TemporaryDirectory(prefix="entroly_remote_") as tmpdir:
        dest = Path(tmpdir) / parsed["repo"]
        _shallow_clone(parsed["clone_url"], parsed["ref"], dest, depth=depth)

        # Narrow to subpath if specified
        root = dest / parsed["subpath"] if parsed["subpath"] else dest

        # Collect files
        files = list(_collect_files(
            root,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            max_file_size=max_file_size,
        ))

        # Git extras
        git_extras = ""
        if include_git_log:
            git_extras += _git_log(dest)
        if include_git_diff:
            git_extras += _git_diff(dest)

        # Format output
        packed_text = _format_output(
            files=files,
            repo_info=parsed,
            output_format=output_format,
            git_extras=git_extras,
        )

        # Token estimate (4 chars ≈ 1 token)
        token_estimate = len(packed_text) // 4

        result: dict = {
            "repo": f"{parsed['owner']}/{parsed['repo']}",
            "ref": parsed["ref"],
            "files": len(files),
            "token_estimate": token_estimate,
            "format": output_format,
            "text": packed_text,
        }

        if split_output:
            result["chunks"] = _split(packed_text, split_output)

        if generate_skills:
            result["skills_file"] = _generate_skills_file(parsed, files)

        return result


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

_DEFAULT_EXCLUDES = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", "target", ".mypy_cache", ".ruff_cache",
    "*.pyc", "*.pyo", "*.so", "*.dll", "*.exe", "*.bin",
    "*.jpg", "*.jpeg", "*.png", "*.gif", "*.webp", "*.ico",
    "*.pdf", "*.zip", "*.tar", "*.gz", "package-lock.json",
    "yarn.lock", "Cargo.lock",
}


def _should_exclude(path: Path, exclude_patterns: list[str] | None) -> bool:
    name = path.name
    for pat in _DEFAULT_EXCLUDES:
        if "*" in pat:
            if path.match(pat):
                return True
        elif name == pat or pat in str(path):
            return True
    if exclude_patterns:
        for pat in exclude_patterns:
            if path.match(pat):
                return True
    return False


def _collect_files(
    root: Path,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    max_file_size: int,
) -> Iterator[dict]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if _should_exclude(path, exclude_patterns):
            continue
        if include_patterns and not any(path.match(p) for p in include_patterns):
            continue
        if path.stat().st_size > max_file_size:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        yield {"path": str(rel), "content": content, "size": path.stat().st_size}


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _format_output(
    files: list[dict],
    repo_info: dict,
    output_format: str,
    git_extras: str,
) -> str:
    repo = f"{repo_info['owner']}/{repo_info['repo']}"
    if output_format == "xml":
        return _fmt_xml(files, repo, git_extras)
    elif output_format == "json":
        return _fmt_json(files, repo, git_extras)
    elif output_format == "plain":
        return _fmt_plain(files, repo, git_extras)
    else:  # markdown (default)
        return _fmt_markdown(files, repo, git_extras)


def _fmt_markdown(files: list[dict], repo: str, extras: str) -> str:
    parts = [f"# Repository: {repo}\n\n"]
    parts.append("## File Tree\n\n```\n")
    for f in files:
        parts.append(f"  {f['path']}\n")
    parts.append("```\n\n## Files\n\n")
    for f in files:
        ext = Path(f["path"]).suffix.lstrip(".")
        lang = ext or "text"
        parts.append(f"### `{f['path']}`\n\n```{lang}\n{f['content']}\n```\n\n")
    if extras:
        parts.append(f"## Git History\n\n```\n{extras}\n```\n")
    return "".join(parts)


def _fmt_xml(files: list[dict], repo: str, extras: str) -> str:
    import xml.sax.saxutils as su
    parts = [f'<?xml version="1.0" encoding="UTF-8"?>\n<repository name="{su.escape(repo)}">\n']
    for f in files:
        path_attr = su.escape(f["path"])
        parts.append(f'  <file path="{path_attr}">\n    <content><![CDATA[{f["content"]}]]></content>\n  </file>\n')
    if extras:
        parts.append(f"  <git_extras><![CDATA[{extras}]]></git_extras>\n")
    parts.append("</repository>")
    return "".join(parts)


def _fmt_json(files: list[dict], repo: str, extras: str) -> str:
    import json
    return json.dumps({
        "repository": repo,
        "files": files,
        "git_extras": extras,
    }, indent=2, ensure_ascii=False)


def _fmt_plain(files: list[dict], repo: str, extras: str) -> str:
    parts = [f"Repository: {repo}\n{'='*60}\n\n"]
    for f in files:
        parts.append(f"--- {f['path']} ---\n{f['content']}\n\n")
    if extras:
        parts.append(f"--- git extras ---\n{extras}\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Git extras
# ---------------------------------------------------------------------------

def _git_log(repo_dir: Path, n: int = 20) -> str:
    result = subprocess.run(
        ["git", "log", f"-{n}", "--oneline", "--no-decorate"],
        cwd=repo_dir, capture_output=True, text=True, timeout=10,
    )
    return f"\n=== Recent Commits ===\n{result.stdout}\n" if result.returncode == 0 else ""


def _git_diff(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "HEAD~1..HEAD", "--stat"],
        cwd=repo_dir, capture_output=True, text=True, timeout=10,
    )
    return f"\n=== Latest Diff ===\n{result.stdout}\n" if result.returncode == 0 else ""


# ---------------------------------------------------------------------------
# Split output
# ---------------------------------------------------------------------------

def _split(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of at most *chunk_size* bytes."""
    encoded = text.encode("utf-8")
    chunks = []
    for i in range(0, len(encoded), chunk_size):
        chunks.append(encoded[i : i + chunk_size].decode("utf-8", errors="replace"))
    return chunks


# ---------------------------------------------------------------------------
# Agent Skills generation
# ---------------------------------------------------------------------------

def _generate_skills_file(repo_info: dict, files: list[dict]) -> str:
    """Generate a Claude Agent Skills YAML file for the repository."""
    repo = f"{repo_info['owner']}/{repo_info['repo']}"
    file_list = "\n".join(f"  - {f['path']}" for f in files[:30])
    return f"""# Claude Agent Skill — {repo}
# Generated by Entroly (https://github.com/juyterman1000/entroly)

name: {repo_info['repo']}
description: |
  AI skill for the {repo} repository. Provides context about the
  codebase structure, key files, and conventions.

version: "1.0"
source: "{repo_info['clone_url']}"
ref: "{repo_info['ref']}"

key_files:
{file_list}

instructions: |
  When answering questions about {repo_info['repo']}, use the packed
  repository context to provide accurate, code-grounded answers.
  Refer to specific file paths when citing code.
"""
