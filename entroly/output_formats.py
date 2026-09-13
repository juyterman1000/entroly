"""
entroly.output_formats — multiple output format serializers.

Closes competitive gap vs Repomix which supports XML (default), Markdown,
JSON, and Plain Text output.

Also provides --split-output chunking for large repositories.
"""
from __future__ import annotations

import json
import xml.sax.saxutils as _su
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------

FORMATS = ("markdown", "xml", "json", "plain", "text")


def serialize(
    files: list[dict[str, Any]],
    *,
    fmt: str = "markdown",
    repo_name: str = "",
    metadata: dict[str, Any] | None = None,
    git_log: str = "",
    git_diff: str = "",
) -> str:
    """
    Serialize a list of file dicts to the requested output format.

    Parameters
    ----------
    files : list[dict]
        Each dict must have ``path`` (str) and ``content`` (str).
    fmt : str
        One of ``markdown``, ``xml``, ``json``, ``plain`` / ``text``.
    repo_name : str
        Optional repository identifier shown in headers.
    metadata : dict | None
        Optional extra metadata to embed in formats that support it.
    git_log : str
        Optional recent git log to append.
    git_diff : str
        Optional diff to append.

    Returns
    -------
    str — formatted text
    """
    fmt = fmt.lower()
    if fmt in ("plain", "text"):
        return _plain(files, repo_name, git_log, git_diff)
    elif fmt == "xml":
        return _xml(files, repo_name, metadata or {}, git_log, git_diff)
    elif fmt == "json":
        return _json(files, repo_name, metadata or {}, git_log, git_diff)
    else:  # markdown (default)
        return _markdown(files, repo_name, git_log, git_diff)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _markdown(files: list[dict], repo: str, git_log: str, git_diff: str) -> str:
    lines = []
    if repo:
        lines.append(f"# {repo}\n")
        lines.append("> Packed by [Entroly](https://github.com/juyterman1000/entroly)\n")
    lines.append("## File Tree\n\n```")
    for f in files:
        lines.append(f"  {f['path']}")
    lines.append("```\n")
    lines.append("## Files\n")
    for f in files:
        ext = Path(f["path"]).suffix.lstrip(".")
        lang = _lang_hint(ext)
        lines.append(f"### `{f['path']}`\n")
        lines.append(f"```{lang}")
        lines.append(f["content"])
        lines.append("```\n")
    if git_log:
        lines.append(f"## Git Log\n\n```\n{git_log}\n```\n")
    if git_diff:
        lines.append(f"## Latest Diff\n\n```diff\n{git_diff}\n```\n")
    return "\n".join(lines)


def _xml(files: list[dict], repo: str, meta: dict, git_log: str, git_diff: str) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8"?>']
    attrs = f' name="{_su.escape(repo)}"' if repo else ""
    parts.append(f"<repository{attrs}>")
    if meta:
        parts.append("  <metadata>")
        for k, v in meta.items():
            parts.append(f"    <{k}>{_su.escape(str(v))}</{k}>")
        parts.append("  </metadata>")
    parts.append("  <files>")
    for f in files:
        path_attr = _su.escape(f["path"])
        parts.append(f'    <file path="{path_attr}">')
        parts.append(f"      <content><![CDATA[{f['content']}]]></content>")
        parts.append("    </file>")
    parts.append("  </files>")
    if git_log:
        parts.append(f"  <git_log><![CDATA[{git_log}]]></git_log>")
    if git_diff:
        parts.append(f"  <git_diff><![CDATA[{git_diff}]]></git_diff>")
    parts.append("</repository>")
    return "\n".join(parts)


def _json(files: list[dict], repo: str, meta: dict, git_log: str, git_diff: str) -> str:
    payload: dict[str, Any] = {
        "repository": repo,
        "generated_by": "entroly",
        "metadata": meta,
        "files": files,
    }
    if git_log:
        payload["git_log"] = git_log
    if git_diff:
        payload["git_diff"] = git_diff
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _plain(files: list[dict], repo: str, git_log: str, git_diff: str) -> str:
    sep = "=" * 60
    parts = []
    if repo:
        parts.append(f"Repository: {repo}")
        parts.append(sep)
        parts.append("")
    for f in files:
        parts.append(f"--- {f['path']} ---")
        parts.append(f["content"])
        parts.append("")
    if git_log:
        parts.append("--- git log ---")
        parts.append(git_log)
        parts.append("")
    if git_diff:
        parts.append("--- git diff ---")
        parts.append(git_diff)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Language hint map
# ---------------------------------------------------------------------------

_LANG: dict[str, str] = {
    "py": "python", "rs": "rust", "js": "javascript", "ts": "typescript",
    "tsx": "tsx", "jsx": "jsx", "go": "go", "java": "java", "rb": "ruby",
    "cpp": "cpp", "cc": "cpp", "c": "c", "h": "c", "hpp": "cpp",
    "cs": "csharp", "sh": "bash", "bash": "bash", "zsh": "zsh",
    "fish": "fish", "toml": "toml", "yaml": "yaml", "yml": "yaml",
    "json": "json", "md": "markdown", "html": "html", "css": "css",
    "scss": "scss", "sql": "sql", "kt": "kotlin", "swift": "swift",
    "r": "r", "lua": "lua", "php": "php", "ex": "elixir", "exs": "elixir",
}


def _lang_hint(ext: str) -> str:
    return _LANG.get(ext.lower(), ext or "text")


# ---------------------------------------------------------------------------
# Split output
# ---------------------------------------------------------------------------

def split_output(text: str, max_bytes: int) -> list[str]:
    """
    Split *text* into chunks no larger than *max_bytes* bytes.

    Splits at newline boundaries where possible to keep file blocks intact.

    Parameters
    ----------
    text : str
        The serialized repository text to split.
    max_bytes : int
        Maximum chunk size in bytes.

    Returns
    -------
    list[str] — ordered chunks, all ≤ max_bytes encoded bytes.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")

    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]

    chunks: list[str] = []
    lines = text.splitlines(keepends=True)
    current: list[str] = []
    current_bytes = 0

    for line in lines:
        line_bytes = len(line.encode("utf-8"))
        if current_bytes + line_bytes > max_bytes and current:
            chunks.append("".join(current))
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += line_bytes

    if current:
        chunks.append("".join(current))

    return chunks


def write_split_output(
    text: str,
    output_path: str | Path,
    max_bytes: int,
) -> list[Path]:
    """
    Write split chunks to ``<output_path>.part001``, ``<output_path>.part002``, …

    Returns list of written paths.
    """
    output_path = Path(output_path)
    chunks = split_output(text, max_bytes)
    if len(chunks) == 1:
        output_path.write_text(text, encoding="utf-8")
        return [output_path]

    written: list[Path] = []
    for i, chunk in enumerate(chunks, 1):
        part_path = output_path.with_suffix(f".part{i:03d}{output_path.suffix}")
        part_path.write_text(chunk, encoding="utf-8")
        written.append(part_path)
    return written


def parse_size(size_str: str) -> int:
    """Parse a human size string like '2mb', '500kb', '1gb' into bytes."""
    size_str = size_str.strip().lower()
    if size_str.endswith("gb"):
        return int(float(size_str[:-2]) * 1024 ** 3)
    elif size_str.endswith("mb"):
        return int(float(size_str[:-2]) * 1024 ** 2)
    elif size_str.endswith("kb"):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith("b"):
        return int(size_str[:-1])
    else:
        return int(size_str)
