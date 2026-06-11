"""Cross-document ingestion for Context Receipts."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .models import (
    SCHEMA_VERSION,
    ContextIndex,
    DocumentChunk,
    DocumentRecord,
    stable_hash,
    text_fingerprint,
)

DOCUMENT_EXTENSIONS = frozenset({
    ".txt",
    ".md",
    ".markdown",
    ".rst",
})

SOURCE_EXTENSIONS = frozenset({
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".kt",
    ".mjs",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".xml",
    ".yaml",
    ".yml",
})

SUPPORTED_FILENAMES = frozenset({
    "Containerfile",
    "Dockerfile",
    "Justfile",
    "Makefile",
    "Procfile",
    "Rakefile",
})

SUPPORTED_FILENAME_PREFIXES = (
    "Containerfile.",
    "Dockerfile.",
)

SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | SOURCE_EXTENSIONS

SKIP_DIRECTORIES = frozenset({
    ".entroly",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "venv",
})

SKIP_DIRECTORY_SUFFIXES = (
    ".egg-info",
)

SKIP_FILENAMES = frozenset({
    "Cargo.lock",
    "Gemfile.lock",
    "Pipfile.lock",
    "bun.lockb",
    "composer.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "yarn.lock",
})

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*|[^\w\s]", re.UNICODE)
HEADING_RE = re.compile(
    r"^\s{0,3}(#{1,6}\s+.+|"
    r"(?:section|article|clause|exhibit|schedule|addendum)\s+[A-Za-z0-9.\-]+.*|"
    r"\d+(?:\.\d+)*\s+.+)$",
    re.IGNORECASE,
)
PAGE_RE = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)


def estimate_tokens(text: str) -> int:
    return max(1, len(TOKEN_RE.findall(text)))


def _byte_offset(text: str, char_offset: int) -> int:
    return len(text[:char_offset].encode("utf-8"))


def _clean_heading(line: str) -> str:
    return line.strip().lstrip("#").strip()


def _title_for_path(source_path: str) -> str:
    name = Path(source_path).name
    return Path(name).stem.replace("_", " ").replace("-", " ").strip() or name


def _is_skipped_directory(name: str) -> bool:
    return name in SKIP_DIRECTORIES or name.endswith(SKIP_DIRECTORY_SUFFIXES)


def _is_skipped_filename(name: str) -> bool:
    return name in SKIP_FILENAMES


def _has_supported_filename(name: str) -> bool:
    return name in SUPPORTED_FILENAMES or name.startswith(SUPPORTED_FILENAME_PREFIXES)


def _is_supported_document_path(path: Path) -> bool:
    return (
        path.is_file()
        and not _is_skipped_filename(path.name)
        and (path.suffix.lower() in SUPPORTED_EXTENSIONS or _has_supported_filename(path.name))
    )


def _discover_supported_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if _is_supported_document_path(root) else []

    paths: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _is_skipped_directory(d))
        current_path = Path(current_root)
        for filename in sorted(filenames):
            candidate = current_path / filename
            if _is_supported_document_path(candidate):
                paths.append(candidate)
    return paths


def supported_documents_hint() -> str:
    extensions = ", ".join(sorted(DOCUMENT_EXTENSIONS | SOURCE_EXTENSIONS))
    filenames = ", ".join(sorted(SUPPORTED_FILENAMES | {"Dockerfile.*", "Containerfile.*"}))
    return f"Use supported text/source files ({extensions}) or filenames ({filenames})."


def read_documents_from_path(path: str | Path) -> list[tuple[str, str]]:
    root = Path(path)
    paths = _discover_supported_paths(root)
    documents: list[tuple[str, str]] = []
    for p in paths:
        source_path = p.as_posix()
        try:
            documents.append((source_path, p.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            documents.append((source_path, p.read_text(encoding="utf-8", errors="replace")))
    return documents


def _paragraph_blocks(text: str) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    current: list[str] = []
    start: int | None = None
    line_start = 0
    heading: str | None = None
    page: int | None = None

    def flush(end: int) -> None:
        nonlocal current, start
        if not current or start is None:
            current = []
            start = None
            return
        raw = "".join(current).strip()
        if raw:
            blocks.append(
                {
                    "text": raw,
                    "start": start,
                    "end": end,
                    "heading": heading,
                    "page": page,
                }
            )
        current = []
        start = None

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        match = PAGE_RE.search(stripped)
        if match:
            try:
                page = int(match.group(1))
            except ValueError:
                pass
        if "\f" in line:
            page = 1 if page is None else page + line.count("\f")

        if stripped and HEADING_RE.match(stripped):
            flush(line_start)
            heading = _clean_heading(stripped)

        if not stripped:
            flush(line_start)
        else:
            if start is None:
                start = line_start
            current.append(line)
        line_start += len(line)
    flush(len(text))
    return blocks


def _token_spans(text: str) -> list[re.Match[str]]:
    return list(TOKEN_RE.finditer(text))


def _split_large_block(
    source_text: str,
    block: dict[str, object],
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, object]]:
    text = str(block["text"])
    tokens = _token_spans(text)
    if not tokens:
        return []
    chunks: list[dict[str, object]] = []
    step = max(1, chunk_tokens - max(0, overlap_tokens))
    block_start = int(block["start"])
    for token_start in range(0, len(tokens), step):
        token_end = min(len(tokens), token_start + chunk_tokens)
        start = tokens[token_start].start()
        end = tokens[token_end - 1].end()
        chunks.append(
            {
                "text": text[start:end],
                "start": block_start + start,
                "end": block_start + end,
                "heading": block.get("heading"),
                "page": block.get("page"),
            }
        )
        if token_end >= len(tokens):
            break
    return chunks


def _chunk_document(
    source_path: str,
    text: str,
    *,
    document_id: str,
    document_fingerprint: str,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[DocumentChunk]:
    blocks = _paragraph_blocks(text)
    chunks_raw: list[dict[str, object]] = []
    pending: list[str] = []
    start: int | None = None
    end = 0
    heading: str | None = None
    page: int | None = None
    token_count = 0

    def flush() -> None:
        nonlocal pending, start, end, heading, page, token_count
        if not pending or start is None:
            pending = []
            start = None
            token_count = 0
            return
        chunks_raw.append(
            {
                "text": "\n\n".join(pending).strip(),
                "start": start,
                "end": end,
                "heading": heading,
                "page": page,
            }
        )
        pending = []
        start = None
        token_count = 0

    for block in blocks:
        btokens = estimate_tokens(str(block["text"]))
        if btokens > chunk_tokens:
            flush()
            chunks_raw.extend(
                _split_large_block(
                    text,
                    block,
                    chunk_tokens=chunk_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            continue
        if pending and token_count + btokens > chunk_tokens:
            flush()
        if start is None:
            start = int(block["start"])
            heading = block.get("heading") if isinstance(block.get("heading"), str) else None
            page = block.get("page") if isinstance(block.get("page"), int) else None
        pending.append(str(block["text"]))
        end = int(block["end"])
        token_count += btokens
    flush()

    title = _title_for_path(source_path)
    result: list[DocumentChunk] = []
    running_tokens = 0
    for idx, raw in enumerate(chunks_raw):
        chunk_text = str(raw["text"])
        count = estimate_tokens(chunk_text)
        start_char = int(raw["start"])
        end_char = int(raw["end"])
        fp = text_fingerprint(f"{document_fingerprint}\n{start_char}:{end_char}\n{chunk_text}")
        chunk_id = "chk_" + stable_hash(
            {
                "doc": document_id,
                "start": start_char,
                "end": end_char,
                "fingerprint": fp,
            }
        )[:12]
        result.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                source_path=source_path,
                title=title,
                section_heading=raw.get("heading") if isinstance(raw.get("heading"), str) else None,
                page_number=raw.get("page") if isinstance(raw.get("page"), int) else None,
                chunk_index=idx,
                byte_start=_byte_offset(text, start_char),
                byte_end=_byte_offset(text, end_char),
                token_start=running_tokens,
                token_end=running_tokens + count,
                token_count=count,
                fingerprint=fp,
                text=chunk_text,
            )
        )
        running_tokens += count
    return result


def ingest_documents(
    documents: list[tuple[str, str]],
    *,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
) -> ContextIndex:
    records: list[DocumentRecord] = []
    chunks: list[DocumentChunk] = []
    source_fingerprints: dict[str, str] = {}

    for source_path, text in sorted(documents, key=lambda item: item[0]):
        doc_fp = text_fingerprint(text)
        doc_id = "doc_" + stable_hash({"source": source_path, "fingerprint": doc_fp})[:12]
        doc_chunks = _chunk_document(
            source_path,
            text,
            document_id=doc_id,
            document_fingerprint=doc_fp,
            chunk_tokens=max(40, int(chunk_tokens)),
            overlap_tokens=max(0, int(overlap_tokens)),
        )
        source_fingerprints[source_path] = doc_fp
        records.append(
            DocumentRecord(
                document_id=doc_id,
                source_path=source_path,
                title=_title_for_path(source_path),
                fingerprint=doc_fp,
                token_count=estimate_tokens(text),
                byte_count=len(text.encode("utf-8")),
                chunk_ids=[c.chunk_id for c in doc_chunks],
            )
        )
        chunks.extend(doc_chunks)

    return ContextIndex(
        schema_version=SCHEMA_VERSION,
        documents=records,
        chunks=chunks,
        chunk_token_limit=max(40, int(chunk_tokens)),
        chunk_overlap=max(0, int(overlap_tokens)),
        source_fingerprints=source_fingerprints,
    )
