"""Shared Unicode-aware text features and preservation guards.

The helpers are deterministic, dependency-free, and intentionally conservative.
They improve multilingual relevance without claiming semantic understanding.
"""
from __future__ import annotations

import re
from pathlib import PurePath
from typing import Iterable

_LATIN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with",
}
_INSTRUCTION_BASENAMES = {
    "agents.md",
    "claude.md",
    "gemini.md",
    "skill.md",
    "skills.md",
    "rules.md",
    ".cursorrules",
    "copilot-instructions.md",
}
_INSTRUCTION_DIRECTORIES = {
    "skills",
    ".cursor/rules",
    ".claude/rules",
    ".github/instructions",
}
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def is_cjk_character(ch: str) -> bool:
    """Return whether *ch* is in a commonly used CJK/Kana/Hangul range."""
    if not ch:
        return False
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0xAC00 <= code <= 0xD7AF
    )


def _identifier_parts(token: str) -> Iterable[str]:
    for coarse in re.split(r"[_.:/\\-]+", token):
        if not coarse:
            continue
        yield coarse
        for part in _CAMEL_BOUNDARY_RE.split(coarse):
            if part and part != coarse:
                yield part


def text_terms(text: str) -> set[str]:
    """Extract case-folded lexical terms plus CJK uni/bi-grams.

    CJK strings often omit whitespace, so whole-word tokenization misses query
    substrings. Emitting bounded uni/bi-grams preserves deterministic matching
    without introducing an external tokenizer.
    """
    terms: set[str] = set()
    current: list[str] = []
    cjk_run: list[str] = []

    def flush_current() -> None:
        if not current:
            return
        raw_token = "".join(current)
        current.clear()
        token = raw_token.casefold()
        if len(token) >= 2 or token.isnumeric():
            terms.add(token)
        for part in _identifier_parts(raw_token):
            folded = part.casefold()
            if len(folded) >= 2 or folded.isnumeric():
                terms.add(folded)

    def flush_cjk() -> None:
        if not cjk_run:
            return
        run = "".join(cjk_run)
        cjk_run.clear()
        terms.add(run)
        terms.update(run)
        terms.update(run[index : index + 2] for index in range(len(run) - 1))

    for ch in text:
        if is_cjk_character(ch):
            flush_current()
            cjk_run.append(ch)
        elif ch.isalnum() or ch == "_":
            flush_cjk()
            current.append(ch)
        else:
            flush_current()
            flush_cjk()
    flush_current()
    flush_cjk()
    return {term for term in terms if term}


def query_terms(text: str) -> set[str]:
    """Extract discriminative query terms while retaining non-Latin signals."""
    return {
        term
        for term in text_terms(text)
        if not (term.isascii() and term in _LATIN_STOPWORDS)
    }


def is_instruction_path(path: str | PurePath | None) -> bool:
    """Detect agent instruction/rule files that should be delivered in full."""
    if path is None:
        return False
    normalised = str(path).replace("\\", "/").strip("/").casefold()
    if not normalised:
        return False
    parts = [part for part in normalised.split("/") if part]
    if parts and parts[-1] in _INSTRUCTION_BASENAMES:
        return True
    joined = "/".join(parts)
    return any(
        joined == directory
        or joined.startswith(directory + "/")
        or f"/{directory}/" in f"/{joined}/"
        for directory in _INSTRUCTION_DIRECTORIES
    )


def protected_input_reason(
    text: str,
    *,
    budget_tokens: int,
    content_type: str,
    source_path: str | PurePath | None = None,
) -> str | None:
    """Return a conservative bypass reason, if one applies.

    Large minified JSON or code is not classified as a short input merely
    because it occupies one line. The bounded token condition protects small
    command outputs and instruction snippets without disabling useful
    compression of genuinely large single-line payloads.
    """
    if is_instruction_path(source_path):
        return "instruction_file_full_fidelity"
    estimated_tokens = max(0, (len(text) + 3) // 4)
    if estimated_tokens <= budget_tokens:
        return "already_fits"
    kind = content_type.casefold()
    if kind in {"json", "json_text", "jsonl", "table"}:
        return None
    non_empty_lines = sum(bool(line.strip()) for line in text.splitlines())
    short_limit = max(64, min(512, budget_tokens * 2))
    if non_empty_lines < 5 and estimated_tokens <= short_limit:
        return "short_input_full_fidelity"
    return None


__all__ = [
    "is_cjk_character",
    "is_instruction_path",
    "protected_input_reason",
    "query_terms",
    "text_terms",
]
