"""Byte-canonical evidence coordinates for re-derivable context receipts (schema v2).

A `SourceSpan` pins a fragment to an immutable byte interval of a specific source
snapshot. Byte offsets are the canonical coordinate; line numbers are derived
convenience metadata. Verification is exact and fail-closed:

    fragment_bytes == source_bytes[byte_start:byte_end]
    sha256(fragment_bytes) == fragment_digest
    sha256(source_bytes)   == source_digest

so a receipt fails when the file changed, the revision differs, offsets are out
of bounds, or normalization altered content. There is NO fuzzy recovery — a span
that cannot be located by a unique byte match is reported, never guessed.

Rationale for bytes over characters/lines: line endings differ (LF/CRLF), Unicode
makes character offsets ambiguous, identical text can recur, and line numbers
shift after edits. Bytes are unambiguous and independently hashable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256

SCHEMA_VERSION = 2  # v1 = file-level provenance (no offsets); v2 = byte-range provenance


class Representation(str, Enum):
    """What kind of source region a fragment represents (chunking is explicit)."""

    WHOLE_FILE = "whole_file"
    SYNTAX_BLOCK = "syntax_block"
    LINE_WINDOW = "line_window"
    SENTENCE_SEGMENT = "sentence_segment"
    MERGED = "merged"
    SKELETON = "skeleton"
    REFERENCE_ONLY = "reference_only"


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def derive_lines(source_bytes: bytes, byte_start: int, byte_end: int) -> tuple[int, int]:
    """1-indexed inclusive line range a byte interval [start, end) touches.

    line(pos) = (newlines before pos) + 1. line_end is the line of the last byte
    in the span (end-1); an empty span reports a single line.
    """
    line_start = source_bytes.count(b"\n", 0, byte_start) + 1
    last = max(byte_start, byte_end - 1)
    line_end = source_bytes.count(b"\n", 0, last) + 1
    return line_start, line_end


@dataclass(frozen=True)
class SourceSpan:
    """An immutable, independently-verifiable pointer into one source snapshot."""

    source_path: str          # canonical repo-relative POSIX path
    source_digest: str        # sha256 of the WHOLE source file bytes (identity)
    byte_start: int           # canonical coordinate (inclusive)
    byte_end: int             # canonical coordinate (exclusive)
    line_start: int           # derived, 1-indexed inclusive
    line_end: int             # derived, 1-indexed inclusive
    fragment_digest: str      # sha256 of source_bytes[byte_start:byte_end]
    representation: str = Representation.WHOLE_FILE.value
    source_commit: str = ""   # optional VCS revision the snapshot came from

    def verify(self, source_bytes: bytes) -> bool:
        """Fail-closed byte re-derivation against a candidate source snapshot."""
        if digest(source_bytes) != self.source_digest:
            return False  # file changed / wrong revision
        if not (0 <= self.byte_start <= self.byte_end <= len(source_bytes)):
            return False  # offsets out of bounds
        return digest(source_bytes[self.byte_start:self.byte_end]) == self.fragment_digest

    def byte_len(self) -> int:
        return self.byte_end - self.byte_start

    def to_dict(self) -> dict:
        d = {
            "schema_version": SCHEMA_VERSION,
            "source_path": self.source_path,
            "source_digest": self.source_digest,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "fragment_digest": self.fragment_digest,
            "representation": self.representation,
        }
        if self.source_commit:
            d["source_commit"] = self.source_commit
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SourceSpan:
        return cls(
            source_path=str(d["source_path"]),
            source_digest=str(d["source_digest"]),
            byte_start=int(d["byte_start"]),
            byte_end=int(d["byte_end"]),
            line_start=int(d["line_start"]),
            line_end=int(d["line_end"]),
            fragment_digest=str(d["fragment_digest"]),
            representation=str(d.get("representation", Representation.WHOLE_FILE.value)),
            source_commit=str(d.get("source_commit", "")),
        )


def compute_span(
    source_bytes: bytes,
    block_bytes: bytes,
    source_path: str,
    *,
    representation: str = Representation.WHOLE_FILE.value,
    source_commit: str = "",
) -> SourceSpan | str:
    """Locate a contiguous block by a UNIQUE byte match, or return a fail reason.

    Whole-file blocks map to [0, len). Zero or multiple occurrences fail closed
    ("not_found" / "ambiguous_duplicate") rather than guessing a location.
    """
    if not block_bytes:
        return "empty_block"
    if block_bytes == source_bytes:
        start, end = 0, len(source_bytes)
    else:
        occurrences = source_bytes.count(block_bytes)
        if occurrences == 0:
            return "not_found"
        if occurrences > 1:
            return "ambiguous_duplicate"
        start = source_bytes.index(block_bytes)
        end = start + len(block_bytes)
    line_start, line_end = derive_lines(source_bytes, start, end)
    return SourceSpan(
        source_path=source_path,
        source_digest=digest(source_bytes),
        byte_start=start,
        byte_end=end,
        line_start=line_start,
        line_end=line_end,
        fragment_digest=digest(block_bytes),
        representation=representation,
        source_commit=source_commit,
    )


def merge_spans(
    spans: list[SourceSpan], source_by_path: dict[str, bytes]
) -> list[SourceSpan]:
    """Merge overlapping/adjacent byte intervals of the SAME source, staying verifiable.

    A merged span's fragment_digest is recomputed from `source_by_path[path]` so it
    still re-derives exactly. A path is only merged when its source bytes are
    provided AND all its spans share one source_digest (consistent snapshot);
    otherwise its spans are passed through untouched (fail-safe). An interval that
    still corresponds to a single original span keeps that span's representation.
    """
    by_path: dict[str, list[SourceSpan]] = {}
    for s in spans:
        by_path.setdefault(s.source_path, []).append(s)
    out: list[SourceSpan] = []
    for path, group in by_path.items():
        src = source_by_path.get(path)
        if src is None or len({s.source_digest for s in group}) > 1:
            out.extend(group)
            continue
        group.sort(key=lambda s: (s.byte_start, s.byte_end))
        # coalesce byte intervals, tracking how many originals contributed to each
        intervals: list[tuple[int, int, int, str]] = []  # start, end, count, first_repr
        cs, ce, cnt, rep = group[0].byte_start, group[0].byte_end, 1, group[0].representation
        for s in group[1:]:
            if s.byte_start <= ce:
                ce, cnt = max(ce, s.byte_end), cnt + 1
            else:
                intervals.append((cs, ce, cnt, rep))
                cs, ce, cnt, rep = s.byte_start, s.byte_end, 1, s.representation
        intervals.append((cs, ce, cnt, rep))
        src_digest = digest(src)
        for start, end, count, first_repr in intervals:
            ls, le = derive_lines(src, start, end)
            out.append(SourceSpan(
                source_path=path,
                source_digest=src_digest,
                byte_start=start,
                byte_end=end,
                line_start=ls,
                line_end=le,
                fragment_digest=digest(src[start:end]),
                representation=first_repr if count == 1 else Representation.MERGED.value,
                source_commit=group[0].source_commit,
            ))
    return out
