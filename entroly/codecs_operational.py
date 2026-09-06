"""Recoverable codecs for diffs, search results, and static HTML.

These codecs are deliberately extractive.  They never synthesize facts and
only offer a compact representation when every protected string remains
verbatim and the exact original has been written to the recovery store.
"""

from __future__ import annotations

import html
import re
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any

from .codec import (
    RecoveryStore,
    Representation,
    SupportDecision,
    content_digest,
    estimate_tokens,
)


_QUERY_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_DIFF_HEADER = re.compile(r"^(?:diff --git |index |--- |\+\+\+ |@@ )")
_SEARCH_LINE = re.compile(r"^(.+?):([1-9]\d*)(?::([1-9]\d*))?:(.*)$")
_FAILURE = re.compile(r"\b(?:error|failed|failure|panic|exception|fatal)\b", re.IGNORECASE)


def _looks_like_search_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    leaf = normalized.rsplit("/", 1)[-1]
    return "/" in normalized or bool(re.search(r"\.[A-Za-z0-9_-]{1,12}$", leaf))


def _full(text: str, source_id: str, content_type: str, codec: str, version: str) -> Representation:
    return Representation(
        representation_id=f"{source_id}#{codec}.full",
        source_id=source_id,
        content_type=content_type,
        text=text,
        token_cost=estimate_tokens(text),
        codec=codec,
        codec_version=version,
        source_sha256=content_digest(text),
        distortion_risk=0.0,
    )


def _compact(
    *,
    text: str,
    compact: str,
    source_id: str,
    content_type: str,
    codec: str,
    version: str,
    store: RecoveryStore,
    protected: tuple[str, ...],
    omitted_count: int,
    item_label: str,
) -> list[Representation]:
    full = _full(text, source_id, content_type, codec, version)
    if not compact or estimate_tokens(compact) >= full.token_cost:
        return [full]
    if any(value not in compact for value in protected):
        return [full]
    recovery = store.put(
        text,
        item_count=max(0, omitted_count),
        item_label=item_label,
        note=f"complete original {content_type} for {source_id or 'input'}",
    )
    if store.recover(recovery) != text:
        return [full]
    compressed = Representation(
        representation_id=f"{source_id}#{codec}.extractive",
        source_id=source_id,
        content_type=content_type,
        text=compact,
        token_cost=estimate_tokens(compact),
        codec=codec,
        codec_version=version,
        source_sha256=content_digest(text),
        protected_evidence=protected,
        distortion_risk=1.0 - len(compact) / max(1, len(text)),
        recovery=recovery,
    )
    return [full, compressed]


class DiffCodec:
    name = "diff"
    version = "1"

    def __init__(self, store: RecoveryStore) -> None:
        self.store = store

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type.lower() in {"diff", "patch", "unified_diff"}:
            return SupportDecision(True, 1.0, "declared diff content type")
        sample = text[:8000]
        signals = sum(1 for line in sample.splitlines() if _DIFF_HEADER.match(line))
        return SupportDecision(signals >= 3, min(0.96, 0.55 + signals * 0.05), "unified diff structure")

    def representations(self, text: str, source_id: str = "", **options: Any) -> list[Representation]:
        lines = text.splitlines(keepends=True)
        context = max(0, min(10, int(options.get("context_lines", 2))))
        keep: set[int] = set()
        protected: list[str] = []
        for index, line in enumerate(lines):
            stripped = line.rstrip("\r\n")
            header = bool(_DIFF_HEADER.match(stripped))
            changed = stripped.startswith(("+", "-")) and not stripped.startswith(("+++", "---"))
            if header or changed or _FAILURE.search(stripped):
                keep.add(index)
                if header or changed:
                    protected.append(stripped)
                if changed:
                    keep.update(range(max(0, index - context), min(len(lines), index + context + 1)))
        compact = "".join(line for index, line in enumerate(lines) if index in keep)
        return _compact(
            text=text,
            compact=compact,
            source_id=source_id,
            content_type="diff",
            codec=self.name,
            version=self.version,
            store=self.store,
            protected=tuple(dict.fromkeys(value for value in protected if value)),
            omitted_count=len(lines) - len(keep),
            item_label="diff context line(s) restored",
        )


class SearchResultCodec:
    name = "search-results"
    version = "1"

    def __init__(self, store: RecoveryStore) -> None:
        self.store = store

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type.lower() in {"search", "search_result", "search-results", "rg"}:
            return SupportDecision(True, 1.0, "declared search-result content type")
        lines = [line for line in text[:12000].splitlines() if line.strip()]
        matches = 0
        for line in lines:
            parsed = _SEARCH_LINE.match(line)
            if parsed and _looks_like_search_path(parsed.group(1)):
                matches += 1
        confidence = matches / max(1, len(lines))
        return SupportDecision(matches >= 3 and confidence >= 0.6, min(0.94, confidence), "path/line search-result structure")

    def representations(self, text: str, source_id: str = "", **options: Any) -> list[Representation]:
        query_terms = {
            match.group(0).lower()
            for match in _QUERY_WORD.finditer(str(options.get("query") or ""))
            if len(match.group(0)) > 1
        }
        max_per_file = max(1, min(100, int(options.get("max_hits_per_file", 8))))
        parsed: list[tuple[str, str]] = []
        unparsed: list[str] = []
        for line in text.splitlines():
            match = _SEARCH_LINE.match(line)
            if match:
                parsed.append((match.group(1), line))
            elif line.strip():
                unparsed.append(line)
        grouped: dict[str, list[str]] = defaultdict(list)
        for path, line in parsed:
            grouped[path].append(line)
        selected: list[str] = []
        protected: list[str] = []
        # dict insertion order preserves the source's file order. Reordering
        # evidence can itself change downstream long-context behavior.
        for path in grouped:
            hits = grouped[path]
            ranked = sorted(
                enumerate(hits),
                key=lambda item: (
                    -sum(term in item[1].lower() for term in query_terms),
                    -int(bool(_FAILURE.search(item[1]))),
                    item[0],
                ),
            )
            chosen_indices = sorted(index for index, _ in ranked[:max_per_file])
            chosen = [hits[index] for index in chosen_indices]
            selected.extend(chosen)
            protected.extend(line for line in chosen if _FAILURE.search(line))
            if len(hits) > len(chosen):
                selected.append(f"[entroly: {len(hits) - len(chosen)} additional hit(s) in this file are recoverable]")
        selected.extend(unparsed[:3])
        selected.extend(
            line for line in unparsed[3:] if _FAILURE.search(line) and line not in selected
        )
        compact = "\n".join(selected)
        if text.endswith("\n") and compact:
            compact += "\n"
        return _compact(
            text=text,
            compact=compact,
            source_id=source_id,
            content_type="search-results",
            codec=self.name,
            version=self.version,
            store=self.store,
            protected=tuple(dict.fromkeys(protected)),
            omitted_count=max(0, len(text.splitlines()) - len(selected)),
            item_label="search result line(s) restored",
        )


class _EvidenceHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.stack: list[str] = []
        self.fragments: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden_depth += 1
        self.stack.append(tag)
        if self.hidden_depth:
            return
        values = {key.lower(): value or "" for key, value in attrs}
        if tag in {"a", "button", "input", "select", "textarea"}:
            label = values.get("aria-label") or values.get("name") or values.get("title") or values.get("value")
            if label:
                self.fragments.append((tag, html.unescape(label).strip()))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"} and self.hidden_depth:
            self.hidden_depth -= 1
        if tag in self.stack:
            reverse_index = self.stack[::-1].index(tag)
            del self.stack[len(self.stack) - reverse_index - 1 :]

    def handle_data(self, data: str) -> None:
        if self.hidden_depth:
            return
        value = " ".join(data.split())
        if not value:
            return
        tag = self.stack[-1] if self.stack else "text"
        role = tag if tag in {"title", "h1", "h2", "h3", "h4", "li", "p", "a", "button", "label", "th", "td"} else "text"
        self.fragments.append((role, value))


class HtmlCodec:
    name = "html-evidence"
    version = "1"

    def __init__(self, store: RecoveryStore) -> None:
        self.store = store

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type.lower() in {"html", "text/html"}:
            return SupportDecision(True, 1.0, "declared HTML content type")
        sample = text[:4000].lower()
        signals = sum(marker in sample for marker in ("<!doctype html", "<html", "<body", "</html>"))
        return SupportDecision(signals >= 2, 0.93 if signals >= 3 else 0.82, "HTML document structure")

    def representations(self, text: str, source_id: str = "", **options: Any) -> list[Representation]:
        parser = _EvidenceHTMLParser()
        try:
            parser.feed(text)
            parser.close()
        except (ValueError, RecursionError):
            return [_full(text, source_id, "html", self.name, self.version)]
        terms = {
            match.group(0).lower()
            for match in _QUERY_WORD.finditer(str(options.get("query") or ""))
            if len(match.group(0)) > 1
        }
        budget = max(64, int(options.get("budget", 2000)))
        ranked: list[tuple[int, int, str]] = []
        for index, (role, value) in enumerate(parser.fragments):
            lower = value.lower()
            score = 100 * sum(term in lower for term in terms)
            if role in {"title", "h1", "h2", "h3", "button", "label", "a"}:
                score += 25
            if _FAILURE.search(value):
                score += 40
            rendered = f"{role}: {value}"
            ranked.append((score, index, rendered))
        selected: dict[int, str] = {}
        used = 0
        for _score, index, rendered in sorted(ranked, key=lambda item: (-item[0], item[1])):
            cost = estimate_tokens(rendered + "\n")
            if used + cost > budget:
                continue
            selected[index] = rendered
            used += cost
        compact = "\n".join(selected[index] for index in sorted(selected))
        # Every query term that exists in the source must remain addressable in
        # active context. Otherwise the safe behavior is full pass-through.
        source_lower = text.lower()
        compact_lower = compact.lower()
        required_terms = {term for term in terms if term in source_lower}
        if not required_terms.issubset({term for term in required_terms if term in compact_lower}):
            compact = text
        protected = tuple(
            rendered for rendered in selected.values() if _FAILURE.search(rendered)
        )
        return _compact(
            text=text,
            compact=compact,
            source_id=source_id,
            content_type="html",
            codec=self.name,
            version=self.version,
            store=self.store,
            protected=protected,
            omitted_count=max(0, len(parser.fragments) - len(selected)),
            item_label="HTML evidence fragment(s) restored",
        )


__all__ = ["DiffCodec", "HtmlCodec", "SearchResultCodec"]
