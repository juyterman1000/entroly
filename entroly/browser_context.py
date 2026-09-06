"""Recoverable, query-conditioned browser accessibility evidence.

Rendered accessibility snapshots are smaller and more action-relevant than raw
DOM, but they are still untrusted web content.  This module never executes text
from a page as instructions, never persists browser credentials, and passes the
full snapshot through whenever query coverage or exact recovery cannot be
proved.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .codec import RecoveryReference, RecoveryStore, content_digest, estimate_tokens


_INTERACTIVE = re.compile(
    r"\b(button|checkbox|combobox|dialog|link|menuitem|radio|searchbox|tab|textbox)\b",
    re.IGNORECASE,
)
_STRUCTURAL = re.compile(r"\b(banner|heading|main|navigation|region|table)\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_STOPWORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}


@dataclass(frozen=True)
class BrowserContextResult:
    text: str
    original_tokens: int
    active_tokens: int
    recoverable_tokens: int
    mode: str
    query_term_count: int
    retained_query_term_count: int
    source_sha256: str
    recovery: RecoveryReference | None

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "entroly.browser-evidence.v1",
            "mode": self.mode,
            "tokens": {
                "active": self.active_tokens,
                "recoverable": self.recoverable_tokens,
                "original": self.original_tokens,
            },
            "query_coverage": {
                "required_terms": self.query_term_count,
                "retained_terms": self.retained_query_term_count,
                "complete": self.query_term_count == self.retained_query_term_count,
            },
            "source_sha256": self.source_sha256,
            "exact_recovery": bool(self.recovery),
            "recovery_digest": self.recovery.digest if self.recovery else None,
            "claim_boundary": (
                "Accessibility evidence was selected extractively. This receipt does not "
                "prove task success or visual equivalence to the rendered page."
            ),
        }


def _query_terms(query: str) -> tuple[str, ...]:
    return tuple(sorted({
        match.group(0).lower()
        for match in _WORD.finditer(query)
        if len(match.group(0)) > 1 and match.group(0).lower() not in _STOPWORDS
    }))


def _passthrough(snapshot: str, mode: str, terms: tuple[str, ...]) -> BrowserContextResult:
    tokens = estimate_tokens(snapshot)
    source_lower = snapshot.lower()
    present = sum(term in source_lower for term in terms)
    return BrowserContextResult(
        snapshot,
        tokens,
        tokens,
        0,
        mode,
        len(terms),
        present,
        content_digest(snapshot),
        None,
    )


def compress_accessibility_snapshot(
    snapshot: str,
    *,
    query: str = "",
    budget: int = 2_000,
    store: RecoveryStore | None = None,
    source_id: str = "browser",
) -> BrowserContextResult:
    """Select an extractive evidence envelope or return the complete snapshot."""
    original_tokens = estimate_tokens(snapshot)
    terms = _query_terms(query)
    if not snapshot or budget <= 0 or original_tokens <= budget:
        return _passthrough(snapshot, "passthrough", terms)

    lines = snapshot.splitlines()
    source_lower = snapshot.lower()
    if terms and any(term not in source_lower for term in terms):
        return _passthrough(snapshot, "passthrough-query-miss", terms)

    scored: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        lower = line.lower()
        matches = {term for term in terms if term in lower}
        score = 100 * len(matches)
        if _INTERACTIVE.search(line):
            score += 30
        if _STRUCTURAL.search(line):
            score += 12
        if line.strip().startswith(("- alert", "- status")):
            score += 40
        if score:
            scored.append((score, index, line))

    selected: dict[int, str] = {}
    used = 0
    for _score, index, line in sorted(scored, key=lambda item: (-item[0], item[1])):
        indent = len(line) - len(line.lstrip())
        candidates = [(index, line)]
        for ancestor in range(index - 1, max(-1, index - 16), -1):
            ancestor_line = lines[ancestor]
            ancestor_indent = len(ancestor_line) - len(ancestor_line.lstrip())
            if ancestor_line.strip() and ancestor_indent < indent:
                candidates.append((ancestor, ancestor_line))
                indent = ancestor_indent
                if indent == 0:
                    break
        for line_index, candidate in reversed(candidates):
            if line_index in selected:
                continue
            cost = estimate_tokens(candidate + "\n")
            if used + cost > budget:
                continue
            selected[line_index] = candidate
            used += cost

    compact = "\n".join(selected[index] for index in sorted(selected))
    retained = {term for term in terms if term in compact.lower()}
    if terms and retained != set(terms):
        return _passthrough(snapshot, "passthrough-budget-insufficient", terms)
    if not compact or estimate_tokens(compact) >= original_tokens:
        return _passthrough(snapshot, "passthrough-no-gain", terms)

    recovery_store = store if store is not None else RecoveryStore()
    recovery = recovery_store.put(
        snapshot,
        item_count=max(0, len(lines) - len(selected)),
        item_label="accessibility line(s) restored",
        note=f"complete browser accessibility snapshot for {source_id}",
    )
    try:
        recovered = recovery_store.recover(recovery)
    except (KeyError, ValueError):
        return _passthrough(snapshot, "passthrough-recovery-failed", terms)
    if recovered != snapshot:
        return _passthrough(snapshot, "passthrough-recovery-failed", terms)
    active = estimate_tokens(compact)
    return BrowserContextResult(
        compact,
        original_tokens,
        active,
        max(0, original_tokens - active),
        "compressed",
        len(terms),
        len(retained),
        content_digest(snapshot),
        recovery,
    )


def _validate_url(url: str, *, allow_private_network: bool) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("browser URL must use http or https and include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("credentials in browser URLs are not accepted")
    if allow_private_network:
        return
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError(f"browser hostname could not be resolved: {parsed.hostname}") from exc
    for raw in addresses:
        address = ipaddress.ip_address(raw.split("%", 1)[0])
        if not address.is_global:
            raise ValueError(
                "private, loopback, link-local, and reserved browser targets require "
                "--allow-private-network"
            )


def capture_accessibility_snapshot(
    url: str,
    *,
    timeout_ms: int = 30_000,
    allow_private_network: bool = False,
    max_snapshot_bytes: int = 16 * 1024 * 1024,
) -> str:
    """Capture an ephemeral Playwright ARIA snapshot with no stored profile."""
    _validate_url(url, allow_private_network=allow_private_network)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - exercised at CLI boundary
        raise RuntimeError(
            "browser support is not installed; run `pip install 'entroly[browser]'` "
            "and `playwright install chromium`"
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                accept_downloads=False,
                ignore_https_errors=False,
                service_workers="block",
            )
            if not allow_private_network:
                def guard(route: Any) -> None:
                    try:
                        _validate_url(route.request.url, allow_private_network=False)
                    except ValueError:
                        route.abort("blockedbyclient")
                        return
                    route.continue_()

                context.route("**/*", guard)
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            snapshot = page.locator("body").aria_snapshot(timeout=timeout_ms)
            if len(snapshot.encode("utf-8", "surrogatepass")) > max_snapshot_bytes:
                raise ValueError(
                    f"rendered accessibility snapshot exceeds {max_snapshot_bytes} bytes"
                )
            return snapshot
        finally:
            browser.close()


def query_fingerprint(query: str) -> str:
    """Return a stable privacy-safe identifier for a query without logging it."""
    return hashlib.sha256(query.encode("utf-8", "surrogatepass")).hexdigest()[:16]


__all__ = [
    "BrowserContextResult",
    "capture_accessibility_snapshot",
    "compress_accessibility_snapshot",
    "query_fingerprint",
]
