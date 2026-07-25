"""Shared exact-recovery contract for agent integrations.

Discovery and recovery are deliberately separate operations. Retrieval/ranking may
help an agent discover *which* evidence matters, but once Entroly emits a ``ccr:``
handle, recovery is an exact content-addressed lookup. No query, ranking, fuzzy
matching, or relevance threshold is accepted on this path.
"""

from __future__ import annotations

import re
from typing import Any

_EXACT_HANDLE_RE = re.compile(r"^(?:ccr:)?([0-9a-f]{24})$", re.IGNORECASE)


class ExactRecoveryError(ValueError):
    """Raised when an exact-recovery request violates the handle contract."""


def normalize_exact_handle(value: str) -> str:
    """Return a canonical lowercase ``ccr:<24-hex>`` handle.

    Raw 24-character hexadecimal digests are accepted for clients that strip the
    visible ``ccr:`` prefix while serializing tool arguments. Source paths,
    natural-language queries, empty strings, and partial hashes are rejected.
    """

    if not isinstance(value, str):
        raise ExactRecoveryError("recovery hash must be a string")
    match = _EXACT_HANDLE_RE.fullmatch(value.strip())
    if match is None:
        raise ExactRecoveryError(
            "recovery requires one exact ccr:<24-hex> handle; source paths and queries are not accepted"
        )
    return f"ccr:{match.group(1).lower()}"


def retrieve_exact(handle: str, *, store: Any | None = None) -> dict[str, Any]:
    """Return the complete original content for an exact CCR handle.

    The function intentionally calls ``store.retrieve`` rather than
    ``retrieve_or_materialize``. A missing historical hash remains a miss; it can
    never silently resolve to a newer source revision.
    """

    canonical = normalize_exact_handle(handle)
    if store is None:
        from ..ccr import get_ccr_store

        store = get_ccr_store()
    entry = store.retrieve(canonical)
    if entry is None:
        raise ExactRecoveryError(
            f"exact recovery handle {canonical!r} is unavailable or was evicted"
        )
    original = entry.get("original")
    if not isinstance(original, str):
        raise ExactRecoveryError("stored recovery entry has no exact original content")
    return {
        "status": "recovered",
        "lookup": "hash_only",
        "full_content": True,
        "retrieval_handle": canonical,
        "source": str(entry.get("source", "")),
        "content_sha256": str(entry.get("content_sha256", "")),
        "resolution": str(entry.get("resolution", "")),
        "original_tokens": int(entry.get("original_tokens", 0) or 0),
        "compressed_tokens": int(entry.get("compressed_tokens", 0) or 0),
        "original_content": original,
    }


def exact_recovery_tool_schema() -> dict[str, Any]:
    """Return the portable Hermes/OpenAI-style tool schema."""

    return {
        "name": "entroly_retrieve",
        "description": (
            "Recover the complete exact original content referenced by an Entroly "
            "ccr hash. This is a hash-only lookup: pass no query or source path."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "hash": {
                    "type": "string",
                    "pattern": r"^(?:[cC][cC][rR]:)?[0-9a-fA-F]{24}$",
                    "description": "Exact ccr:<24-hex> recovery handle.",
                }
            },
            "required": ["hash"],
        },
    }
