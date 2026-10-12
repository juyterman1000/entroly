"""Apply Entroly's own reduction thesis to Entroly's own tool responses.

Measured on this repository: the public MCP tool catalog costs about 1,593
tokens, carried on every turn, and one ``optimize_context`` call that selected
**zero** fragments returned about 1,967 tokens across 34 top-level keys -- one
token of which was the answer. That call was already in ``compact`` mode.

The catalog is cacheable, sitting at the front of the prompt prefix, so its
real cost is amortised. The response is not. It is fresh content on every call,
so ten calls is roughly twenty thousand uncacheable tokens of ``online_prism``,
``selection_pressure``, ``crystallization``, ``ravs_shadow``, ``ravs_routing``
and ``task_dream`` -- from the tool whose purpose is reducing tokens.

Entroly's claim is *reduce, and keep a handle that recovers the exact bytes*.
Its own responses did the opposite: they shipped the whole receipt inline, every
call, whether or not anything read it. So this applies the claim inward. The
diagnostics are stored in the same content-addressed store the product already
exposes, and replaced by one handle plus a list of what moved.

Three properties this has to hold, in order of how much they bind:

1. **No existing caller may break.** ``provenance`` is read in 15 files in this
   repository, ``context_assurance`` and ``online_prism`` in 7 each. Removing
   keys from the default response is therefore not an option. This runs only
   under ``ENTROLY_MCP_PROFILE=public`` -- the profile every agent install
   manifest already sets, and which already trims the tool list. Library, SDK,
   proxy and Node callers default to ``full`` and see byte-identical output.
2. **Elision must be reversible before it happens.** If the store will not take
   the diagnostics, the blocks stay inline. Reducing without a redeemable
   handle is the destructive behaviour the product exists to avoid, and the
   browser-snapshot path already sets this precedent by passing the full
   snapshot through whenever exact recovery cannot be proved.
3. **The response must say what it did.** ``diagnostics.status`` separates
   ``inline`` from ``deferred`` from ``unavailable``, and a deferred response
   names every key it moved. An agent that silently receives fewer keys cannot
   tell a reduced answer from a degraded one.
"""

from __future__ import annotations

from typing import Any, Mapping

from .tokens import count_tokens

__all__ = [
    "AGENT_ESSENTIAL_KEYS",
    "DIAGNOSTIC_KEY",
    "apply_response_profile",
]

DIAGNOSTIC_KEY = "diagnostics"

# What an agent needs to act on this turn. Everything else is a receipt: real,
# worth keeping, and not worth re-reading on every call.
#
# Scalars stay even where their nested block leaves -- `tokens_saved` is the one
# number that shows the tool did something, and it costs single digits inline
# while the `savings` block it belongs to costs over a hundred.
AGENT_ESSENTIAL_KEYS = frozenset(
    {
        "selected_fragments",
        "selected",
        "selected_count",
        "tokens_used",
        "total_tokens",
        "total_fragments",
        "tokens_saved",
        "tokens_saved_this_call",
        "effective_budget",
        "user_budget",
        "budget_utilization",
        "method",
        "selector",
        "status",
        "no_match",
        "guidance",
        "error",
        "request_id",
        "response",
        DIAGNOSTIC_KEY,
    }
)


def _store_diagnostics(
    deferred: Mapping[str, Any], *, request_id: str, store: Any | None
) -> str:
    """Put the receipt in the content-addressed store, or return "".

    An empty return means the caller must keep the blocks inline. Every failure
    mode lands here deliberately: no store, a store that refuses, a collision,
    a serialisation error. None of them may produce a response that dropped
    diagnostics without a way to get them back.
    """
    try:
        import json

        if store is None:
            from .ccr import get_ccr_store

            store = get_ccr_store()
        payload = json.dumps(
            dict(deferred), ensure_ascii=False, sort_keys=True, indent=2, default=str
        )
        handle = store.store(
            source=f"entroly:optimize_context_diagnostics/{request_id or 'unknown'}",
            original_content=payload,
            compressed_content=f"[diagnostics for {request_id or 'unknown'}]",
            resolution="reference",
            original_tokens=count_tokens(payload),
            compressed_tokens=0,
        )
        return str(handle or "")
    except Exception:
        # Including CcrHandleCollision: a handle that already addresses other
        # content must not be reused, and the honest response then keeps the
        # diagnostics inline rather than pointing at someone else's bytes.
        return ""


def apply_response_profile(
    payload: Mapping[str, Any],
    *,
    profile: str,
    store: Any | None = None,
) -> dict[str, Any]:
    """Return ``payload`` shaped for ``profile``.

    ``full`` is returned unchanged, which is what every non-agent caller gets.
    ``public`` moves the receipt behind a handle and records the move.
    """
    original = dict(payload)
    if profile != "public":
        return original

    deferred = {
        key: value
        for key, value in original.items()
        if key not in AGENT_ESSENTIAL_KEYS
    }
    if not deferred:
        original[DIAGNOSTIC_KEY] = {"status": "inline", "reason": "nothing to defer"}
        return original

    kept = {
        key: value for key, value in original.items() if key in AGENT_ESSENTIAL_KEYS
    }
    request_id = str(original.get("request_id") or "")
    handle = _store_diagnostics(deferred, request_id=request_id, store=store)

    if not handle:
        # Fail safe, not fail small. Unreduced and honest beats reduced and
        # unrecoverable.
        original[DIAGNOSTIC_KEY] = {
            "status": "unavailable",
            "reason": "diagnostics could not be stored for recovery, so they were kept inline",
            "keys": sorted(deferred),
        }
        return original

    import json

    kept[DIAGNOSTIC_KEY] = {
        "status": "deferred",
        "handle": handle,
        "keys": sorted(deferred),
        "tokens_deferred": count_tokens(
            json.dumps(deferred, ensure_ascii=False, sort_keys=True, default=str)
        ),
        "redeem": "retrieve_context",
        "note": (
            "Full selection diagnostics, receipts and assurance records were "
            "stored verbatim. Pass the handle to retrieve_context to read them; "
            "nothing was discarded."
        ),
    }
    return kept
