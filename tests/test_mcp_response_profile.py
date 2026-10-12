"""Agents should not pay for the receipt on every call, and nobody should break.

Measured on this repository: the public MCP tool catalog is about 1,593 tokens
carried every turn, and one ``optimize_context`` call that selected **zero**
fragments returned about 1,967 tokens across 34 top-level keys -- one token of
which was the answer. That was already ``compact`` mode. The catalog sits at the
front of the prompt prefix and caches; the response does not, so it is fresh
uncacheable content on every call.

Entroly's claim is *reduce, and keep a handle that recovers the exact bytes*.
Its own responses shipped the whole receipt inline regardless of whether
anything read it. This applies the claim inward.

The constraint that shapes the design is compatibility, not size.
``provenance`` is read in 15 files in this repository, ``context_assurance``
and ``online_prism`` in 7 each, so removing keys from the default response is
not available. The reduction therefore runs only under
``ENTROLY_MCP_PROFILE=public`` -- the profile every agent install manifest
already sets and which already trims the tool list -- while the ``full``
default that SDK, proxy, library and Node callers receive stays byte-identical.

So most of what follows is about the two ways this could do harm: changing the
default, and reducing something it cannot give back.
"""
from __future__ import annotations

import json

import pytest

from entroly.ccr import CompressedContextStore
from entroly.integrations.exact_recovery import retrieve_exact
from entroly.mcp_response import (
    AGENT_ESSENTIAL_KEYS,
    DIAGNOSTIC_KEY,
    apply_response_profile,
)
from entroly.tokens import count_tokens


def _payload() -> dict:
    """A response shaped like the real one, including the heavy receipt keys."""
    return {
        "selected_fragments": [{"source": "a.py", "text": "def f(): pass"}],
        "selected_count": 1,
        "tokens_used": 12,
        "total_tokens": 400,
        "tokens_saved": 388,
        "effective_budget": 400,
        "status": "ok",
        "method": "qccr",
        "request_id": "req-abc123",
        "response": {"mode": "compact"},
        # The receipt: real, worth keeping, not worth re-reading every call.
        "context_assurance": {
            "schema": "entroly.context-assurance.v1",
            "verdict": "uncertain",
            "query_commitment": "sha256:" + "a" * 64,
            "structural_assurance": {"budget": {"status": "passed"}},
        },
        "provenance": {"turn": 1, "fragment_count": 1, "hallucination_risk": "low"},
        "online_prism": {"reward": 0.0019, "weights": {"w_recency": 0.30}},
        "selection_pressure": {"resonance": 0.1, "content_gini": 0.0763},
        "crystallization": {"active_clusters": 1},
        "ravs_shadow": {"total_nodes": 1, "estimated_cost_usd": 0.0},
        "ravs_routing": {"use_original": True, "reason": "router_disabled"},
        "task_dream": {"status": "skipped", "reason": "budget too small"},
        "savings": {"lifetime": {"tokens_saved": 98246896}},
        "injection_scan": {"matches": []},
    }


@pytest.fixture()
def store() -> CompressedContextStore:
    return CompressedContextStore(max_entries=32)


# ── the default must not move ─────────────────────────────────────────


def test_the_full_profile_is_returned_byte_identical(store) -> None:
    """Every non-agent caller is on this path; it may not change at all."""
    payload = _payload()

    result = apply_response_profile(payload, profile="full", store=store)

    assert result == payload
    assert json.dumps(result, sort_keys=True) == json.dumps(payload, sort_keys=True)


@pytest.mark.parametrize("profile", ["full", "", "unknown", "research", "FULL"])
def test_only_the_public_profile_reduces(store, profile) -> None:
    """An unrecognised profile must behave like full, never like public."""
    payload = _payload()

    result = apply_response_profile(payload, profile=profile, store=store)

    assert result == payload, (
        "anything other than an explicit public profile must be left alone; "
        "guessing here would change output for callers who never opted in"
    )


def test_the_keys_other_code_reads_survive_in_full(store) -> None:
    payload = _payload()

    result = apply_response_profile(payload, profile="full", store=store)

    for key in ("provenance", "context_assurance", "online_prism"):
        assert key in result


# ── the public profile, and what it promises ──────────────────────────


def test_the_public_profile_defers_the_receipt_and_keeps_the_answer(store) -> None:
    payload = _payload()

    result = apply_response_profile(payload, profile="public", store=store)

    assert result["selected_fragments"] == payload["selected_fragments"]
    assert result["tokens_used"] == 12
    assert result["tokens_saved"] == 388, "the one number showing it worked stays"
    assert result["status"] == "ok"

    diagnostics = result[DIAGNOSTIC_KEY]
    assert diagnostics["status"] == "deferred"
    assert "context_assurance" in diagnostics["keys"]
    assert "provenance" in diagnostics["keys"]
    for key in diagnostics["keys"]:
        assert key not in result, f"{key} was reported deferred but is still inline"


def test_a_deferred_receipt_comes_back_byte_exact(store) -> None:
    """Reduction is only honest if the bytes are recoverable."""
    payload = _payload()

    result = apply_response_profile(payload, profile="public", store=store)
    recovered = retrieve_exact(result[DIAGNOSTIC_KEY]["handle"], store=store)

    assert recovered["integrity"] == "verified"
    restored = json.loads(recovered["original_content"])
    assert restored["context_assurance"] == payload["context_assurance"]
    assert restored["provenance"] == payload["provenance"]
    assert sorted(restored) == result[DIAGNOSTIC_KEY]["keys"]


def test_it_actually_costs_less(store) -> None:
    payload = _payload()

    full = apply_response_profile(payload, profile="full", store=store)
    lean = apply_response_profile(payload, profile="public", store=store)

    before = count_tokens(json.dumps(full, indent=2))
    after = count_tokens(json.dumps(lean, indent=2))
    assert after < before, "a reduction that does not reduce is overhead"
    assert after < before * 0.75, (
        f"expected a material reduction; {before} -> {after}"
    )


def test_the_response_says_what_it_removed_and_how_to_get_it(store) -> None:
    """An agent receiving fewer keys must be able to tell reduced from broken."""
    result = apply_response_profile(_payload(), profile="public", store=store)

    diagnostics = result[DIAGNOSTIC_KEY]
    assert diagnostics["redeem"] == "retrieve_context"
    assert diagnostics["handle"].startswith("ccr:")
    assert diagnostics["tokens_deferred"] > 0
    assert "nothing was discarded" in diagnostics["note"]


# ── the failure direction ─────────────────────────────────────────────


def test_an_unstorable_receipt_stays_inline(monkeypatch) -> None:
    """Fail safe, not fail small: unreduced and honest beats unrecoverable."""

    class Refuses:
        def store(self, **_kwargs):
            raise RuntimeError("store unavailable")

    payload = _payload()
    result = apply_response_profile(payload, profile="public", store=Refuses())

    assert result["context_assurance"] == payload["context_assurance"], (
        "if the diagnostics cannot be made recoverable they must not be removed"
    )
    assert result["provenance"] == payload["provenance"]
    assert result[DIAGNOSTIC_KEY]["status"] == "unavailable"
    assert "kept inline" in result[DIAGNOSTIC_KEY]["reason"]


def test_a_handle_collision_also_keeps_it_inline() -> None:
    """A handle addressing other content must never be reused."""
    from entroly.ccr import CcrHandleCollision

    class Collides:
        def store(self, **_kwargs):
            raise CcrHandleCollision("handle already addresses different content")

    result = apply_response_profile(_payload(), profile="public", store=Collides())

    assert result[DIAGNOSTIC_KEY]["status"] == "unavailable"
    assert "provenance" in result


def test_a_response_with_nothing_to_defer_says_so(store) -> None:
    lean = {key: 1 for key in list(AGENT_ESSENTIAL_KEYS)[:4]}

    result = apply_response_profile(lean, profile="public", store=store)

    assert result[DIAGNOSTIC_KEY]["status"] == "inline"
    assert result[DIAGNOSTIC_KEY]["reason"] == "nothing to defer"


def test_an_error_response_is_never_reduced_away(store) -> None:
    """A failure must reach the agent, not a handle pointing at the failure."""
    result = apply_response_profile(
        {"error": "File not found: nope.py", "status": "failed", "provenance": {"turn": 1}},
        profile="public",
        store=store,
    )

    assert result["error"] == "File not found: nope.py"
    assert result["status"] == "failed"
