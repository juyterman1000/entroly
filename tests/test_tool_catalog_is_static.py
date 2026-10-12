"""The tool catalog must not adapt per request, and the arithmetic says why.

Entroly ships tool-schema deferral, which filters inactive schemas out of an
*outbound provider request* and is the largest measured context lever available:
21K-57K tokens per turn, against about 2% from compressing their content. The
obvious next thought is to point it at Entroly's own MCP catalog, which costs
about 1,593 tokens. That thought is wrong, and expensively so.

The catalog sits at position 0 of the prompt prefix. Prompt caching charges
1.25x to write a prefix and 0.1x to read it, so a cached catalog costs a tenth
of its size per turn, while *changing* it costs twelve and a half times its
size once. Replacing a cached set ``S`` with a smaller ``S'`` therefore only
pays off over the remaining ``m`` turns when

    1.25*S' + 0.1*S'*(m-1)  <  0.1*S*m        =>   m > 11.5 * S' / (S - S')

Verified against brute force. A 5% reduction needs 219 remaining turns to repay
the single write it costs; 10% needs 104; even 50% needs 12. And a policy that
re-decides every turn pays that write every turn: measured at 50 turns with a
30% reduction, static-full costs 9,797 effective tokens, static-trimmed 6,858,
and per-turn dynamic 69,694 -- **7.1x worse than doing nothing at all.**

So the rule, which this test enforces:

    Any reduction applied at prefix position 0 must be static. Adaptive
    selection there converts a cacheable fixed cost into an uncacheable
    variable one, and loses by a factor of five to seven.

This is also why the catalog was left at full length rather than trimmed. After
caching, 300 tokens of prose is about 30 effective tokens per turn, while a
description an agent misreads costs a wrong tool call. The measured upside did
not justify the correctness risk, and 11 of the 13 descriptions carry "makes no
network call" -- a trust statement that ``openWorldHint`` does not encode.

Request-varying content belongs in the response, where it is already handled:
``entroly/mcp_response.py`` defers diagnostics behind a recoverable handle and
measured 1,967 -> 657 tokens. Responses are never cached, so reduction there is
pure gain with none of the arithmetic above.
"""
from __future__ import annotations

import inspect

from entroly import server


def _break_even_turns(full: float, reduced: float) -> float:
    """Remaining turns a runtime reduction must survive to repay its write."""
    return 11.5 * reduced / (full - reduced)


def test_the_break_even_formula_matches_the_cost_model() -> None:
    """Pin the arithmetic, so the rule below is not folklore."""
    write, read = 1.25, 0.10

    def cost(size: float, turns: int, changes: int) -> float:
        return changes * write * size + read * size * (turns - changes)

    full = 1593.0
    for fraction in (0.05, 0.10, 0.20, 0.30, 0.50):
        reduced = full * (1 - fraction)
        predicted = _break_even_turns(full, reduced)
        empirical = next(
            turns
            for turns in range(1, 5000)
            if cost(reduced, turns, changes=1) < read * full * turns
        )
        assert abs(predicted - empirical) <= 1, (
            f"{fraction:.0%}: formula says {predicted:.1f}, brute force {empirical}"
        )


def test_per_turn_adaptation_loses_to_doing_nothing() -> None:
    """The result that forecloses dynamic catalog filtering."""
    write, read = 1.25, 0.10
    full, turns, fraction = 1593.0, 50, 0.30
    reduced = full * (1 - fraction)

    static_full = write * full + read * full * (turns - 1)
    dynamic = turns * write * reduced

    assert dynamic > static_full * 5, (
        "re-deciding the catalog every turn pays a cache write every turn; it "
        f"must be recorded as far worse than doing nothing ({dynamic:.0f} vs "
        f"{static_full:.0f})"
    )


def test_the_public_catalog_is_a_constant_not_a_computation() -> None:
    """A per-request allowlist would be the dynamic policy the math rejects."""
    assert isinstance(server._PUBLIC_MCP_TOOLS, frozenset), (
        "the public tool set must be a module-level constant; building it per "
        "request is how adaptive prefix selection gets introduced by accident"
    )
    assert server._PUBLIC_MCP_TOOLS, "and it must not be empty"


def test_the_allowlist_depends_only_on_the_profile() -> None:
    """Its signature is the guard: no request, no session, no history."""
    signature = inspect.signature(server._mcp_profile_allowed_tools)

    assert list(signature.parameters) == ["profile"], (
        "tool-set selection may not take request state. Any extra parameter "
        "here means the catalog can vary per turn, which costs a 1.25x prefix "
        "write to save 0.1x reads."
    )


def test_the_same_profile_always_yields_the_same_catalog() -> None:
    for profile in ("full", "public"):
        first = server._mcp_profile_allowed_tools(profile)
        second = server._mcp_profile_allowed_tools(profile)
        assert first == second
        if first is not None:
            assert sorted(first) == sorted(second), "and in a stable order"


def test_reduction_still_happens_where_it_is_free() -> None:
    """Responses are never cached, so the arithmetic does not apply there."""
    from entroly.mcp_response import apply_response_profile

    payload = {"selected_fragments": [], "request_id": "r", "provenance": {"turn": 1}}

    assert apply_response_profile(payload, profile="full") == payload
    lean = apply_response_profile(payload, profile="public", store=None)
    assert "diagnostics" in lean
