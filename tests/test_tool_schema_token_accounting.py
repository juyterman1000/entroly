"""The one number this module publishes must be the number it measured.

``defer_tool_schemas`` reports ``tokens_deferred``, and the proxy publishes it
to the caller as ``X-Entroly-Tool-Schema-Tokens-Deferred``. It was computed by
dividing the encoded byte length by four.

Measured against the o200k encoder on realistic schemas, that reads about 33%
high. Tool schemas are repetitive English descriptions over repeated JSON keys
and tokenize near 5.3 characters per token, not 4 -- the opposite direction
from dense minified data, which tokenizes near 3.3 and would have read low. So
the error is neither small nor consistently signed, and the one figure this
module exists to produce was inflated by a third.

The counter is now the canonical one. Because ``count_tokens`` itself degrades
to the same character heuristic when the encoder is unavailable, the result
carries ``token_accounting`` so a caller doing cost arithmetic can tell a
measurement from an approximation. An estimate labelled as an estimate is
useful; an estimate labelled as a measurement is the overclaim.

These tests do not re-check the filter's fail-closed behaviour, which
``test_tool_schema_savings_observability.py`` already covers. They check the
accounting.
"""
from __future__ import annotations

import json

import pytest

from entroly.tokens import count_tokens, tokenizer_is_exact
from entroly.tool_schema_deferral import defer_tool_schemas

requires_tokenizer = pytest.mark.skipif(
    not tokenizer_is_exact(),
    reason="exact token accounting requires the o200k encoder",
)


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Search the indexed repository for source spans matching a "
                "natural-language query. Returns ranked file paths with byte "
                "offsets and relevance scores. Does not modify the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "description": "Cap."},
                    "path_filter": {"type": "string", "description": "Glob."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def _canonical(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


@requires_tokenizer
def test_the_reported_saving_equals_the_measured_difference() -> None:
    tools = [_tool(f"tool_{index}") for index in range(12)]
    body = {"tools": tools}

    decision = defer_tool_schemas(body, "tool_0,tool_1,tool_2")

    assert decision.changed
    kept = decision.body["tools"]
    expected = count_tokens(_canonical(tools)) - count_tokens(_canonical(kept))
    assert decision.tokens_deferred == expected, (
        "the published saving must be the difference the encoder measured, not "
        "a character-length approximation of it"
    )


@requires_tokenizer
def test_the_character_heuristic_would_have_overstated_this() -> None:
    """Pins the defect's size and direction so a regression is obvious."""
    tools = [_tool(f"tool_{index}") for index in range(12)]

    decision = defer_tool_schemas({"tools": tools}, "tool_0")
    kept = decision.body["tools"]

    def heuristic(value) -> int:
        return (len(_canonical(value).encode("utf-8")) + 3) // 4

    inflated = heuristic(tools) - heuristic(kept)
    measured = decision.tokens_deferred

    assert inflated > measured, "the old counter must be the high one"
    overstatement = (inflated - measured) / measured
    assert 0.20 < overstatement < 0.50, (
        "tool schemas tokenize near 5.3 chars/token, so dividing by 4 reads "
        f"roughly a third high; measured {overstatement:.1%}"
    )


@requires_tokenizer
def test_an_exact_count_is_labelled_exact() -> None:
    decision = defer_tool_schemas(
        {"tools": [_tool("keep"), _tool("drop")]}, "keep"
    )

    assert decision.changed
    assert decision.token_accounting == "exact_o200k"


def test_an_estimated_count_is_not_labelled_exact(monkeypatch) -> None:
    """Without the encoder the figure is an estimate and must say so."""
    import entroly.tool_schema_deferral as module

    monkeypatch.setattr(module, "tokenizer_is_exact", lambda: False)

    decision = defer_tool_schemas(
        {"tools": [_tool("keep"), _tool("drop")]}, "keep"
    )

    assert decision.changed
    assert decision.token_accounting == "character_estimate", (
        "a caller doing cost arithmetic on this figure must be able to tell "
        "that the encoder was unavailable"
    )


def test_paths_that_publish_no_figure_claim_no_measurement() -> None:
    """Only the changed path reports a saving, so only it asserts accounting."""
    for body, header in (
        ({}, "anything"),
        ({"tools": []}, "anything"),
        ({"tools": [_tool("only")]}, ""),
        ({"tools": [_tool("only")]}, "only"),
    ):
        decision = defer_tool_schemas(body, header)
        assert decision.changed is False
        assert decision.tokens_deferred == 0


@requires_tokenizer
def test_a_non_ascii_description_is_counted_not_approximated() -> None:
    """Byte length and token count diverge hardest here."""
    wide = _tool("wide")
    wide["function"]["description"] = "検索インデックスを照会する。" * 12
    tools = [wide, _tool("narrow")]

    decision = defer_tool_schemas({"tools": tools}, "narrow")

    assert decision.changed
    kept = decision.body["tools"]
    assert decision.tokens_deferred == (
        count_tokens(_canonical(tools)) - count_tokens(_canonical(kept))
    )
    # The heuristic divides UTF-8 *bytes* by four, and CJK costs three bytes a
    # character, so here it errs the other way. Either way it is not a count.
    heuristic = (len(_canonical(tools).encode("utf-8")) + 3) // 4 - (
        len(_canonical(kept).encode("utf-8")) + 3
    ) // 4
    assert heuristic != decision.tokens_deferred
