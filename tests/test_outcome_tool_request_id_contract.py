"""An outcome tool must name a field the caller can actually find.

Found by dogfooding. Three outcome-recording tools documented their first
argument as "the trace_id from the optimize_context call". ``optimize_context``
returns no ``trace_id``; the field is ``request_id``. A caller who follows the
docstring searches the response for a key that is not there, and then passes
something plausible instead -- which is what happened: the call was accepted
and silently landed in ``memory_promotion: {"status": "skipped", "reason":
"request_id does not match the active optimization"}``.

That is the worst shape for this defect. The tool does not reject the call, so
the loop looks closed while no outcome ever reaches the learning layer. Nothing
in the response says the id was unusable rather than merely unmatched.

So this pins the only thing that makes the loop closable: the name in the
docstring, the name in the response, and the key the matcher compares must all
be the same string.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1] / "entroly" / "server.py"

OUTCOME_TOOLS = ("record_test_result", "record_command_exit")


@pytest.fixture(scope="module")
def source() -> str:
    if not SERVER.is_file():
        pytest.skip(f"server module is absent: {SERVER}")
    return SERVER.read_text(encoding="utf-8")


def test_no_outcome_tool_points_at_a_field_named_trace_id(source: str) -> None:
    """``trace_id`` is not a key of any tool response in this server."""
    assert "trace_id" not in source, (
        "a docstring names trace_id, but optimize_context returns request_id. "
        "A caller who searches the response for trace_id finds nothing and "
        "passes a wrong id, which is accepted and then silently skipped."
    )


def test_the_documented_field_is_the_one_the_matcher_compares(source: str) -> None:
    """Docstring, response key and matcher key must be one string."""
    # The matcher that decides whether an outcome reaches the learning layer.
    assert (
        'str((_last_opt_ctx or {}).get("request_id", "")) == request_id' in source
    ), "the active-optimization matcher moved; re-point this guard at it"

    # The response field a caller reads it from.
    assert '"request_id": request_id' in source, (
        "optimize_context must return request_id for the outcome loop to be "
        "closable from the MCP surface"
    )


@pytest.mark.parametrize("tool", OUTCOME_TOOLS)
def test_each_outcome_tool_documents_where_the_id_comes_from(
    source: str, tool: str
) -> None:
    # Take the whole definition up to the next tool, rather than matching on
    # indentation: the closing ``) -> str:`` is indented less than the
    # parameters, so an indentation-anchored pattern stops before the docstring
    # and the guard passes on the signature alone.
    start = source.find(f"def {tool}(")
    assert start != -1, f"{tool} was not found; re-point this guard"
    rest = source[start:]
    end = re.search(r"\n    @mcp\.tool\(\)", rest)
    body = rest[: end.start()] if end else rest[:4000]
    assert '"""' in body, f"{tool}'s docstring was not captured; fix the guard"

    assert "request_id" in body, f"{tool} must take request_id"
    assert "optimize_context" in body, (
        f"{tool}'s docstring must say which call supplies request_id; without "
        "it the caller guesses, and a guess is accepted and then skipped"
    )
