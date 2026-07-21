"""Fail-loud guidance when optimize_context has no indexed codebase.

Regression cover for the dogfood finding: an unrooted/empty entroly server
returned selected: [] with hallucination_risk high and no explanation, which
an agent reads as "no relevant context" rather than "misconfigured".
"""

from __future__ import annotations

from entroly.server import _empty_context_guidance


def test_guidance_emitted_when_nothing_indexed():
    g = _empty_context_guidance(0, r"C:\some\app\dir")
    assert g is not None
    assert g["status"] == "no_codebase_indexed"
    assert g["resolved_source_root"] == r"C:\some\app\dir"
    # Must name the concrete fix, not just report failure.
    joined = " ".join(g["resolve"]).lower()
    assert "entroly_source" in joined
    assert "restart" in joined


def test_no_guidance_when_fragments_present():
    # A genuinely empty query match (with a populated session) is not an error.
    assert _empty_context_guidance(1, "/repo") is None
    assert _empty_context_guidance(900, "/repo") is None


def test_guidance_is_json_safe():
    import json

    g = _empty_context_guidance(0, "/repo")
    # Serializes cleanly for the MCP string return path.
    assert json.loads(json.dumps(g))["status"] == "no_codebase_indexed"
