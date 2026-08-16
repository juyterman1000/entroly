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


def test_no_guidance_when_fragments_present(tmp_path, monkeypatch):
    # A genuinely empty query match (with a populated session) is not an error.
    # The root must look like a project: this assertion previously used a
    # non-existent "/repo", which is indistinguishable from the mis-rooted case
    # the suspicious-root check now catches.
    monkeypatch.delenv("ENTROLY_SOURCE", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert _empty_context_guidance(1, str(tmp_path)) is None
    assert _empty_context_guidance(900, str(tmp_path)) is None


def test_populated_index_at_an_inherited_non_project_root_is_flagged(
    tmp_path, monkeypatch
):
    """The dangerous half of the dogfood finding.

    A server rooted at the MCP host's app bundle walks up plenty of files, so
    ingested_count is healthy and every emptiness check passes -- while recall
    answers from a corpus unrelated to the user's repository. Ranked results
    always look plausible, so nothing downstream can notice.
    """
    monkeypatch.delenv("ENTROLY_SOURCE", raising=False)
    bundle = tmp_path / "app-1.20186.0" / "resources"
    bundle.mkdir(parents=True)
    (bundle / "vendor.js").write_text("const a=1")

    guidance = _empty_context_guidance(20, str(tmp_path / "app-1.20186.0"))
    assert guidance is not None, "populated index at a non-project root must warn"
    assert guidance["status"] == "suspicious_source_root"
    joined = " ".join(guidance["resolve"]).lower()
    assert "entroly_source" in joined
    assert "restart" in joined


def test_explicit_entroly_source_is_respected(tmp_path, monkeypatch):
    # An operator who names the root deliberately is not second-guessed, even
    # when it carries no project marker.
    monkeypatch.setenv("ENTROLY_SOURCE", str(tmp_path))
    assert _empty_context_guidance(20, str(tmp_path)) is None


def test_guidance_is_json_safe():
    import json

    g = _empty_context_guidance(0, "/repo")
    # Serializes cleanly for the MCP string return path.
    assert json.loads(json.dumps(g))["status"] == "no_codebase_indexed"
