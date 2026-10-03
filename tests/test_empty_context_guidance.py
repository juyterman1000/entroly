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


def test_guidance_does_not_blame_the_working_directory():
    """The message must not assert a cause that was measured not to be one.

    It used to say an empty index "usually means the MCP server's working
    directory is not your project root", and offered ENTROLY_SOURCE as the first
    remedy. Measured against a freshly spawned `python -m entroly.server` on this
    repository, with cwd at the repository root and ENTROLY_SOURCE set to it,
    recall_relevant still returned count 0 -- so the stated cause was wrong and
    the first remedy sent the reader to restart a server that came back equally
    empty. Ingesting is what actually populated the index.
    """
    g = _empty_context_guidance(0, "/repo", tool="recall_relevant")

    assert "working directory is not your project root" not in g["message"], (
        "the message asserts a cause that a correct root does not fix"
    )
    # The first remedy must be the one that demonstrably works.
    first = g["resolve"][0].lower()
    assert any(t in first for t in ("ingest", "remember_fragment", "read_source_file")), (
        f"first remedy should be ingestion, got: {g['resolve'][0]!r}"
    )
    # And it must still say an empty index is not proof the code is absent,
    # which is the misreading that made this worth fixing at all.
    assert "not evidence" in g["message"] or "nothing has been read" in g["message"]


def test_message_names_the_tool_that_returned_nothing():
    """The message is read by an agent choosing its next action.

    It was hardcoded to ``optimize_context``, so reusing this guidance from any
    other tool would tell the reader to go inspect a tool it never called.
    """
    assert "optimize_context" in _empty_context_guidance(0, "/repo")["message"]
    recall = _empty_context_guidance(0, "/repo", tool="recall_relevant")["message"]
    assert "recall_relevant" in recall
    assert "optimize_context" not in recall


def test_recall_relevant_reports_an_unindexed_server(monkeypatch, tmp_path):
    """``recall_relevant`` is the first call the server instructions prescribe.

    On a server that indexed nothing it returned ``count: 0`` plus a hint to
    retry with ``full=True`` -- the one parameter that cannot help, since there
    are no bodies to expand. An agent reads that as "this repository does not
    contain the code", which is the opposite of the truth, and the fix (point
    the server at the repo root) is never surfaced. optimize_context had warned
    about exactly this since the original dogfood finding; recall_relevant was
    simply never wired to the same check.
    """
    import json

    from entroly import server as srv

    monkeypatch.delenv("ENTROLY_SOURCE", raising=False)
    monkeypatch.chdir(tmp_path)

    class _EmptyEngine:
        _use_rust = False
        _total_fragments_ingested = 0

        def recall_relevant(self, query, top_k):
            return []

    payload = json.loads(
        srv._recall_relevant_payload(_EmptyEngine(), "anything at all", 3, False)
    )

    assert payload["count"] == 0
    assert "hint" not in payload, (
        "an empty result must not advertise full=True, which cannot produce "
        f"results; got {payload.get('hint')!r}"
    )
    assert payload["guidance"]["status"] == "no_codebase_indexed"
    assert "recall_relevant" in payload["guidance"]["message"]


def test_recall_relevant_keeps_the_slim_hint_when_it_found_something():
    """The hint is still correct whenever there is a body to expand."""
    import json

    from entroly import server as srv

    class _OneHitEngine:
        _use_rust = False
        _total_fragments_ingested = 7

        def recall_relevant(self, query, top_k):
            return [{"source": "a.py", "score": 0.9, "content": "def a(): pass"}]

    payload = json.loads(
        srv._recall_relevant_payload(_OneHitEngine(), "a", 3, False)
    )
    assert payload["count"] == 1
    assert "full=True" in payload["hint"]
