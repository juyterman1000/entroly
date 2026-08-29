"""An empty reply must say why it is empty.

Found by driving the MCP server the way an editor does: a tools/call issued
straight after the handshake came back with selected_fragments: [] and, across
29 payload keys, nothing indicating that indexing was still running. To the
agent that is indistinguishable from "this repository has nothing relevant",
and an agent told that stops asking.

The reasoning to explain it already existed. It sat behind a guard that
returned early whenever the selection was empty -- the exact case it was
written for -- so it could only ever fire for a non-empty selection drawn from
zero candidates, which does not happen.
"""

from __future__ import annotations

import pytest

from entroly.engine import apply_no_match_contract


def _empty(total_fragments: int = 0) -> dict:
    return {"selected_fragments": [], "total_fragments": total_fragments}


class TestEmptySelectionIsExplained:
    def test_an_empty_selection_is_not_silent(self):
        result = _empty()
        apply_no_match_contract(result, "where is the certificate enforced")

        assert result.get("status") == "no_match", (
            "an empty reply with no explanation reads as 'nothing is relevant'"
        )
        assert result["no_match"]["reason"]
        assert result["no_match"]["remediation"]

    def test_indexing_in_progress_is_reported_as_such(self, monkeypatch):
        """Not yet read is not the same as nothing matched."""
        monkeypatch.setattr("entroly.engine._indexing_still_running", lambda: True)
        result = _empty()
        apply_no_match_contract(result, "any query")

        assert result["no_match"]["indexing_in_progress"] is True
        assert "still running" in result["no_match"]["reason"]
        assert "retry" in result["no_match"]["remediation"]

    def test_a_settled_empty_index_does_not_blame_indexing(self, monkeypatch):
        monkeypatch.setattr("entroly.engine._indexing_still_running", lambda: False)
        monkeypatch.setattr("entroly.engine._usable_core_absent", lambda: False)
        result = _empty()
        apply_no_match_contract(result, "any query")

        assert result["no_match"]["indexing_in_progress"] is False
        assert "still running" not in result["no_match"]["reason"]
        assert "not the limiting factor" in result["no_match"]["remediation"], (
            "the user must not be told to reword a query that was never ranked"
        )

    def test_ranked_candidates_that_missed_do_advise_rephrasing(self, monkeypatch):
        """With candidates ranked, the query genuinely is the variable."""
        monkeypatch.setattr("entroly.engine._indexing_still_running", lambda: False)
        result = _empty(total_fragments=1849)
        apply_no_match_contract(result, "zzzz nonexistent token")

        assert result["no_match"]["candidates_considered"] == 1849
        assert "rephrase" in result["no_match"]["remediation"]

    def test_a_non_empty_selection_is_untouched_by_this_path(self):
        """The trimming contract must keep working for real selections."""
        result = {
            "selected_fragments": [
                {"path": "entroly/proxy.py", "content": "certificate enforced here",
                 "relevance": 1.0, "token_count": 5}
            ],
            "total_fragments": 1,
        }
        apply_no_match_contract(result, "certificate enforced")

        assert result["selected_fragments"], "a genuine match must survive"

    def test_no_query_still_returns_early(self):
        result = _empty()
        apply_no_match_contract(result, "")
        assert "status" not in result


class TestIndexingProbeFailsSafe:
    def test_the_probe_never_raises(self):
        from entroly.engine import _indexing_still_running

        assert _indexing_still_running() in (True, False)

    def test_the_server_exposes_the_state(self):
        from entroly.server import indexing_in_progress

        assert indexing_in_progress() in (True, False)
