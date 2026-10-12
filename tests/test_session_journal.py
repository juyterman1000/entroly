"""A journal is only worth keeping if it can be read back, and deleted.

Compaction loses the frontier rather than the objective -- measured elsewhere at
3.1% key-state omission against 8.2% next-action omission -- so this records
what was touched, what ran, and what failed, and ranks it against a query.

Two defects here were found by running it, not by reading it, and both are
pinned below because each silently destroyed the feature while every unit
looked correct:

* the path policy returns ``ordinary`` / ``sensitive`` / ``generated``, and an
  earlier draft compared against ``"allowed"``. Nothing matched, so *every*
  path was redacted, retrieval had no target text left to rank on, and a query
  for a file in the journal returned the wrong events;
* ``record_from_hook_payload`` writes ``"exit:1"`` while the classifier
  compared against the literal ``"error"``, so the single most valuable kind --
  a failing command, the clearest statement of where the frontier is -- was
  never labelled and ``kinds=["error"]`` returned nothing.

Neither would have been caught by asserting that the functions return
something.
"""
from __future__ import annotations

import json

import pytest

from entroly.session_journal import (
    forget_session,
    journal_path,
    recall,
    record_event,
    record_from_hook_payload,
)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    return tmp_path


def _session(home) -> None:
    for payload in (
        {"tool_name": "Edit", "tool_input": {"file_path": "entroly/ccr.py"}, "exit_code": 0},
        {
            "tool_name": "Bash",
            "tool_input": {"command": "pytest tests/test_ccr.py"},
            "exit_code": 1,
            "tool_output": "FAILED tests/test_ccr.py::test_retrieve - AssertionError",
        },
        {"tool_name": "Read", "tool_input": {"file_path": "entroly/tokens.py"}, "exit_code": 0},
        {"tool_name": "Edit", "tool_input": {"file_path": "entroly/checkpoint_relevance.py"}, "exit_code": 0},
        {"tool_name": "Bash", "tool_input": {"command": "ruff check entroly/"}, "exit_code": 0},
    ):
        assert record_from_hook_payload(payload)


# ── the two defects found by running it ───────────────────────────────


def test_an_ordinary_path_is_recorded_not_redacted(home) -> None:
    """The regression that gutted retrieval."""
    record_event(tool="Edit", target="entroly/checkpoint_relevance.py")

    (event,) = recall("", limit=5).events

    assert event.target == "entroly/checkpoint_relevance.py", (
        "the policy vocabulary is ordinary/sensitive/generated; comparing "
        "against 'allowed' redacted every path and left nothing to rank on"
    )


def test_a_query_matches_a_path_component(home) -> None:
    """The shared tokenizer treats a whole path as one term."""
    _session(home)

    result = recall("checkpoint", limit=3)

    assert result.matched >= 1, (
        "a journal containing entroly/checkpoint_relevance.py must match a "
        "query for 'checkpoint'; the shared tokenizer emits the whole path as "
        "a single term, so the event text carries its parts too"
    )
    assert result.events[0].target == "entroly/checkpoint_relevance.py"
    assert result.reason == "lexical_rank"


def test_a_failing_command_is_labelled_an_error(home) -> None:
    """The frontier is the point of the journal, so it must be selectable."""
    _session(home)

    result = recall("", kinds=["error"], limit=5)

    assert len(result.events) == 1
    event = result.events[0]
    assert event.kind == "error"
    assert event.status == "exit:1"
    assert "FAILED" in event.detail, "the failure text is what makes it useful"


# ── retrieval honesty ─────────────────────────────────────────────────


def test_an_unmatched_query_says_so_rather_than_implying_relevance(home) -> None:
    _session(home)

    result = recall("kubernetes helm chart rollout", limit=3)

    assert result.matched == 0
    assert result.reason == "no_lexical_match"
    assert result.events, "returning nothing would be less useful than recency"
    assert any("recency, not relevance" in note for note in result.notes), (
        "an agent must be able to tell a ranked answer from a fallback"
    )


def test_recall_reports_what_it_withheld(home) -> None:
    for index in range(30):
        record_event(tool="Edit", target=f"src/module_{index}.py")

    result = recall("module", limit=5)

    assert len(result.events) == 5
    assert result.matched == 30
    assert result.withheld == 25, (
        "five events out of thirty matching is a different answer from five "
        "out of five, and the caller cannot otherwise tell"
    )


def test_an_empty_journal_is_not_an_empty_answer(home) -> None:
    result = recall("anything")

    assert result.events == ()
    assert result.reason == "journal_empty"


def test_an_unknown_kind_is_distinguished_from_no_events(home) -> None:
    _session(home)

    result = recall("", kinds=["deployment"], limit=5)

    assert result.reason == "no_events_of_requested_kind"
    assert result.events == ()


# ── privacy and deletion ──────────────────────────────────────────────


def test_a_secret_shaped_path_is_never_written_down(home) -> None:
    for secret in (".env.production", "id_rsa", "secrets/credentials.json"):
        record_event(tool="Write", target=secret)

    raw = journal_path().read_text(encoding="utf-8")

    for secret in (".env.production", "id_rsa", "credentials.json"):
        assert secret not in raw, f"{secret} reached disk"
    assert raw.count("[sensitive:") == 3, "and each was replaced by a stable id"


def test_the_same_secret_gets_a_stable_id(home) -> None:
    """Stable so a reader can correlate, opaque so it discloses nothing."""
    record_event(tool="Write", target=".env.production")
    record_event(tool="Read", target=".env.production")

    targets = {event.target for event in recall("", limit=5).events}

    assert len(targets) == 1
    assert next(iter(targets)).startswith("[sensitive:")


def test_forgetting_removes_the_bytes(home) -> None:
    """A fresh session means a clean slate, not a filtered read."""
    _session(home)
    assert journal_path().exists()

    assert forget_session() is True

    assert not journal_path().exists()
    assert recall("ccr").reason == "journal_empty"


def test_forgetting_an_absent_journal_succeeds(home) -> None:
    assert forget_session() is True


# ── it must not break the call it observes ────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"tool_name": ""},
        {"tool_name": "Bash", "tool_input": None, "exit_code": "nope"},
        {"tool_name": "Edit", "tool_input": {"file_path": None}},
        {"tool_name": "X" * 500, "tool_input": {"command": "y" * 5000}},
    ],
)
def test_a_malformed_payload_never_raises(home, payload) -> None:
    record_from_hook_payload(payload)  # must not raise


def test_a_torn_line_costs_one_event_not_the_journal(home) -> None:
    _session(home)
    with journal_path().open("a", encoding="utf-8") as stream:
        stream.write('{"schema":"x","ts":1,"kind":"edit"\n')  # truncated
        stream.write("\n")
    record_event(tool="Edit", target="entroly/after_the_tear.py")

    result = recall("after_the_tear", limit=3)

    assert result.matched >= 1
    assert result.events[0].target == "entroly/after_the_tear.py"


def test_a_long_detail_is_bounded_on_disk(home) -> None:
    record_from_hook_payload(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "pytest"},
            "exit_code": 2,
            "tool_output": "E" * 20_000,
        }
    )

    line = journal_path().read_text(encoding="utf-8").strip()
    payload = json.loads(line)

    assert len(payload["detail"]) < 400, "one failing command must not dump 20 KB"
    assert payload["kind"] == "error"


# ── the frontier must survive length normalisation ────────────────────


def test_a_matching_error_is_never_ranked_out_of_the_result(home) -> None:
    """BM25 buries the event this journal exists to preserve.

    Length normalisation penalises an error *because* it carries the failure
    text. Measured directly: for the query "ccr", a routine edit of
    entroly/ccr.py scores 1.0000 while an error on the same file scores 0.5413
    and ranks last of 21. The adversarial arm of
    benchmarks/session_journal_retrieval.py went from 0 wins and 3 losses
    against plain recency to 6 wins and 0 losses once a slot was reserved.
    """
    for _ in range(20):
        record_event(tool="Edit", target="entroly/ccr.py", status="ok")
    record_event(
        tool="Bash",
        target="entroly/ccr.py",
        status="exit:1",
        detail="FAILED test_retrieve - AssertionError: digest drift",
    )
    for _ in range(20):
        record_event(tool="Edit", target="entroly/ccr.py", status="ok")

    result = recall("ccr", limit=8)

    errors = [event for event in result.events if event.kind == "error"]
    assert errors, (
        "a failure matching the query must reach the result; ranking alone "
        "puts it below every shorter event that shares its terms"
    )
    assert "digest drift" in errors[0].detail
    assert any("length normalisation" in note for note in result.notes), (
        "and the promotion must be disclosed, not silently applied"
    )


def test_the_reserved_slot_does_not_displace_an_error_already_present(home) -> None:
    record_event(tool="Bash", target="entroly/ccr.py", status="exit:1", detail="boom")
    record_event(tool="Edit", target="entroly/ccr.py", status="ok")

    result = recall("ccr", limit=8)

    assert len([e for e in result.events if e.kind == "error"]) == 1
    assert result.notes == (), "nothing was promoted, so nothing to disclose"
