"""One tool call must be one event, and a command must survive its preamble.

Both defects here were found by calling ``resume_state`` against the merged
code, not by reading it, and neither was visible to the existing suite --
which writes events one at a time with already-clean targets.

Measured on a real 84-event journal:

* **12 byte-identical duplicate lines**, in adjacent runs of two and four. A
  host fans one tool call out to every matching hook registration, and
  Entroly's own onboarding wrote a user-scope hook matching ``Bash`` while the
  repository shipped one matching ``Bash|Read|Grep|Glob|Edit|Write|TodoWrite``.
  Eight returned slots held three distinct commands.
* **83% of targets hit the 200-char cap**, and 61 of 69 began with the same
  ``cd <repo>`` prefix, so the ranker scored boilerplate every event shared
  and the actual command was truncated away.

A timestamp cannot separate these cases: ``time.time()`` returned 8 distinct
values across 200,000 calls on the measured host.
"""
from __future__ import annotations

import json

import pytest

from entroly.session_journal import (
    DEDUP_WINDOW_SECONDS,
    MAX_REDELIVERY_RUN,
    MAX_TARGET_CHARS,
    SCHEMA,
    journal_path,
    recall,
    record_event,
    record_from_hook_payload,
)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    return tmp_path


def _write_raw(rows: list[dict]) -> None:
    path = journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps({"schema": SCHEMA, "detail": "", "session": "", **row})
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


# ── one call, recorded by every matching hook, is one event ───────────


def test_a_call_recorded_by_two_hooks_is_one_event(home) -> None:
    """The defect that spent 5 of 8 slots on repeats."""
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": "pytest tests/test_ccr.py -q"},
        "exit_code": 0,
    }
    assert record_from_hook_payload(payload)
    assert record_from_hook_payload(payload)  # the second registration

    result = recall("pytest", limit=8)

    assert len(result.events) == 1, "two deliveries of one call are one event"
    assert result.redelivered_collapsed == 1, (
        "and the collapsing must be reported, not silently applied"
    )


def test_four_deliveries_collapse_to_one(home) -> None:
    """Observed in the wild: two registrations times a retried call."""
    for _ in range(4):
        record_event(tool="Bash", target="git checkout -b feature main")

    result = recall("checkout", limit=8)

    assert len(result.events) == 1
    assert result.redelivered_collapsed == 3


def test_the_same_command_run_again_later_is_two_events(home) -> None:
    """Re-running a command is real work, not a redelivery.

    This is why identity cannot be content alone, and why the window exists.
    """
    base = 1_700_000_000.0
    _write_raw(
        [
            {"ts": base, "kind": "command", "tool": "Bash",
             "target": "pytest tests/", "status": "ok"},
            {"ts": base + DEDUP_WINDOW_SECONDS + 60, "kind": "command",
             "tool": "Bash", "target": "pytest tests/", "status": "ok"},
        ]
    )

    result = recall("pytest", limit=8)

    assert len(result.events) == 2, (
        "the same command minutes apart is two runs; collapsing it would "
        "erase the retry that is often the whole point"
    )
    assert result.redelivered_collapsed == 0


def test_duplicates_separated_by_other_work_are_both_kept(home) -> None:
    """Adjacency is part of the identity; a later repeat is not a redelivery."""
    base = 1_700_000_000.0
    _write_raw(
        [
            {"ts": base, "kind": "command", "tool": "Bash",
             "target": "ruff check entroly/", "status": "ok"},
            {"ts": base + 0.1, "kind": "edit", "tool": "Edit",
             "target": "entroly/ccr.py", "status": "ok"},
            {"ts": base + 0.2, "kind": "command", "tool": "Bash",
             "target": "ruff check entroly/", "status": "ok"},
        ]
    )

    result = recall("ruff", limit=8)

    assert len([e for e in result.events if e.tool == "Bash"]) == 2
    assert result.redelivered_collapsed == 0


def test_a_run_too_long_for_fan_out_is_left_alone(home) -> None:
    """Collapsing real repeated work is the worse failure, so the bound errs.

    Fan-out multiplicity is the number of matching hook registrations, which is
    small. A burst longer than that is an agent genuinely touching one file
    repeatedly, and flattening it to a single event would under-report the
    work. This bound is what keeps the error-promotion ranking case -- which
    depends on a long run of look-alike edits -- intact.
    """
    for _ in range(MAX_REDELIVERY_RUN + 4):
        record_event(tool="Edit", target="entroly/ccr.py", status="ok")

    result = recall("ccr", limit=MAX_REDELIVERY_RUN + 8)

    assert len(result.events) == MAX_REDELIVERY_RUN + 4
    assert result.redelivered_collapsed == 0


def test_a_failure_is_not_collapsed_into_the_success_before_it(home) -> None:
    """Status is part of the identity, so the frontier survives."""
    record_event(tool="Bash", target="pytest tests/", status="ok")
    record_event(tool="Bash", target="pytest tests/", status="exit:1",
                 detail="FAILED test_retrieve")

    result = recall("pytest", kinds=["error"], limit=8)

    assert len(result.events) == 1
    assert result.events[0].status == "exit:1"


# ── a command must outlive its shell preamble ─────────────────────────


def test_a_command_is_recorded_without_its_shell_preamble(home) -> None:
    """The cap truncated the verb away and left the shared prefix behind."""
    record_from_hook_payload(
        {
            "tool_name": "Bash",
            "tool_input": {
                "command": (
                    "cd /c/Users/abhis/entroly-codebase-intelligence\n"
                    "export ENTROLY_NO_SELF_HEAL=1\n"
                    "pytest tests/test_session_journal.py -q --tb=short"
                )
            },
            "exit_code": 0,
        }
    )

    (event,) = recall("", limit=5).events

    assert event.target == "pytest tests/test_session_journal.py -q --tb=short"
    assert "cd " not in event.target
    assert "export" not in event.target


def test_a_query_reaches_a_command_buried_past_the_cap(home) -> None:
    """The user-visible consequence, stated as retrieval.

    Stripping the preamble does not shorten a genuinely long command -- that
    is the cap doing its job. What it fixes is a *setup* preamble longer than
    the cap, which is the measured case: the repository path plus a few
    exports, repeated identically on every call in the session.
    """
    preamble = " && ".join(
        [
            "cd /c/Users/abhis/entroly-codebase-intelligence",
            "export ENTROLY_NO_SELF_HEAL=1",
            "export ENTROLY_DIR=/c/Users/abhis/scratch/entroly-test-directory",
            "export PYTHONPATH=/c/Users/abhis/entroly-codebase-intelligence",
            "export RUST_LOG=debug",
        ]
    )
    assert len(preamble) > MAX_TARGET_CHARS, (
        "the fixture only reproduces the defect if the setup alone overruns "
        "the cap, which is what pushed the verb out of the record"
    )

    record_from_hook_payload(
        {
            "tool_name": "Bash",
            "tool_input": {"command": f"{preamble} && cargo clippy --all-targets"},
            "exit_code": 0,
        }
    )

    result = recall("clippy", limit=5)

    assert result.matched >= 1, (
        "before the preamble was stripped this command's identifying verb was "
        "pushed past MAX_TARGET_CHARS and the query matched nothing"
    )
    assert result.events[0].target == "cargo clippy --all-targets"


def test_an_env_assignment_prefix_keeps_the_command(home) -> None:
    """``FOO=1 pytest`` runs pytest; dropping the segment loses the verb."""
    record_from_hook_payload(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "RUST_LOG=debug cargo test --lib"},
            "exit_code": 0,
        }
    )

    (event,) = recall("", limit=5).events

    assert event.target == "cargo test --lib"


def test_a_setup_only_call_is_still_recorded(home) -> None:
    """An empty target is worse than a boilerplate one."""
    record_from_hook_payload(
        {"tool_name": "Bash", "tool_input": {"command": "cd /repo"}, "exit_code": 0}
    )

    (event,) = recall("", limit=5).events

    assert event.target, "a call that only positions the shell still happened"


def test_a_file_path_is_never_rewritten(home) -> None:
    """Only a command is salience-stripped. A path is already the unit."""
    record_from_hook_payload(
        {
            "tool_name": "Edit",
            "tool_input": {"file_path": "entroly/set/export/cd_helper.py"},
            "exit_code": 0,
        }
    )

    (event,) = recall("", limit=5).events

    assert event.target == "entroly/set/export/cd_helper.py", (
        "path segments that happen to spell shell verbs must survive intact"
    )
