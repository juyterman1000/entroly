"""An allowlist must be earned from evidence, and refused without it.

Tool-schema deferral withholds a schema only when a caller supplies an
allowlist, because withholding a needed schema is a correctness failure rather
than a cost one. That left one question: what evidence could justify an
allowlist? Measured answer: none existed.

The ``PostToolUse`` hook fires on every ``Bash|Read|Grep|Glob|Edit|Write|
TodoWrite`` call and receives the harness tool name. It read only
``tool_input.command``, classified that against CLI verifier patterns, and
returned early for anything unclassifiable. On this machine that produced 5,252
outcomes across 19 "tools" -- all verifiers, ``pytest``/``ruff``/``cargo_test``
-- and no record of the tool calls an allowlist would rank.

So these tests are mostly about refusal. A frequency table that happily
recommends a narrow allowlist from four observations would be worse than no
table at all, because the filter downstream is trusted to be evidence-based.
"""
from __future__ import annotations

import json

import pytest

from entroly.tool_activity import (
    MAX_TRACKED_TOOLS,
    activity_path,
    compact_activity,
    journal_path,
    load_activity,
    record_tool_use,
)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    return tmp_path


def _observe(counts: dict[str, int]) -> None:
    for name, times in counts.items():
        for _ in range(times):
            assert record_tool_use(name)


# ── the counter itself ────────────────────────────────────────────────


def test_only_a_name_is_recorded(home) -> None:
    """Counting that an Edit happened must not record what was edited.

    Checked on disk across both files, because the privacy claim is about what
    is persisted, not about which of the two files happens to hold it.
    """
    record_tool_use("Edit")
    assert journal_path().read_text(encoding="utf-8") == "Edit\n"

    compact_activity()
    payload = json.loads(activity_path().read_text(encoding="utf-8"))
    assert payload["counts"] == {"Edit": 1}
    assert set(payload) == {"schema", "counts"}

    on_disk = payload_text = json.dumps(payload) + journal_path().name
    for leak in ("/", "\\", "command", "argument", ".py"):
        assert leak not in payload_text.replace(journal_path().name, ""), (
            f"a persisted record must not contain {leak!r}"
        )
    assert on_disk  # the files exist and were inspected


def test_an_increment_does_not_rewrite_the_table(home) -> None:
    """The measurement that forced this design: no table write per call.

    Read-modify-write measured 16.6 ms per call because replacing the table
    created a new inode and the next read was a cold open. An append must not
    touch the table at all.
    """
    for _ in range(5):
        record_tool_use("Read")

    assert not activity_path().exists(), (
        "appending must not write the table; that write is what cost 14 ms"
    )
    assert load_activity().counts == {"Read": 5}, "and the count must still be right"


def test_compaction_folds_the_journal_and_clears_it(home) -> None:
    _observe({"Read": 7, "Edit": 3})
    assert journal_path().exists()

    assert compact_activity() is True

    assert not journal_path().exists(), "a folded journal must not be re-counted"
    assert json.loads(activity_path().read_text(encoding="utf-8"))["counts"] == {
        "Read": 7,
        "Edit": 3,
    }
    assert load_activity().counts == {"Read": 7, "Edit": 3}


def test_counts_survive_many_compactions(home) -> None:
    """Folding is amortised, so it happens repeatedly and must not drift."""
    for _ in range(3):
        _observe({"Read": 40, "Edit": 10})
        compact_activity()

    activity = load_activity()
    assert activity.counts == {"Read": 120, "Edit": 30}
    assert activity.total == 150


def test_a_table_and_a_journal_are_summed(home) -> None:
    _observe({"Read": 5})
    compact_activity()
    _observe({"Read": 2, "Edit": 1})

    assert load_activity().counts == {"Read": 7, "Edit": 1}


def test_a_damaged_journal_line_is_skipped_not_fatal(home) -> None:
    _observe({"Read": 3})
    with journal_path().open("a", encoding="utf-8") as fh:
        fh.write("has space\n\n\x00bad\nEdit\n")

    activity = load_activity()

    assert activity.counts == {"Read": 3, "Edit": 1}, (
        "a torn or hostile line costs one count and must not poison the table"
    )


@pytest.mark.parametrize(
    "bad", ["", "   ", None, 7, {"a": 1}, "has space", "semi;colon", "a" * 200]
)
def test_an_unusable_name_is_not_counted(home, bad) -> None:
    assert record_tool_use(bad) is False
    assert load_activity().total == 0


def test_the_table_is_bounded(home) -> None:
    for index in range(MAX_TRACKED_TOOLS + 25):
        record_tool_use(f"tool_{index}")

    activity = load_activity()

    assert activity.distinct == MAX_TRACKED_TOOLS
    # An already-counted tool must keep counting after the cap is reached.
    assert record_tool_use("tool_0") is True
    assert load_activity().counts["tool_0"] == 2


def test_a_damaged_table_reads_as_no_evidence(home) -> None:
    activity_path().parent.mkdir(parents=True, exist_ok=True)
    activity_path().write_text("{not json", encoding="utf-8")

    assert load_activity().total == 0


def test_hostile_values_in_the_table_are_dropped(home) -> None:
    activity_path().parent.mkdir(parents=True, exist_ok=True)
    activity_path().write_text(
        json.dumps({"counts": {"Read": 5, "bad name": 9, "Edit": "lots", "Neg": -3}}),
        encoding="utf-8",
    )

    activity = load_activity()

    assert activity.counts == {"Read": 5}


def test_counting_never_raises_when_the_directory_is_unusable(
    home, monkeypatch
) -> None:
    """This runs inside a host hook and may not fail the call it observes."""
    monkeypatch.setattr(
        "entroly.tool_activity.activity_path",
        lambda *a, **k: home / "nope" / "\x00" / "tool_activity.json",
    )

    assert record_tool_use("Read") is False
