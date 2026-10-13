"""A reader that drops content must say so, and say what can restore it.

Found by dogfooding. ``read_source_file`` was asked for the two functions whose
bodies mattered, and returned both as signatures only -- 11 of 17 blocks
reduced -- alongside a field named ``recovery`` set to null. The natural reading
is "content was dropped and cannot be recovered". Both halves of that were
wrong in different ways:

* ``recovery`` on this path meant "re-read past the session cache", unrelated
  to the content recovery the same word denotes everywhere else in Entroly
  (``receipt_id``/``span_id`` in codec receipts, ``retrieval_handles`` in the
  claims report);
* the cache-hit branch returns before that dictionary is built, so the field
  could only ever be null. It was a field that existed to be null.

The reduced bodies genuinely have no content handle -- this reader never
registers one, unlike the optimize and proxy paths -- but they are obtainable
by re-reading at full resolution or by line range. That is the distinction the
response now draws, rather than leaving the caller to infer it and infer it
wrongly.

No handle is fabricated for spans this path never registered. A handle that
cannot be redeemed is worse than an honest refusal, because the caller stops
looking for the text.
"""
from __future__ import annotations

import pytest

from entroly.server import _smart_read_elision


def test_a_complete_read_reports_nothing_to_recover() -> None:
    report = _smart_read_elision(6, {"full": 6})

    assert report["blocks_reduced"] == 0
    assert report["content_recovery"] == "not_required"
    assert report["obtain_full_text"] is None, (
        "a read that dropped nothing must not advertise a remedy"
    )


def test_a_reduced_read_states_the_omission_and_the_route() -> None:
    """The shape that misled: 11 of 17 blocks reduced to signatures."""
    report = _smart_read_elision(17, {"full": 6, "medium": 11})

    assert report["blocks_total"] == 17
    assert report["blocks_delivered_full"] == 6
    assert report["blocks_reduced"] == 11
    assert report["content_recovery"] == "unavailable_on_this_path"
    assert "does not register recovery handles" in report["reason"]
    assert 'resolution="full"' in report["obtain_full_text"]
    assert "line_start" in report["obtain_full_text"]


def test_omission_is_never_reported_as_a_recovery_promise() -> None:
    """Four states, and only one of them promises bytes can be produced."""
    report = _smart_read_elision(17, {"full": 6, "medium": 11})

    assert report["content_recovery"] in {
        "not_required",
        "unavailable_on_this_path",
    }
    assert "handle" not in str(report.get("content_recovery", "")), (
        "this path registers no handle; it must not imply one exists"
    )


def test_skipped_blocks_count_as_reduced() -> None:
    """SKIP drops a block entirely and must not read as delivered."""
    report = _smart_read_elision(10, {"full": 2, "medium": 3, "skip": 5})

    assert report["blocks_reduced"] == 8
    assert report["content_recovery"] == "unavailable_on_this_path"


@pytest.mark.parametrize(
    "total, counts",
    [
        (None, None),
        ("", {}),
        (0, {"full": 0}),
        (5, {"full": "not a number"}),
        (3, ["not", "a", "dict"]),
    ],
)
def test_a_malformed_reader_result_does_not_raise(total, counts) -> None:
    """This runs inside a tool response; it may not be the thing that fails."""
    report = _smart_read_elision(total, counts)

    assert report["blocks_reduced"] >= 0
    assert report["content_recovery"] in {
        "not_required",
        "unavailable_on_this_path",
    }


def test_delivered_is_never_reported_above_total() -> None:
    """A nonsensical count must not produce a negative omission."""
    report = _smart_read_elision(3, {"full": 9})

    assert report["blocks_reduced"] == 0
    assert report["content_recovery"] == "not_required"


def test_the_misleading_field_name_is_gone() -> None:
    """``recovery`` must not reappear on this reader's response."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "entroly" / "server.py"
    ).read_text(encoding="utf-8")

    assert '"recovery": "call again with fresh=true"' not in source, (
        "the dead, misleadingly named field is back; it can only ever be null "
        "because the cache-hit branch returns earlier, and it collides with "
        "Entroly's content-recovery vocabulary"
    )
    assert '"cache_recovery"' in source
    assert '"elision": _smart_read_elision(' in source
