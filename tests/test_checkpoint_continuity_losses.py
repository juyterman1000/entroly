"""A handoff may bound what it carries, but not hide what it dropped.

Found by dogfooding: a checkpoint written with sixteen decisions resumed with
several of them severed mid-sentence -- "...48 row", "P1b = collision-". The
bound was 500 characters per decision, applied silently. A resuming agent
receives an instruction that stops mid-word and cannot tell it from a complete
one, which is the precise failure a handoff exists to prevent.

The same function dropped decisions past the twentieth without a word, and
keyed deduplication on the *bounded* text, so two long decisions differing only
after character 500 collided and the second was discarded as a duplicate. That
last one is a correctness bug: the longer the decisions, the likelier it fires,
and nothing reports it.

Bounds are kept -- an unbounded checkpoint is its own failure. What changes is
that every bound now reports itself.
"""
from __future__ import annotations

from entroly.checkpoint_relevance import (
    DECISION_CHAR_LIMIT,
    bounded_text,
    normalize_decisions,
)


def test_a_severed_decision_says_it_was_severed() -> None:
    long_decision = "P1b = collision-safe storage because " + "x" * DECISION_CHAR_LIMIT

    (carried,) = normalize_decisions([long_decision])

    assert carried.startswith("P1b = collision-safe storage because")
    assert "truncated" in carried, (
        "a decision cut mid-sentence must say so; otherwise the next agent "
        "cannot distinguish a complete instruction from a severed one"
    )
    assert str(len(long_decision)) in carried, "report the size that did not fit"


def test_a_decision_that_fits_is_untouched() -> None:
    """The marker must not appear on content that was carried whole."""
    decision = "Recovery stays hash-only; discovery is a separate verb."

    assert normalize_decisions([decision]) == [decision]
    assert bounded_text(decision, DECISION_CHAR_LIMIT) == decision


def test_decisions_dropped_past_the_limit_are_counted() -> None:
    decisions = [f"decision {index}" for index in range(25)]

    carried = normalize_decisions(decisions, limit=20)

    assert carried[:20] == decisions[:20]
    assert carried[-1] == "[5 further decision(s) not carried: raise max_decisions]"


def test_an_earlier_loss_survives_a_later_merge() -> None:
    """Re-normalising must not re-hide a loss recorded upstream."""
    already_lost = normalize_decisions(
        [f"decision {index}" for index in range(25)], limit=20
    )

    # A later merge re-normalises the stored list, which now fits the limit.
    again = normalize_decisions(already_lost, limit=20)

    assert again[-1] == "[5 further decision(s) not carried: raise max_decisions]", (
        "the note is parsed and carried forward, not skipped, or the count of "
        "what was lost disappears at the next checkpoint"
    )


def test_notes_do_not_compound_across_merges() -> None:
    """Carrying the count forward must not keep inflating it."""
    first = normalize_decisions([f"decision {i}" for i in range(25)], limit=20)
    second = normalize_decisions(first, limit=20)
    third = normalize_decisions(second, limit=20)

    assert second[-1] == third[-1] == (
        "[5 further decision(s) not carried: raise max_decisions]"
    )


def test_long_decisions_differing_past_the_bound_are_not_merged() -> None:
    """The correctness bug: dedup keyed on the bounded text, so these collided."""
    shared = "y" * (DECISION_CHAR_LIMIT + 50)
    first = f"{shared} ... therefore take option (b)"
    second = f"{shared} ... therefore take option (c)"

    carried = normalize_decisions([first, second])

    assert len(carried) == 2, (
        "two distinct decisions were silently collapsed into one because "
        "deduplication compared only the first 500 characters"
    )


def test_identical_decisions_are_still_deduplicated() -> None:
    """Keying on full text must not lose case-insensitive dedup."""
    carried = normalize_decisions(
        ["Use canonical provider adapters", "use canonical provider adapters"]
    )

    assert carried == ["Use canonical provider adapters"]
