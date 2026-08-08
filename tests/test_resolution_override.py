"""A pinned resolution must be honoured exactly, or reported — never quietly changed.

Automatic resolution selection is the right default, but it cannot be right for
every question: measured on this repository, a signature-level view answered
12/12 questions whose evidence lives in a signature and 0/20 whose evidence
lives in a function body. A caller that knows which kind of question it is
asking needs a way to say so.

The dangerous failure is not "too many tokens" — it is returning a lower
resolution than the caller asked for while reporting success, because the caller
then reasons about source it never received.
"""

from __future__ import annotations

import pytest

from entroly.semantic_resolution import Resolution, resolve

SOURCE = '''\
"""Module docstring."""


def alpha(a, b):
    """Add two numbers."""
    total = a + b
    for _ in range(10):
        total += 1
    return total


def beta(x):
    """Multiply by two."""
    scratch = x * 2
    scratch += 1
    return scratch


class Gamma:
    """A class."""

    def method(self, value):
        """Do a thing."""
        return value + 1
'''


def test_default_is_still_automatic() -> None:
    result = resolve(SOURCE, query="alpha", budget=1000)
    assert result.forced_resolution is None


def test_full_returns_verbatim_bodies() -> None:
    """The 'verbatim on demand' case — the escape hatch that was missing."""
    result = resolve(SOURCE, query="", budget=1000, resolution=Resolution.FULL)

    assert result.forced_resolution == "full"
    assert set(result.resolution_counts) == {"full"}
    # Body lines, not just signatures.
    assert "total += 1" in result.output
    assert "scratch = x * 2" in result.output


def test_low_returns_only_stubs() -> None:
    result = resolve(SOURCE, query="", budget=1000, resolution=Resolution.LOW)
    assert set(result.resolution_counts) == {"low"}
    assert "total += 1" not in result.output
    assert "alpha" in result.output


def test_pinned_resolution_is_not_demoted_by_a_tiny_budget() -> None:
    """The load-bearing property.

    A caller asking for FULL under an impossible budget must receive FULL and
    be told it overran — not a silently downgraded MEDIUM that looks like a
    successful read.
    """
    result = resolve(SOURCE, query="", budget=1, resolution=Resolution.FULL)

    assert set(result.resolution_counts) == {"full"}
    assert "total += 1" in result.output
    assert result.over_budget is True
    assert result.total_tokens > result.budget


def test_automatic_mode_still_demotes_to_fit() -> None:
    """The override must not disturb existing budget behaviour."""
    result = resolve(SOURCE, query="", budget=1)
    assert result.forced_resolution is None
    assert result.resolution_counts.get("full", 0) == 0


def test_over_budget_is_false_when_it_fits() -> None:
    result = resolve(SOURCE, query="", budget=100_000, resolution=Resolution.FULL)
    assert result.over_budget is False


@pytest.mark.parametrize("level", ["full", "medium", "diff", "low"])
def test_every_documented_level_is_accepted(level: str) -> None:
    result = resolve(SOURCE, query="", budget=1000, resolution=level)
    assert result.forced_resolution == level


def test_skip_is_rejected_rather_than_returning_an_empty_document() -> None:
    with pytest.raises(ValueError, match="full/medium/diff/low"):
        resolve(SOURCE, query="", budget=1000, resolution="skip")


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="full/medium/diff/low"):
        resolve(SOURCE, query="", budget=1000, resolution="signatures")
