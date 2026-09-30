"""A whole line of prose is not one claim just because it mentions an identifier.

`_candidate_claim_segments` runs two passes. The sentence pass splits prose into
sentences. The line pass then adds a whole line as a single claim when
`_is_list_or_table_row` or `_looks_like_code_claim` matches, so that a bare code
line -- `result = compute_score(x)` -- becomes a claim even though it is not a
sentence.

`_looks_like_code_claim` is satisfied by any text containing ``_``. Prose that
merely mentions an identifier therefore matches, and a single-line multi-sentence
response gets the entire line added on top of its per-sentence claims:

    "If the model cannot be loaded, nli_score returns neutral. The threshold is 0.60."
      -> the whole two-sentence line, as one claim
      -> the first sentence
      -> the second sentence

The extra segment is not a duplicate -- it is a longer, different string, which
is why the `seen` set in `add()` does not catch it. It is a conjunction of
several assertions, which is precisely what atomic claim extraction exists to
avoid: a verifier cannot attribute support or contradiction to a claim that
asserts three things at once.

Consequences: `total_claims` overstates how many assertions were checked, and
`summary_score` (1 - mean certificate risk) is skewed because the same content is
weighted twice.

The discriminator is sentence count. Every genuine code line is one sentence, so
the whole-line add is only redundant when the line holds more than one -- in
which case the sentence pass has already covered it.
"""
from __future__ import annotations

import pytest

from entroly.witness import (
    _candidate_claim_segments,
    _looks_like_code_claim,
    extract_claims,
)

#: Lines that are genuinely not sentences and must still become claims.
CODE_LINES = (
    "result = compute_score(x)",
    "self._audit.append(record)",
    "entroly/witness.py:938",
    "from entroly.witness import extract_claims",
)


def _segment_texts(text: str) -> list[str]:
    return [seg for _start, seg in _candidate_claim_segments(text)]


# ── The defect ───────────────────────────────────────────────────────


def test_a_multi_sentence_line_is_not_also_added_whole():
    text = (
        "If the model cannot be loaded, nli_score returns neutral. "
        "The threshold is 0.60."
    )

    segments = _segment_texts(text)

    whole = [s for s in segments if "nli_score" in s and "threshold" in s]
    assert not whole, (
        f"the entire multi-sentence line was added as one claim: {whole}"
    )


def test_each_sentence_of_such_a_line_is_still_its_own_claim():
    """Removing the whole-line segment must not remove the real ones."""
    text = (
        "If the model cannot be loaded, nli_score returns neutral. "
        "The threshold is 0.60."
    )

    claims = [c.text for c in extract_claims(text, force_python=True)]

    assert any("nli_score" in c for c in claims), claims
    assert any("threshold" in c for c in claims), claims


def test_claim_count_matches_the_number_of_assertions():
    text = (
        "nli_score returns neutral. The threshold is 0.60. "
        "The other threshold is 0.65."
    )

    claims = extract_claims(text, force_python=True)

    assert len(claims) == 3, (
        f"expected one claim per sentence, got {len(claims)}: "
        f"{[c.text for c in claims]}"
    )


# ── Guards: the code-claim path must keep working ────────────────────


@pytest.mark.parametrize("line", CODE_LINES)
def test_a_single_statement_code_line_is_still_a_claim(line):
    """These are not sentences; the line pass is the only thing that admits them.

    Without this guard the fix could delete the code-claim path entirely and
    every assertion above would still pass.
    """
    assert _looks_like_code_claim(line)

    segments = _segment_texts(line)

    assert segments, f"code line produced no claim segment: {line!r}"


def test_a_list_row_is_still_a_claim():
    segments = _segment_texts("- the retry limit is three attempts")

    assert segments, "list row produced no claim segment"


def test_prose_without_identifiers_is_unaffected():
    """This shape never triggered the line pass; it must not change."""
    text = "The cache is disabled by default. The retry limit is three attempts."

    claims = extract_claims(text, force_python=True)

    assert len(claims) == 2, [c.text for c in claims]


def test_a_single_sentence_line_with_an_identifier_still_yields_its_claim():
    """The boundary case: one sentence, so the line pass is not redundant."""
    claims = extract_claims(
        "The nli_score function returns unavailable.", force_python=True
    )

    assert claims
    assert any("nli_score" in c.text for c in claims), [c.text for c in claims]
