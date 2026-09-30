"""A declarative sentence that opens with a wh-word is still a claim.

`_is_question_like` treated any sentence whose first word is in
`_QUESTION_STARTERS` as a question, with no requirement that it end in `?`.
English uses the same words to open declarative clauses:

    When the lock times out, the append is unserialized.
    Where the path is absolute, resolution is skipped.
    What the receipt omits is listed under omitted_context.

All of these were rejected by `_is_claim_like` and never certified. For a
verifier that is a silent loss: the claim is not checked, and nothing reports
that it went unchecked.

The defect hid behind the dialogue fallback. When such a sentence is the whole
output, sentence-level extraction yields nothing, the fallback wraps the entire
response as one claim, and the claim appears to be extracted. It is only lost
when a second, ordinary sentence follows -- then the fallback does not fire:

    S1 alone      -> 1 claim   (the fallback's, not the sentence's)
    S1 + S2       -> 1 claim   (S1 gone)
    S2 + S1       -> 2 claims  (S1 present)

So the single-sentence test anyone would write cannot see it.

Measured through `verify_response` before this fix: a response whose only false
claim sat in that position reported `total_claims: 2` and `flagged_claims: []`
-- the contradiction was never named. The aggregate risk still flagged the
response, so this is a defect in the audit trail rather than a demonstrated
false pass.

Only the wh-words are affected. Subject-auxiliary inversion ("Is the cache
enabled") is a genuine question marker without a `?` and must keep being
rejected, so the two classes are separated rather than the check weakened.
"""
from __future__ import annotations

import pytest

from entroly.witness import (
    _is_claim_like,
    _is_question_like,
    extract_claims,
)

_FOLLOWER = "The entailment threshold is 0.60."

#: Declaratives that open with a wh-word. Each is a factual assertion a
#: verifier must check, not a question.
WH_DECLARATIVES = (
    "When the lock times out, the append is unserialized.",
    "Where the path is absolute, resolution is skipped.",
    "How this differs is in the token budget.",
    "What the receipt omits is listed under omitted_context.",
    "Which engine serves is decided by native_status.",
    "Why this matters is that the record is unverifiable.",
)

#: Real questions, which must still be rejected.
QUESTIONS = (
    "When is the cache cold?",
    "Where does the receipt store omissions?",
    "What is the entailment threshold?",
    "Is the cache enabled by default",      # inversion, no '?'
    "Does the proxy inject context",        # inversion, no '?'
    "Can the engine run without Rust",      # inversion, no '?'
)


@pytest.mark.parametrize("sentence", WH_DECLARATIVES)
def test_a_wh_declarative_is_not_treated_as_a_question(sentence):
    assert not _is_question_like(sentence), (
        "a declarative opening with a wh-word was classified as a question, so "
        "it never reaches claim extraction"
    )


@pytest.mark.parametrize("sentence", WH_DECLARATIVES)
def test_a_wh_declarative_is_claim_like(sentence):
    assert _is_claim_like(sentence)


@pytest.mark.parametrize("sentence", QUESTIONS)
def test_a_real_question_is_still_rejected(sentence):
    """Without this the fix would let questions through as claims."""
    assert _is_question_like(sentence), (
        "a genuine question stopped being recognised; subject-auxiliary "
        "inversion and a trailing '?' must both still mark a question"
    )


@pytest.mark.parametrize("sentence", WH_DECLARATIVES)
def test_a_wh_declarative_survives_a_following_sentence(sentence):
    """The case the dialogue fallback was hiding.

    Alone, the fallback manufactures a claim and the loss is invisible. With an
    ordinary sentence after it, the fallback does not fire and the claim has to
    stand on its own.
    """
    claims = extract_claims(sentence + " " + _FOLLOWER, force_python=True)

    texts = " ".join(c.text for c in claims)
    anchor = sentence.split(",")[0].split()[-1].rstrip(".")
    assert anchor in texts, (
        f"the wh-declarative was dropped when followed by another sentence; "
        f"extracted {[c.text for c in claims]}"
    )


def test_both_sentences_are_extracted_not_just_the_follower():
    claims = extract_claims(WH_DECLARATIVES[0] + " " + _FOLLOWER, force_python=True)

    assert len(claims) >= 2, (
        f"expected a claim for each sentence, got {[c.text for c in claims]}"
    )


def test_a_question_followed_by_a_claim_still_yields_only_the_claim():
    """The complement: fixing declaratives must not admit interrogatives."""
    claims = extract_claims("When is the cache cold? " + _FOLLOWER, force_python=True)

    assert all("cold" not in c.text for c in claims), (
        f"a question was extracted as a claim: {[c.text for c in claims]}"
    )
