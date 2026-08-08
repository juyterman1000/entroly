"""Precomputed corpus tokens must not change an IDF value.

`_attach_sufficiency` tokenises every candidate file to compute its anchors,
then called `_idf_map`, which tokenised the identical corpus a second time.
Certificate construction measured ~47% of a selection call and half of that was
the duplicate pass. The optimisation hands the already-computed sets in.

That is only safe if the values are identical, which is what this asserts. A
faster certificate that reports different numbers is not an optimisation.
"""

from __future__ import annotations

import pytest

from entroly.sufficiency import _idf_map, _lexical_terms

CORPUS = [
    "def compute_budget(tokens, limit):\n    return min(tokens, limit)",
    "class BudgetError(Exception):\n    '''Raised when the budget is exhausted.'''",
    "SETTING = 42\n# unrelated module with no budget wording at all",
    "",
]
TERMS = ["budget", "tokens", "exhausted", "absent"]


def test_precomputed_terms_give_identical_idf() -> None:
    recomputed = _idf_map(TERMS, CORPUS)
    reused = _idf_map(TERMS, CORPUS, [_lexical_terms(text) for text in CORPUS])
    assert recomputed == reused


def test_misaligned_precomputed_terms_fail_visibly() -> None:
    """Silently scoring against the wrong documents would corrupt the verdict."""
    with pytest.raises(ValueError, match="align 1:1"):
        _idf_map(TERMS, CORPUS, [_lexical_terms(CORPUS[0])])


def test_empty_corpus_still_returns_a_value_per_term() -> None:
    assert _idf_map(TERMS, [], []) == {term: 1.0 for term in TERMS}
