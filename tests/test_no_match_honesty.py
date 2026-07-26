"""Selection must not fabricate relevance, and savings must not count withholding.

Dogfooding found that a query matching nothing still produced a ranked list:
score normalization floors and uniform fallbacks give every fragment a middling
score, so `optimize_context` returned confident-looking unrelated files and then
billed the omitted corpus as "tokens saved". Measured on this repository, the
nonsense query "zzqqxx blorptastic wubbleflux" selected 8 files at ~0.6
relevance and shared its top hits with a real query, while `tokens_saved` stayed
~2.65M regardless of query — it was simply `corpus - selected`.

The contract these tests pin:
  * unrelated context is worse than none -> report `no_match`
  * withholding context because nothing matched is not a saving -> credit zero
  * pinned/required evidence is operator policy, not relevance -> always kept
"""

from __future__ import annotations

from entroly.server import _evidence_backed, _honest_tokens_saved


def test_positive_relevance_is_evidence_backed():
    assert _evidence_backed([{"source": "a.py", "relevance": 0.62}]) is True
    assert _evidence_backed([{"source": "a.py", "relevance_score": 1.4}]) is True


def test_zero_relevance_is_not_evidence_backed():
    assert _evidence_backed([{"source": "a.py", "relevance": 0.0}]) is False
    assert _evidence_backed([{"source": "a.py", "relevance_score": 0}]) is False
    assert _evidence_backed([]) is False


def test_pinned_fragments_alone_are_not_evidence():
    # Pinned content is included by operator policy, not because it matched.
    # Counting it would make every no-match look matched.
    pinned_only = [{"source": "always.md", "is_pinned": True, "relevance": 0.9}]
    assert _evidence_backed(pinned_only) is False
    mixed = pinned_only + [{"source": "hit.py", "relevance": 0.3}]
    assert _evidence_backed(mixed) is True


def test_malformed_relevance_never_counts_as_evidence():
    for bad in (None, "", "not-a-number", [], {}):
        assert _evidence_backed([{"source": "a.py", "relevance": bad}]) is False


def test_savings_are_zero_when_nothing_matched():
    # The core accounting defect: the WORSE the match, the LARGER the reported
    # saving, because `corpus - selected` grows as selection shrinks.
    assert _honest_tokens_saved([{"relevance": 0.0}], 2_654_766) == 0
    assert _honest_tokens_saved([], 2_654_766) == 0
    assert _honest_tokens_saved(
        [{"is_pinned": True, "relevance": 1.0}], 2_654_766
    ) == 0


def test_savings_are_credited_when_evidence_was_found():
    assert _honest_tokens_saved([{"relevance": 0.62}], 5000) == 5000


def test_savings_are_never_negative():
    assert _honest_tokens_saved([{"relevance": 0.62}], -17) == 0
    assert _honest_tokens_saved([{"relevance": 0.62}], None) == 0


# ── Lexical verification (independent of the ranker's scores) ────────────────
# Relevance scores cannot detect no-match: the native ranker floors and
# normalizes, so a query matching nothing returns ~0.6. Checking the delivered
# text is the one signal the scoring pipeline cannot fabricate.

def test_selection_matching_requires_a_query_term_in_the_delivered_text():
    from entroly.server import _selection_matches_query

    hit = [{"source": "entroly/proxy.py", "content": "def inject_context(): ..."}]
    assert _selection_matches_query("inject context proxy", hit) is True
    # High score, zero lexical overlap -> the fabricated-relevance case.
    miss = [{"source": "docs/readme.md", "content": "unrelated prose",
             "relevance": 0.6}]
    assert _selection_matches_query("zzqqxx blorptastic wubbleflux", miss) is False


def test_pinned_content_cannot_satisfy_the_query_match():
    from entroly.server import _selection_matches_query

    pinned = [{"source": "always.md", "content": "inject context proxy",
               "is_pinned": True}]
    assert _selection_matches_query("inject context proxy", pinned) is False


def test_stopword_only_query_is_not_treated_as_a_no_match():
    from entroly.server import _selection_matches_query

    # No lexical intent to satisfy; must not flip everything to no_match.
    assert _selection_matches_query("what is the", [{"content": "anything"}]) is True
    assert _selection_matches_query("", [{"content": "anything"}]) is True


def test_source_path_counts_as_evidence():
    from entroly.server import _selection_matches_query

    assert _selection_matches_query(
        "checkpoint", [{"source": "entroly/checkpoint.py", "content": ""}]
    ) is True
