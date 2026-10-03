"""The receipt scorer and the engine must agree on what a function word is.

Two stopword lists existed. The Rust query path filtered 77 words; the receipt
chunk scorer filtered 17, so `not`, `is`, `how`, `are` and 57 others were
content terms to the surface that produces the auditable receipt.

Measured consequence on budget-constrained selection, which is what
`create_context_receipt` performs. Seven gold query/file pairs lifted from
`benchmarks/evidence_retention.py`, scored over a 19-file corpus:

    budget   precision before   after
       600              0.545   0.574
      1500              0.500   0.537
      4000              0.454   0.488

More gold chunks selected (24->27, 41->43, 79->84) out of the same or fewer
total, at every budget, in the same direction.

Ranking alone does not show this and that is the point: the same seven queries
scored 0.9048 MRR before and 0.8929 after, slightly *worse*. Spurious matches on
function words score too low to displace rank 1 -- but they consume budget once
the real answers are exhausted, which is the mechanism. Measuring the wrong
surface would have falsified a change that works.

`all` and `any` are deliberately excluded from the import. They are Python
builtins and legitimate query terms -- "where do we use any() vs all()" must
keep them. Excluding them costs nothing measurable: precision is identical to
importing the full list at all three budgets.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from entroly.context_receipts.retrieval import STOPWORDS, tokenize

RUST_QUERY_SOURCE = Path(__file__).resolve().parent.parent / "entroly-engine" / "src" / "query.rs"

#: Python builtins that are real identifiers in a code query. The engine filters
#: them because it refines natural-language queries; the receipt scorer ranks
#: source chunks, where they carry signal.
CODE_IDENTIFIERS = {"all", "any"}


def _rust_stopwords() -> set[str]:
    text = RUST_QUERY_SOURCE.read_text(encoding="utf-8")
    block = re.search(r"static STOP_WORDS: &\[&str\] = &\[(.*?)\];", text, re.S)
    assert block, "STOP_WORDS not found in entroly-engine/src/query.rs"
    return set(re.findall(r'"([a-z]+)"', block.group(1)))


# ── Parity, which is the part that stops the drift recurring ─────────


def test_the_receipt_scorer_filters_what_the_engine_filters():
    """Read from the Rust source, so adding a word there cannot leave this behind.

    The two lists drifted to 17 against 77 because nothing compared them. A
    hardcoded copy here would drift the same way.
    """
    expected = _rust_stopwords() - CODE_IDENTIFIERS

    missing = sorted(expected - set(STOPWORDS))

    assert not missing, (
        "the engine filters these but the receipt scorer does not, so they act "
        f"as content terms on the auditable surface: {missing}"
    )


@pytest.mark.parametrize("word", sorted(CODE_IDENTIFIERS))
def test_python_builtins_are_not_filtered(word):
    """Guards the exclusion. Without this the parity test invites a blanket copy."""
    assert word not in STOPWORDS, (
        f"{word!r} is a Python builtin and a legitimate query term; filtering it "
        'breaks a query like "where do we use any() vs all()"'
    )

    assert tokenize(f"where do we use {word}() in the selector") == [word, "selector"] or (
        word in tokenize(f"where do we use {word}() in the selector")
    )


# ── The defect the parity closes ─────────────────────────────────────


def test_a_function_word_is_not_a_content_term():
    """`is not None` made an unrelated chunk match a query about an NLI scorer."""
    for word in ("not", "is", "in", "it", "be", "to"):
        assert word not in tokenize(f"the value {word} present"), (
            f"{word!r} survived tokenization and can match any chunk containing it"
        )


def test_a_real_query_keeps_its_content_terms():
    """Over-filtering is the opposite failure and would be just as bad."""
    terms = tokenize("where is the global checkpoint cap enforced")

    assert "checkpoint" in terms
    assert "enforced" in terms
    assert "global" in terms


def test_an_all_stopword_query_still_produces_terms_downstream():
    """Degenerate queries must not leave the ranker with nothing.

    `rank_chunks` falls back to unfiltered tokens when `tokenize` returns empty,
    so a query made entirely of function words still scores. This pins that the
    widened list does not defeat that fallback.
    """
    from entroly.context_receipts.retrieval import TOKEN_RE

    query = "is it that we are not in there"

    assert tokenize(query) == [], "fixture is no longer an all-stopword query"
    assert TOKEN_RE.findall(query), "the raw-token fallback has nothing to fall back to"
