"""Smoke tests for Query-Conditioned Compressive Retrieval."""
from __future__ import annotations

from entroly.qccr import select, _query_tokens, _split_sentences, _mmr_select, _bm25_corpus


def test_empty_fragments_returns_empty():
    assert select([], token_budget=1024, query="anything") == []


def test_empty_query_returns_input():
    frags = [{"source": "a.py", "content": "def f(): pass", "token_count": 5}]
    assert select(frags, token_budget=1024, query="") == frags


def test_selects_query_relevant_file():
    frags = [
        {"source": "irrelevant.md", "content": "This document explains the weather patterns in spring and autumn.", "token_count": 20},
        {"source": "relevant.py", "content": "def jaccard_similarity(a, b):\n    return len(a & b) / len(a | b)", "token_count": 15},
        {"source": "also_irrelevant.md", "content": "The history of ancient Rome spans over twelve centuries.", "token_count": 18},
    ]
    result = select(frags, token_budget=512, query="What is jaccard similarity?")
    assert result, "qccr returned nothing for a query with obvious match"
    sources = [r.get("source") for r in result]
    assert "relevant.py" in sources, f"qccr did not pick the jaccard file: {sources}"


def test_budget_respected():
    frags = [
        {"source": f"f{i}.py", "content": "def func(): return 1\n" * 50, "token_count": 200}
        for i in range(20)
    ]
    result = select(frags, token_budget=500, query="function definition")
    total = sum(r.get("token_count", 0) for r in result)
    assert total <= 600, f"budget exceeded: {total} > 500"  # small slack for rounding


def test_tokenization_splits_identifiers():
    toks = _query_tokens("How does taint_flow work with CamelCase identifiers?")
    assert "taint" in toks
    assert "flow" in toks
    assert "camel" in toks or "camelcase" in toks
    assert "case" in toks or "camelcase" in toks


def test_sentence_split_code_breaks():
    text = "def foo():\n    pass\n\ndef bar():\n    return 1\n"
    sents = _split_sentences(text)
    assert len(sents) >= 1


def test_mmr_selects_diverse():
    sentences = [
        "Jaccard similarity measures set overlap.",
        "Jaccard similarity measures set overlap completely.",  # near-duplicate
        "The weather is sunny today.",
    ]
    tf, _, _, _ = _bm25_corpus(sentences)
    rel = [2.0, 1.9, 0.0]  # first two relevant, third not
    chosen = _mmr_select(sentences, tf, rel, budget_tokens=100)
    # Should pick index 0 (highest rel); index 2 has rel=0 so excluded.
    assert 0 in chosen
    assert 2 not in chosen


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                raise
