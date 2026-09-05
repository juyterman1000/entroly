"""The injected context must be byte-identical for identical input.

CLAUDE.md states the invariant: "Cache stability: prompt prefixes should remain
byte-stable unless intentionally changed." Provider KV caches match an *exact*
prefix, and ``CacheAligner`` only reuses a block when its SHA-256 is identical,
so any instability silently costs the discount -- the request still succeeds, it
just never hits cache. A non-reproducible selection also cannot be re-derived
from a receipt.

``EntrolyEngine.export_fragments`` iterated a Rust ``HashMap`` unsorted, and
that list is what the default QCCR selector groups by "first-seen order". Score
ties were therefore broken in HashMap iteration order, which Rust randomizes per
map instance. Measured before the fix, with 12 fragments all scoring exactly
1.0: the same set was selected every run, in a different order every run, so the
bytes and the digest changed on every process start and between two engines in
the same process.

The sibling code path already guarded this -- ``lib.rs`` sorts its cogops
candidate vector with a comment naming the same hazard -- but the default path
did not.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUERY = "how does the payment refund processor validate a declined transaction"


def _selected(monkeypatch) -> tuple[str, list[str], int]:
    """Ingest a fixed corpus into a fresh engine and return (digest, sources, n).

    A fresh ENTROLY_DIR per engine is required: otherwise the engine warm-starts
    from a persisted index and the run measures the previous corpus.
    """
    monkeypatch.setenv("ENTROLY_DIR", tempfile.mkdtemp(prefix="bytestab_"))
    from entroly.server import EntrolyEngine

    engine = EntrolyEngine()

    # All twelve are near-identical so they score equally -- ties are exactly
    # the condition that exposed the ordering entropy. The trailing tag keeps
    # them distinct enough to survive SimHash dedup; without it the corpus
    # collapses and the test proves nothing.
    for i in range(12):
        engine.ingest_fragment(
            content=(
                "def validate_declined_transaction(payment, refund):\n"
                "    if payment.status == 'declined':\n"
                "        return refund.processor.validate(payment)\n"
                f"    # tag hot_{i}_{i * 37 % 91}\n"
            ),
            source=f"file:pkg/hot_{i:02d}.py",
            token_count=60,
            is_pinned=False,
        )

    result = engine.optimize_context(token_budget=600, query=QUERY)
    frags = result.get("selected_fragments") or result.get("selected") or []

    def field(f, name):
        return f.get(name, "") if isinstance(f, dict) else getattr(f, name, "")

    sources = [field(f, "source") for f in frags]
    blob = "\n".join(field(f, "content") for f in frags)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), sources, len(frags)


def test_two_engines_produce_byte_identical_context(monkeypatch):
    """Same corpus, same query, two engines -> identical bytes.

    Rust reseeds ``HashMap`` per map instance, so two engines in one process is
    enough to expose the entropy; it does not need separate processes.
    """
    digest_a, sources_a, n_a = _selected(monkeypatch)
    digest_b, sources_b, n_b = _selected(monkeypatch)

    # The fixture must actually have selected something under ties, or this
    # asserts nothing.
    assert n_a >= 2, f"only {n_a} fragment(s) selected; the fixture is not exercising ties"
    assert n_a == n_b

    assert sources_a == sources_b, (
        "two engines selected the same fragments in a different order:\n"
        f"  A: {[s.rsplit('/', 1)[-1] for s in sources_a]}\n"
        f"  B: {[s.rsplit('/', 1)[-1] for s in sources_b]}"
    )
    assert digest_a == digest_b, (
        f"identical input produced different context bytes: {digest_a} vs {digest_b}"
    )


def test_export_fragments_is_ordered(monkeypatch):
    """The ordering contract, asserted where it is established.

    ``export_fragments`` is the boundary the selector reads. Pinning the order
    here localises a regression to this function rather than leaving it to
    surface as a mysterious cache miss.
    """
    monkeypatch.setenv("ENTROLY_DIR", tempfile.mkdtemp(prefix="bytestab_ex_"))
    from entroly.server import EntrolyEngine

    engine = EntrolyEngine()
    if not getattr(engine, "_use_rust", False):
        pytest.skip("native engine unavailable; the Python path preserves dict order")

    for i in range(25):
        engine.ingest_fragment(
            content=f"def fn_{i}():\n    return {i * 17 % 83}\n",
            source=f"file:pkg/mod_{i:02d}.py",
            token_count=20,
            is_pinned=False,
        )

    ids = [dict(f)["fragment_id"] for f in engine._rust.export_fragments()]
    assert ids, "export_fragments returned nothing"
    assert ids == sorted(ids), (
        "export_fragments returned fragments in unsorted (HashMap) order; the "
        "default QCCR selector groups by first-seen order over this list, so "
        f"ties inherit the entropy. First few: {ids[:5]}"
    )
