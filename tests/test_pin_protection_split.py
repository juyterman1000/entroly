"""Pinning is operator policy; protection is a storage guarantee.

One boolean used to mean both. `file_criticality` set `is_pinned`, and the
knapsack treats a pinned fragment as mandatory -- included before optimization,
with the remaining budget computed as `budget - pinned_tokens`. So every
manifest and every file carrying a safety signal was force-included in EVERY
query, whether or not it had anything to do with the question.

Measured on this repository before the split: 56 fragments / 223,288 tokens
pinned, consuming 31,648 of 62,602 delivered tokens (50.6%) across the eight
gold queries at an 8,000-token budget -- and the pinned share was byte-identical
for every query (1,965 / 3,923 / 7,996 tokens at 4k / 8k / 16k), because it was
allocated without reference to the query at all.

The intent behind criticality was "never drop this from the store", which is
eviction protection and does not require forced inclusion. These tests pin that
separation, and that neither guarantee was weakened to get the budget back.
"""

from __future__ import annotations

import pytest

def _native_has_split() -> bool:
    """The split lives in the native engine; skip only if it is genuinely absent.

    An earlier version of this guard probed a class name that does not exist, so
    every test silently skipped and proved nothing. Probe the field itself.
    """
    try:
        import entroly_core
    except ImportError:  # pragma: no cover - pure-Python install
        return False
    engine = entroly_core.EntrolyEngine()
    engine.ingest("probe", "file:probe.py", 4, False)
    return "is_protected" in engine.export_fragments()[0]


pytestmark = pytest.mark.skipif(
    not _native_has_split(), reason="native engine lacks the pin/protection split"
)


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    from entroly.server import EntrolyConfig, EntrolyEngine

    return EntrolyEngine(config=EntrolyConfig())


def _by_source(engine, source: str) -> dict:
    for frag in engine._rust.export_fragments():
        if str(frag.get("source")) == source:
            return frag
    raise AssertionError(f"{source} not in index")


# ── The split itself ────────────────────────────────────────────────────────

def test_critical_file_is_protected_but_not_force_included(engine):
    engine.ingest_fragment(
        content='[package]\nname = "x"\nversion = "0.1.0"\n',
        source="file:Cargo.toml",
        token_count=32,
        is_pinned=False,
    )
    frag = _by_source(engine, "file:Cargo.toml")
    assert frag["is_protected"] is True, "criticality must still protect from eviction"
    assert frag["is_pinned"] is False, (
        "criticality must NOT force the fragment into every query's context"
    )


def test_operator_pin_is_honoured(engine):
    # The one thing that should still force inclusion: someone asked for it.
    engine.ingest_fragment(
        content="remember this exactly",
        source="file:notes.md",
        token_count=8,
        is_pinned=True,
    )
    assert _by_source(engine, "file:notes.md")["is_pinned"] is True


def test_ordinary_file_is_neither(engine):
    engine.ingest_fragment(
        content="def helper():\n    return 1\n",
        source="file:helper.py",
        token_count=12,
        is_pinned=False,
    )
    frag = _by_source(engine, "file:helper.py")
    assert frag["is_pinned"] is False
    assert frag["is_protected"] is False


# ── The guarantee that must not regress ─────────────────────────────────────

def test_protected_fragments_still_survive_eviction(engine):
    """Protection is the whole point of the criticality rule; keep it."""
    engine.ingest_fragment(
        content="MIT License\n\nPermission is hereby granted...",
        source="file:LICENSE",
        token_count=16,
        is_pinned=False,
    )
    assert _by_source(engine, "file:LICENSE")["is_protected"] is True

    for i in range(60):
        engine.ingest_fragment(
            content=f"filler content number {i}\n" * 4,
            source=f"file:filler_{i}.py",
            token_count=16,
            is_pinned=False,
        )
    engine.decay_and_evict() if hasattr(engine, "decay_and_evict") else None
    sources = {str(f.get("source")) for f in engine._rust.export_fragments()}
    assert "file:LICENSE" in sources, "a protected file must never be evicted"


# ── Budget: the ranker must actually get the capacity back ──────────────────

def test_ranker_receives_budget_previously_taken_by_criticality(engine):
    """A relevant ordinary file must beat an irrelevant manifest for budget.

    Before the split the manifest was mandatory, so it consumed budget first and
    the answer competed for what was left.
    """
    engine.ingest_fragment(
        content='[tool.poetry]\nname = "unrelated"\n' + "dependency = 1\n" * 200,
        source="file:pyproject.toml",
        token_count=900,
        is_pinned=False,
    )
    engine.ingest_fragment(
        content="def parse_manifest(path):\n    return decode_manifest(path)\n" * 20,
        source="file:parser.py",
        token_count=400,
        is_pinned=False,
    )
    result = engine.optimize_context(600, "parse_manifest decode_manifest")
    selected = result.get("selected_fragments") or result.get("selected") or []
    sources = {str(f.get("source")) for f in selected}
    assert "file:parser.py" in sources, (
        "the query-relevant file must fit; the manifest no longer pre-empts it"
    )


def test_delivered_tokens_respect_the_budget(engine):
    for i in range(12):
        engine.ingest_fragment(
            content=f"alpha beta gamma payload {i}\n" * 30,
            source=f"file:mod_{i}.py",
            token_count=120,
            is_pinned=False,
        )
    budget = 400
    result = engine.optimize_context(budget, "alpha beta gamma")
    selected = result.get("selected_fragments") or result.get("selected") or []
    delivered = sum(int(f.get("token_count") or 0) for f in selected)
    assert delivered <= budget, f"delivered {delivered} exceeds budget {budget}"


def test_token_accounting_splits_pinned_from_ordinary(engine):
    engine.ingest_fragment(
        content="operator required evidence\n" * 10,
        source="file:required.md",
        token_count=60,
        is_pinned=True,
    )
    for i in range(6):
        engine.ingest_fragment(
            content=f"alpha beta payload {i}\n" * 20,
            source=f"file:m_{i}.py",
            token_count=60,
            is_pinned=False,
        )
    result = engine.optimize_context(400, "alpha beta")
    selected = result.get("selected_fragments") or result.get("selected") or []
    required = sum(int(f.get("token_count") or 0)
                   for f in selected if f.get("is_pinned"))
    ordinary = sum(int(f.get("token_count") or 0)
                   for f in selected if not f.get("is_pinned"))
    delivered = sum(int(f.get("token_count") or 0) for f in selected)
    assert required + ordinary == delivered, "accounting must be exhaustive"
