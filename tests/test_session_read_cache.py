"""A re-read must be free, and a changed file must never be suppressed.

The second property is the one that matters. A cache that returns a reference
for content the agent has not actually seen, or for a file that has since
changed, is worse than no cache: it makes the agent confident about bytes that
no longer exist.
"""

from __future__ import annotations

from entroly.session_read_cache import SessionReadCache

BIG = "\n".join(f"def function_{i}(alpha, beta):\n    return alpha + beta" for i in range(200))


def test_first_delivery_is_verbatim() -> None:
    cache = SessionReadCache()
    decision = cache.deliver("a.py", BIG)
    assert not decision.suppressed
    assert decision.text == BIG
    assert decision.tokens_saved == 0


def test_unchanged_re_read_on_a_later_turn_is_suppressed() -> None:
    cache = SessionReadCache()
    cache.deliver("a.py", BIG)
    cache.advance_turn()
    decision = cache.deliver("a.py", BIG)

    assert decision.suppressed
    assert decision.delivered_tokens < decision.original_tokens / 20
    assert decision.tokens_saved > 0


def test_changed_content_is_never_suppressed() -> None:
    """The load-bearing safety property."""
    cache = SessionReadCache()
    cache.deliver("a.py", BIG)
    cache.advance_turn()

    changed = BIG + "\ndef newly_added(gamma):\n    return gamma"
    decision = cache.deliver("a.py", changed)

    assert not decision.suppressed
    assert decision.text == changed, "a modified file must be delivered in full"


def test_reverting_a_file_restores_suppression() -> None:
    """Digest keying, not path keying: content identity is what counts."""
    cache = SessionReadCache()
    cache.deliver("a.py", BIG)
    cache.advance_turn()
    cache.deliver("a.py", BIG + "\n# edit")
    cache.advance_turn()
    reverted = cache.deliver("a.py", BIG + "\n# edit")
    assert reverted.suppressed


def test_same_turn_re_delivery_is_not_suppressed() -> None:
    """Within one turn the agent may not have seen the first copy yet."""
    cache = SessionReadCache()
    cache.deliver("a.py", BIG)
    decision = cache.deliver("a.py", BIG)
    assert not decision.suppressed


def test_small_content_is_not_worth_a_reference() -> None:
    cache = SessionReadCache()
    tiny = "x = 1\n"
    cache.deliver("t.py", tiny)
    cache.advance_turn()
    decision = cache.deliver("t.py", tiny)
    assert not decision.suppressed, "a reference must not cost more than the content"


def test_reference_is_actionable_not_a_bare_placeholder() -> None:
    """An agent must be able to tell what was withheld and how to get it."""
    cache = SessionReadCache()
    cache.deliver("pkg/mod.py", BIG)
    cache.advance_turn()
    text = cache.deliver("pkg/mod.py", BIG).text

    assert "pkg/mod.py" in text
    assert "L" in text, "must state the size it stands for"
    assert "#" in text, "must carry a digest so the agent can verify"

    # The recovery instructions are identical for every entry, so they are
    # emitted once per turn instead of per line. Repeating them cost ~46
    # tokens per file against ~6 for a comparable external system.
    assert "re-read" in SessionReadCache.PREAMBLE


def test_eviction_is_bounded() -> None:
    cache = SessionReadCache(max_entries=8)
    for i in range(50):
        cache.deliver(f"f{i}.py", BIG)
    assert len(cache._entries) <= 8


def test_stats_label_the_baseline() -> None:
    cache = SessionReadCache()
    cache.deliver("a.py", BIG)
    cache.advance_turn()
    cache.deliver("a.py", BIG)
    stats = cache.stats()
    assert stats["suppressed_deliveries"] == 1
    assert stats["tokens_saved"] > 0
    assert "not a provider bill delta" in stats["baseline"]
