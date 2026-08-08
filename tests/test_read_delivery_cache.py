from __future__ import annotations

from entroly.read_delivery_cache import ReadDeliveryCache


CONTRACT = {
    "path": "pkg/mod.py",
    "query": "",
    "budget": 1000,
    "resolution": "full",
    "previous_source_sha256": "",
    "line_start": None,
    "line_end": None,
}
SOURCE = "\n".join(f"def function_{i}():\n    return {i}" for i in range(200))


def _deliver(
    cache: ReadDeliveryCache,
    *,
    session: str = "session-a",
    contract: dict[str, object] | None = None,
    source: str = SOURCE,
    output: str = SOURCE,
    fresh: bool = False,
):
    return cache.deliver(
        session_id=session,
        path="pkg/mod.py",
        mode=str((contract or CONTRACT)["resolution"]),
        contract=contract or CONTRACT,
        source=source,
        output=output,
        fresh=fresh,
    )


def test_exact_repeat_in_the_same_session_is_suppressed() -> None:
    cache = ReadDeliveryCache()
    first = _deliver(cache)
    repeated = _deliver(cache)

    assert not first.cache_hit
    assert repeated.cache_hit
    assert repeated.output_sha256 == first.output_sha256
    assert repeated.delivered_tokens < first.delivered_tokens / 20


def test_sessions_are_isolated() -> None:
    cache = ReadDeliveryCache()
    _deliver(cache, session="primary")
    subagent = _deliver(cache, session="subagent")

    assert not subagent.cache_hit
    assert subagent.text == SOURCE


def test_mode_query_budget_and_line_range_are_part_of_the_contract() -> None:
    cache = ReadDeliveryCache()
    _deliver(cache)

    variants = [
        CONTRACT | {"resolution": "low"},
        CONTRACT | {"query": "different"},
        CONTRACT | {"budget": 200},
        CONTRACT | {"line_start": 1, "line_end": 10},
    ]
    for variant in variants:
        assert not _deliver(cache, contract=variant).cache_hit


def test_changed_source_is_never_suppressed_even_when_output_matches() -> None:
    cache = ReadDeliveryCache()
    _deliver(cache, output="same rendered selection")
    changed = _deliver(
        cache,
        source=SOURCE + "\n# changed outside the selected range",
        output="same rendered selection",
    )

    assert not changed.cache_hit


def test_changed_rendered_output_is_never_suppressed() -> None:
    cache = ReadDeliveryCache()
    _deliver(cache)
    changed = _deliver(cache, output=SOURCE + "\n# renderer changed")

    assert not changed.cache_hit


def test_fresh_bypasses_a_valid_hit_and_refreshes_the_entry() -> None:
    cache = ReadDeliveryCache()
    _deliver(cache)
    fresh = _deliver(cache, fresh=True)
    next_read = _deliver(cache)

    assert not fresh.cache_hit
    assert fresh.text == SOURCE
    assert next_read.cache_hit


def test_tiny_output_passes_through_when_the_reference_would_cost_more() -> None:
    cache = ReadDeliveryCache()
    _deliver(cache, source="x", output="x")
    repeated = _deliver(cache, source="x", output="x")

    assert not repeated.cache_hit
    assert repeated.text == "x"


def test_cache_is_bounded_by_session_and_entry() -> None:
    cache = ReadDeliveryCache(max_sessions=2, max_entries_per_session=2)
    for session in ("one", "two", "three"):
        for index in range(3):
            _deliver(
                cache,
                session=session,
                contract=CONTRACT | {"query": str(index)},
            )

    stats = cache.stats()
    assert stats["sessions"] == 2
    assert stats["entries"] == 4
    assert "not a provider bill delta" in stats["baseline"]
