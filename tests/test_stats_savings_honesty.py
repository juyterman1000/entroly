"""MCP stats must not publish a fabricated dollar-savings figure.

The native core emits `estimated_cost_saved_usd` derived from
`total_tokens_saved`, which accumulates (all candidates - selected) on every
call. A live session reported 6.07M tokens "saved" against 2.58M tokens tracked,
and one 3,000-token request claimed 2.5M tokens saved. `optimize_context`
already documents that MCP results "never ... support a dollar-savings claim",
and the dashboard already strips the field; the MCP stats surface did not.
"""

from __future__ import annotations

from entroly.server import EntrolyEngine

_honest = EntrolyEngine._honest_savings_block


def test_fabricated_dollar_figure_is_removed():
    out = _honest({
        "total_tokens_saved": 6071075,
        "total_duplicates_caught": 402,
        "total_optimizations": 1,
        "estimated_cost_saved_usd": 18.2132,
    })
    assert "estimated_cost_saved_usd" not in out, (
        "MCP stats must not publish a dollar figure the code itself disclaims"
    )


def test_token_counter_is_renamed_to_say_what_it_measures():
    out = _honest({"total_tokens_saved": 6071075})
    assert out["dedup_tokens_avoided"] == 6071075
    assert "total_tokens_saved" not in out, (
        "the old name reads as LLM savings; it is dedup/selection telemetry"
    )


def test_baseline_is_stated_explicitly():
    out = _honest({"total_tokens_saved": 1})
    assert "baseline" in out
    assert "not a dollar-savings claim" in out["baseline"].lower()


def test_other_telemetry_is_preserved():
    out = _honest({
        "total_tokens_saved": 10,
        "total_duplicates_caught": 402,
        "total_optimizations": 7,
        "total_fragments_ingested": 1387,
    })
    assert out["total_duplicates_caught"] == 402
    assert out["total_optimizations"] == 7
    assert out["total_fragments_ingested"] == 1387


def test_missing_or_malformed_block_is_safe():
    assert _honest(None) == {}
    assert _honest({}) == {"baseline": _honest({})["baseline"]}
    assert _honest("unexpected") == "unexpected"  # pass through, never crash


def test_get_stats_does_not_leak_the_dollar_figure():
    class _Rust:
        @staticmethod
        def stats():
            return {"savings": {"total_tokens_saved": 5, "estimated_cost_saved_usd": 1.23}}

        @staticmethod
        def dep_graph_stats():
            return {"nodes": 0, "edges": 0}

    class _Prefetch:
        @staticmethod
        def stats():
            return {}

    class _Ckpt:
        @staticmethod
        def stats():
            return {}

    class _FakeEngine:
        _use_rust = True
        _rust = _Rust()
        _prefetch = _Prefetch()
        _checkpoint_mgr = _Ckpt()
        _honest_savings_block = staticmethod(_honest)
        _build_stamp = EntrolyEngine._build_stamp

    stats = EntrolyEngine.get_stats(_FakeEngine())  # type: ignore[arg-type]
    assert "estimated_cost_saved_usd" not in stats["savings"]
    assert stats["savings"]["dedup_tokens_avoided"] == 5


def test_stats_resource_counters_work_on_the_native_shape():
    # The native core emits a `savings` block with total_-prefixed names; the
    # pure-Python path emits `engine`. Reading only `engine` reported 0 for
    # every counter on native installs, i.e. every real deployment.
    from entroly.server import EntrolyEngine

    native = EntrolyEngine._honest_savings_block({
        "total_fragments_ingested": 1387,
        "total_duplicates_caught": 402,
        "total_optimizations": 7,
        "total_tokens_saved": 5,
        "estimated_cost_saved_usd": 1.23,
    })
    # After normalization the native block still uses total_-prefixed names for
    # everything except the renamed token counter.
    assert native["total_fragments_ingested"] == 1387
    assert native["dedup_tokens_avoided"] == 5
    assert "estimated_cost_saved_usd" not in native


def test_token_counter_is_always_an_int_never_none():
    # The raw value used to pass through verbatim, so a None or non-numeric
    # counter reached consumers and any arithmetic on it raised TypeError.
    from entroly.server import EntrolyEngine

    for raw in (None, "", "not-a-number", [], 7):
        out = EntrolyEngine._honest_savings_block({"total_tokens_saved": raw})
        value = out["dedup_tokens_avoided"]
        assert isinstance(value, int), f"{raw!r} -> {value!r} is not an int"
        assert value + 1 > 0 or value == 0   # arithmetic must not raise
    assert EntrolyEngine._honest_savings_block(
        {"total_tokens_saved": 7})["dedup_tokens_avoided"] == 7
