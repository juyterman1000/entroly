"""Regression tests: autotuned `tuning_config.json` must reach the engine.

Guards the bug where the runtime loader read flat keys (`weight_recency`)
while `entroly autotune` writes the nested schema (`weights.recency`), so
autotuned values were silently dropped and the engine always used defaults.
"""
from __future__ import annotations

from entroly.config import EntrolyConfig, resolve_tuning_kwargs

_DEFAULTS = EntrolyConfig()


def test_nested_autotune_schema_reaches_config():
    """The exact bug: a nested autotuned config must change the engine config."""
    cfg = {
        "weights": {"recency": 0.50, "frequency": 0.20, "semantic_sim": 0.20, "entropy": 0.10},
        "decay": {"half_life_turns": 30, "min_relevance_threshold": 0.10},
        # extra autotune sections that the runtime engine doesn't consume:
        "knapsack": {"exploration_rate": 0.2}, "ios": {"skeleton_info_factor": 0.6},
    }
    kw = resolve_tuning_kwargs(cfg)
    assert kw["weight_recency"] == 0.50
    assert kw["weight_entropy"] == 0.10
    assert kw["decay_half_life_turns"] == 30
    assert kw["min_relevance_threshold"] == 0.10

    conf = EntrolyConfig(**kw)
    assert conf.weight_recency == 0.50          # NOT the 0.30 default
    assert conf.decay_half_life_turns == 30     # NOT the 15 default


def test_flat_keys_still_work_back_compat():
    cfg = {"weight_recency": 0.42, "decay_half_life_turns": 25}
    kw = resolve_tuning_kwargs(cfg)
    assert kw["weight_recency"] == 0.42
    assert kw["decay_half_life_turns"] == 25


def test_nested_wins_over_flat_when_both_present():
    cfg = {"weights": {"recency": 0.55}, "weight_recency": 0.11}
    assert resolve_tuning_kwargs(cfg)["weight_recency"] == 0.55


def test_missing_or_empty_falls_back_to_defaults():
    for cfg in ({}, {"weights": {}}, None, "not-a-dict", []):
        kw = resolve_tuning_kwargs(cfg)
        assert kw["weight_recency"] == _DEFAULTS.weight_recency
        assert kw["decay_half_life_turns"] == _DEFAULTS.decay_half_life_turns


def test_malformed_and_out_of_range_fall_back_without_raising():
    cfg = {
        "weights": {"recency": "oops", "frequency": 5.0, "entropy": -1},
        "decay": {"half_life_turns": -5, "min_relevance_threshold": 2.0},
    }
    kw = resolve_tuning_kwargs(cfg)  # must not raise
    assert kw["weight_recency"] == _DEFAULTS.weight_recency       # non-numeric
    assert kw["weight_frequency"] == _DEFAULTS.weight_frequency   # > 1.0
    assert kw["weight_entropy"] == _DEFAULTS.weight_entropy       # < 0
    assert kw["decay_half_life_turns"] == _DEFAULTS.decay_half_life_turns  # < 1
    assert kw["min_relevance_threshold"] == _DEFAULTS.min_relevance_threshold  # > 1.0


def test_zero_is_a_valid_value_not_treated_as_missing():
    """0.0 is falsy — the mapper must distinguish 'absent' from 'zero'."""
    cfg = {"weights": {"entropy": 0.0}, "decay": {"min_relevance_threshold": 0.0}}
    kw = resolve_tuning_kwargs(cfg)
    assert kw["weight_entropy"] == 0.0
    assert kw["min_relevance_threshold"] == 0.0


def test_returns_only_known_config_fields():
    """Every returned key must be a real EntrolyConfig field (no TypeError on **)."""
    kw = resolve_tuning_kwargs({"weights": {"recency": 0.3}})
    EntrolyConfig(**kw)  # must construct cleanly
    valid = set(EntrolyConfig.__dataclass_fields__)
    assert set(kw).issubset(valid)
