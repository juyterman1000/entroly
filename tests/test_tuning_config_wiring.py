"""Regression tests: autotuned `tuning_config.json` must reach the engine.

Guards the bug where the runtime loader read flat keys (`weight_recency`)
while `entroly autotune` writes the nested schema (`weights.recency`), so
autotuned values were silently dropped and the engine always used defaults.
"""
from __future__ import annotations

import json

from entroly.config import EntrolyConfig, load_active_tuning_config, resolve_tuning_kwargs, writable_tuning_config_path

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



def test_non_finite_boolean_and_fractional_tuning_values_fall_back():
    cfg = {
        "weights": {
            "recency": float("nan"),
            "frequency": float("inf"),
            "semantic_sim": True,
        },
        "decay": {"half_life_turns": 3.5, "min_relevance_threshold": False},
        "dedup": {"hamming_threshold": float("inf")},
    }

    kw = resolve_tuning_kwargs(cfg)

    assert kw["weight_recency"] == _DEFAULTS.weight_recency
    assert kw["weight_frequency"] == _DEFAULTS.weight_frequency
    assert kw["weight_semantic_sim"] == _DEFAULTS.weight_semantic_sim
    assert kw["decay_half_life_turns"] == _DEFAULTS.decay_half_life_turns
    assert kw["min_relevance_threshold"] == _DEFAULTS.min_relevance_threshold
    assert kw["dedup_hamming_threshold"] == _DEFAULTS.dedup_hamming_threshold


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


def test_all_engine_dimensions_are_threaded():
    """Every dimension the runtime engine consumes must flow from a nested
    autotune config — not just the 4 weights + 2 decay params."""
    cfg = {
        "weights": {"recency": 0.4, "frequency": 0.3, "semantic_sim": 0.2, "entropy": 0.1},
        "decay": {"half_life_turns": 22, "min_relevance_threshold": 0.08},
        "knapsack": {"exploration_rate": 0.25},
        "dedup": {"hamming_threshold": 5},
        "ios": {"skeleton_info_factor": 0.6, "reference_info_factor": 0.2, "diversity_floor": 0.05},
    }
    c = EntrolyConfig(**resolve_tuning_kwargs(cfg))
    assert c.exploration_rate == 0.25
    assert c.dedup_hamming_threshold == 5
    assert c.ios_skeleton_info_factor == 0.6
    assert c.ios_reference_info_factor == 0.2
    assert c.ios_diversity_floor == 0.05


def test_new_dimensions_flat_back_compat_and_defaults():
    flat = {"exploration_rate": 0.3, "hamming_threshold": 4, "ios_diversity_floor": 0.2}
    c = EntrolyConfig(**resolve_tuning_kwargs(flat))
    assert c.exploration_rate == 0.3
    assert c.dedup_hamming_threshold == 4
    assert c.ios_diversity_floor == 0.2
    # missing -> defaults
    d = EntrolyConfig(**resolve_tuning_kwargs({}))
    assert d.exploration_rate == _DEFAULTS.exploration_rate
    assert d.dedup_hamming_threshold == _DEFAULTS.dedup_hamming_threshold


def test_rust_engine_accepts_threaded_config():
    """If the native engine is present, an autotuned config must construct it
    without error (proves the extended kwargs are wired to the real core)."""
    pytest = __import__("pytest")
    try:
        import entroly_core  # noqa: F401
    except ImportError:
        pytest.skip("native engine not installed")
    from entroly.server import _build_rust_engine
    cfg = {
        "weights": {"recency": 0.4, "frequency": 0.3, "semantic_sim": 0.2, "entropy": 0.1},
        "knapsack": {"exploration_rate": 0.25}, "dedup": {"hamming_threshold": 5},
        "ios": {"skeleton_info_factor": 0.6, "reference_info_factor": 0.2, "diversity_floor": 0.05},
    }
    eng = _build_rust_engine(EntrolyConfig(**resolve_tuning_kwargs(cfg)))
    assert eng is not None


def test_project_scoped_tuning_config_precedes_packaged_defaults(tmp_path, monkeypatch):
    """Profile/role writes must be read by runtime startup, not just by CLI."""
    state_dir = tmp_path / "state"
    monkeypatch.setenv("ENTROLY_DIR", str(state_dir))

    path = writable_tuning_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"weights": {"recency": 0.77, "frequency": 0.11}}),
        encoding="utf-8",
    )

    loaded = load_active_tuning_config()
    assert loaded is not None
    loaded_path, cfg = loaded
    assert loaded_path == path
    assert cfg["weights"]["recency"] == 0.77
    assert resolve_tuning_kwargs(cfg)["weight_recency"] == 0.77
