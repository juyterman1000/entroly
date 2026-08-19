import json
from pathlib import Path

from entroly.config import EntrolyConfig


def test_shipped_exploration_is_deterministic_by_default():
    assert EntrolyConfig.__dataclass_fields__["exploration_rate"].default == 0.0

    root = Path(__file__).resolve().parents[1]
    for rel in (
        "entroly/data/tuning_defaults.json",
        "entroly-wasm/data/tuning_defaults.json",
    ):
        data = json.loads((root / rel).read_text(encoding="utf-8"))
        assert data["knapsack"]["exploration_rate"] == 0.0

    core = (root / "entroly-core/src/lib.rs").read_text(encoding="utf-8")
    wasm = (root / "entroly-wasm/src/lib.rs").read_text(encoding="utf-8")
    assert "exploration_rate=0.0" in core
    assert "exploration_rate: 0.0," in wasm


def test_exploration_remains_explicitly_configurable():
    cfg = EntrolyConfig(exploration_rate=0.25)
    assert cfg.exploration_rate == 0.25
