import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "competitive_gate", Path(__file__).resolve().parents[1] / "benchmarks" / "competitive_gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def rows():
    return [
        dict(task="a", arm="native", success=True, activated=True,
             input_tokens=100, output_tokens=20, cost_usd=1, latency_ms=100),
        dict(task="a", arm="entroly", success=True, activated=True,
             input_tokens=60, output_tokens=20, cost_usd=.7, latency_ms=90),
    ]


def test_pass():
    assert gate.evaluate(rows(), "entroly", "native", 1)["screening_gate_passed"]


@pytest.mark.parametrize("key,value", [
    ("success", False), ("activated", False), ("cost_usd", 1),
    ("input_tokens", 100),
])
def test_regression(key, value):
    data = rows()
    data[1][key] = value
    assert not gate.evaluate(data, "entroly", "native", 1)["screening_gate_passed"]


@pytest.mark.parametrize("data", [rows()[:1], rows() + rows(), []])
def test_incomplete_or_duplicate(data):
    with pytest.raises(ValueError):
        gate.evaluate(data, "entroly", "native", 1)


def test_invalid_metric():
    data = rows()
    data[1]["cost_usd"] = float("nan")
    with pytest.raises(ValueError):
        gate.evaluate(data, "entroly", "native", 1)
