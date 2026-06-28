from __future__ import annotations

from examples.memory_fabric_e2e_demo import run_demo


def test_memory_fabric_e2e_demo_passes() -> None:
    result = run_demo()

    assert result["passed"] is True
    assert result["unsafe_blocked"] is True
    assert set(result["expected_memory_ids"]).issubset(set(result["selected_ids"]))
    assert result["used_tokens"] <= result["budget"]
    assert "memory_os" in result["layer_status"]
    assert "schipc" in result["layer_status"]
    assert "pollination" in result["layer_status"]
    assert result["prompt_evidence"]
