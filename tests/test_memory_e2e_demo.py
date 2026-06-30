from __future__ import annotations

from examples.memory_os_e2e_demo import run_demo


def test_memory_os_e2e_demo_passes() -> None:
    result = run_demo()

    assert result["passed"] is True
    assert result["unsafe_blocked"] is True
    assert set(result["expected_memories"]).issubset(set(result["selected_ids"]))
    assert result["receipt"]["used_tokens"] <= result["receipt"]["budget"]
    assert result["prompt_evidence"]
