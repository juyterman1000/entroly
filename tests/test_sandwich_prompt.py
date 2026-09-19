"""Unit tests for 5-Zone Sandwich Layout and U-Curve attention distribution."""

from __future__ import annotations

from entroly.stable_prefix import (
    PrefixZone,
    build_sandwich_prompt,
    u_curve_reorder,
)


def test_u_curve_reorder():
    """Verify that highest-ranked fragments occupy the primacy and recency peaks."""
    # 5 fragments ranked from best (rank0) to worst (rank4)
    ranked = ["rank0_best", "rank1_second", "rank2_third", "rank3_fourth", "rank4_worst"]
    reordered = u_curve_reorder(ranked)

    # Primacy peak (first item) should be rank0 (best)
    assert reordered[0] == "rank0_best"
    # Recency peak (last item) should be rank1 (second best)
    assert reordered[-1] == "rank1_second"
    # The middle item should be the lowest-ranked item (rank4)
    assert reordered[2] == "rank4_worst"


def test_build_sandwich_prompt():
    """Verify that the 5-zone sandwich prompt preserves prefix stability and guards user intent."""
    system = "You are an autonomous code assistant."
    history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    topology = "Repository contains: auth.py, db.py, main.py"
    evidence = ["def get_db(): ...", "def authenticate_user(): ..."]
    query = "Add JWT token validation to authenticate_user"

    prompt = build_sandwich_prompt(
        system_prompt=system,
        history_turns=history,
        topology_summary=topology,
        evidence_chunks=evidence,
        active_query=query,
    )

    # Verify all 5 zones are present
    assert "[SYSTEM]" in prompt
    assert "You are an autonomous code assistant." in prompt
    assert "[HISTORY]" in prompt
    assert "USER: hello" in prompt
    assert "[TOPOLOGY]" in prompt
    assert "Repository contains: auth.py" in prompt
    assert "[CODE_EVIDENCE]" in prompt
    assert "<active_task>" in prompt
    assert "<query>Add JWT token validation to authenticate_user</query>" in prompt
    assert "<instruction_guard>" in prompt

    # Verify that the prefix (System + History) is byte-stable even if query changes
    prompt_query2 = build_sandwich_prompt(
        system_prompt=system,
        history_turns=history,
        topology_summary=topology,
        evidence_chunks=["def different_chunk(): ..."],
        active_query="Completely different query",
    )

    # Prefix up to [HISTORY] end must be byte-identical (for 100% provider prompt caching)
    prefix1 = prompt.split("[TOPOLOGY]")[0]
    prefix2 = prompt_query2.split("[TOPOLOGY]")[0]
    assert prefix1 == prefix2
