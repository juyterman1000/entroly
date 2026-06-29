from __future__ import annotations

import pytest

from entroly.harness_budget import (
    CodingHarnessBudgetController,
    SubagentBudgetAllocator,
    SubagentDemand,
)


def test_concave_allocator_favors_higher_marginal_utility() -> None:
    allocator = SubagentBudgetAllocator(quantum_tokens=128)
    result = allocator.allocate(
        [
            SubagentDemand.create(
                "implementer",
                max_tokens=4_096,
                utility_weight=2.0,
                saturation_tokens=1_024,
            ),
            SubagentDemand.create(
                "researcher",
                max_tokens=4_096,
                utility_weight=1.0,
                saturation_tokens=1_024,
            ),
        ],
        total_tokens=4_096,
    )

    assert result["implementer"].tokens > result["researcher"].tokens
    assert sum(item.tokens for item in result.values()) == 4_096


def test_harness_reserves_primary_and_retrieval_context() -> None:
    plan = CodingHarnessBudgetController().plan(
        total_context_tokens=10_000,
        stable_prefix_tokens=2_000,
        dynamic_request_tokens=1_000,
        retrieval_reserve_tokens=1_000,
        subagents=[
            SubagentDemand.create(
                "coder",
                min_tokens=1_000,
                max_tokens=8_000,
                price_per_million=5,
            )
        ],
        spend_ceiling_usd="0.01",
    )

    assert plan.subagent_budget_tokens == 6_000
    assert plan.subagent_allocations["coder"].tokens <= 2_000
    assert plan.total_cost_micro_usd <= 10_000
    assert plan.allocated_subagent_tokens + plan.unused_tokens == 6_000


def test_impossible_mandatory_minima_fail_closed() -> None:
    allocator = SubagentBudgetAllocator()
    with pytest.raises(ValueError, match="minima exceed token budget"):
        allocator.allocate(
            [
                SubagentDemand.create("a", min_tokens=1_000),
                SubagentDemand.create("b", min_tokens=1_000),
            ],
            total_tokens=1_500,
        )
