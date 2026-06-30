"""Budget allocation for coding-agent harnesses.

Subagents have diminishing returns: the first useful context tokens are more
valuable than the last. The allocator uses deterministic discrete water-filling
over concave utility curves while respecting minimum/maximum allocations, a
global token budget, and an optional microdollar ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class SubagentDemand:
    agent_id: str
    min_tokens: int
    max_tokens: int
    utility_weight: float = 1.0
    saturation_tokens: int = 1024
    price_per_million: Decimal = Decimal("0")

    @classmethod
    def create(
        cls,
        agent_id: str,
        *,
        min_tokens: int = 0,
        max_tokens: int = 4096,
        utility_weight: float = 1.0,
        saturation_tokens: int = 1024,
        price_per_million: str | int | float = 0,
    ) -> "SubagentDemand":
        return cls(
            agent_id=agent_id,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            utility_weight=utility_weight,
            saturation_tokens=saturation_tokens,
            price_per_million=Decimal(str(price_per_million)),
        )

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id is required")
        if self.min_tokens < 0 or self.max_tokens < self.min_tokens:
            raise ValueError("require 0 <= min_tokens <= max_tokens")
        if self.utility_weight <= 0:
            raise ValueError("utility_weight must be positive")
        if self.saturation_tokens <= 0:
            raise ValueError("saturation_tokens must be positive")
        if self.price_per_million < 0:
            raise ValueError("price_per_million cannot be negative")

    def utility(self, tokens: int) -> float:
        bounded = max(0, min(tokens, self.max_tokens))
        return self.utility_weight * math.log1p(
            bounded / self.saturation_tokens
        )


@dataclass(frozen=True, slots=True)
class SubagentAllocation:
    agent_id: str
    tokens: int
    cost_micro_usd: int
    utility: float


@dataclass(frozen=True, slots=True)
class HarnessBudgetPlan:
    total_context_tokens: int
    stable_prefix_tokens: int
    dynamic_request_tokens: int
    retrieval_reserve_tokens: int
    subagent_budget_tokens: int
    subagent_allocations: Mapping[str, SubagentAllocation]
    total_cost_micro_usd: int
    unused_tokens: int

    @property
    def allocated_subagent_tokens(self) -> int:
        return sum(item.tokens for item in self.subagent_allocations.values())


def _micro_cost(tokens: int, rate_per_million: Decimal) -> int:
    return int(
        (Decimal(tokens) * rate_per_million).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


class SubagentBudgetAllocator:
    """Discrete concave allocator with deterministic tie breaking."""

    def __init__(self, *, quantum_tokens: int = 128) -> None:
        if quantum_tokens <= 0:
            raise ValueError("quantum_tokens must be positive")
        self.quantum_tokens = quantum_tokens

    def allocate(
        self,
        demands: Iterable[SubagentDemand],
        *,
        total_tokens: int,
        spend_ceiling_micro_usd: int | None = None,
    ) -> dict[str, SubagentAllocation]:
        items = sorted(list(demands), key=lambda item: item.agent_id)
        if len({item.agent_id for item in items}) != len(items):
            raise ValueError("agent_id values must be unique")
        if total_tokens < 0:
            raise ValueError("total_tokens cannot be negative")
        if spend_ceiling_micro_usd is not None and spend_ceiling_micro_usd < 0:
            raise ValueError("spend ceiling cannot be negative")

        allocations = {item.agent_id: item.min_tokens for item in items}
        minimum_tokens = sum(allocations.values())
        minimum_cost = sum(
            _micro_cost(item.min_tokens, item.price_per_million)
            for item in items
        )
        if minimum_tokens > total_tokens:
            raise ValueError("mandatory subagent minima exceed token budget")
        if (
            spend_ceiling_micro_usd is not None
            and minimum_cost > spend_ceiling_micro_usd
        ):
            raise ValueError("mandatory subagent minima exceed spend ceiling")

        remaining_tokens = total_tokens - minimum_tokens
        remaining_spend = (
            None
            if spend_ceiling_micro_usd is None
            else spend_ceiling_micro_usd - minimum_cost
        )

        while remaining_tokens > 0:
            candidates: list[tuple[float, str, int, int]] = []
            for item in items:
                current = allocations[item.agent_id]
                capacity = item.max_tokens - current
                grant = min(self.quantum_tokens, remaining_tokens, capacity)
                if grant <= 0:
                    continue
                cost = _micro_cost(grant, item.price_per_million)
                if remaining_spend is not None and cost > remaining_spend:
                    continue

                marginal = item.utility(current + grant) - item.utility(current)
                token_fraction = grant / max(1, remaining_tokens)
                if remaining_spend is None or cost == 0:
                    resource_fraction = token_fraction
                else:
                    spend_fraction = cost / max(1, remaining_spend)
                    resource_fraction = token_fraction + spend_fraction
                score = marginal / max(resource_fraction, 1e-15)
                candidates.append((score, item.agent_id, grant, cost))

            if not candidates:
                break

            _, agent_id, grant, cost = max(
                candidates,
                key=lambda candidate: (candidate[0], candidate[1]),
            )
            allocations[agent_id] += grant
            remaining_tokens -= grant
            if remaining_spend is not None:
                remaining_spend -= cost

        by_id = {item.agent_id: item for item in items}
        return {
            agent_id: SubagentAllocation(
                agent_id=agent_id,
                tokens=tokens,
                cost_micro_usd=_micro_cost(
                    tokens,
                    by_id[agent_id].price_per_million,
                ),
                utility=by_id[agent_id].utility(tokens),
            )
            for agent_id, tokens in sorted(allocations.items())
        }


class CodingHarnessBudgetController:
    """Reserve primary context first, then optimize subagent allocations."""

    def __init__(self, allocator: SubagentBudgetAllocator | None = None) -> None:
        self.allocator = allocator or SubagentBudgetAllocator()

    def plan(
        self,
        *,
        total_context_tokens: int,
        stable_prefix_tokens: int,
        dynamic_request_tokens: int,
        retrieval_reserve_tokens: int,
        subagents: Iterable[SubagentDemand],
        spend_ceiling_usd: str | int | float | None = None,
    ) -> HarnessBudgetPlan:
        fixed = (
            stable_prefix_tokens
            + dynamic_request_tokens
            + retrieval_reserve_tokens
        )
        values = (
            total_context_tokens,
            stable_prefix_tokens,
            dynamic_request_tokens,
            retrieval_reserve_tokens,
        )
        if any(value < 0 for value in values):
            raise ValueError("context token budgets cannot be negative")
        if fixed > total_context_tokens:
            raise ValueError("primary context and reserves exceed context window")

        subagent_budget = total_context_tokens - fixed
        spend_micro = None
        if spend_ceiling_usd is not None:
            spend_micro = int(
                (Decimal(str(spend_ceiling_usd)) * Decimal("1000000")).quantize(
                    Decimal("1"),
                    rounding=ROUND_HALF_UP,
                )
            )
            if spend_micro < 0:
                raise ValueError("spend ceiling cannot be negative")

        allocations = self.allocator.allocate(
            subagents,
            total_tokens=subagent_budget,
            spend_ceiling_micro_usd=spend_micro,
        )
        allocated_tokens = sum(item.tokens for item in allocations.values())
        total_cost = sum(item.cost_micro_usd for item in allocations.values())
        return HarnessBudgetPlan(
            total_context_tokens=total_context_tokens,
            stable_prefix_tokens=stable_prefix_tokens,
            dynamic_request_tokens=dynamic_request_tokens,
            retrieval_reserve_tokens=retrieval_reserve_tokens,
            subagent_budget_tokens=subagent_budget,
            subagent_allocations=allocations,
            total_cost_micro_usd=total_cost,
            unused_tokens=subagent_budget - allocated_tokens,
        )


__all__ = [
    "CodingHarnessBudgetController",
    "HarnessBudgetPlan",
    "SubagentAllocation",
    "SubagentBudgetAllocator",
    "SubagentDemand",
]
