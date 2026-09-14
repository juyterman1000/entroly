"""Tests for Spec 4: Prefix zone layout and zone-aware budget allocation."""

from __future__ import annotations

from entroly.models.registry import AttentionArchitecture
from entroly.stable_prefix import (
    PrefixZone,
    ZoneBudget,
    compute_zone_budgets,
)


class TestPrefixZoneEnum:
    def test_three_zones_exist(self):
        assert PrefixZone.SYSTEM.value == "system"
        assert PrefixZone.HISTORY.value == "history"
        assert PrefixZone.LIVE.value == "live"


class TestZoneBudgetAllocation:
    def test_full_budget_to_live_when_no_fixed_costs(self):
        zb = compute_zone_budgets(10000)
        assert zb.system == 0
        assert zb.history == 0
        assert zb.live == 10000
        assert zb.total == 10000

    def test_fixed_costs_reduce_live(self):
        zb = compute_zone_budgets(10000, system_tokens=2000, history_tokens=3000)
        assert zb.system == 2000
        assert zb.history == 3000
        assert zb.live == 5000

    def test_fixed_costs_exceeding_budget_give_zero_live(self):
        zb = compute_zone_budgets(5000, system_tokens=3000, history_tokens=4000)
        assert zb.live == 0

    def test_linear_hybrid_clamps_live_zone(self):
        zb = compute_zone_budgets(
            20000,
            attention_profile=AttentionArchitecture.LINEAR_HYBRID_GDN,
            system_tokens=1000,
            history_tokens=1000,
        )
        assert zb.live <= 2048

    def test_mla_allows_expanded_live_zone(self):
        zb = compute_zone_budgets(
            20000,
            attention_profile=AttentionArchitecture.LATENT_MLA,
            system_tokens=1000,
            history_tokens=1000,
        )
        assert zb.live <= 12288
        assert zb.live > 2048

    def test_sparse_allows_expanded_live_zone(self):
        zb = compute_zone_budgets(
            20000,
            attention_profile=AttentionArchitecture.SPARSE_DSA,
            system_tokens=1000,
            history_tokens=1000,
        )
        assert zb.live <= 12288

    def test_standard_attention_uses_full_remaining(self):
        zb = compute_zone_budgets(
            20000,
            attention_profile=AttentionArchitecture.FULL_MHA_GQA,
            system_tokens=2000,
            history_tokens=3000,
        )
        assert zb.live == 15000

    def test_unknown_profile_uses_full_remaining(self):
        zb = compute_zone_budgets(
            10000,
            attention_profile=None,
            system_tokens=1000,
            history_tokens=1000,
        )
        assert zb.live == 8000

    def test_attention_profile_in_result(self):
        zb = compute_zone_budgets(
            10000,
            attention_profile=AttentionArchitecture.LATENT_MLA,
        )
        assert zb.attention_profile == "latent_mla"

    def test_none_profile_in_result(self):
        zb = compute_zone_budgets(5000)
        assert zb.attention_profile is None
