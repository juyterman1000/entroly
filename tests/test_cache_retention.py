from __future__ import annotations

from entroly.cache_retention import CacheRetentionForecaster, anthropic_retention_plans


def test_long_pause_history_can_justify_one_hour_retention() -> None:
    forecaster = CacheRetentionForecaster()
    for timestamp in (0.0, 1800.0, 3600.0, 5400.0, 7200.0):
        forecaster.observe_activity("conversation", observed_at=timestamp)
    forecast = forecaster.forecast(
        "conversation",
        prefix_tokens=100_000,
        input_micro_usd_per_million=3_000_000,
        plans=anthropic_retention_plans(),
        baseline_plan="anthropic_5m",
        expected_resumes=3,
    )
    assert forecast.recommended_plan == "anthropic_1h"
    assert forecast.projected_savings_micro_usd > 0
    assert forecast.tier == "estimated"
    assert forecast.sample_count == 4


def test_short_pause_history_keeps_lower_write_cost_plan() -> None:
    forecaster = CacheRetentionForecaster()
    for timestamp in (0.0, 60.0, 120.0, 180.0, 240.0):
        forecaster.observe_activity("conversation", observed_at=timestamp)
    forecast = forecaster.forecast(
        "conversation",
        prefix_tokens=100_000,
        input_micro_usd_per_million=3_000_000,
        plans=anthropic_retention_plans(),
        baseline_plan="anthropic_5m",
    )
    assert forecast.recommended_plan == "anthropic_5m"


def test_forecaster_never_performs_provider_io() -> None:
    forecaster = CacheRetentionForecaster()
    forecast = forecaster.forecast(
        "new",
        prefix_tokens=10_000,
        input_micro_usd_per_million=1_000_000,
        plans=anthropic_retention_plans(include_one_hour=False),
    )
    assert forecast.recommended_plan == "anthropic_5m"
    assert forecast.confidence == 0.0
