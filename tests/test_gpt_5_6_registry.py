from __future__ import annotations

import pytest

from entroly.models.registry import RegistryTrust, get_model_registry


@pytest.mark.parametrize(
    ("alias", "model_id", "input_price", "output_price"),
    [
        ("gpt-5.6-sol", "openai/gpt-5.6-sol", 5.0, 30.0),
        ("gpt-5.6-terra", "openai/gpt-5.6-terra", 2.5, 15.0),
        ("gpt-5.6-luna", "openai/gpt-5.6-luna", 1.0, 6.0),
    ],
)
def test_gpt_5_6_family_has_verified_budget_and_pricing_metadata(
    alias: str,
    model_id: str,
    input_price: float,
    output_price: float,
) -> None:
    result = get_model_registry().resolve(alias)

    assert result.capability is not None
    assert result.capability.id == model_id
    assert result.trust is RegistryTrust.VERIFIED
    assert result.warning is None
    assert result.context_window == 1_050_000
    assert result.capability.max_output_tokens == 128_000
    assert result.capability.input_price_per_million == input_price
    assert result.capability.output_price_per_million == output_price
    assert result.capability.supports_tools is True
    assert result.capability.supports_vision is True
    assert result.capability.supports_reasoning is True
    assert result.capability.reasoning_levels == (
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )


def test_unsuffixed_gpt_5_6_alias_resolves_to_sol() -> None:
    result = get_model_registry().resolve("gpt-5.6")

    assert result.capability is not None
    assert result.capability.id == "openai/gpt-5.6-sol"
    assert result.context_window == 1_050_000
    assert result.trust is RegistryTrust.VERIFIED
