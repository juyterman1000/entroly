"""Current Claude models resolve to their real window and their real price.

Every Claude 4.6+ and 5.x id fell through to the registry's 128K fallback, so
session rescue started compacting history at ~113K tokens on 1M-window models,
and the bundled price table priced Opus 4.5+ at the retired $15 Opus card and
every 5.x model at the $3 default (Haiku 5.5 30x high, Fable 5.1 3.3x low).

Figures are from platform.claude.com (models overview, per-model pages,
pricing and context-windows), verified 2026-10-31.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from entroly import value_tracker as vt
from entroly.models.registry import RegistryTrust, get_model_registry
from entroly.proxy_config import context_window_for_model

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "entroly" / "models" / "registry.json"
JS_TRACKER = ROOT / "entroly-wasm" / "js" / "value_tracker.js"

# (api id, input $/MTok, output $/MTok)
CLAUDE_5 = [
    ("claude-fable-5-1", 10.0, 50.0),
    ("claude-opus-5-5", 4.0, 20.0),
    ("claude-opus-5", 5.0, 25.0),
    ("claude-sonnet-5-5", 2.0, 10.0),
    ("claude-sonnet-5", 2.0, 10.0),
    ("claude-haiku-5-5", 0.10, 0.50),
]


@pytest.fixture(autouse=True)
def _bundled_pricing(monkeypatch, tmp_path):
    # A developer's ~/.entroly/pricing.json must not decide these assertions.
    monkeypatch.setenv("ENTROLY_PRICING_FILE", str(tmp_path / "absent.json"))
    vt.reset_pricing_cache()
    yield
    vt.reset_pricing_cache()


@pytest.mark.parametrize(("model_id", "input_price", "output_price"), CLAUDE_5)
@pytest.mark.parametrize("form", ["{}", "anthropic/{}", "anthropic.{}"])
def test_claude_5_models_resolve_verified_with_1m_window(
    model_id: str, input_price: float, output_price: float, form: str
) -> None:
    result = get_model_registry().resolve(form.format(model_id))

    assert result.capability is not None
    assert result.capability.id == f"anthropic/{model_id}"
    assert result.trust is RegistryTrust.VERIFIED
    assert result.warning is None
    assert result.context_window == 1_000_000
    assert result.capability.max_output_tokens == 128_000
    assert result.capability.input_price_per_million == input_price
    assert result.capability.output_price_per_million == output_price
    assert context_window_for_model(form.format(model_id)) == 1_000_000


# Opus 4.8, Opus 4.7 and Sonnet 4.6: 1M window and 128K output per the
# context-windows page; Bedrock ids from model-ids-and-versions (no "-v1"
# suffix after Opus 4.6).
CLAUDE_4_CURRENT = [
    ("claude-opus-4-8", "claude-opus-4.8", 5.0, 25.0),
    ("claude-opus-4-7", "claude-opus-4.7", 5.0, 25.0),
    ("claude-sonnet-4-6", "claude-sonnet-4.6", 3.0, 15.0),
]


@pytest.mark.parametrize(("model_id", "dotted", "input_price", "output_price"), CLAUDE_4_CURRENT)
@pytest.mark.parametrize("form", ["{}", "anthropic/{}", "anthropic.{}", "dotted"])
def test_current_claude_4_models_resolve_verified_with_1m_window(
    model_id: str, dotted: str, input_price: float, output_price: float, form: str
) -> None:
    name = dotted if form == "dotted" else form.format(model_id)
    result = get_model_registry().resolve(name)

    assert result.capability is not None
    assert result.capability.id == f"anthropic/{dotted}"
    assert result.trust is RegistryTrust.VERIFIED
    assert result.warning is None
    assert result.context_window == 1_000_000
    assert result.capability.max_output_tokens == 128_000
    assert result.capability.input_price_per_million == input_price
    assert result.capability.output_price_per_million == output_price
    assert context_window_for_model(name) == 1_000_000
    assert vt.estimate_cost(1_000_000, model_id) == pytest.approx(input_price)


def test_opus_4_6_has_its_1m_window() -> None:
    # Recorded as 200K, which made session rescue fire at a fifth of the window.
    assert context_window_for_model("claude-opus-4-6") == 1_000_000
    assert context_window_for_model("claude-opus-4.6") == 1_000_000


@pytest.mark.parametrize(("model_id", "input_price", "output_price"), CLAUDE_5)
@pytest.mark.parametrize(
    "form", ["{}", "anthropic/{}", "anthropic.{}", "us.anthropic.{}"]
)
def test_claude_5_savings_are_priced_at_the_published_rate(
    model_id: str, input_price: float, output_price: float, form: str
) -> None:
    model = form.format(model_id)
    assert vt._has_priced_model(model)
    assert vt.estimate_cost(1_000_000, model) == pytest.approx(input_price)
    assert vt.estimate_cost(1_000_000, model, "output") == pytest.approx(output_price)


@pytest.mark.parametrize(
    ("model", "input_price", "output_price"),
    [
        # Opus 4 and 4.1 keep the $15/$75 card ...
        ("claude-opus-4-20250514", 15.0, 75.0),
        ("claude-opus-4-1-20250805", 15.0, 75.0),
        # ... Opus 4.5 and later were repriced to $5/$25.
        ("claude-opus-4-5-20251101", 5.0, 25.0),
        ("claude-opus-4-6", 5.0, 25.0),
        ("claude-opus-4-7", 5.0, 25.0),
        ("claude-opus-4-8", 5.0, 25.0),
        ("claude-haiku-4-5-20251001", 1.0, 5.0),
        ("claude-sonnet-4-5-20250929", 3.0, 15.0),
    ],
)
def test_claude_4_generation_rates(model: str, input_price: float, output_price: float) -> None:
    assert vt.estimate_cost(1_000_000, model) == pytest.approx(input_price)
    assert vt.estimate_cost(1_000_000, model, "output") == pytest.approx(output_price)


def test_provider_prefix_is_only_stripped_before_a_claude_id() -> None:
    # A name that merely contains "claude-" must not inherit a Claude rate.
    assert not vt._has_priced_model("acme/not-claude-opus-5-5")
    assert vt.estimate_cost(1_000_000, "acme/not-claude-opus-5-5") == pytest.approx(3.0)


def test_every_registered_anthropic_model_has_a_bundled_price() -> None:
    """A verified Anthropic model must never be priced at the default."""
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    unpriced = [
        model["id"]
        for model in payload["models"]
        if model["provider"] == "anthropic"
        and not vt._has_priced_model(model["id"].split("/", 1)[1])
    ]
    assert not unpriced, f"Anthropic models priced at the default rate: {unpriced}"


def _js_cost_table() -> dict[str, float]:
    source = JS_TRACKER.read_text(encoding="utf-8")
    block = source.split("const COST_PER_M = {", 1)[1].split("};", 1)[0]
    return {
        key: float(value)
        for key, value in re.findall(r"'([a-z0-9.\-]+)':\s*([0-9.]+)", block)
    }


def test_js_tracker_matches_python_input_rates_for_anthropic() -> None:
    js = _js_cost_table()
    python = {
        key: rates["input"] * 1000
        for key, rates in vt._MODEL_PRICING.items()
        if key.startswith("claude-")
    }
    assert set(python) <= set(js), f"missing in JS: {sorted(set(python) - set(js))}"
    mismatched = {key: (python[key], js[key]) for key in python if js[key] != pytest.approx(python[key])}
    assert not mismatched, f"Python vs JS $/M input rates differ: {mismatched}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_tracker_normalises_provider_prefixes() -> None:
    script = (
        "const v=require(process.argv[1]);"
        "console.log(JSON.stringify(["
        "v.estimateCost(1e6,'us.anthropic.claude-haiku-5-5'),"
        "v.estimateCost(1e6,'anthropic/claude-opus-5-5'),"
        "v.estimateCost(1e6,'claude-opus-4-20250514')]))"
    )
    out = subprocess.run(
        ["node", "-e", script, str(JS_TRACKER)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert json.loads(out) == pytest.approx([0.1, 4.0, 15.0])


# ── injected context is budgeted against at most a 200K window ──────────


@pytest.fixture
def _proxy_config(monkeypatch):
    from entroly.proxy_config import ProxyConfig

    monkeypatch.delenv("ENTROLY_INJECTION_REFERENCE_WINDOW", raising=False)
    return ProxyConfig()


@pytest.mark.parametrize(
    "model", ["claude-opus-5-5", "claude-haiku-5-5", "claude-opus-4-6", "gpt-5.6-sol"]
)
def test_1m_window_models_get_the_200k_injection_budget(model, _proxy_config) -> None:
    """A real 1M window must not quintuple the repo context sent per request."""
    from entroly.proxy_transform import compute_dynamic_budget, compute_token_budget

    reference = "claude-sonnet-4-5-20250929"  # a 200K-window model
    for vagueness, fragments in ((0.2, 100), (0.5, 1_000), (0.9, 5_000)):
        assert compute_dynamic_budget(
            model, _proxy_config, vagueness=vagueness, total_fragments=fragments
        ) == compute_dynamic_budget(
            reference, _proxy_config, vagueness=vagueness, total_fragments=fragments
        )
    assert compute_token_budget(model, _proxy_config) == compute_token_budget(
        reference, _proxy_config
    )
    # The cap is on the injection budget only; capacity checks see 1M.
    assert context_window_for_model(model) > 200_000


def test_smaller_windows_are_not_affected_by_the_cap(_proxy_config) -> None:
    from entroly.proxy_transform import compute_token_budget

    assert compute_token_budget("gpt-4o", _proxy_config) == int(
        context_window_for_model("gpt-4o") * _proxy_config.context_fraction
    )


def test_injection_reference_window_is_configurable(monkeypatch, _proxy_config) -> None:
    from entroly.proxy_transform import compute_token_budget, injection_budget_window

    monkeypatch.setenv("ENTROLY_INJECTION_REFERENCE_WINDOW", "0")
    assert injection_budget_window(1_000_000) == 1_000_000
    assert compute_token_budget("claude-opus-5-5", _proxy_config) == int(
        1_000_000 * _proxy_config.context_fraction
    )
    monkeypatch.setenv("ENTROLY_INJECTION_REFERENCE_WINDOW", "500000")
    assert injection_budget_window(1_000_000) == 500_000
    monkeypatch.setenv("ENTROLY_INJECTION_REFERENCE_WINDOW", "not-a-number")
    assert injection_budget_window(1_000_000) == 200_000
