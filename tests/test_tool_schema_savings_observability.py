from __future__ import annotations

import sys
from types import SimpleNamespace

from entroly import proxy_traffic_session as attribution
from entroly import proxy_value_otel as otel
from entroly import proxy_value_projection as projection
from entroly.proxy_traffic_receipt import (
    _TrafficRequestContext,
    _apply_tool_schema_deferral,
)
from entroly.tool_schema_deferral import defer_tool_schemas


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Use {name} for a deliberately descriptive operation.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }


def _rows() -> list[dict]:
    return [
        {
            "source": "context_optimization",
            "tier": "measured",
            "role": "additive",
            "evidence_source": "local_observation",
            "headline_included": True,
            "events": 1,
            "tokens": 600,
            "micro_usd": 0,
            "priced_events": 0,
        },
        {
            "source": "tool_schema_deferral",
            "tier": "measured",
            "role": "explanatory",
            "evidence_source": "local_observation",
            "headline_included": False,
            "events": 1,
            "tokens": 140,
            "micro_usd": 0,
            "priced_events": 0,
        },
    ]


def test_tool_schema_deferral_is_explicit_and_fail_closed() -> None:
    body = {"model": "gpt-test", "tools": [_tool("read"), _tool("write")]}

    absent = defer_tool_schemas(body, "")
    unknown = defer_tool_schemas(body, "missing")

    assert absent.changed is False
    assert unknown.changed is False
    assert absent.body["tools"] == body["tools"]
    assert unknown.body["tools"] == body["tools"]


def test_forced_tool_choice_is_retained_with_active_allowlist() -> None:
    body = {
        "model": "gpt-test",
        "tools": [_tool("read"), _tool("write"), _tool("delete")],
        "tool_choice": {"type": "function", "function": {"name": "write"}},
    }

    result = defer_tool_schemas(body, "read")
    names = [tool["function"]["name"] for tool in result.body["tools"]]

    assert result.changed is True
    assert names == ["read", "write"]
    assert result.schemas_before == 3
    assert result.schemas_after == 2
    assert result.tokens_deferred > 0
    assert len(body["tools"]) == 3  # input was not mutated


def test_gemini_function_declarations_are_filtered_without_dropping_builtins() -> None:
    body = {
        "model": "gemini-test",
        "tools": [
            {
                "functionDeclarations": [
                    {"name": "search", "description": "Search the catalog."},
                    {"name": "purchase", "description": "Purchase an item."},
                ]
            },
            {"googleSearch": {}},
        ],
    }

    result = defer_tool_schemas(body, "search")

    assert result.changed is True
    assert result.body["tools"][0]["functionDeclarations"] == [
        {"name": "search", "description": "Search the catalog."}
    ]
    assert result.body["tools"][1] == {"googleSearch": {}}


def test_final_outbound_seam_records_component_and_response_headers() -> None:
    state = attribution.AttributionState(request_id="req-tools")
    token = attribution.CURRENT_ATTRIBUTION.set(state)
    try:
        context = _TrafficRequestContext(
            proxy=object(),
            request_id="req-tools",
            request_correlation="corr",
            client="test",
            provider="openai",
            path="/v1/chat/completions",
            headers={"x-entroly-active-tools": "read"},
            requested_model="gpt-test",
            original_context_tokens=1000,
        )
        kwargs: dict = {}
        output = _apply_tool_schema_deferral(
            context,
            {"model": "gpt-test", "tools": [_tool("read"), _tool("write")]},
            kwargs,
        )
    finally:
        attribution.CURRENT_ATTRIBUTION.reset(token)

    assert len(output["tools"]) == 1
    assert int(kwargs["extra_headers"]["X-Entroly-Tool-Schema-Tokens-Deferred"]) > 0
    row = attribution.aggregate_contributions(state.contributions)[0]
    assert row["source"] == "tool_schema_deferral"
    assert row["tier"] == "measured"
    assert row["headline_included"] is False


def test_prometheus_total_partitions_compression_and_schema_components(monkeypatch) -> None:
    monkeypatch.setattr(
        projection._value,
        "build_traffic_value_snapshot",
        lambda: {
            "windows": {
                "lifetime": {
                    "extra_provider_cost_usd": 0.0,
                    "value_by_source": _rows(),
                }
            }
        },
    )

    text = projection._prometheus_rows()

    assert "entroly_proxy_tokens_saved_total 600" in text
    assert "entroly_proxy_compression_tokens_saved_total 460" in text
    assert "entroly_proxy_tool_schema_tokens_saved_total 140" in text


def test_otel_emits_aggregate_and_component_counters(monkeypatch) -> None:
    created: dict[str, SimpleNamespace] = {}

    class Instrument:
        def __init__(self, name: str):
            self.name = name
            self.values: list[int | float] = []

        def add(self, value, attributes=None):
            self.values.append(value)

    class Meter:
        def create_counter(self, name: str, **kwargs):
            created[name] = Instrument(name)
            return created[name]

        def create_up_down_counter(self, name: str, **kwargs):
            created[name] = Instrument(name)
            return created[name]

    fake_metrics = SimpleNamespace(get_meter=lambda name: Meter())
    monkeypatch.setitem(sys.modules, "opentelemetry", SimpleNamespace(metrics=fake_metrics))
    monkeypatch.setattr(otel, "_INSTRUMENTS", None)
    otel._SEEN.clear()

    otel.emit_value_otel("receipt-components", _rows())

    assert created["entroly.proxy.tokens.saved"].values == [600]
    assert created["entroly.proxy.tokens.compression_saved"].values == [460]
    assert created["entroly.proxy.tokens.tool_schema_saved"].values == [140]
