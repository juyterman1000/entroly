"""End to end: a real request through the proxy carries a conservative net saving.

Drives the full Starlette app with a mocked Anthropic upstream, buffered and
streamed, and reads the sealed Traffic Receipt the request produced.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from entroly import net_savings

LOG = "".join(f"tests/test_mod.py::test_case_{i} PASSED [{i % 100}%]\n" for i in range(600))


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_SOURCE", str(tmp_path))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ENTROLY_AIR_GAP", "1")
    monkeypatch.setenv("ENTROLY_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("ENTROLY_VAULT_COUPLING", "0")
    monkeypatch.delenv("ENTROLY_PRICING_CATALOG", raising=False)
    monkeypatch.delenv("ENTROLY_BUNDLED_PRICING", raising=False)
    monkeypatch.delenv("ENTROLY_USAGE_LEDGER", raising=False)
    net_savings._reset_catalog_cache_for_tests()
    net_savings.SEEN.clear()
    net_savings.GOVERNOR.clear()
    monkeypatch.delenv("ENTROLY_NET_SAVING_GOVERNOR", raising=False)
    from entroly.proxy import create_proxy_app
    from entroly.proxy_config import ProxyConfig

    return create_proxy_app(
        SimpleNamespace(stats=lambda: {}),
        ProxyConfig(),
        start_dashboard=False,
        start_autotune=False,
    )


def _anthropic_json(input_tokens: int, cache_read: int = 0) -> dict:
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": input_tokens, "output_tokens": 3,
                  "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": 0},
    }


def _anthropic_sse(input_tokens: int) -> bytes:
    start = {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-6",
        "content": [], "usage": {"input_tokens": input_tokens, "output_tokens": 1,
                                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}}
    delta = {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 3}}
    events = [
        ("message_start", start),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", delta),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def _body(model: str = "claude-opus-4-6", *, tool_output: str = "ok", stream: bool = False) -> dict:
    return {
        "model": model, "max_tokens": 64, "stream": stream,
        "system": "You are a coding agent.",
        "messages": [
            {"role": "user", "content": "run the tests and tell me what failed"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "pytest -q"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": tool_output}]},
        ],
    }


def _run(app, body: dict, *, stream: bool = False, input_tokens: int = 1_000):
    seen: dict = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        if stream:
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=_anthropic_sse(input_tokens))
        return httpx.Response(200, json=_anthropic_json(input_tokens))

    async def go():
        proxy = app.state.proxy
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://proxy.test") as client:
            response = await client.post(
                "/v1/messages", json=body,
                headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            )
            await response.aread()
        snapshot = proxy._traffic_receipt_ledger.snapshot()
        return response, snapshot["latest"], seen

    return asyncio.run(go())


def _net_row(receipt: dict) -> dict | None:
    rows = [r for r in receipt.get("value_contributions", []) if r["source"] == "net_saving_conservative"]
    return rows[0] if rows else None


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_untouched_request_carries_a_zero_net_saving(app, stream) -> None:
    response, receipt, upstream = _run(app, _body(stream=stream), stream=stream)

    assert response.status_code == 200
    assert receipt is not None
    assert receipt["net_saving_conservative_micro_usd"] == 0
    assert receipt["net_saving_basis"]["pricing_source"].startswith("bundled:")
    row = _net_row(receipt)
    assert row is not None
    assert row["headline_included"] is False
    assert row["tier"] == "estimated"
    assert row["evidence_source"] == "provider_usage"


def test_receipt_stays_digest_verifiable_with_the_net_saving(app) -> None:
    from entroly.proxy_traffic_receipt import _canonical_json
    import hashlib

    _, receipt, _ = _run(app, _body())
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_digest"}
    assert hashlib.sha256(_canonical_json(unsigned)).hexdigest() == receipt["receipt_digest"]


def test_unpriced_model_has_no_net_saving_claim(app) -> None:
    _, receipt, _ = _run(app, _body(model="claude-unreleased-9"))
    assert receipt["net_saving_conservative_micro_usd"] is None
    assert _net_row(receipt) is None


def test_compressed_tool_output_is_a_positive_provider_calibrated_saving(app, monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "0")
    from entroly.proxy import create_proxy_app
    from entroly.proxy_config import ProxyConfig

    live = create_proxy_app(SimpleNamespace(stats=lambda: {}), ProxyConfig(),
                            start_dashboard=False, start_autotune=False)
    _, receipt, upstream = _run(live, _body(tool_output=LOG), input_tokens=900)

    forwarded = upstream["body"]["messages"][-1]["content"][0]["content"]
    assert len(json.dumps(forwarded)) < len(LOG)  # Entroly really changed the request
    basis = receipt["net_saving_basis"]
    assert basis["removed_new_tokens"] > 0
    assert receipt["net_saving_conservative_micro_usd"] > 0


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_receipt_correlates_with_the_request_id_the_client_sees(app, stream) -> None:
    """Without a client x-request-id, the receipt minted its own id, so it
    matched neither X-Entroly-Request-Id nor the usage recorded under it."""
    from entroly.proxy_traffic_receipt import _correlation_digest

    response, receipt, _ = _run(app, _body(stream=stream), stream=stream)
    client_id = response.headers["X-Entroly-Request-Id"]
    assert receipt["request_correlation"] == _correlation_digest(client_id)


# ── the governor: a conversation that stops paying stops being optimized ──


def _live_app(monkeypatch):
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "0")
    from entroly.proxy import create_proxy_app
    from entroly.proxy_config import ProxyConfig

    return create_proxy_app(SimpleNamespace(stats=lambda: {}), ProxyConfig(),
                            start_dashboard=False, start_autotune=False)


def _seed_losses(app, body: dict, *, requests: int = 3, micro_usd: int = -10_000) -> None:
    conversation = app.state.proxy._routing_conversation_id(body, "anthropic")
    assert conversation
    for _ in range(requests):
        net_savings.GOVERNOR.record(conversation, micro_usd)


def _rows(receipt: dict, source: str) -> list[dict]:
    return [r for r in receipt.get("value_contributions", []) if r["source"] == source]


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_losing_conversation_forwards_the_clients_own_request(app, monkeypatch, stream) -> None:
    live = _live_app(monkeypatch)
    body = _body(tool_output=LOG, stream=stream)
    _seed_losses(live, body)

    response, receipt, upstream = _run(live, body, stream=stream, input_tokens=9_000)

    assert upstream["body"]["messages"] == body["messages"]  # nothing optional applied
    assert response.headers["X-Entroly-Net-Saving-Governor"] == "hold"
    assert receipt["net_saving_conservative_micro_usd"] == 0
    governor = receipt["net_saving_governor"]
    assert governor["applied"] == "hold"
    assert governor["priced_requests"] == 4
    assert governor["cumulative_micro_usd"] == -30_000
    (row,) = _rows(receipt, "net_saving_governor_hold")
    assert row["role"] == "protected"
    assert row["headline_included"] is False


def test_governor_observes_a_conversation_that_is_paying(app, monkeypatch) -> None:
    live = _live_app(monkeypatch)
    body = _body(tool_output=LOG)
    _seed_losses(live, body, micro_usd=+10_000)

    response, receipt, upstream = _run(live, body, input_tokens=900)

    forwarded = upstream["body"]["messages"][-1]["content"][0]["content"]
    assert len(json.dumps(forwarded)) < len(LOG)
    assert response.headers["X-Entroly-Net-Saving-Governor"] == "observe"
    assert receipt["net_saving_governor"]["applied"] == "observe"
    assert receipt["net_saving_governor"]["action"] == "observe"
    assert not _rows(receipt, "net_saving_governor_hold")


def test_governor_hold_does_not_leak_into_another_conversation(app, monkeypatch) -> None:
    live = _live_app(monkeypatch)
    losing = _body(tool_output=LOG)
    _seed_losses(live, losing)
    other = _body(tool_output=LOG)
    other["messages"][0] = {"role": "user", "content": "lint the package"}

    response, _, upstream = _run(live, other, input_tokens=900)

    forwarded = upstream["body"]["messages"][-1]["content"][0]["content"]
    assert len(json.dumps(forwarded)) < len(LOG)
    assert response.headers["X-Entroly-Net-Saving-Governor"] == "observe"


def test_governor_can_be_switched_off(app, monkeypatch) -> None:
    live = _live_app(monkeypatch)
    body = _body(tool_output=LOG)
    _seed_losses(live, body)
    monkeypatch.setenv("ENTROLY_NET_SAVING_GOVERNOR", "0")

    response, receipt, upstream = _run(live, body, input_tokens=900)

    forwarded = upstream["body"]["messages"][-1]["content"][0]["content"]
    assert len(json.dumps(forwarded)) < len(LOG)
    assert "X-Entroly-Net-Saving-Governor" not in response.headers
    assert receipt["net_saving_governor"] is None
