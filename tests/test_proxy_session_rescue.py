from __future__ import annotations

import asyncio
import json

from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig


class _EmptyEngine:
    def advance_turn(self) -> None:
        return None

    def optimize_context(self, token_budget: int, query: str) -> dict:
        return {"selected_fragments": [], "query_analysis": {}}


def _proxy_config() -> ProxyConfig:
    config = ProxyConfig(witness_mode="off")
    config.enable_adaptive_budget = False
    config.enable_dynamic_budget = False
    config.enable_hierarchical_compression = False
    config.enable_passive_feedback = False
    config.enable_context_scaffold = False
    return config


def test_live_proxy_compresses_tool_output_with_recovery_handle(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    heavy = "\n".join(
        [f"2026-07-24 INFO worker={index % 5} complete" for index in range(500)]
        + ["FATAL payment-service E_CONNRESET"]
    )

    async def run():
        proxy = PromptCompilerProxy(_EmptyEngine(), _proxy_config())
        captured = {}

        async def capture(_url, _headers, body, *_args, **kwargs):
            captured["body"] = json.loads(json.dumps(body))
            return JSONResponse(
                {"ok": True},
                headers=kwargs.get("extra_headers") or {},
            )

        proxy._forward_response = capture
        app = Starlette(
            routes=[
                Route(
                    "/v1/chat/completions",
                    proxy.handle_proxy,
                    methods=["POST"],
                )
            ]
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer sk-test"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": "why did payment fail?"},
                        {"role": "tool", "content": heavy},
                    ],
                },
            )
        return response, captured, proxy

    response, captured, proxy = asyncio.run(run())

    forwarded = captured["body"]["messages"][1]["content"]
    assert response.status_code == 200
    assert "FATAL payment-service E_CONNRESET" in forwarded
    assert "[entroly-recovery:" in forwarded
    assert response.headers["x-entroly-compression-mode"] == "elc"
    assert response.headers["x-entroly-session-rescue"] == "passthrough"
    assert proxy._session_rescue_store is not None
    assert (tmp_path / "session_rescue_recovery.json").exists()


def test_live_proxy_blocks_unrecoverable_overflow_before_upstream(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))

    async def run():
        proxy = PromptCompilerProxy(_EmptyEngine(), _proxy_config())
        forwarded = False

        async def capture(*_args, **_kwargs):
            nonlocal forwarded
            forwarded = True
            return JSONResponse({"unexpected": True})

        proxy._forward_response = capture
        app = Starlette(
            routes=[
                Route(
                    "/v1/chat/completions",
                    proxy.handle_proxy,
                    methods=["POST"],
                )
            ]
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer sk-test"},
                json={
                    "model": "gpt-4",
                    "messages": [
                        {
                            "role": "user",
                            "content": "essential user evidence " * 2_000,
                        }
                    ],
                },
            )
        return response, forwarded

    response, forwarded = asyncio.run(run())

    assert response.status_code == 413
    assert response.json()["error"] == "session_context_rescue_required"
    assert "not forwarded" not in response.json()["action"]
    assert forwarded is False


def test_recovery_store_init_failure_does_not_fall_back_to_lossy_pruning(
    tmp_path,
    monkeypatch,
) -> None:
    store_path = tmp_path / "full-store.json"
    store_path.write_text("x" * 256, encoding="utf-8")
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE_STORE", str(store_path))
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE_STORE_MAX_BYTES", "128")
    heavy = "\n".join(f"raw terminal line {index}" for index in range(500))

    async def run():
        proxy = PromptCompilerProxy(_EmptyEngine(), _proxy_config())
        captured = {}

        async def capture(_url, _headers, body, *_args, **kwargs):
            captured["body"] = body
            return JSONResponse(
                {"ok": True},
                headers=kwargs.get("extra_headers") or {},
            )

        proxy._forward_response = capture
        app = Starlette(
            routes=[
                Route(
                    "/v1/chat/completions",
                    proxy.handle_proxy,
                    methods=["POST"],
                )
            ]
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer sk-test"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": "inspect output"},
                        {"role": "tool", "content": heavy},
                    ],
                },
            )
        return response, captured["body"], proxy

    response, forwarded, proxy = asyncio.run(run())

    assert response.status_code == 200
    assert forwarded["messages"][1]["content"] == heavy
    assert proxy._session_rescue is None
    assert "exceeds its configured byte limit" in proxy._session_rescue_init_error
