"""Request-anchored context boundary and its live non-streaming proxy path."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import httpx

from entroly.codec import RecoveryStore
from entroly.context_boundary import (
    compact_chat_history,
    compact_request_context,
    estimate_request_tokens,
)
from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig
from entroly.tokens import count_messages_tokens


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Keep the current task intact."},
        {"role": "user", "content": "Earlier file: " + "old_code " * 400},
        {"role": "assistant", "content": "I read that file."},
        {"role": "user", "content": "Fix the current bug."},
    ]


def test_boundary_preserves_active_turn_and_recovers_complete_history(tmp_path):
    original = _messages()
    decision = compact_chat_history(original, max_tokens=140, store_path=tmp_path / "recovery.json")

    assert decision is not None
    assert decision.messages[0] == original[0]
    assert decision.messages[-1] == original[-1]
    assert decision.omitted_messages == 2
    assert decision.estimated_tokens == estimate_request_tokens(decision.body) <= 140
    assert "entroly recover " + decision.recovery_digest in decision.messages[1]["content"]
    reopened = RecoveryStore(tmp_path / "recovery.json")
    reference = reopened.reference_for(decision.recovery_digest)
    assert reference is not None
    assert json.loads(reopened.recover(reference)) == original[1:3]


def test_public_recover_command_reads_boundary_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    original = _messages()
    decision = compact_chat_history(original, max_tokens=140)
    assert decision is not None
    output_path = tmp_path / "recovered.json"
    completed = subprocess.run(
        [sys.executable, "-m", "entroly", "recover", decision.recovery_digest,
         "--out", str(output_path)],
        cwd=os.getcwd(),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == original[1:3]


def test_boundary_preserves_oversized_active_turn_and_unsupported_shapes(tmp_path):
    original = _messages()
    original[-1] = {"role": "user", "content": "latest " * 600}
    assert compact_chat_history(original, max_tokens=100, store_path=tmp_path / "recovery.json") is None
    assert original[-1]["content"] == "latest " * 600
    assert not (tmp_path / "recovery.json").exists()

    with_tool_call = _messages()
    with_tool_call[2] = {"role": "tool", "content": "result"}
    assert compact_chat_history(with_tool_call, max_tokens=100, store_path=tmp_path / "recovery.json") is None


def test_boundary_refuses_unrecoverable_omission(tmp_path, monkeypatch):
    def fail_put(self, content, **kwargs):
        raise OSError("simulated local storage failure")

    monkeypatch.setattr(RecoveryStore, "put", fail_put)
    assert compact_chat_history(_messages(), max_tokens=140, store_path=tmp_path / "recovery.json") is None


def _proxy() -> PromptCompilerProxy:
    proxy = PromptCompilerProxy(object(), ProxyConfig())
    proxy._witness_enabled = False
    proxy._witness_analyzer = None
    proxy._enable_passive_feedback = False
    return proxy


def test_proxy_preflight_sends_bounded_recoverable_history(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setattr("entroly.proxy.context_window_for_model", lambda model: 165)
    captured = []

    def upstream(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run():
        proxy = _proxy()
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            return await proxy._forward_response(
                "https://provider.example/v1/chat/completions",
                {},
                {"model": "test-model", "messages": _messages()},
                provider="openai",
            )
        finally:
            await proxy._client.aclose()

    response = asyncio.run(run())
    assert response.status_code == 200
    assert len(captured) == 1
    assert captured[0]["messages"][-1] == _messages()[-1]
    assert count_messages_tokens(captured[0]["messages"]) <= int(165 * 0.85)
    assert response.headers["X-Entroly-Preflight-Compacted"] == "true"


def test_proxy_forwards_oversized_active_turn_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setattr("entroly.proxy.context_window_for_model", lambda model: 165)
    original = _messages()
    original[-1] = {"role": "user", "content": "latest " * 600}
    captured = []

    def upstream(request):
        captured.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})

    async def run():
        proxy = _proxy()
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            return await proxy._forward_response(
                "https://provider.example/v1/chat/completions",
                {},
                {"model": "test-model", "messages": original},
                provider="openai",
            )
        finally:
            await proxy._client.aclose()

    response = asyncio.run(run())
    assert response.status_code == 400
    assert len(captured) == 1
    assert captured[0]["messages"] == original
    assert "X-Entroly-Preflight-Compacted" not in response.headers


def test_proxy_context_error_retries_once_with_complete_older_turn_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setattr("entroly.proxy.context_window_for_model", lambda model: 500)
    original = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Old task " * 60},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Middle task " * 60},
        {"role": "assistant", "content": "Middle answer"},
        {"role": "user", "content": "Current task"},
    ]
    captured = []

    def upstream(request):
        captured.append(json.loads(request.content))
        if len(captured) == 1:
            return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run():
        proxy = _proxy()
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            return await proxy._forward_response(
                "https://provider.example/v1/chat/completions",
                {},
                {"model": "test-model", "messages": original},
                provider="openai",
            )
        finally:
            await proxy._client.aclose()

    response = asyncio.run(run())
    assert response.status_code == 200
    assert len(captured) == 2
    assert captured[0]["messages"] == original
    assert captured[1]["messages"][-1] == original[-1]
    assert original[1] not in captured[1]["messages"]
    assert original[2] not in captured[1]["messages"]
    assert response.headers["X-Entroly-Preflight-Compacted"] == "true"


def test_proxy_compacts_anthropic_in_native_system_field(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setattr("entroly.proxy.context_window_for_model", lambda model: 165)
    original = _messages()[1:]
    captured = []

    def upstream(request):
        captured.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"type": "invalid_request_error"}})

    async def run():
        proxy = _proxy()
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            return await proxy._forward_response(
                "https://provider.example/v1/messages",
                {},
                {"model": "test-model", "messages": original},
                provider="anthropic",
            )
        finally:
            await proxy._client.aclose()

    response = asyncio.run(run())
    assert response.status_code == 400
    assert len(captured) == 1
    assert captured[0]["messages"][-1] == original[-1]
    assert captured[0]["messages"] == [original[-1]]
    assert "entroly recover sha256:" in captured[0]["system"]
    assert estimate_request_tokens(captured[0]) <= int(165 * 0.85)


def test_gemini_compacts_native_contents_and_preserves_instruction(tmp_path):
    body = {
        "systemInstruction": {"parts": [{"text": "Original system instruction"}]},
        "contents": [
            {"role": "user", "parts": [{"text": "old " * 400}]},
            {"role": "model", "parts": [{"text": "old answer"}]},
            {"role": "user", "parts": [{"text": "current request"}]},
        ],
    }
    decision = compact_request_context(
        body, provider="gemini", max_tokens=140,
        store_path=tmp_path / "recovery.json",
    )
    assert decision is not None
    assert decision.body["contents"] == body["contents"][-1:]
    assert decision.body["systemInstruction"]["parts"][0] == body["systemInstruction"]["parts"][0]
    assert "entroly recover sha256:" in decision.body["systemInstruction"]["parts"][1]["text"]
    assert decision.estimated_tokens <= 140
    store = RecoveryStore(tmp_path / "recovery.json")
    ref = store.reference_for(decision.recovery_digest)
    assert ref is not None
    assert json.loads(store.recover(ref)) == body["contents"][:2]


def test_proxy_gemini_preflight_uses_native_contents(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setattr("entroly.proxy.context_window_for_model", lambda model: 165)
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "old " * 400}]},
            {"role": "model", "parts": [{"text": "old answer"}]},
            {"role": "user", "parts": [{"text": "current request"}]},
        ],
    }
    captured = []

    def upstream(request):
        captured.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "invalid parameter"}})

    async def run():
        proxy = _proxy()
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            return await proxy._forward_response(
                "https://provider.example/v1beta/models/gemini-test:generateContent",
                {}, body, provider="gemini",
            )
        finally:
            await proxy._client.aclose()

    response = asyncio.run(run())
    assert response.status_code == 400
    assert len(captured) == 1
    assert captured[0]["contents"] == body["contents"][-1:]
    assert "entroly recover sha256:" in captured[0]["systemInstruction"]["parts"][-1]["text"]
    assert response.headers["X-Entroly-Preflight-Compacted"] == "true"


def test_streaming_preflight_shares_provider_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setattr("entroly.proxy.context_window_for_model", lambda model: 165)
    captured = []

    def upstream(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    async def run():
        proxy = _proxy()
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            response = await proxy._stream_response(
                "https://provider.example/v1/chat/completions",
                {}, {"model": "test-model", "messages": _messages(), "stream": True},
                provider="openai",
            )
            data = b"".join([chunk async for chunk in response.body_iterator])
            return response, data
        finally:
            await proxy._client.aclose()

    response, data = asyncio.run(run())
    assert b"[DONE]" in data
    assert len(captured) == 1
    assert captured[0]["messages"][-1] == _messages()[-1]
    assert captured[0]["messages"][1]["role"] == "system"
    assert response.headers["X-Entroly-Preflight-Compacted"] == "true"


def test_responses_input_uses_same_anchored_boundary(tmp_path):
    body = {
        "model": "test-model",
        "instructions": "Keep this policy",
        "input": [
            {"role": "user", "content": "old " * 400},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current request"},
        ],
    }
    decision = compact_request_context(
        body, provider="openai", max_tokens=150,
        store_path=tmp_path / "recovery.json",
    )
    assert decision is not None
    assert decision.body["instructions"].startswith(body["instructions"])
    assert "entroly recover sha256:" in decision.body["instructions"]
    assert decision.body["input"][-1] == body["input"][-1]
    assert decision.body["input"] == body["input"][-1:]
