"""Offline sidecar concurrency checks with explicit worker/release handshakes."""

from __future__ import annotations

import asyncio
import socket
import threading
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_SOURCE", str(tmp_path))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ENTROLY_AIR_GAP", "1")
    monkeypatch.setenv("ENTROLY_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "0")
    monkeypatch.setenv("ENTROLY_VAULT_COUPLING", "0")
    monkeypatch.delenv("ENTROLY_SIDECAR_TOKEN", raising=False)
    from entroly.proxy import create_proxy_app
    from entroly.proxy_config import ProxyConfig

    return create_proxy_app(
        SimpleNamespace(stats=lambda: {}),
        ProxyConfig(),
        start_dashboard=False,
        start_autotune=False,
    )


def _run_offline(coroutine, monkeypatch):
    async def run():
        def deny_network(*args, **kwargs):
            raise AssertionError("sidecar concurrency tests must not access the network")

        # Let asyncio create its internal wakeup socket before forbidding connects.
        with monkeypatch.context() as offline:
            offline.setattr(socket.socket, "connect", deny_network)
            offline.setattr(socket, "create_connection", deny_network)
            return await coroutine

    return asyncio.run(run())


async def _send(client, operation, source):
    if operation == "prepare":
        return await client.post("/proof/prepare", json={
            "path": str(source), "query": "How is evidence preserved?",
        })
    if operation == "advance":
        return await client.post("/proof/advance", json={
            "session_id": "pgs_" + "a" * 24,
            "model_output": "Evidence keeps its fingerprint.",
            "idempotency_key": "delivery-0",
            "expected_revision": 0,
            "request_sha256": "b" * 64,
        })
    return await client.get("/proof/inspect", params={
        "session_id": "pgs_" + "a" * 24,
    })


class _BlockedWork:
    def __init__(self):
        self.loop_thread = threading.get_ident()
        self.entered = threading.Event()
        self.release = threading.Event()

    def block(self):
        self.entered.set()
        assert threading.get_ident() != self.loop_thread, "proof work ran on the event loop"
        assert self.release.wait(10), "test did not release the blocked proof worker"

    async def wait_until_entered(self):
        assert await asyncio.wait_for(
            asyncio.to_thread(self.entered.wait, 5), timeout=6,
        ), "proof worker never reached the blocking point"


@pytest.mark.parametrize("operation", ["prepare", "advance", "inspect"])
@pytest.mark.parametrize("blocked_stage", ["factory", "runtime"])
def test_health_progresses_while_proof_work_is_held(
    app, tmp_path, monkeypatch, operation, blocked_stage,
):
    import entroly.context_receipts.ingest as ingest
    import entroly.proof_guided_runtime as runtime_module

    source = tmp_path / "evidence.md"
    source.write_text("Evidence keeps its fingerprint.\n", encoding="utf-8")
    gate = _BlockedWork()
    calls = []
    original_read = ingest.read_documents_from_path

    def read_documents(path):
        assert threading.get_ident() != gate.loop_thread
        return original_read(path)

    def invoke(*args, **kwargs):
        assert threading.get_ident() != gate.loop_thread
        calls.append((args, kwargs))
        if blocked_stage == "runtime":
            gate.block()
        return {"status": "ok", "operation": operation}

    runtime = SimpleNamespace(**{operation: invoke})

    def factory(*args, **kwargs):
        assert threading.get_ident() != gate.loop_thread
        if blocked_stage == "factory":
            gate.block()
        return runtime

    monkeypatch.setattr(ingest, "read_documents_from_path", read_documents)
    monkeypatch.setattr(runtime_module, "ProofGuidedRuntime", factory)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:9377",
        ) as client:
            proof = asyncio.create_task(_send(client, operation, source))
            try:
                await gate.wait_until_entered()
                health = await asyncio.wait_for(client.get("/health"), timeout=5)
                assert health.status_code == 200
                assert health.json()["status"] == "ok"
                assert not gate.release.is_set()
                assert not proof.done(), "proof completed before the held work was released"
            finally:
                gate.release.set()
                response = await asyncio.wait_for(proof, timeout=5)
            assert response.status_code == 200, response.text
            assert response.json() == {"status": "ok", "operation": operation}

    _run_offline(run(), monkeypatch)
    assert app.state.proxy._proof_runtime is runtime
    assert len(calls) == 1
    if operation == "advance":
        assert type(calls[0][1]["expected_revision"]) is int
        assert calls[0][1]["expected_revision"] == 0
        assert calls[0][1]["request_sha256"] == "b" * 64


def test_concurrent_cold_requests_share_one_lazy_runtime(app, tmp_path, monkeypatch):
    import entroly.proof_guided_runtime as runtime_module

    gate = _BlockedWork()
    second_factory_entered = threading.Event()
    entry_lock = threading.Lock()
    factory_entries = 0
    constructed = []
    runtime = SimpleNamespace(inspect=lambda session_id: {"session_id": session_id})
    original_factory = app.state.proxy.proof_runtime

    def observed_factory():
        nonlocal factory_entries
        with entry_lock:
            factory_entries += 1
            if factory_entries == 2:
                second_factory_entered.set()
        return original_factory()

    def factory(*args, **kwargs):
        constructed.append(runtime)
        gate.block()
        return runtime

    monkeypatch.setattr(runtime_module, "ProofGuidedRuntime", factory)
    monkeypatch.setattr(app.state.proxy, "proof_runtime", observed_factory)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:9377",
        ) as client:
            requests = [asyncio.create_task(_send(client, "inspect", tmp_path))]
            try:
                await gate.wait_until_entered()
                requests.append(asyncio.create_task(_send(client, "inspect", tmp_path)))
                assert await asyncio.wait_for(
                    asyncio.to_thread(second_factory_entered.wait, 5), timeout=6,
                ), "second request never entered the contended lazy factory"
                health = await asyncio.wait_for(client.get("/health"), timeout=5)
                assert health.status_code == 200
                assert not gate.release.is_set()
                assert all(not request.done() for request in requests)
                assert constructed == [runtime]
            finally:
                gate.release.set()
                responses = await asyncio.wait_for(asyncio.gather(*requests), timeout=5)
            assert all(response.status_code == 200 for response in responses)
            assert responses[0].json() == responses[1].json()

    _run_offline(run(), monkeypatch)
    assert factory_entries == 2
    assert constructed == [runtime]
    assert app.state.proxy._proof_runtime is runtime


@pytest.mark.parametrize("failed_stage", ["factory", "runtime"])
@pytest.mark.parametrize("error_name, status", [
    ("ValueError", 400),
    ("ProofGuidedSessionNotFound", 404),
    ("ProofGuidedSessionConflict", 409),
    ("RuntimeError", 500),
])
def test_worker_errors_keep_sidecar_transport_semantics(
    app, tmp_path, monkeypatch, failed_stage, error_name, status,
):
    import entroly.proof_guided_runtime as runtime_module

    error_type = {"ValueError": ValueError, "RuntimeError": RuntimeError}.get(error_name)
    if error_type is None:
        error_type = getattr(runtime_module, error_name)
    loop_thread = threading.get_ident()

    def fail(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        raise error_type("private diagnostic")

    def factory(*args, **kwargs):
        if failed_stage == "factory":
            return fail()
        return SimpleNamespace(inspect=fail)

    monkeypatch.setattr(runtime_module, "ProofGuidedRuntime", factory)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:9377",
        ) as client:
            response = await _send(client, "inspect", tmp_path)
        assert response.status_code == status, response.text
        if status == 500:
            assert response.json()["error"] == "proof_guided_operation_failed"
            assert "private diagnostic" not in response.text
        else:
            assert response.json() == {"error": error_name, "detail": "private diagnostic"}

    _run_offline(run(), monkeypatch)
