from __future__ import annotations

import asyncio
import hashlib
import json
import re
from types import SimpleNamespace

from starlette.requests import Request

import entroly.proxy as proxy_module
from entroly.compression_retrieval_store_secure import CompressionRetrievalStore


def _request(
    body: dict,
    *,
    store: CompressionRetrievalStore | None,
    query: bytes = b"",
) -> Request:
    encoded = json.dumps(body).encode("utf-8")
    messages = [{"type": "http.request", "body": encoded, "more_body": False}]

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    app = SimpleNamespace(
        state=SimpleNamespace(proxy=SimpleNamespace(_session_rescue_store=store))
    )
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/compress",
            "raw_path": b"/v1/compress",
            "query_string": query,
            "headers": [
                (b"host", b"127.0.0.1:9377"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(encoded)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 43123),
            "server": ("127.0.0.1", 9377),
            "app": app,
        },
        receive,
    )


def test_compress_endpoint_returns_forwardable_body_and_stores_recovery(
    tmp_path,
) -> None:
    heavy = "\n".join(
        ["compile dependency ok" for _ in range(400)]
        + ["ERROR final link failed in src/main.rs:91"]
    )
    body = {
        "model": "gpt-test",
        "messages": [
            {"role": "user", "content": "why did the build fail?"},
            {"role": "tool", "content": heavy},
        ],
    }
    store = CompressionRetrievalStore(tmp_path / "recovery.json")

    response = asyncio.run(
        proxy_module._compress_only(
            _request(body, store=store, query=b"budget_tokens=160")
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["model"] == body["model"]
    assert payload["messages"][0] == body["messages"][0]
    assert "ERROR final link failed" in payload["messages"][1]["content"]
    assert "[entroly-recovery:" in payload["messages"][1]["content"]
    assert response.headers["x-entroly-changed"] == "true"
    assert int(response.headers["x-entroly-tokens-saved"]) > 0
    assert response.headers["x-entroly-recovery"] == "stored"
    assert len(store.list_receipts()) == 1


def test_compress_endpoint_fails_closed_without_recovery_store() -> None:
    body = {"messages": [{"role": "tool", "content": "x" * 10_000}]}

    response = asyncio.run(
        proxy_module._compress_only(_request(body, store=None))
    )

    assert response.status_code == 503
    assert json.loads(response.body)["error"] == "compression_recovery_unavailable"


def test_compress_endpoint_rejects_invalid_budget(tmp_path) -> None:
    response = asyncio.run(
        proxy_module._compress_only(
            _request(
                {"messages": []},
                store=CompressionRetrievalStore(tmp_path / "recovery.json"),
                query=b"budget_tokens=true",
            )
        )
    )

    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "invalid_budget"


def test_compress_route_is_live_and_precedes_catch_all(
    tmp_path,
    monkeypatch,
) -> None:
    from httpx import ASGITransport, AsyncClient

    class FakeEngine:
        def stats(self):
            return {}

    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "1")
    app = proxy_module.create_proxy_app(
        FakeEngine(),
        proxy_module.ProxyConfig(host="127.0.0.1"),
        start_dashboard=False,
        start_autotune=False,
    )
    paths = [route.path for route in app.routes]
    assert paths.index("/v1/compress") < paths.index("/{path:path}")

    async def run():
        heavy = "\n".join(
            ["compile dependency ok" for _ in range(400)]
            + ["ERROR final link failed in src/main.rs:91"]
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://127.0.0.1:9377",
        ) as client:
            compressed = await client.post(
                "/v1/compress?budget_tokens=160",
                json={
                    "messages": [
                        {"role": "user", "content": "why did the build fail?"},
                        {"role": "tool", "content": heavy},
                    ]
                },
            )
            marker = re.search(
                r"\[entroly-recovery:([^:\]]+):([^\]]+)\]",
                compressed.json()["messages"][1]["content"],
            )
            assert marker is not None
            recovered = await client.get(
                "/retrieve",
                params={
                    "receipt_id": marker.group(1),
                    "span_id": marker.group(2),
                    "retrieval_id": "gateway-test-1",
                },
            )
            return compressed, recovered

    response, recovered = asyncio.run(run())

    assert response.status_code == 200
    assert response.headers["x-entroly-changed"] == "true"
    assert recovered.status_code == 200
    assert "ERROR final link failed" in recovered.json()["content"]
    assert recovered.json()["content_sha256"] == hashlib.sha256(
        recovered.json()["content"].encode("utf-8")
    ).hexdigest()
    assert recovered.headers["x-entroly-retrieval-id"] == "gateway-test-1"
