import asyncio
import json
from dataclasses import dataclass

import httpx

from entroly.ccr import get_ccr_store
from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig


@dataclass
class _Result:
    rejected: bool

    @property
    def summary_score(self):
        return 0.1 if self.rejected else 1.0

    @property
    def certificates(self):
        return []

    @property
    def n_grounded(self):
        return 0 if self.rejected else 1

    @property
    def n_unsupported(self):
        return 1 if self.rejected else 0

    n_contradicted = 0
    n_unknown = 0
    latency_ms = 0.1

    def flagged(self):
        return ["unsupported"] if self.rejected else []

    def as_dict(self):
        return {
            "summary_score": self.summary_score,
            "n_unsupported": self.n_unsupported,
        }


@dataclass
class _Rewrite:
    output: str
    mode: str = "audit"
    changed: bool = False
    suppressed_count: int = 0
    warned_count: int = 0

    def as_dict(self):
        return {
            "mode": self.mode,
            "changed": self.changed,
            "suppressed_count": self.suppressed_count,
            "warned_count": self.warned_count,
        }


class _Analyzer:
    def analyze(self, context, _output):
        return _Result(rejected="--- Exact Recovery Context ---" not in context)

    def analyze_and_rewrite(self, context, output, mode="audit"):
        return self.analyze(context, output), _Rewrite(output=output, mode=mode)

    def nli_usage(self):
        return {}


class _AlwaysRejectAnalyzer(_Analyzer):
    def analyze(self, _context, _output):
        return _Result(rejected=True)


class _Engine:
    def __init__(self):
        self.resolution_outcomes = []
        self.fragments = {
            "auth-1": {
                "id": "auth-1",
                "source": "file:auth.py",
                "content": "def validate_token(token):\n    return token == 'known'\n",
                "token_count": 14,
                "entropy_score": 0.8,
            }
        }

    def _get_fragment(self, key):
        if key in self.fragments:
            return self.fragments[key]
        return next(
            (
                fragment for fragment in self.fragments.values()
                if fragment["source"] == key
            ),
            None,
        )

    def record_resolution_outcome(self, resolutions, success):
        self.resolution_outcomes.append((resolutions, success))


def _proxy():
    proxy = PromptCompilerProxy(_Engine(), ProxyConfig(witness_mode="audit"))
    proxy._witness_analyzer = _Analyzer()
    proxy._enable_distill = False
    proxy._enable_passive_feedback = False
    proxy._eicv_enabled = False
    proxy._ece = None
    return proxy


def _recoverable_fragment():
    store = get_ccr_store()
    store.clear()
    handle = store.store(
        source="file:auth.py",
        original_content="def validate_token(token):\n    return token == 'known'\n",
        compressed_content="def validate_token(token): ...",
        resolution="skeleton",
        original_tokens=14,
        compressed_tokens=5,
        fragment_id="auth-1",
        relevance=0.9,
    )
    return {
        "id": "auth-1",
        "source": "file:auth.py",
        "content": "def validate_token(token): ...",
        "variant": "skeleton",
        "retrieval_handle": handle,
        "relevance": 0.9,
    }


def _openai_sse(text):
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }
        ]
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


def test_build_recovery_context_materializes_exact_original():
    proxy = _proxy()
    fragment = _recoverable_fragment()

    context, sources = proxy._build_recovery_context([fragment])

    assert sources == ["file:auth.py"]
    assert "def validate_token(token):" in context
    assert "return token == 'known'" in context
    assert "sha256=" in context


def test_prepare_auto_recovery_is_bounded_to_one_retry():
    proxy = _proxy()
    fragment = _recoverable_fragment()
    body = {"model": "test", "messages": [{"role": "user", "content": "fix auth"}]}

    first = proxy._prepare_auto_recovery_retry(
        body,
        "openai",
        "fix auth",
        [fragment],
        _Result(rejected=True),
        recovery_depth=0,
    )
    second = proxy._prepare_auto_recovery_retry(
        body,
        "openai",
        "fix auth",
        [fragment],
        _Result(rejected=True),
        recovery_depth=1,
    )

    assert first is not None
    assert "return token == 'known'" in json.dumps(first[0])
    assert second is None


def test_build_recovery_context_enforces_strict_token_cap():
    proxy = _proxy()
    proxy._auto_recovery_max_tokens = 4
    fragment = _recoverable_fragment()

    context, sources = proxy._build_recovery_context([fragment])

    assert context == ""
    assert sources == []


def test_build_recovery_context_slices_oversized_source_with_retrieval_handle():
    proxy = _proxy()
    proxy._auto_recovery_max_tokens = 64
    original = (
        "Configuration reference\n"
        + ("introductory material\n" * 30)
        + "rotation_interval = 45\n"
        + ("trailing material\n" * 20)
    )
    store = get_ccr_store()
    store.clear()
    handle = store.store(
        source="file:settings.py",
        original_content=original,
        compressed_content="rotation_interval = ...",
        resolution="reference",
        original_tokens=(len(original) + 3) // 4,
        compressed_tokens=6,
        fragment_id="settings-1",
        relevance=0.9,
    )
    fragment = {
        "id": "settings-1",
        "source": "file:settings.py",
        "content": "rotation_interval = ...",
        "variant": "reference",
        "retrieval_handle": handle,
        "relevance": 0.9,
    }

    context, sources = proxy._build_recovery_context(
        [fragment],
        query="find rotation_interval",
    )

    assert sources == ["file:settings.py"]
    assert "exact bounded excerpts" in context
    assert "omitted gaps marked" in context
    assert f"retrieve={handle}" in context
    assert "rotation_interval = 45" in context


def test_build_recovery_context_keeps_distinct_handles_from_same_source():
    proxy = _proxy()
    store = get_ccr_store()
    store.clear()
    fragments = []
    for fragment_id, original, relevance in [
        ("auth-1", "def validate_token(token): return token == 'known'\n", 0.9),
        ("auth-2", "ROTATION_INTERVAL = 17\n", 0.8),
    ]:
        handle = store.store(
            source="file:auth.py",
            original_content=original,
            compressed_content=f"[ref] {fragment_id}",
            resolution="reference",
            original_tokens=(len(original) + 3) // 4,
            compressed_tokens=3,
            fragment_id=fragment_id,
            relevance=relevance,
        )
        fragments.append({
            "id": fragment_id,
            "source": "file:auth.py",
            "content": f"[ref] {fragment_id}",
            "variant": "reference",
            "retrieval_handle": handle,
            "relevance": relevance,
        })

    context, sources = proxy._build_recovery_context(fragments)

    assert sources == ["file:auth.py", "file:auth.py"]
    assert "def validate_token" in context
    assert "ROTATION_INTERVAL = 17" in context


def test_build_recovery_context_does_not_displace_selected_detail_with_standby():
    proxy = _proxy()
    proxy._auto_recovery_max_tokens = 24
    store = get_ccr_store()
    store.clear()

    def fragment(fragment_id, source, original, relevance, *, standby=False):
        handle = store.store(
            source=source,
            original_content=original,
            compressed_content=f"[ref] {fragment_id}",
            resolution="reference",
            original_tokens=(len(original) + 3) // 4,
            compressed_tokens=3,
            fragment_id=fragment_id,
            relevance=relevance,
        )
        return {
            "id": fragment_id,
            "source": source,
            "content": f"[ref] {fragment_id}",
            "variant": "reference",
            "retrieval_handle": handle,
            "relevance": relevance,
            "recovery_candidate": standby,
        }

    selected = fragment(
        "selected-1",
        "file:selected.py",
        "SELECTED_DETAIL\n" * 5,
        0.1,
    )
    standby = fragment(
        "standby-1",
        "file:standby.py",
        "STANDBY_FILLER\n" * 6,
        1.0,
        standby=True,
    )

    context, sources = proxy._build_recovery_context([standby, selected])

    assert sources == ["file:selected.py"]
    assert "SELECTED_DETAIL" in context
    assert "STANDBY_FILLER" not in context


def test_non_streaming_verification_retry_recovers_exact_context():
    proxy = _proxy()
    fragment = _recoverable_fragment()
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        text = (
            "Use validate_token(token) from auth.py."
            if len(requests) == 2
            else "Use invented_auth_check(token)."
        )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": text}}
                ]
            },
        )

    async def run():
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            return await proxy._forward_response(
                "https://example.test/v1/chat/completions",
                {},
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "fix auth"}],
                },
                witness_context="fix auth",
                provider="openai",
                recoverable_fragments=[fragment],
            )
        finally:
            await proxy._client.aclose()

    response = asyncio.run(run())

    assert len(requests) == 2
    assert "return token == 'known'" in json.dumps(requests[1])
    assert response.headers["X-Entroly-Recovery-Attempted"] == "true"
    assert response.headers["X-Entroly-Recovery-Verified"] == "true"
    assert response.headers["X-Entroly-Recovery-Fragments"] == "1"
    assert response.headers["X-Entroly-Recovered"] == "true"
    assert response.headers["X-Entroly-Recovered-Fragments"] == "1"
    assert json.loads(response.body)["choices"][0]["message"]["content"] == (
        "Use validate_token(token) from auth.py."
    )
    assert proxy._auto_recovery_attempted == 1
    assert proxy._auto_recovery_succeeded == 1
    assert proxy._auto_recovery_failed == 0
    assert proxy.engine.resolution_outcomes == [(["skeleton"], False)]


def test_non_streaming_recovery_does_not_claim_unverified_retry():
    proxy = _proxy()
    proxy._witness_analyzer = _AlwaysRejectAnalyzer()
    fragment = _recoverable_fragment()
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Use invented_auth_check(token).",
                        }
                    }
                ]
            },
        )

    async def run():
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            return await proxy._forward_response(
                "https://example.test/v1/chat/completions",
                {},
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "fix auth"}],
                },
                witness_context="fix auth",
                provider="openai",
                recoverable_fragments=[fragment],
            )
        finally:
            await proxy._client.aclose()

    response = asyncio.run(run())

    assert len(requests) == 2
    assert "X-Entroly-Recovered" not in response.headers
    assert response.headers["X-Entroly-Recovery-Attempted"] == "true"
    assert response.headers["X-Entroly-Recovery-Verified"] == "false"
    assert response.headers["X-Entroly-Recovery-Fragments"] == "1"
    assert proxy._auto_recovery_attempted == 1
    assert proxy._auto_recovery_succeeded == 0
    assert proxy._auto_recovery_failed == 1


def test_buffered_streaming_verification_retry_recovers_exact_context():
    proxy = _proxy()
    fragment = _recoverable_fragment()
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        text = (
            "Use validate_token(token) from auth.py."
            if len(requests) == 2
            else "Use invented_auth_check(token)."
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_openai_sse(text),
        )

    async def run():
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            response = await proxy._buffered_witness_stream_response(
                "https://example.test/v1/chat/completions",
                {},
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "fix auth"}],
                },
                witness_context="fix auth",
                provider="openai",
                recoverable_fragments=[fragment],
            )
            chunks = [chunk async for chunk in response.body_iterator]
            return response, b"".join(chunks)
        finally:
            await proxy._client.aclose()

    response, raw = asyncio.run(run())

    assert len(requests) == 2
    assert "return token == 'known'" in json.dumps(requests[1])
    assert response.headers["X-Entroly-Recovery-Attempted"] == "true"
    assert response.headers["X-Entroly-Recovery-Verified"] == "true"
    assert response.headers["X-Entroly-Recovered"] == "true"
    assert b"Use validate_token(token) from auth.py." in raw
    assert proxy._auto_recovery_attempted == 1
    assert proxy._auto_recovery_succeeded == 1
