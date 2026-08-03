from __future__ import annotations

import asyncio

import entroly.proxy as proxy_module
import entroly.proxy_transport_final as transport_final
from entroly.proxy_transport_final import (
    _bounded_stream_iterator,
    _bounded_stream_response,
)
from starlette.responses import StreamingResponse


class TrackedStream:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        try:
            for chunk in self.chunks:
                yield chunk
        finally:
            self.closed = True

    async def aclose(self):
        self.closed = True


def test_stream_under_limit_is_preserved_and_closed() -> None:
    async def run():
        source = TrackedStream([b"data: one\n\n", b"data: [DONE]\n\n"])
        output = []
        async for chunk in _bounded_stream_iterator(source, 1024):
            output.append(chunk)
        return source, output

    source, output = asyncio.run(run())
    assert output == [b"data: one\n\n", b"data: [DONE]\n\n"]
    assert source.closed


def test_stream_overflow_stops_before_excess_and_closes_upstream() -> None:
    overflow = []

    async def run():
        source = TrackedStream([b"1234", b"5678", b"never-forwarded"])
        output = []
        async for chunk in _bounded_stream_iterator(
            source,
            6,
            on_overflow=lambda: overflow.append(True),
        ):
            output.append(chunk)
        return source, output

    source, output = asyncio.run(run())
    assert output[0] == b"1234"
    assert b"upstream_response_too_large" in output[1]
    assert b"never-forwarded" not in b"".join(output)
    assert overflow == [True]
    assert source.closed


def test_client_close_propagates_to_upstream_iterator() -> None:
    async def run():
        source = TrackedStream([b"first", b"second"])
        wrapped = _bounded_stream_iterator(source, 1024)
        assert await anext(wrapped) == b"first"
        await wrapped.aclose()
        return source

    source = asyncio.run(run())
    assert source.closed


def test_text_chunks_are_counted_as_utf8_bytes() -> None:
    async def run():
        source = TrackedStream(["é", "é"])
        output = []
        async for chunk in _bounded_stream_iterator(source, 3):
            output.append(chunk)
        return output

    output = asyncio.run(run())
    assert output[0] == "é"
    assert b"upstream_response_too_large" in output[1]


def test_bounded_stream_wrapper_is_active() -> None:
    assert proxy_module.PromptCompilerProxy._stream_response is _bounded_stream_response


def test_response_wrapper_sets_limit_and_enforces_it(monkeypatch) -> None:
    failures = []

    async def original(_self, *_args, **_kwargs):
        return StreamingResponse(TrackedStream([b"1234", b"5678"]))

    monkeypatch.setattr(transport_final, "_ORIGINAL_STREAM_RESPONSE", original)
    monkeypatch.setattr(
        transport_final._safe,
        "_bounded_positive_int",
        lambda _name, _default: 6,
    )

    class Breaker:
        def record_failure(self):
            failures.append(True)

    async def run():
        response = await _bounded_stream_response(
            type("Proxy", (), {"_breaker": Breaker()})()
        )
        output = []
        async for chunk in response.body_iterator:
            output.append(chunk)
        return response, output

    response, output = asyncio.run(run())
    assert response.headers["X-Entroly-Stream-Limit"] == "6"
    assert output[0] == b"1234"
    assert b"upstream_response_too_large" in output[1]
    assert failures == [True]
