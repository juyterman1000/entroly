"""Outcome attribution must affect the beliefs of the request it belongs to.

The defect: `proxy._last_injected_claim_ids` was one instance attribute that
every request overwrote. `/outcome` is a separate HTTP request, so

    request A -> claims A
    request B -> claims B   (overwrites)
    outcome A -> Bayesian update applied to B's beliefs

It broke sequentially, not only under concurrency, and on failure it also
enqueued the wrong beliefs for reverification.

These tests exercise the store directly for state semantics and the ASGI app for
transport semantics. Direct-call tests cannot catch an interleaving bug, so A8
and A13 drive real concurrency.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from entroly.request_attribution import (
    CONSUMED,
    PENDING,
    RequestAttributionStore,
    safe_request_id,
)


def test_a1_ab_isolation():
    """A1: an outcome for A must not surface B's claims."""
    store = RequestAttributionStore()
    store.record("req-a", ["claim-a1", "claim-a2"])
    store.record("req-b", ["claim-b1"])

    assert store.claim("req-a") == ("claim-a1", "claim-a2")
    # B is untouched and still attributable.
    assert store.state_of("req-b") == PENDING


def test_a2_reverse_outcome_ordering():
    """A2: outcomes arriving out of order each hit their own request."""
    store = RequestAttributionStore()
    store.record("req-a", ["a"])
    store.record("req-b", ["b"])

    assert store.claim("req-b") == ("b",)
    store.commit("req-b")
    assert store.claim("req-a") == ("a",)
    store.commit("req-a")

    assert store.state_of("req-a") == CONSUMED
    assert store.state_of("req-b") == CONSUMED


def test_a3_duplicate_outcome_attributes_exactly_once():
    """A3: a redelivered outcome must not reapply the Bayesian update."""
    store = RequestAttributionStore()
    store.record("req", ["c1"])

    first = store.claim("req")
    assert first == ("c1",)
    store.commit("req")

    assert store.claim("req") is None, "second delivery must abstain"


def test_a4_failed_mutation_stays_retryable():
    """A4: a throwing update must leave the entry attributable.

    Popping before mutating would make a legitimate retry impossible, which is
    why the store uses claim -> mutate -> commit.
    """
    store = RequestAttributionStore()
    store.record("req", ["c1"])

    assert store.claim("req") == ("c1",)
    store.release("req")  # simulates attribute_outcome() raising

    assert store.state_of("req") == PENDING
    assert store.claim("req") == ("c1",), "retry must see the claims again"
    store.commit("req")
    assert store.claim("req") is None


def test_a5_unknown_request_id_abstains():
    store = RequestAttributionStore()
    store.record("known", ["c"])
    assert store.claim("does-not-exist") is None


def test_a6_expired_request_id_abstains():
    store = RequestAttributionStore(ttl_seconds=0.05)
    store.record("req", ["c"])
    time.sleep(0.12)
    assert store.claim("req") is None


def test_a7_missing_request_id_abstains():
    """A7: an empty id must never resolve to 'the most recent claims'."""
    store = RequestAttributionStore()
    store.record("req", ["c"])
    assert store.claim("") is None
    assert store.claim(None) is None  # type: ignore[arg-type]


def test_a11_bounded_deterministic_eviction():
    """A11: oldest-insertion-first, so behaviour is reproducible."""
    store = RequestAttributionStore(max_entries=3)
    for i in range(5):
        store.record(f"req-{i}", [f"c{i}"])

    assert len(store) == 3
    # The two oldest are gone, the three newest remain.
    assert store.claim("req-0") is None
    assert store.claim("req-1") is None
    assert store.state_of("req-4") == PENDING


def test_a12_hostile_request_id_cannot_inject_headers():
    """A12: the reflected id is validated, never echoed."""
    hostile = [
        "bad\r\nX-Injected: yes",
        "bad\nSet-Cookie: a=b",
        "with space",
        "A" * 500,
        "",
        None,
        12345,
        "semi;colon",
    ]
    for value in hostile:
        out = safe_request_id(value)
        assert "\r" not in out and "\n" not in out
        assert 1 <= len(out) <= 128
        assert out != value, f"{value!r} must not be reflected verbatim"

    # A well-formed id is preserved so real correlation still works.
    assert safe_request_id("req-abc.123:xy_Z") == "req-abc.123:xy_Z"


def test_a13_concurrent_outcomes_for_same_request_mutate_once():
    """A13: the race, not just sequential redelivery.

    Many threads attempt the same request simultaneously; exactly one may win
    the PENDING -> IN_FLIGHT transition.
    """
    store = RequestAttributionStore()
    store.record("req", ["c1", "c2"])

    winners: list[tuple[str, ...]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(16)

    def worker() -> None:
        barrier.wait()
        got = store.claim("req")
        if got is not None:
            with lock:
                winners.append(got)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1, f"exactly one claim must win, got {len(winners)}"
    assert winners[0] == ("c1", "c2")


def test_a8_async_interleaving_keeps_requests_separate():
    """A8: concurrent coroutines recording and claiming interleaved ids."""

    store = RequestAttributionStore()

    async def scenario() -> list[tuple[str, tuple[str, ...] | None]]:
        async def record(rid: str, claims: list[str]) -> None:
            await asyncio.sleep(0)
            store.record(rid, claims)

        await asyncio.gather(*(record(f"r{i}", [f"c{i}"]) for i in range(32)))

        async def claim(rid: str):
            await asyncio.sleep(0)
            return rid, store.claim(rid)

        return list(await asyncio.gather(*(claim(f"r{i}") for i in range(32))))

    results = asyncio.run(scenario())
    for rid, claims in results:
        index = rid[1:]
        assert claims == (f"c{index}",), f"{rid} resolved to {claims}"


def test_global_attribution_attribute_is_gone():
    """The old causal state must not be reintroducible.

    A compatibility fallback to "latest claims" would recreate the defect, so
    the attribute itself should no longer exist on the proxy class.
    """
    import entroly.proxy as proxy_module

    source = __import__("pathlib").Path(proxy_module.__file__).read_text(
        encoding="utf-8"
    )
    # Comments explaining the removal are fine; executable references are not.
    code_lines = [
        line
        for line in source.splitlines()
        if "_last_injected_claim_ids" in line and not line.strip().startswith("#")
    ]
    assert not code_lines, f"live references remain: {code_lines}"


@pytest.mark.parametrize("rid", ["", None])
def test_record_without_request_id_stores_nothing(rid):
    """Without an id there is nothing to attribute to; do not store it."""
    store = RequestAttributionStore()
    store.record(rid, ["c"])  # type: ignore[arg-type]
    assert len(store) == 0
