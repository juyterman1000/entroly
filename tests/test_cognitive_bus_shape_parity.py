"""The two Cognitive Bus drain shapes are a cross-runtime contract.

`drain()` returns the full event; `drain_memory_bridge()` returns the narrower
payload `hippocampus.remember()` consumes. They are not the same object with
fields removed -- the bridge uses ``source`` where the full event uses
``source_agent`` -- so a binding that serializes both through one helper does not
merely return extra keys, it renames one.

That is exactly what happened when the implementation moved into
`entroly-engine`: the rewritten PyO3 binding routed both methods through a shared
`event_dict`, widening the bridge payload to eight keys and replacing ``source``
with ``source_agent``. Nothing caught it, because no Python caller had adopted
`drain_memory_bridge()` yet -- `context_bridge.py` only mentions it in a comment.
A break with no current caller is still a break, and it would have put Python and
npm on different shapes the moment the WASM binding landed.

Both bindings now delegate to the engine's own serializers. These tests pin the
Python side of that; `cognitive_bus::tests::drain_shapes_are_pinned_for_every_runtime`
pins the engine side.
"""

from __future__ import annotations

import pytest

entroly_core = pytest.importorskip("entroly_core")

CognitiveBus = getattr(entroly_core, "CognitiveBus", None)

pytestmark = pytest.mark.skipif(
    CognitiveBus is None,
    reason="native entroly_core does not export CognitiveBus",
)

FULL_EVENT_KEYS = {
    "content",
    "emotional_tag",
    "event_type",
    "id",
    "is_spike",
    "salience",
    "source_agent",
    "timestamp",
}

BRIDGE_KEYS = {
    "content",
    "emotional_tag",
    "event_type",
    "salience",
    "source",
}


def _bus_with_one_event():
    # Threshold 0 so the published event is guaranteed to reach the bridge
    # queue; this test is about payload shape, not about salience policy.
    bus = CognitiveBus(0.0)
    bus.subscribe("reader", ["observation"])
    bus.publish("writer", "observation", "payload text", 3, 99.0)
    return bus


def test_drain_returns_the_full_event_shape() -> None:
    bus = _bus_with_one_event()
    drained = bus.drain("reader", 10)

    assert len(drained) == 1
    assert set(drained[0]) == FULL_EVENT_KEYS
    assert drained[0]["source_agent"] == "writer"


def test_memory_bridge_returns_the_narrow_shape() -> None:
    bus = _bus_with_one_event()
    bridged = bus.drain_memory_bridge()

    assert len(bridged) == 1
    assert set(bridged[0]) == BRIDGE_KEYS


def test_memory_bridge_uses_source_not_source_agent() -> None:
    """The precise regression: one renamed key.

    Asserting the absence matters as much as the presence. A payload carrying
    both would satisfy a naive "has source" check while still having changed
    shape for anyone reading it as a fixed record.
    """
    bus = _bus_with_one_event()
    bridged = bus.drain_memory_bridge()

    assert bridged[0]["source"] == "writer"
    assert "source_agent" not in bridged[0]


def test_the_two_shapes_are_not_interchangeable() -> None:
    """Guards against a future "simplification" that merges them again."""
    bus = _bus_with_one_event()
    full = bus.drain("reader", 10)[0]

    bus2 = _bus_with_one_event()
    bridged = bus2.drain_memory_bridge()[0]

    assert set(full) != set(bridged)
    assert set(bridged) < set(full) | {"source"}
    # `id`, `timestamp` and `is_spike` are full-event only; the bridge consumer
    # has no use for them and they should not be paid for.
    for key in ("id", "timestamp", "is_spike"):
        assert key in full
        assert key not in bridged
