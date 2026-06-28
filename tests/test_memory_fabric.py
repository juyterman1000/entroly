from __future__ import annotations

from entroly.memory_fabric import MemoryFabric


def test_memory_fabric_recalls_from_memory_os() -> None:
    fabric = MemoryFabric(enable_long_term=False, enable_native=False)
    mem_id = fabric.remember(
        "Auth timeout was fixed in auth/session.py by increasing refresh slack.",
        agent_id="coder",
        importance=0.9,
        source="incident/auth-timeout",
        tags=["auth", "timeout"],
    )

    result = fabric.recall("login timeout refresh", agent_id="coder", budget=1200)

    assert mem_id in {m.id for m in result.context.selected}
    assert "auth/session.py" in result.as_text()
    assert result.receipt()["memory_os"]["used_tokens"] <= 1200


def test_memory_fabric_reports_layer_capabilities() -> None:
    fabric = MemoryFabric(enable_long_term=False, enable_native=False)
    layers = {layer.name: layer for layer in fabric.capabilities()}

    assert layers["memory_os"].status == "active"
    assert layers["hippocampus_bridge"].status == "disabled"
    assert layers["rust_memory_manager"].status == "disabled"
    assert "schipc" in layers
    assert "pollination" in layers
    assert "federation" in layers
    assert "receipts_witness" in layers


def test_memory_fabric_save_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "fabric-memory.json"
    fabric = MemoryFabric(enable_long_term=False, enable_native=False)
    fabric.remember("Persist through fabric", agent_id="coder", importance=0.8)
    fabric.save(path)

    restored = MemoryFabric.load(path, enable_long_term=False, enable_native=False)
    result = restored.recall("persist fabric", agent_id="coder", budget=100)

    assert result.context.selected
    assert "Persist through fabric" in result.as_text()


def test_memory_fabric_safety_still_blocks_secret() -> None:
    fabric = MemoryFabric(enable_long_term=False, enable_native=False)

    try:
        fabric.remember("sk-abcdefghijklmnopqrstuvwxyz123456", agent_id="coder")
    except ValueError as exc:
        assert "safety policy" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("secret should have been blocked")


def test_memory_fabric_consolidate_and_stats_are_structured() -> None:
    fabric = MemoryFabric(enable_long_term=False, enable_native=False)
    fabric.remember("Important recurring auth lesson", agent_id="coder", importance=0.9)
    result = fabric.consolidate()
    stats = fabric.stats()

    assert "memory_os_promoted" in result
    assert "memory_os" in stats
    assert "layers" in stats
