from __future__ import annotations

import sys
import types

from entroly.memory_fabric import MemoryFabric


class FakeMemoryManager:
    def stats(self):
        return {"episode_count": 0}


class FakeIpcBus:
    def __init__(self):
        self.messages = []

    def send(self, sender_id, receiver_id, content):
        self.messages.append((sender_id, receiver_id, content))
        return {"delivered": True, "reason": "novel", "sender_id": sender_id, "receiver_id": receiver_id}

    def stats(self):
        return {"total_sent": len(self.messages), "total_delivered": len(self.messages)}


class FakeComplianceGate:
    def __init__(self):
        self.checked = []

    def check_message(self, sender_id, receiver_id, content):
        self.checked.append((sender_id, receiver_id, content))
        if "sk-" in content:
            return {"allowed": False, "reason": "pii_detected", "pii_types": ["APIKey"]}
        return {"allowed": True, "reason": "ok", "pii_types": []}

    def stats(self):
        return {"total_allowed": 1, "blocked_pii": 1}


class FakePollinationEngine:
    def __init__(self):
        self.agents = set()
        self.lessons = []
        self.shares = []
        self.rewards = []
        self.ticks = 0

    def register_agent(self, agent_id):
        self.agents.add(agent_id)

    def record_lesson(self, agent_id, description, success, surprise, domain):
        self.lessons.append((agent_id, description, success, surprise, domain))

    def share(self, from_agent, to_agent):
        self.shares.append((from_agent, to_agent))
        return len(self.lessons)

    def reward(self, sharer_id, receiver_id, reward):
        self.rewards.append((sharer_id, receiver_id, reward))

    def tick(self):
        self.ticks += 1

    def stats(self):
        return {"total_agents": len(self.agents), "total_lessons_shared": len(self.shares)}


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


def test_memory_fabric_uses_native_kernels_when_exported(monkeypatch) -> None:
    fake_core = types.SimpleNamespace(
        MemoryManager=FakeMemoryManager,
        IpcBus=FakeIpcBus,
        ComplianceGate=FakeComplianceGate,
        PollinationEngine=FakePollinationEngine,
    )
    monkeypatch.setitem(sys.modules, "entroly_core", fake_core)

    fabric = MemoryFabric(enable_long_term=False, enable_native=True)
    layers = {layer.name: layer for layer in fabric.capabilities()}

    assert layers["rust_memory_manager"].status == "available"
    assert layers["schipc"].status == "available"
    assert layers["compliance_gate"].status == "available"
    assert layers["pollination"].status == "available"

    delivered = fabric.send_agent_message(1, 2, "novel auth lesson")
    blocked = fabric.send_agent_message(1, 2, "secret sk-abcdefghijklmnopqrstuvwxyz123456")

    assert delivered["delivered"] is True
    assert blocked["delivered"] is False
    assert blocked["reason"] == "pii_detected"

    recorded = fabric.record_agent_lesson("coder", "auth retry worked", success=True, surprise=0.2)
    shared = fabric.share_agent_lessons("coder", "reviewer")
    fabric.reward_agent_share("coder", "reviewer", 1.0)
    fabric.tick()
    stats = fabric.stats()

    assert recorded["recorded"] is True
    assert shared["shared"] is True
    assert "native" in stats
    assert "ipc" in stats["native"]
    assert "compliance" in stats["native"]
    assert "pollination" in stats["native"]
