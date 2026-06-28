from __future__ import annotations

import json

import pytest

from entroly.memory import MemoryOS


def test_memory_os_recalls_relevant_memory_under_budget() -> None:
    mem = MemoryOS(default_budget=40)
    auth_id = mem.remember(
        "Login timeout was fixed in auth/session.py by increasing refresh slack.",
        agent_id="coder",
        importance=0.9,
        source="incident/auth-timeout",
        tags=["critical"],
    )
    mem.remember(
        "The marketing page headline was changed last Friday.",
        agent_id="coder",
        importance=0.2,
        source="notes/marketing",
    )

    ctx = mem.recall("why is login timeout happening again", agent_id="coder", budget=40)

    assert ctx.selected
    assert ctx.selected[0].id == auth_id
    assert ctx.used_tokens <= ctx.budget
    assert "auth/session.py" in ctx.as_text()
    receipt = ctx.receipt()
    assert receipt["risk"]["selected_count"] >= 1
    assert "selected" in receipt and "omitted" in receipt


def test_memory_os_deduplicates_and_reinforces_exact_memory() -> None:
    mem = MemoryOS()
    first = mem.remember("Same lesson", agent_id="coder", source="tool")
    second = mem.remember("Same lesson", agent_id="coder", source="tool")

    assert first == second
    assert mem.stats()["total_entries"] == 1


def test_memory_os_forgets_weak_working_memory_but_keeps_semantic() -> None:
    mem = MemoryOS(death_threshold=0.2)
    mem.remember("temporary note", agent_id="coder", importance=0.1, tier="working")
    semantic_id = mem.remember("stable architectural invariant", agent_id="coder", importance=0.1, tier="semantic")

    # Advance far enough for the weak working memory to decay, but below the
    # auto-consolidation interval so this assertion measures explicit forget().
    mem.tick(20)
    forgotten = mem.forget()
    ctx = mem.recall("architectural invariant", agent_id="coder", budget=100)

    assert forgotten == 1
    assert [m.id for m in ctx.selected] == [semantic_id]


def test_memory_os_save_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "memory.json"
    mem = MemoryOS(default_budget=333, max_entries=10, max_tokens=1000)
    mem.remember("Persist this memory", agent_id="coder", importance=0.8)
    mem.save(path)

    restored = MemoryOS.load(path)
    ctx = restored.recall("persist memory", agent_id="coder", budget=100)

    assert path.exists()
    assert restored.stats()["default_budget"] == 333
    assert ctx.selected
    assert "Persist this memory" in ctx.as_text()


def test_memory_os_snapshot_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError):
        MemoryOS.from_snapshot({"entries": "not-a-list"})


def test_memory_os_blocks_secrets_by_default() -> None:
    mem = MemoryOS()
    with pytest.raises(ValueError, match="safety policy"):
        mem.remember("deploy key sk-abcdefghijklmnopqrstuvwxyz123456")


def test_memory_os_redacts_secrets_when_requested() -> None:
    mem = MemoryOS(safety_policy="redact")
    mem_id = mem.remember("contact me at dev@example.com", agent_id="coder")
    ctx = mem.recall("contact", agent_id="coder")

    assert ctx.selected[0].id == mem_id
    assert "EMAIL_REDACTED" in ctx.selected[0].content
    assert "dev@example.com" not in ctx.selected[0].content


def test_memory_os_capacity_evicts_weak_non_semantic_memory() -> None:
    mem = MemoryOS(max_entries=2, max_tokens=10_000)
    mem.remember("weak temporary note", agent_id="coder", importance=0.1, tier="working")
    semantic_id = mem.remember("semantic invariant", agent_id="coder", importance=0.1, tier="semantic")
    strong_id = mem.remember("critical auth timeout lesson", agent_id="coder", importance=0.9, tier="working")

    ctx = mem.recall("auth semantic temporary", agent_id="coder", budget=100)
    selected_ids = {m.id for m in ctx.selected}

    assert mem.stats()["total_entries"] == 2
    assert semantic_id in selected_ids
    assert strong_id in selected_ids


def test_memory_os_over_budget_omits_with_reason() -> None:
    mem = MemoryOS(default_budget=2)
    mem.remember("short high value auth", agent_id="coder", importance=0.9)
    mem.remember("another relevant but too long memory entry for auth timeout investigation", agent_id="coder", importance=0.8)

    ctx = mem.recall("auth timeout", agent_id="coder", budget=2)

    assert ctx.used_tokens <= 2
    assert any(item.reason == "over_budget" for item in ctx.omitted)


def test_memory_os_json_snapshot_is_plain_data() -> None:
    mem = MemoryOS()
    mem.remember("plain json", agent_id="coder", importance=0.7)
    encoded = json.dumps(mem.snapshot())

    assert "plain json" in encoded
