from __future__ import annotations

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

    mem.tick(1000)
    forgotten = mem.forget()
    ctx = mem.recall("architectural invariant", agent_id="coder", budget=100)

    assert forgotten >= 1
    assert [m.id for m in ctx.selected] == [semantic_id]


def test_memory_os_snapshot_roundtrip() -> None:
    mem = MemoryOS()
    mem.remember("Persist this memory", agent_id="coder", importance=0.8)
    snap = mem.snapshot()

    restored = MemoryOS.from_snapshot(snap)
    ctx = restored.recall("persist memory", agent_id="coder", budget=100)

    assert ctx.selected
    assert "Persist this memory" in ctx.as_text()
