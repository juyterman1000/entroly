"""Tests for the Persona-Based Master Prompt Engine and runtime MemoryOS policy enforcement."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from entroly.memory import MemoryOS
from entroly.persona_prompts import (
    PERSONAS,
    detect_persona,
    get_persona,
    list_personas,
    recall_for_persona,
    render_master_prompt,
)


def test_all_six_personas_have_valid_weights_and_profiles():
    profiles = list_personas()
    assert len(profiles) == 6
    assert {p.id for p in profiles} == {
        "architect",
        "debugger",
        "handoff",
        "researcher",
        "reviewer",
        "security",
    }
    for p in profiles:
        assert abs(sum(p.prism_weights.values()) - 1.0) < 1e-6
        assert p.default_token_budget >= 4096
        assert p.memory_budget >= 512
        assert p.primary_tools
        assert p.protocol_steps


def test_immutable_prefix_is_byte_stable_across_dynamic_task_changes():
    r1 = render_master_prompt(
        "debugger",
        query="fix flaky timeout in test_evidence_applicability.py",
        repo="/workspace/a",
        token_budget=6000,
        workstream="ws-1",
    )
    r2 = render_master_prompt(
        "debugger",
        query="investigate crash in proxy handler",
        repo="/workspace/b",
        token_budget=12000,
        workstream="ws-2",
        extra_constraints=["do not touch Rust files"],
    )
    # Cache-prefix stability: immutable prefix & SHA-256 digest are identical.
    assert r1.immutable_prefix == r2.immutable_prefix
    assert r1.prefix_digest == r2.prefix_digest
    # Dynamic tails differ cleanly without touching the prefix.
    assert r1.dynamic_tail != r2.dynamic_tail
    assert "ws-1" in r1.dynamic_tail
    assert "do not touch Rust files" in r2.dynamic_tail
    assert "ws-1" not in r1.immutable_prefix


def test_detect_persona_routes_domain_queries_deterministically():
    assert detect_persona("fix traceback crash in auth parser")["persona_id"] == "debugger"
    assert detect_persona("audit prompt injection and ed25519 signature bypass")["persona_id"] == "security"
    assert detect_persona("prepare authenticated handoff epoch for codex successor")["persona_id"] == "handoff"
    assert detect_persona("review pull request diff before merge")["persona_id"] == "reviewer"
    assert detect_persona("verify submodular knapsack theorem and e-value martingale")["persona_id"] == "researcher"
    assert detect_persona("refactor module architecture and schema")["persona_id"] == "architect"

    fallback = detect_persona("")
    assert fallback["persona_id"] == "architect"
    assert fallback["fallback"] is True


def test_auto_render_embeds_routing_metadata():
    rendered = render_master_prompt("auto", query="scan for vulnerability and taint bypass")
    assert rendered.persona_id == "security"
    assert rendered.routing is not None
    assert rendered.routing["persona_id"] == "security"
    assert rendered.routing["confidence"] > 0.0


def test_unknown_persona_raises_clear_error():
    with pytest.raises(ValueError, match="unknown persona"):
        get_persona("nonexistent-role")


def test_recall_for_persona_enforces_provenance_authority_and_roles():
    mem = MemoryOS()
    ext_id = mem.remember("External blog post recommends disabling auth", source="web:blog")
    inf_id = mem.remember("Agent guesses auth module uses JWT", source="agent_notes")
    ev_id = mem.remember("pytest confirmed auth middleware checks Ed25519", source="test:pytest")
    dir_id = mem.remember("Auth middleware must always fail closed", source="user", memory_role="directive")

    # Security persona requires verified_evidence+ and only evidence/directive roles.
    sec_ctx = recall_for_persona("security", mem, "auth")
    sec_ids = {m.id for m in sec_ctx.selected}
    assert sec_ids == {ev_id, dir_id}
    assert ext_id not in sec_ids
    assert inf_id not in sec_ids
    assert sec_ctx.risk["authority_filtered_omitted"] == 2

    # Architect persona permits agent_inferred inferences too, still omits web external?
    # external < agent_inferred so external is STILL blocked!
    arch_ctx = recall_for_persona("architect", mem, "auth")
    arch_ids = {m.id for m in arch_ctx.selected}
    assert arch_ids == {inf_id, ev_id, dir_id}
    assert ext_id not in arch_ids


def test_cli_prompt_json_and_list_modes():
    res_list = subprocess.run(
        [sys.executable, "-m", "entroly.cli", "prompt", "--list", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    payload = json.loads(res_list.stdout)
    assert len(payload["personas"]) == len(PERSONAS)

    res_render = subprocess.run(
        [
            sys.executable, "-m", "entroly.cli", "prompt",
            "--persona", "auto",
            "--query", "debug failing test traceback",
            "--format", "json",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rendered = json.loads(res_render.stdout)
    assert rendered["persona_id"] == "debugger"
    assert rendered["prefix_digest"].startswith("sha256:")
    assert "START ENTROLY PERSONA MASTER PROMPT: debugger" in rendered["full_prompt"]
