from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from entroly.repository_intelligence.graph_identity import file_node_id
from entroly.work_graph import WorkGraph, WorkGraphUnavailableError
from entroly.work_graph_content_digest import enrich_worktree_content_digests
from entroly.work_graph_repo import discover_repository_observation
from entroly.work_graph_store import WorkGraphStore


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _require_native() -> None:
    try:
        WorkGraph("native-probe")
    except WorkGraphUnavailableError as exc:
        pytest.skip(str(exc))


def test_receiving_agent_verifies_explicit_handoff_and_detects_later_edit(
    tmp_path: Path,
) -> None:
    """Scenarios C/K/L: explicit continuation is stronger than inference."""
    _require_native()
    import entroly_core

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "src").mkdir()
    (repo / "src" / "auth.py").write_text(
        "def authenticate(token):\n    return token == 'v1'\n",
        encoding="utf-8",
    )
    (repo / "src" / "tokens.py").write_text(
        "def issue_token():\n    return 'v1'\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    _git(repo, "checkout", "-b", "feature/auth-v2")
    (repo / "src" / "auth.py").write_text(
        "from .tokens import issue_token\n\ndef authenticate(token):\n    return token == issue_token()\n",
        encoding="utf-8",
    )
    (repo / "src" / "tokens.py").write_text(
        "def issue_token():\n    return 'v2'\n",
        encoding="utf-8",
    )

    state_root = tmp_path / "shared-state"
    sender = WorkGraphStore.for_repository(repo, root=state_root)
    observation = discover_repository_observation(
        repo,
        agent_id="claude",
        session_id="claude-session",
        task_hint={
            "task_id": "auth-v2",
            "title": "Upgrade authentication tokens",
            "trust": "observed",
            "explicit_status": "in_progress",
            "remaining_work": ["run package tests"],
            "source_kind": "user_statement",
            "source_ref": "user:auth-v2",
        },
        observed_at_ms=1_000,
    )
    observation["decisions"] = [{
        "decision_id": "decision:token-v2",
        "text": "Issue and validate the same token version",
        "source_ref": "checkpoint:claude-session",
        "source_kind": "checkpoint",
        "trust": "observed",
    }]
    observation["verifications"] = [{
        "verification_id": "test:auth-unit",
        "name": "authentication unit test",
        "state": "passed",
        "evidence_kind": "test_result",
        "source_ref": "pytest:tests/test_auth.py",
        "digest": "sha256:auth-test-output",
        "observed_at_ms": 1_050,
    }]
    enrich_worktree_content_digests(repo, observation)
    graph = sender.submit_repository_observation(observation, repository_path=repo)
    workstream = graph.unfinished()[0]
    workstream_id = workstream["node_id"]

    receipt = json.loads(
        entroly_core.context_receipt_build_json(
            sender.repo_id,
            observation["branch"]["head_sha"],
            graph.graph_commitment,
            workstream_id,
            "sha256:handoff-sources",
            ["src/auth.py#0:200", "src/tokens.py#0:100"],
            [],
            ["test:auth-unit"],
            [],
            ["rh_auth_sources"],
            ["test:auth-unit"],
            512,
            "work-scope/v1",
            "execution:pending",
            1_100,
        )
    )
    graph, _ = sender.record_context_receipt(
        receipt,
        agent_id="claude",
        session_id="claude-session",
    )
    handoff = sender.handoff(
        workstream_id,
        "claude",
        "codex",
        generated_at_ms=1_200,
    )
    proof = sender.continuation_proof(
        handoff,
        outstanding_work_refs=["run package tests"],
        recovery_handle_ids=["rh_auth_sources"],
        created_at_ms=1_300,
    )

    receiver = WorkGraphStore.for_repository(repo, root=state_root)
    received_graph = receiver.load()
    resumed = receiver.resume(workstream_id)
    snapshot = received_graph.snapshot()
    nodes = {node["node_id"]: node for node in snapshot["nodes"]}

    assert WorkGraph.verify_handoff_integrity(handoff)
    assert received_graph.verify_handoff(handoff)
    assert resumed["selected_workstream"]["symbol_ids"]
    assert resumed["changed_paths"] == ["src/auth.py", "src/tokens.py"]
    assert proof["recovery_handle_ids"] == ["rh_auth_sources"]
    assert proof["outstanding_work_refs"] == ["run package tests"]
    assert any(node["kind"] == "decision" for node in nodes.values())
    assert any(node["kind"] == "test" for node in nodes.values())
    for path in resumed["changed_paths"]:
        assert nodes[file_node_id(sender.repo_id, path)]["attributes"]["sha256"]

    tampered = dict(handoff)
    tampered["to_agent"] = "attacker"
    assert not WorkGraph.verify_handoff_integrity(tampered)
    assert not received_graph.verify_handoff(tampered)

    # Keep the Git head unchanged but alter exact worktree bytes. The original
    # artifact remains internally sealed, while graph-bound verification fails.
    (repo / "src" / "tokens.py").write_text(
        "def issue_token():\n    return 'v3'\n",
        encoding="utf-8",
    )
    refreshed = receiver.update_repository(repo, observed_at_ms=1_400)
    assert WorkGraph.verify_handoff_integrity(handoff)
    assert not refreshed.verify_handoff(handoff)
    with pytest.raises(ValueError, match="integrity mismatch"):
        receiver.continuation_proof(handoff, created_at_ms=1_500)
