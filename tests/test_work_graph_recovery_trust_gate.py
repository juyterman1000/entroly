"""A recovering agent must accept the trust level before it may act.

`work_resume` labelled its output `untrusted_recovered_work_state` and recorded
`unknown:previous-agent-intent`, but nothing stopped an agent reading the label
and immediately claiming work as though reconstructed state were observed fact.
The label described the risk; nothing made anyone carry it.

Also covers the index/worktree commitment split: staged and conflicted paths
used to receive no digest at all, on the correct reasoning that one worktree
digest cannot represent both index and worktree state -- but the answer is two
digests, not none.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from entroly.work_graph_content_digest import enrich_worktree_content_digests
from entroly.work_graph_recovery_ack import (
    RecoveryAcknowledgementRequired,
    acknowledge,
    arm,
    pending,
    recovery_token,
    require_acknowledged,
)
from entroly.work_graph_repo import discover_repository_observation


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.test")
    _git(repo, "config", "user.name", "t")
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


class TestTokenIdentity:
    def test_same_state_yields_the_same_token(self):
        view = {"selected_workstream": {"node_id": "w1"}, "changed_paths": ["a.py"]}
        assert recovery_token(view) == recovery_token(dict(view))

    def test_key_order_does_not_change_the_token(self):
        # Key order carries no meaning; it must not demand a second
        # acknowledgement for identical facts.
        left = {"a": 1, "b": 2}
        right = {"b": 2, "a": 1}
        assert recovery_token(left) == recovery_token(right)

    def test_different_state_yields_a_different_token(self):
        base = {"changed_paths": ["a.py"]}
        moved = {"changed_paths": ["a.py", "b.py"]}
        assert recovery_token(base) != recovery_token(moved)


class TestGate:
    def test_no_gate_means_no_refusal(self, tmp_path):
        # Asserted rather than relying on "it did not raise": a test whose only
        # failure mode is an exception says nothing about what it checked, and
        # would still pass if require_acknowledged became a no-op.
        assert pending(tmp_path) is None

        require_acknowledged(tmp_path)

        # Reading the gate must not create one, or the first agent to look
        # would arm a gate against itself.
        assert pending(tmp_path) is None
        assert not list(tmp_path.glob("pending-recovery-ack.json"))

    def test_arming_blocks_and_acknowledging_unblocks(self, tmp_path):
        token = recovery_token({"changed_paths": ["a.py"]})
        arm(tmp_path, token, ["unknown:previous-agent-intent"])

        with pytest.raises(RecoveryAcknowledgementRequired):
            require_acknowledged(tmp_path)

        acknowledge(tmp_path, token)
        require_acknowledged(tmp_path)  # cleared

    def test_acknowledging_stale_state_is_refused(self, tmp_path):
        arm(tmp_path, recovery_token({"changed_paths": ["a.py"]}), [])
        with pytest.raises(ValueError, match="does not match"):
            acknowledge(tmp_path, recovery_token({"changed_paths": ["b.py"]}))
        # Still armed: a rejected acknowledgement must not disarm the gate.
        with pytest.raises(RecoveryAcknowledgementRequired):
            require_acknowledged(tmp_path)

    def test_a_corrupt_marker_counts_as_armed(self, tmp_path):
        arm(tmp_path, recovery_token({"x": 1}), [])
        (tmp_path / "pending-recovery-ack.json").write_text("{not json", encoding="utf-8")

        # Treating corruption as consent would be the wrong default: the gate
        # exists precisely for the case where state cannot be trusted.
        assert pending(tmp_path) is not None
        with pytest.raises(RecoveryAcknowledgementRequired):
            require_acknowledged(tmp_path)

    def test_unknowns_are_carried_to_the_acknowledgement(self, tmp_path):
        token = recovery_token({"x": 1})
        arm(tmp_path, token, ["unknown:previous-agent-intent"])
        result = acknowledge(tmp_path, token)
        assert "unknown:previous-agent-intent" in result["unknowns"]


@pytest.mark.timeout(180)
def test_claim_is_refused_until_recovery_is_acknowledged(tmp_path, monkeypatch):
    """End-to-end through the real MCP surface."""
    repo = _repo(tmp_path)
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
    from entroly.work_graph_mcp import (
        work_acknowledge_recovery,
        work_claim,
        work_resume,
    )

    # Without a recovery outstanding, ordinary work is unaffected.
    assert work_claim(
        project=str(repo), agent_id="a", task_title="t", task_id="t1",
        scope_paths=["app.py"],
    )["status"] == "ok"

    resumed = work_resume(project=str(repo), max_evidence=16)
    context = resumed["context"]
    start = context.find("{")
    ack = json.loads(context[start:context.rfind("}") + 1])["acknowledgement"]
    assert ack["required"] is True
    assert ack["trust"] == "untrusted_recovered_work_state"
    assert "unknown:previous-agent-intent" in ack["unknowns"]

    blocked = work_claim(
        project=str(repo), agent_id="b", task_title="t2", task_id="t2",
        scope_paths=["app.py"],
    )
    assert blocked["status"] == "error"
    assert "acknowledg" in str(blocked.get("detail", "")).lower()

    wrong = work_acknowledge_recovery(project=str(repo), token="recovery:" + "0" * 32)
    assert wrong["status"] == "error"

    accepted = work_acknowledge_recovery(project=str(repo), token=ack["token"])
    assert accepted["status"] == "ok"
    assert accepted["acknowledged"] is True

    assert work_claim(
        project=str(repo), agent_id="b", task_title="t2", task_id="t2",
        scope_paths=["app.py"],
    )["status"] == "ok"


@pytest.mark.timeout(180)
def test_staged_and_worktree_states_receive_separate_commitments(tmp_path):
    """Exact binding for paths one digest cannot describe."""
    repo = _repo(tmp_path)

    # Staged, then edited again: index and worktree genuinely differ.
    (repo / "staged.py").write_text("s = 1\n", encoding="utf-8")
    _git(repo, "add", "staged.py")
    (repo / "staged.py").write_text("s = 2  # edited after add\n", encoding="utf-8")

    observation = enrich_worktree_content_digests(
        repo, discover_repository_observation(repo)
    )
    staged = next(
        change for change in observation["changes"]
        if change["path"] == "staged.py"
    )

    assert staged["index_digest"], "staged bytes were left unbound"
    assert staged["worktree_digest"], "worktree bytes were left unbound"
    assert staged["index_digest"] != staged["worktree_digest"], (
        "index and worktree differ here; identical digests would mean one of "
        "them is not actually being measured"
    )
    # content_digest keeps its original meaning: empty when no single identity
    # can honestly stand for the path.
    assert staged["content_digest"] == ""


@pytest.mark.timeout(180)
def test_conflicted_path_records_base_ours_and_theirs(tmp_path):
    repo = _repo(tmp_path)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    _git(repo, "checkout", "-q", "-b", "other")
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "other")
    _git(repo, "checkout", "-q", main)
    (repo / "app.py").write_text("x = 3\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "mine")
    merged = _git(repo, "merge", "other")
    assert merged.returncode != 0, "fixture did not produce a conflict"

    observation = enrich_worktree_content_digests(
        repo, discover_repository_observation(repo)
    )
    conflicted = next(
        change for change in observation["changes"] if change["path"] == "app.py"
    )

    stages = conflicted.get("conflict_stage_digests", {})
    assert set(stages) == {"base", "ours", "theirs"}
    assert len(set(stages.values())) == 3, "stages must be distinct blobs"
    # The worktree holds merged text with markers, which is none of the three.
    assert conflicted["worktree_digest"] not in set(stages.values())
