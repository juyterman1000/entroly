"""A new CLI process must recover an interrupted claim and enforce its gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from entroly.work_graph import WorkGraph, WorkGraphUnavailableError


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _cli(repo: Path, env: dict[str, str], *args: str) -> tuple[int, dict]:
    result = subprocess.run(
        [sys.executable, "-m", "entroly.work_graph_cli", "--project", str(repo), "--json", *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.stdout, result.stderr
    return result.returncode, json.loads(result.stdout)


def test_new_process_recovers_claim_and_requires_acknowledgement(tmp_path: Path) -> None:
    try:
        WorkGraph("native-probe")
    except WorkGraphUnavailableError as exc:
        pytest.skip(str(exc))

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "checkout", "-b", "feature/interrupted-auth")
    env = dict(os.environ, ENTROLY_DIR=str(tmp_path / "state"))

    code, claimed = _cli(
        repo, env, "claim", "--agent", "claude", "--task", "Fix auth", "--path", "app.py"
    )
    assert code == 0, claimed
    code, state = _cli(repo, env, "state")
    assert code == 0, state
    claimed_workstream = state["unfinished"][0]["node_id"]

    # The first agent stops without leaving a handoff or a remaining-work note.
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    code, recovered = _cli(repo, env, "resume", "--to-agent", "codex")
    assert code == 0, recovered
    assert recovered["resume"]["selected_workstream"]["node_id"] == claimed_workstream
    assert "app.py" in recovered["resume"]["current_worktree_changed_paths"]
    assert "app.py" in recovered["continuation_proof"]["outstanding_work_refs"]
    assert recovered["acknowledgement"]["required"] is True
    token = recovered["acknowledgement"]["token"]

    code, blocked = _cli(repo, env, "claim", "--agent", "codex", "--task", "Finish auth")
    assert code == 1
    assert blocked["error"] == "recovery_acknowledgement_required"
    (repo / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    code, stale = _cli(repo, env, "acknowledge", "--token", token)
    assert code == 1
    assert "repository changed after recovery" in stale["detail"]
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    code, acknowledged = _cli(repo, env, "acknowledge", "--token", token)
    assert code == 0, acknowledged
    assert acknowledged["acknowledged"] is True
    code, successor = _cli(repo, env, "claim", "--agent", "codex", "--task", "Finish auth")
    assert code == 0, successor


def test_new_process_exposes_handoff_and_source_bound_review(tmp_path: Path) -> None:
    try:
        WorkGraph("native-probe")
    except WorkGraphUnavailableError as exc:
        pytest.skip(str(exc))

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "checkout", "-b", "feature/review")
    env = dict(os.environ, ENTROLY_DIR=str(tmp_path / "state"))

    code, claimed = _cli(repo, env, "claim", "--agent", "claude", "--task", "Update value")
    assert code == 0, claimed
    code, state = _cli(repo, env, "state")
    assert code == 0, state
    workstream = state["unfinished"][0]["node_id"]
    source.write_text("VALUE = 2\n", encoding="utf-8")

    code, sent = _cli(
        repo, env, "handoff", "--workstream", workstream,
        "--from-agent", "claude", "--to-agent", "codex",
    )
    assert code == 0, sent
    ref = sent["handoff_ref"]
    assert ref.startswith("handoff:")
    code, inbox = _cli(repo, env, "handoffs", "--to-agent", "codex")
    assert code == 0, inbox
    assert inbox["handoffs"][0]["ref"] == ref
    assert inbox["handoffs"][0]["current"] is True

    code, recovered = _cli(repo, env, "resume", "--workstream", workstream, "--to-agent", "codex")
    assert code == 0, recovered
    code, accepted = _cli(repo, env, "acknowledge", "--token", recovered["acknowledgement"]["token"])
    assert code == 0, accepted

    code, recorded = _cli(
        repo, env, "review", "--handoff-ref", ref, "--reviewer", "codex",
        "--verdict", "changes_requested", "--summary", "Check the value",
        "--finding", "app.py", "1", "high", "Value changed",
        "--check", "focused test", "0",
    )
    assert code == 0, recorded
    assert recorded["review"]["trust_boundary"].startswith("agent_statement")
    assert recorded["review"]["findings"][0]["source_digest"].startswith("sha256:")
    code, listed = _cli(repo, env, "reviews", "--workstream", workstream)
    assert code == 0, listed
    assert listed["reviews"][0]["current"] is True

    source.write_text("VALUE = 3\n", encoding="utf-8")
    code, stale = _cli(repo, env, "reviews", "--workstream", workstream)
    assert code == 0, stale
    assert stale["reviews"][0]["current"] is False
    code, rejected = _cli(
        repo, env, "review", "--handoff-ref", ref, "--reviewer", "codex",
        "--verdict", "approve", "--summary", "Looks good",
    )
    assert code == 1
    assert "stale" in rejected["detail"]

def test_host_accepts_only_the_current_refactor_handoff(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from entroly.receipt_attestation import AttestationKey, AttestationLog

    try:
        WorkGraph("authentication-probe")
    except WorkGraphUnavailableError as exc:
        pytest.skip(str(exc))
    repo = tmp_path / "refactor"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    files = {"app.py": "from dependency import VALUE\n", "dependency.py": "VALUE = 1\n",
             "pytest.ini": "[pytest]\n", "README.md": "Example notes\n"}
    for path, data in files.items():
        (repo / path).write_text(data, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Initial")
    key = AttestationKey.generate()
    env = dict(os.environ, ENTROLY_DIR=str(tmp_path / "state"),
               ENTROLY_HANDOFF_ISSUER="test-host", ENTROLY_HANDOFF_PUBLIC_KEY=key.public_hex())

    def run(*args):
        code, payload = _cli(repo, env, *args)
        assert code == 0, payload
        return payload

    run("claim", "--agent", "writer", "--task", "Refactor value lookup", "--path", "app.py")
    workstream = run("state")["unfinished"][0]["node_id"]
    ref = run("handoff", "--workstream", workstream, "--from-agent", "writer",
              "--to-agent", "successor")["handoff_ref"]
    claims = run("handoff-challenge", "--handoff-ref", ref)["handoff_challenge"]
    now = time.time_ns() // 1_000_000

    def sign(name, signer=key, **updates):
        body = {**claims, "issuer": "test-host", "issued_at_ms": now,
                "expires_at_ms": now + 120_000, **updates}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(AttestationLog(signer).append(body).to_dict()), encoding="utf-8")
        return str(path)

    def accept(path, environment=env):
        return _cli(repo, environment, "accept-handoff", "--handoff-ref", ref, "--assertion", path)

    first = sign("first")
    assert accept(sign("wrong-key", AttestationKey.generate()))[0] == 1
    assert accept(sign("wrong-subject", successor_id="another-agent"))[0] == 1
    assert accept(sign("wrong-audience", audience="other-service"))[0] == 1
    assert accept(sign("expired", issued_at_ms=now - 2000, expires_at_ms=now - 1000))[0] == 1
    assert accept(first, {**env, "ENTROLY_HANDOFF_PUBLIC_KEY": ""})[0] == 1
    for path in files:
        target = repo / path
        target.write_text(files[path] + "# changed\n", encoding="utf-8")
        blocked, detail = accept(first)
        assert blocked == 1 and "stale" in detail["detail"]
        target.write_text(files[path], encoding="utf-8")

    second = sign("second", issued_at_ms=now - 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(accept, (first, second)))
    assert sorted(code for code, _ in outcomes) == [0, 1]
    winner = first if outcomes[0][0] == 0 else second
    receipt = accept(winner)[1]["handoff_acceptance"]
    assert receipt["successor_id"] == "successor"
    assert receipt["verification_evidence"] == "revalidate_before_reuse"
    assert accept(winner)[1]["handoff_acceptance"] == receipt

    evidence = {"check_sha256": "a" * 64, "prior_pass_sha256": "b" * 64,
                "prior_inputs_sha256": "c" * 64, "current_inputs_sha256": "c" * 64}
    for current, action, reason in (
        ("c" * 64, "reuse", "inputs_match"),
        ("d" * 64, "rerun", "inputs_changed"),
        (None, "rerun", "inputs_unavailable"),
    ):
        ref = run("handoff", "--workstream", workstream, "--from-agent", "writer",
                  "--to-agent", "successor")["handoff_ref"]
        claims = run("handoff-challenge", "--handoff-ref", ref)["handoff_challenge"]
        verification = {**evidence, "current_inputs_sha256": current}
        invalid = sign("invalid-" + reason, verification={**verification, "action": "reuse"})
        assert accept(invalid)[0] == 1
        winner = sign("valid-" + reason, verification=verification)
        code, output = accept(winner)
        assert code == 0, output
        decision = output["handoff_acceptance"]["verification"]
        assert decision["action"] == action and decision["reason"] == reason
        assert decision["authority"] == "trusted_host"

    run("claim", "--agent", "other", "--task", "Refresh graph evidence", "--path", "README.md")
    blocked, detail = accept(winner)
    assert blocked == 1 and "revision is no longer current" in detail["detail"]
