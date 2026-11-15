"""Authority over a workstream moves only on a verified host assertion, once.

The numbered tests are the 20 cases preregistered in
``benchmarks/HANDOFF_EVIDENCE_PREREGISTRATION.md`` section 6. Every refusal
also checks that the authority log did not change.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from entroly.handoff_authority import (  # noqa: E402
    AuthorityLog,
    AuthorityRefused,
    accept_handoff,
    build_offer,
    generate_host_key,
    issue_assertion,
    load_trust,
    offer_digest,
    require_write_authority,
)
from entroly.work_graph import WorkGraph, WorkGraphUnavailableError  # noqa: E402
from entroly.work_graph_inbox import save_handoff  # noqa: E402
from entroly.work_graph_store import WorkGraphStore  # noqa: E402

AUDIENCE = "entroly-work"
ISSUER = "mock-host"
SUCCESSOR = "codex:session-b"
PREDECESSOR = "claude-code:session-a"
WORKSPACE_COMMITMENT = "workspace:test"


def _require_native() -> None:
    try:
        WorkGraph("native-probe")
    except WorkGraphUnavailableError as exc:  # pragma: no cover - depends on the install
        pytest.skip(str(exc))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


class World:
    """One repository, one workstream, one mock host key, one offer."""

    def __init__(self, tmp_path: Path) -> None:
        _require_native()
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-b", "main")
        _git(self.repo, "config", "user.email", "t@example.com")
        _git(self.repo, "config", "user.name", "T")
        (self.repo / "app.py").write_text("print('v1')\n", encoding="utf-8")
        _git(self.repo, "add", ".")
        _git(self.repo, "commit", "-m", "base")
        self.store = WorkGraphStore.for_repository(self.repo, root=tmp_path / "state")
        self.store.claim_work(self.repo, agent_id=PREDECESSOR, task_title="refactor")
        self.workstream = str(self.store.load().unfinished()[0]["node_id"])
        self.private, self.public = generate_host_key()
        self.trust_path = tmp_path / "trust.json"
        self.trust_path.write_text(json.dumps({
            "schema": "entroly.handoff-trust.v1",
            "audience": AUDIENCE,
            "issuers": {ISSUER: [self.public]},
            "max_lifetime_seconds": 3600,
        }), encoding="utf-8")
        self.trust = load_trust(str(self.trust_path))
        self.ref, self.digest, self.offer = self.make_offer()

    def make_offer(self, *, to_agent: str = SUCCESSOR, base_epoch: int = 0, from_agent: str = PREDECESSOR):
        from entroly.work_graph_resume import attach_current_worktree_changes
        from entroly.work_graph_repo import discover_repository_observation
        from entroly.work_graph_store import continuation_outstanding_refs

        observation = discover_repository_observation(self.repo)
        receipt = self.store.handoff(self.workstream, from_agent, to_agent)
        view = attach_current_worktree_changes(self.store.resume(self.workstream), observation)
        proof = self.store.continuation_proof(
            receipt,
            outstanding_work_refs=continuation_outstanding_refs(view),
            created_at_ms=int(time.time() * 1000),
        )
        offer = build_offer(
            handoff_receipt=receipt, workstream=self.workstream, from_agent=from_agent,
            to_agent=to_agent, repo_id=self.store.repo_id, base_epoch=base_epoch,
            workspace_commitment=WORKSPACE_COMMITMENT,
        )
        ref = save_handoff(self.store, receipt, proof, offer=offer)
        return ref, offer_digest(offer), offer

    def assertion(self, **overrides):
        options = {
            "issuer": ISSUER, "audience": AUDIENCE, "subject": SUCCESSOR,
            "workspace": self.store.repo_id, "task": self.workstream, "handoff": self.digest,
        }
        key = overrides.pop("private", self.private)
        options.update(overrides)
        return issue_assertion(key, **options)

    def accept(self, assertion, *, ref: str | None = None, now_ms: int | None = None):
        return accept_handoff(
            self.store.repo_dir, repo_id=self.store.repo_id, handoff_ref=ref or self.ref,
            assertion=assertion, trust=self.trust, workspace_commitment=WORKSPACE_COMMITMENT,
            now_ms=now_ms,
        )

    def history(self):
        return AuthorityLog(self.store.repo_dir, self.workstream).history()


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def _refused(world: World, assertion, code: str, **kwargs) -> None:
    before = world.history()
    with pytest.raises(AuthorityRefused) as caught:
        world.accept(assertion, **kwargs)
    assert caught.value.code == code, caught.value
    assert world.history() == before


def test_01_unknown_issuer(world: World) -> None:
    _refused(world, world.assertion(issuer="someone-else"), "unknown-issuer")


def test_02_signature_valid_only_under_an_unpinned_key(world: World) -> None:
    rogue_private, _ = generate_host_key()
    _refused(world, world.assertion(private=rogue_private), "unpinned-key")


def test_03_wrong_audience(world: World) -> None:
    _refused(world, world.assertion(audience="another-verifier"), "wrong-audience")


def test_04_successor_differs_from_the_offer(world: World) -> None:
    _refused(world, world.assertion(subject="rogue:agent"), "wrong-subject")


def test_05_wrong_task(world: World) -> None:
    _refused(world, world.assertion(task="workstream:other"), "wrong-task")


def test_06_wrong_workspace(world: World) -> None:
    _refused(world, world.assertion(workspace="repo:elsewhere"), "wrong-workspace")


def test_07_offer_altered_after_the_assertion_was_issued(world: World) -> None:
    assertion = world.assertion()
    inbox = world.store.repo_dir / "handoffs" / (world.ref.split(":", 1)[1] + ".json")
    payload = json.loads(inbox.read_text(encoding="utf-8"))
    payload["offer"]["evidence"] = [{"record_id": "evidence:" + "0" * 32, "record_digest": "sha256:" + "0" * 64}]
    inbox.write_text(json.dumps(payload), encoding="utf-8")
    _refused(world, assertion, "wrong-handoff")


def test_08_expired(world: World) -> None:
    issued = int(time.time() * 1000) - 3_000_000
    _refused(world, world.assertion(now_ms=issued, ttl_seconds=600), "expired")


def test_09_not_yet_valid(world: World) -> None:
    now = int(time.time() * 1000)
    _refused(world, world.assertion(now_ms=now, not_before_ms=now + 600_000, ttl_seconds=1800), "not-yet-valid")


def test_10_lifetime_longer_than_the_configured_maximum(world: World) -> None:
    _refused(world, world.assertion(ttl_seconds=7200), "lifetime-too-long")


def test_11_attempt_id_reused_after_a_successful_acceptance(world: World) -> None:
    first = world.assertion(attempt_id="attempt-1")
    assert world.accept(first)["outcome"] == "accepted"
    holder_assertion = first
    authority = require_write_authority(
        world.store.repo_dir, repo_id=world.store.repo_id, workstream=world.workstream,
        assertion=holder_assertion, trust=world.trust,
    )
    ref, digest, _ = world.make_offer(to_agent="gemini:c", base_epoch=authority["epoch"], from_agent=SUCCESSOR)
    replay = world.assertion(attempt_id="attempt-1", subject="gemini:c", handoff=digest)
    _refused(world, replay, "replayed-attempt", ref=ref)


def test_12_altered_signature_byte(world: World) -> None:
    assertion = world.assertion()
    signature = assertion["signature"]
    assertion["signature"] = ("0" if signature[0] != "0" else "1") + signature[1:]
    _refused(world, assertion, "bad-signature")


def test_13_missing_claim(world: World) -> None:
    assertion = world.assertion()
    del assertion["claims"]["workspace"]
    _refused(world, assertion, "missing-claim")


def test_14_unknown_extra_claim(world: World) -> None:
    assertion = world.assertion()
    assertion["claims"]["role"] = "admin"
    _refused(world, assertion, "unknown-claim")


def test_15_scope_without_accept(world: World) -> None:
    _refused(world, world.assertion(scope=["work:write"]), "scope-missing")


def test_16_outstanding_recovery_acknowledgement(world: World) -> None:
    from entroly.work_graph_recovery_ack import arm

    arm(world.store.repo_dir, "recovery:outstanding", ["unknown:previous-agent-intent"])
    _refused(world, world.assertion(), "recovery-unacknowledged")


def _accept_in_process(args: tuple) -> str:
    repo_dir, repo_id, ref, assertion, trust_path, start_at = args
    while time.time() < start_at:
        time.sleep(0.001)
    try:
        result = accept_handoff(
            repo_dir, repo_id=repo_id, handoff_ref=ref, assertion=assertion,
            trust=load_trust(trust_path), workspace_commitment=WORKSPACE_COMMITMENT,
        )
    except AuthorityRefused as exc:
        return exc.code
    return result["outcome"]


def test_17_eight_concurrent_acceptances_have_one_winner(world: World) -> None:
    assertions = [world.assertion(attempt_id=f"race-{i}") for i in range(8)]
    start_at = time.time() + 4.0
    jobs = [
        (str(world.store.repo_dir), world.store.repo_id, world.ref, a, str(world.trust_path), start_at)
        for a in assertions
    ]
    with ProcessPoolExecutor(max_workers=8, mp_context=get_context("spawn")) as pool:
        outcomes = list(pool.map(_accept_in_process, jobs))
    assert outcomes.count("accepted") == 1, outcomes
    assert outcomes.count("superseded") == 7, outcomes
    history = world.history()
    assert len(history) == 1 and history[0]["epoch"] == 1


def test_18_interruption_before_and_after_the_commit_point(world: World, monkeypatch) -> None:
    import entroly.handoff_authority as authority

    assertion = world.assertion()
    real_link = authority.os.link

    def crash(*_args, **_kwargs):
        raise OSError("simulated crash before publication")

    monkeypatch.setattr(authority.os, "link", crash)
    with pytest.raises(OSError):
        world.accept(assertion)
    assert world.history() == []
    leftovers = list((world.store.repo_dir / "authority").rglob("*.tmp"))
    assert leftovers == []

    monkeypatch.setattr(authority.os, "link", real_link)
    first = world.accept(assertion)
    assert first["outcome"] == "accepted" and first["epoch"]["epoch"] == 1
    again = world.accept(assertion)  # e.g. the caller crashed before reading the result
    assert again["outcome"] == "already-accepted"
    assert again["epoch"] == first["epoch"]
    assert len(world.history()) == 1


def test_19_fenced_predecessor_and_expired_holder_lose_write_authority(world: World) -> None:
    now = int(time.time() * 1000)
    holder = world.assertion(now_ms=now, ttl_seconds=600)
    world.accept(holder, now_ms=now)

    def write(assertion, at=None):
        return require_write_authority(
            world.store.repo_dir, repo_id=world.store.repo_id, workstream=world.workstream,
            assertion=assertion, trust=world.trust, now_ms=at,
        )

    with pytest.raises(AuthorityRefused) as no_assertion:
        write(None)
    assert no_assertion.value.code == "fenced"
    predecessor = world.assertion(subject=PREDECESSOR, attempt_id="old-session")
    with pytest.raises(AuthorityRefused) as old:
        write(predecessor)
    assert old.value.code == "fenced"
    assert write(holder, now + 1000)["holder"] == SUCCESSOR
    with pytest.raises(AuthorityRefused) as expired:
        write(holder, now + 3_600_000)
    assert expired.value.code == "holder-expired"


def test_20_valid_acceptance_moves_authority_by_one_epoch(world: World) -> None:
    assert world.history() == []
    result = world.accept(world.assertion())
    assert result["outcome"] == "accepted"
    assert result["epoch"]["epoch"] == 1
    assert result["epoch"]["holder"] == SUCCESSOR
    history = world.history()
    assert len(history) == 1
    assert history[0]["offer_digest"] == world.digest
    assert history[0]["prev_digest"] == ""


def test_a_broken_chain_is_refused_not_repaired(world: World) -> None:
    world.accept(world.assertion())
    epoch_file = next((world.store.repo_dir / "authority").rglob("epoch-000001.json"))
    record = json.loads(epoch_file.read_text(encoding="utf-8"))
    record["holder"] = "rogue:agent"
    epoch_file.write_text(json.dumps(record), encoding="utf-8")
    assert world.history()[0]["holder"] == "rogue:agent"  # single record: content is not self-signed
    ref, digest, _ = world.make_offer(to_agent="gemini:c", base_epoch=1)
    (epoch_file.parent / "epoch-000003.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AuthorityRefused) as caught:
        world.accept(world.assertion(subject="gemini:c", handoff=digest), ref=ref)
    assert caught.value.code == "authority-log-corrupt"


def test_without_pinned_keys_nothing_is_accepted(world: World) -> None:
    with pytest.raises(AuthorityRefused) as caught:
        load_trust(None, environ={})
    assert caught.value.code == "no-trust-anchors"


# ── the real entry points ───────────────────────────────────────────


def _cli(capsys, *argv: str) -> tuple[int, dict]:
    from entroly.work_graph_cli import main

    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_cli_handoff_accept_and_fence(tmp_path: Path, monkeypatch, capsys) -> None:
    _require_native()
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    for key in ("ENTROLY_HANDOFF_ASSERTION", "ENTROLY_HANDOFF_TRUST_FILE", "PYTEST_ADDOPTS"):
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    _git(repo.parent, "init", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "NOTES.md").write_text("draft\n", encoding="utf-8")
    (repo / "tests" / "test_calc.py").write_text(
        "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parents[1]))\n"
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    project = ["--project", str(repo)]

    assert _cli(capsys, *project, "claim", "--agent", PREDECESSOR, "--task", "refactor calc")[0] == 0
    _, state = _cli(capsys, *project, "state")
    workstream = state["unfinished"][0]["node_id"]
    code, run = _cli(
        capsys, *project, "evidence", "run", "--name", "calc tests", "--",
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_calc.py",
    )
    assert code == 0 and run["verdict"] == "passed", run
    code, handoff = _cli(
        capsys, *project, "handoff", "--workstream", workstream,
        "--from-agent", PREDECESSOR, "--to-agent", SUCCESSOR, "--evidence", run["record_id"],
    )
    assert code == 0, handoff
    key_file = tmp_path / "host.key"
    _, key = _cli(capsys, "host-keygen", "--private-key-out", str(key_file))
    trust = tmp_path / "trust.json"
    trust.write_text(json.dumps({
        "schema": "entroly.handoff-trust.v1", "audience": AUDIENCE, "issuers": {ISSUER: [key["public_key"]]},
    }), encoding="utf-8")
    assertion_file = tmp_path / "successor.assertion"
    code, _ = _cli(
        capsys, "issue-assertion", "--private-key-file", str(key_file), "--issuer", ISSUER,
        "--audience", AUDIENCE, "--subject", SUCCESSOR, "--workspace", handoff["offer"]["repo_id"],
        "--task", workstream, "--handoff", handoff["offer_digest"], "--out", str(assertion_file),
    )
    assert code == 0
    if os.name == "posix":
        assert (assertion_file.stat().st_mode & 0o777) == 0o600
        assert (key_file.stat().st_mode & 0o777) == 0o600

    (repo / "NOTES.md").write_text("an unrelated edit\n", encoding="utf-8")
    code, accepted = _cli(
        capsys, *project, "accept", "--handoff-ref", handoff["handoff_ref"],
        "--assertion", str(assertion_file), "--trust-file", str(trust),
    )
    assert code == 0, accepted
    assert accepted["outcome"] == "accepted" and accepted["epoch"]["holder"] == SUCCESSOR
    assert [item["decision"] for item in accepted["evidence"]] == ["reuse"]
    assert accepted["next_actions"][0].startswith("reuse")

    (repo / "calc.py").write_text("def add(a, b):\n    return a + b + 0\n", encoding="utf-8")
    code, check = _cli(capsys, *project, "evidence", "check", "--record", run["record_id"])
    assert code == 3 and check["decision"] == "revalidate"  # 0 is reserved for reuse
    assert check["provenance"]["observed_by_entroly"] is True
    assert check["provenance"]["recorded_by_is_authenticated"] is False

    code, fenced = _cli(
        capsys, *project, "handoff", "--workstream", workstream,
        "--from-agent", PREDECESSOR, "--to-agent", "rogue:agent",
    )
    assert code == 1 and fenced["code"] == "fenced"

    from entroly.work_graph_mcp import work_handoff

    monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
    via_mcp = work_handoff(from_agent=PREDECESSOR, to_agent="rogue:agent", workstream_id=workstream)
    assert via_mcp["status"] == "error" and via_mcp["code"] == "fenced"

    code, onward = _cli(
        capsys, *project, "handoff", "--workstream", workstream, "--from-agent", SUCCESSOR,
        "--to-agent", "gemini:c", "--assertion", str(assertion_file), "--trust-file", str(trust),
    )
    assert code == 0 and onward["offer"]["base_epoch"] == 1

    code, authority = _cli(capsys, *project, "authority", "--workstream", workstream)
    assert code == 0 and authority["epoch"] == 1 and authority["current"]["holder"] == SUCCESSOR


def test_cli_accept_with_missing_evidence_says_rerun(tmp_path: Path, monkeypatch, capsys) -> None:
    """The record handed over was deleted: authority moves, the result is not reused."""
    _require_native()
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    (repo / "ok.py").write_text("assert True\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    project = ["--project", str(repo)]
    _cli(capsys, *project, "claim", "--agent", PREDECESSOR, "--task", "t")
    workstream = _cli(capsys, *project, "state")[1]["unfinished"][0]["node_id"]
    run = _cli(capsys, *project, "evidence", "run", "--", sys.executable, "ok.py")[1]
    handoff = _cli(
        capsys, *project, "handoff", "--workstream", workstream, "--from-agent", PREDECESSOR,
        "--to-agent", SUCCESSOR, "--evidence", run["record_id"],
    )[1]
    store = WorkGraphStore.for_repository(repo)
    (store.repo_dir / "evidence" / (run["record_id"].split(":", 1)[1] + ".json")).unlink()
    private, public = generate_host_key()
    trust = tmp_path / "trust.json"
    trust.write_text(json.dumps({
        "schema": "entroly.handoff-trust.v1", "audience": AUDIENCE, "issuers": {ISSUER: [public]},
    }), encoding="utf-8")
    assertion = issue_assertion(
        private, issuer=ISSUER, audience=AUDIENCE, subject=SUCCESSOR,
        workspace=handoff["offer"]["repo_id"], task=workstream, handoff=handoff["offer_digest"],
    )
    assertion_file = tmp_path / "a.json"
    assertion_file.write_text(json.dumps(assertion), encoding="utf-8")
    code, accepted = _cli(
        capsys, *project, "accept", "--handoff-ref", handoff["handoff_ref"],
        "--assertion", str(assertion_file), "--trust-file", str(trust),
    )
    assert code == 0, accepted
    assert accepted["evidence"][0]["decision"] == "unknown"
    assert "record-unavailable" in accepted["evidence"][0]["reasons"]
    assert accepted["next_actions"][0].startswith("rerun")


# ── regressions from the independent security review ────────────────


def test_a_caller_supplied_trust_file_makes_authority_advisory(world: World) -> None:
    """A local process can pin its own key; the acceptance must say so."""
    rogue_private, rogue_public = generate_host_key()
    rogue_trust = world.trust_path.with_name("rogue-trust.json")
    rogue_trust.write_text(json.dumps({
        "schema": "entroly.handoff-trust.v1", "audience": AUDIENCE, "issuers": {ISSUER: [rogue_public]},
    }), encoding="utf-8")
    trust = load_trust(str(rogue_trust))
    assert trust.host_protected is False
    result = accept_handoff(
        world.store.repo_dir, repo_id=world.store.repo_id, handoff_ref=world.ref,
        assertion=world.assertion(private=rogue_private), trust=trust,
        workspace_commitment=WORKSPACE_COMMITMENT,
    )
    assert result["epoch"]["authority"] == "advisory"
    assert world.history()[0]["trust_digest"] == trust.digest
    assert world.history()[0]["key_id"] != __import__("entroly.handoff_authority", fromlist=["key_id"]).key_id(
        world.public
    )


def test_a_future_issue_time_does_not_stretch_the_lifetime(world: World) -> None:
    now = int(time.time() * 1000)
    stretched = world.assertion(now_ms=now + 3_000_000, not_before_ms=now - 60_000, ttl_seconds=600)
    _refused(world, stretched, "not-yet-valid")


def test_evidence_run_exits_with_the_verifications_own_code(tmp_path: Path, monkeypatch, capsys) -> None:
    """Wrapping a failing test run must not turn a red build green."""
    _require_native()
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "fail.py").write_text("raise SystemExit(4)\n", encoding="utf-8")
    code, run = _cli(capsys, "--project", str(repo), "evidence", "run", "--", sys.executable, "fail.py")
    assert code == 4 and run["verdict"] == "failed"
    code, check = _cli(capsys, "--project", str(repo), "evidence", "check", "--record", run["record_id"])
    assert code == 0 and check["decision"] == "reuse"  # the failure still applies
