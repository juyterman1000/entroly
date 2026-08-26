from __future__ import annotations

import json
from types import SimpleNamespace

from entroly import work_graph_cli as c


class FakeGraph:
    repo_id = "repo:test"

    def summary(self):
        return {"event_count": 1}

    def coordination(self, now):
        return {"as_of_ms": now, "conflicts": []}

    def unfinished(self):
        return []


class FakeStore:
    def __init__(self):
        self.observation = None

    def load(self):
        return FakeGraph()

    def submit_repository_observation(self, observation, *, repository_path=None):
        self.observation = observation
        return FakeGraph()

    def resume(self, *args, **kwargs):
        return {"ok": True}

    def handoff(self, *args, **kwargs):
        return {"ok": True}

    def reconstructed_continuation_proof(self, workstream, to_agent, **manifest):
        return {
            "workstream_id": workstream,
            "to_agent": to_agent,
            "from_agent": "",
            "handoff_commitment": "",
            "manifest": manifest,
        }

    def continuation_proof(self, handoff, **manifest):
        return {"handoff": handoff, "manifest": manifest}


def test_cli_claim_uses_user_statement(monkeypatch, capsys, tmp_path):
    fake = FakeStore()
    fingerprinted = []
    monkeypatch.setattr(c, "_store_for_path", lambda _p: fake)
    monkeypatch.setattr(
        c,
        "enrich_worktree_content_digests",
        lambda path, observation: fingerprinted.append((path, observation)),
    )
    monkeypatch.setattr(
        c,
        "discover_repository_observation",
        lambda _path, **kwargs: {
            "repo_id": "repo:test",
            "task_hint": kwargs["task_hint"],
            "leases": [],
        },
    )
    args = SimpleNamespace(
        work_action="claim",
        json_output=True,
        project=str(tmp_path),
        agent="claude",
        task="Fix auth",
        task_id="",
        session="",
        path=[],
        symbol=[],
        ttl=900.0,
    )
    assert c.run(args) == 0
    assert fake.observation["task_hint"]["source_kind"] == "user_statement"
    assert fingerprinted == [(tmp_path, fake.observation)]


def test_standalone_parser_exposes_all_work_actions():
    parser = c.build_parser()
    assert parser.prog == "entroly-work"
    for action in ("state", "claim", "resume", "handoff"):
        if action == "claim":
            args = parser.parse_args([action, "--agent", "codex", "--task", "Fix auth"])
        elif action == "handoff":
            args = parser.parse_args([action, "--workstream", "w", "--from-agent", "claude", "--to-agent", "codex"])
        else:
            args = parser.parse_args([action])
        assert args.work_action == action

    resume = parser.parse_args(["resume", "--to-agent", "codex"])
    assert resume.to_agent == "codex"


def test_cli_human_state_is_compact_and_not_raw_json(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(c, "_store_for_path", lambda _p: FakeStore())
    args = SimpleNamespace(
        work_action="state",
        json_output=False,
        project=str(tmp_path),
    )
    assert c.run(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Entroly Work Graph" in captured.out
    assert "Repository   repo:test" in captured.out
    assert "Events       1" in captured.out
    assert "Unfinished   0" in captured.out
    assert "Conflicts    0" in captured.out
    assert "No unfinished work is currently recorded." in captured.out
    assert '"status"' not in captured.out
    assert "{" not in captured.out


def test_cli_json_state_remains_machine_readable(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(c, "_store_for_path", lambda _p: FakeStore())
    args = SimpleNamespace(
        work_action="state",
        json_output=True,
        project=str(tmp_path),
    )
    assert c.run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["repo_id"] == "repo:test"
    assert payload["summary"] == {"event_count": 1}
    assert payload["unfinished"] == []


def test_cli_human_error_is_actionable(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        c,
        "_store_for_path",
        lambda _p: (_ for _ in ()).throw(ValueError("bad project")),
    )
    args = SimpleNamespace(
        work_action="state",
        json_output=False,
        project=str(tmp_path),
    )
    assert c.run(args) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Entroly Work Graph error [invalid_work_graph_request]: bad project" in captured.err


def test_cli_resume_fingerprints_repo_before_recovery(monkeypatch, tmp_path):
    fake = FakeStore()
    calls = []
    observation = {
        "repo_id": "repo:test",
        "changes": [{"path": "app.py", "content_digest": ""}],
    }
    monkeypatch.setattr(c, "_store_for_path", lambda _p: calls.append("store") or fake)
    monkeypatch.setattr(
        c,
        "discover_repository_observation",
        lambda _p: calls.append("observe") or observation,
    )

    def fingerprint(_path, obs):
        calls.append("fingerprint")
        obs["changes"][0]["content_digest"] = "git-blob:abc"
        return obs

    monkeypatch.setattr(c, "enrich_worktree_content_digests", fingerprint)
    fake.submit_repository_observation = lambda obs, repository_path=None: (
        calls.append("persist") or setattr(fake, "observation", obs) or FakeGraph()
    )
    fake.resume = lambda *args, **kwargs: calls.append("resume") or {"ok": True}
    args = SimpleNamespace(
        work_action="resume",
        json_output=True,
        project=str(tmp_path),
        workstream="",
        max_evidence=32,
    )
    assert c.run(args) == 0
    assert calls == ["store", "observe", "fingerprint", "persist", "resume"]
    assert fake.observation["changes"][0]["content_digest"] == "git-blob:abc"


def test_cli_resume_can_emit_no_handoff_proof(monkeypatch, capsys, tmp_path):
    fake = FakeStore()
    monkeypatch.setattr(c, "_store_for_path", lambda _p: fake)
    monkeypatch.setattr(c, "_passive_observation", lambda _path: {"repo_id": "repo:test"})
    fake.resume = lambda *_args, **_kwargs: {
        "selected_workstream": {"node_id": "workstream:1"},
        "changed_paths": ["app.py"],
        "failures": [],
    }
    args = SimpleNamespace(
        work_action="resume",
        json_output=True,
        project=str(tmp_path),
        workstream="",
        max_evidence=32,
        to_agent="codex",
    )

    assert c.run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["continuation_proof"]["to_agent"] == "codex"
    assert payload["continuation_proof"]["from_agent"] == ""


def test_cli_clean_claim_uses_task_id_in_continuation_proof(monkeypatch, capsys, tmp_path):
    fake = FakeStore()
    monkeypatch.setattr(c, "_store_for_path", lambda _p: fake)
    monkeypatch.setattr(c, "_passive_observation", lambda _path: {"repo_id": "repo:test"})
    fake.resume = lambda *_args, **_kwargs: {
        "selected_workstream": {
            "node_id": "workstream:1",
            "task_ids": ["task:clean", "task:clean"],
        },
        "changed_paths": [],
        "failures": [],
    }
    args = SimpleNamespace(
        work_action="resume",
        json_output=True,
        project=str(tmp_path),
        workstream="",
        max_evidence=32,
        to_agent="codex",
    )

    assert c.run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["continuation_proof"]["manifest"]["outstanding_work_refs"] == [
        "task:clean"
    ]


def test_cli_resume_validates_before_store_or_repository_refresh(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(c, "_store_for_path", lambda _p: calls.append("store") or FakeStore())
    monkeypatch.setattr(
        c,
        "discover_repository_observation",
        lambda _p: calls.append("observe") or {"repo_id": "repo:test"},
    )
    monkeypatch.setattr(
        c,
        "enrich_worktree_content_digests",
        lambda _p, observation: calls.append("fingerprint") or observation,
    )
    args = SimpleNamespace(
        work_action="resume",
        json_output=True,
        project=str(tmp_path),
        workstream="",
        max_evidence=-1,
    )
    assert c.run(args) == 1
    assert calls == []


def test_cli_handoff_fingerprints_repo_before_receipt(monkeypatch, tmp_path):
    fake = FakeStore()
    calls = []
    observation = {
        "repo_id": "repo:test",
        "changes": [{"path": "app.py", "content_digest": ""}],
    }
    monkeypatch.setattr(c, "_store_for_path", lambda _p: calls.append("store") or fake)
    monkeypatch.setattr(
        c,
        "discover_repository_observation",
        lambda _p: calls.append("observe") or observation,
    )

    def fingerprint(_path, obs):
        calls.append("fingerprint")
        obs["changes"][0]["content_digest"] = "git-blob:def"
        return obs

    monkeypatch.setattr(c, "enrich_worktree_content_digests", fingerprint)
    fake.submit_repository_observation = lambda obs, repository_path=None: (
        calls.append("persist") or setattr(fake, "observation", obs) or FakeGraph()
    )
    fake.handoff = lambda *args, **kwargs: calls.append("handoff") or {
        "workstream_id": "workstream:1",
        "from_agent": "claude",
        "to_agent": "codex",
    }
    fake.resume = lambda *_args, **_kwargs: calls.append("resume") or {
        "changed_paths": ["app.py"],
        "failures": [],
    }
    original_proof = fake.continuation_proof
    fake.continuation_proof = (
        lambda handoff, **manifest: calls.append("proof")
        or original_proof(handoff, **manifest)
    )
    args = SimpleNamespace(
        work_action="handoff",
        json_output=True,
        project=str(tmp_path),
        workstream="workstream:1",
        from_agent="claude",
        to_agent="codex",
    )
    assert c.run(args) == 0
    assert calls == ["store", "observe", "fingerprint", "persist", "handoff", "resume", "proof"]
    assert fake.observation["changes"][0]["content_digest"] == "git-blob:def"


def test_cli_handoff_validates_before_store_construction(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(c, "_store_for_path", lambda _p: calls.append("store") or FakeStore())
    monkeypatch.setattr(
        c,
        "enrich_worktree_content_digests",
        lambda _p, observation: calls.append("fingerprint") or observation,
    )
    args = SimpleNamespace(
        work_action="handoff",
        json_output=True,
        project=str(tmp_path),
        workstream="",
        from_agent="claude",
        to_agent="codex",
    )
    assert c.run(args) == 1
    assert calls == []


def test_cli_claim_validates_before_store_construction(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(c, "_store_for_path", lambda _p: calls.append("store") or FakeStore())
    args = SimpleNamespace(
        work_action="claim",
        json_output=True,
        project=str(tmp_path),
        agent="",
        task="Fix auth",
        task_id="",
        session="",
        path=[],
        symbol=[],
        ttl=900.0,
    )
    assert c.run(args) == 1
    assert calls == []
