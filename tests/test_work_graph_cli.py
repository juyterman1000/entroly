from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from entroly import work_graph_cli as c
from entroly.work_graph_resume import attach_current_worktree_changes, recovery_unknowns


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
        self._state_dir = TemporaryDirectory()
        self.repo_dir = Path(self._state_dir.name)

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


def test_live_change_projection_is_bounded_and_marks_unknown_tail():
    view = attach_current_worktree_changes(
        {}, {"changes": [{"path": "b.py"}, {"path": "a.py"}]}, max_paths=1
    )
    assert view["current_worktree_changed_paths"] == ["a.py"]
    assert view["current_worktree_change_count"] == 2
    assert view["current_worktree_changes_truncated"] is True
    assert "unknown:current-worktree-changes-truncated" in recovery_unknowns(view)


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
    for action in ("state", "claim", "resume", "acknowledge", "handoff", "handoffs", "review", "reviews"):
        if action == "claim":
            args = parser.parse_args([action, "--agent", "codex", "--task", "Fix auth"])
        elif action == "handoff":
            args = parser.parse_args([action, "--workstream", "w", "--from-agent", "claude", "--to-agent", "codex"])
        elif action == "acknowledge":
            args = parser.parse_args([action, "--token", "recovery:abc"])
        elif action == "handoffs":
            args = parser.parse_args([action, "--to-agent", "codex"])
        elif action == "review":
            args = parser.parse_args([
                action, "--handoff-ref", "handoff:" + "a" * 64,
                "--reviewer", "codex", "--verdict", "approve", "--summary", "Reviewed",
            ])
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
    # Earlier daemon tests may emit an asynchronous diagnostic during this
    # test's setup. Check only stderr produced by the CLI invocation itself.
    capsys.readouterr()
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
    assert payload["acknowledgement"]["required"] is True
    assert payload["acknowledgement"]["unknowns"] == ["unknown:previous-agent-intent"]


def test_cli_resume_prefers_recorded_work_over_newer_placeholder(monkeypatch, capsys, tmp_path):
    fake = FakeStore()
    selected = []
    monkeypatch.setattr(c, "_store_for_path", lambda _p: fake)
    monkeypatch.setattr(c, "_passive_observation", lambda _p: {"repo_id": "repo:test"})
    fake.load = lambda: SimpleNamespace(unfinished=lambda: [
        {"node_id": "workstream:placeholder", "remaining_work": []},
        {"node_id": "workstream:real", "remaining_work": ["finish test"]},
    ])
    fake.resume = lambda workstream, **_kwargs: (
        selected.append(workstream) or {"selected_workstream": {"node_id": workstream}}
    )
    args = c.build_parser().parse_args(["--project", str(tmp_path), "--json", "resume"])

    assert c.run(args) == 0
    assert selected == ["workstream:real"]
    assert json.loads(capsys.readouterr().out)["resume"]["selected_workstream"]["node_id"] == "workstream:real"


def test_cli_recovery_blocks_claim_until_matching_acknowledgement(monkeypatch, capsys, tmp_path):
    fake = FakeStore()
    monkeypatch.setattr(c, "_store_for_path", lambda _p: fake)
    monkeypatch.setattr(c, "_passive_observation", lambda _p: {"repo_id": "repo:test"})
    fake.resume = lambda *_args, **_kwargs: {
        "selected_workstream": {"node_id": "workstream:real"},
        "changed_paths": ["app.py"],
    }
    parser = c.build_parser()
    options = ["--project", str(tmp_path), "--json"]
    assert c.run(parser.parse_args([*options, "resume"])) == 0
    token = json.loads(capsys.readouterr().out)["acknowledgement"]["token"]

    claim = parser.parse_args([*options, "claim", "--agent", "codex", "--task", "finish"])
    assert c.run(claim) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "recovery_acknowledgement_required"
    handoff = parser.parse_args([
        *options, "handoff", "--workstream", "workstream:real",
        "--from-agent", "claude", "--to-agent", "codex",
    ])
    assert c.run(handoff) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "recovery_acknowledgement_required"
    assert c.run(parser.parse_args([*options, "acknowledge", "--token", "recovery:wrong"])) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert c.run(claim) == 1
    capsys.readouterr()

    assert c.run(parser.parse_args([*options, "acknowledge", "--token", token])) == 0
    assert json.loads(capsys.readouterr().out)["acknowledged"] is True
    monkeypatch.setattr(c, "discover_repository_observation", lambda *_a, **_k: {"repo_id": "repo:test", "leases": []})
    monkeypatch.setattr(c, "enrich_worktree_content_digests", lambda _p, observation: observation)
    assert c.run(claim) == 0


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
    from entroly import work_graph_inbox

    monkeypatch.setattr(
        work_graph_inbox, "save_handoff",
        lambda store, handoff, proof: calls.append("save") or "handoff:" + "a" * 64,
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
    assert calls == ["store", "observe", "fingerprint", "persist", "handoff", "resume", "proof", "save"]
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
