from __future__ import annotations

from types import SimpleNamespace

from entroly import work_graph_cli as c


class FakeGraph:
    repo_id = "repo:test"

    def summary(self):
        return {"event_count": 1}

    def coordination(self, now):
        return {"as_of_ms": now}

    def unfinished(self):
        return []


class FakeStore:
    def __init__(self):
        self.observation = None

    def load(self):
        return FakeGraph()

    def submit_observation(self, observation):
        self.observation = observation
        return FakeGraph()

    def resume(self, *args, **kwargs):
        return {"ok": True}

    def handoff(self, *args, **kwargs):
        return {"ok": True}


def test_cli_claim_uses_user_statement(monkeypatch, capsys, tmp_path):
    fake = FakeStore()
    monkeypatch.setattr(c, "_store_for_path", lambda _p: fake)
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


def test_standalone_parser_exposes_all_work_actions():
    parser = c.build_parser()
    for action in ("state", "claim", "resume", "handoff"):
        if action == "claim":
            args = parser.parse_args([action, "--agent", "codex", "--task", "Fix auth"])
        elif action == "handoff":
            args = parser.parse_args([action, "--workstream", "w", "--from-agent", "claude", "--to-agent", "codex"])
        else:
            args = parser.parse_args([action])
        assert args.work_action == action


def test_cli_resume_refreshes_repo_before_recovery(monkeypatch, tmp_path):
    fake = FakeStore()
    calls = []
    monkeypatch.setattr(c, "_store_for_path", lambda _p: fake)
    monkeypatch.setattr(c, "discover_repository_observation", lambda _p: calls.append("observe") or {"repo_id": "repo:test"})
    fake.resume = lambda *args, **kwargs: calls.append("resume") or {"ok": True}
    args = SimpleNamespace(
        work_action="resume",
        json_output=True,
        project=str(tmp_path),
        workstream="",
        max_evidence=32,
    )
    assert c.run(args) == 0
    assert calls == ["observe", "resume"]
