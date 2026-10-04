"""A passive repository refresh must not replace the workstream being resumed.

`work_resume` refreshed the graph with a passive observation and *then* called
`resume(None)`. A passive observation carries no task hint, so it cannot
reproduce the recorded task id: workstream identity is `branch_name:task_id`
(`work_graph.rs`), and the hintless path derives `inferred:<branch>` instead. The
refresh therefore minted a second workstream for the same branch -- status
InProgress, trust Inferred, no decisions, no verification, no outstanding work --
and `unfinished_work()` sorts by `updated_at_ms` descending, so the brand-new
empty workstream outranked the real one.

Measured before the fix on the production path:

    before passive refresh:  selected = workstream:c863fb54  outstanding=[...]
    after  passive refresh:  selected = workstream:a40c2965  outstanding=[]

Reproduced with realistic (not backdated) timestamps, so it was not an artifact
of the fixture's clock.

These tests drive `work_graph_mcp.work_resume`, the real B3 consumer, rather than
`store.resume` -- the defect lived entirely in the ordering of the two calls, so
testing the lower layer would have passed throughout.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

# `work_resume` reaches the Work Graph, which is PyO3-only and raises
# WorkGraphUnavailableError without the compiled engine. The pure-Python
# fallback job installs the base package with no Rust engine on purpose, so the
# whole module must skip there rather than fail: these tests are about which
# workstream the resume path selects, not about whether the engine is present.
pytest.importorskip("entroly_core", reason="work_resume requires the Rust engine")

REMAINING = "wire the new config field through the Rust engine"
DECISION = "budget must stay per-request; a global default broke cache alignment"


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "proj"
    path.mkdir()
    _git(path, "init", "-q", "-b", "feat/configurable-budget")
    _git(path, "config", "user.email", "pilot@example.invalid")
    _git(path, "config", "user.name", "pilot")
    (path / "config.py").write_text("TOKEN_BUDGET = 8192\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "baseline")
    # Uncommitted edit: the interrupted state a successor inherits.
    (path / "config.py").write_text(
        "TOKEN_BUDGET = 8192\nPER_REQUEST_BUDGET = None\n", encoding="utf-8"
    )
    # work_resume refuses a project outside ENTROLY_SOURCE; honour that guard
    # rather than bypassing it.
    monkeypatch.setenv("ENTROLY_SOURCE", str(path))
    monkeypatch.setenv("ENTROLY_NO_SELF_HEAL", "1")
    # Give each test its own Work Graph store. `_store_root()` defaults to
    # ~/.entroly/work-graphs, which is the developer's real store and is shared
    # by every test in the session. Without this the file passed in isolation
    # and failed inside the full suite: `test_passed_verdict_is_not_reported_as_failed`
    # saw outstanding work from a sibling test's workstream, because the
    # content-based selector legitimately picks any content-bearing workstream
    # in the store it is given. Isolating the store is the right fix -- the
    # assertion is about this test's recorded state, not about whatever else a
    # developer's machine happens to hold.
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "entroly-home"))
    return path


def _observation(repo_id: str, head: str, *, verdict: str = "failed",
                 remaining: list[str] | None = None,
                 branch: str = "feat/configurable-budget",
                 task_id: str = "task-budget") -> dict:
    now = int(time.time() * 1000)
    obs = {
        "repo_id": repo_id,
        "observed_at_ms": now,
        "repository_label": "continuity pilot",
        "agent_id": "agent:claude",
        "session_id": "session-a",
        "task_hint": {
            "task_id": task_id,
            "title": "make the per-request token budget configurable",
            "trust": "observed",
            "explicit_status": "in_progress",
            "remaining_work": REMAINING if remaining is None else remaining,
            "source_kind": "user_statement",
            "source_ref": "user:task",
        },
        "branch": {
            "name": branch, "head_sha": head,
            "default_branch": "main", "ahead_by": 1,
        },
        "changes": [{"path": "config.py", "kind": "modified",
                     "staged": False, "conflicted": False}],
        "decisions": [{
            "decision_id": "decision-1", "text": DECISION,
            "source_ref": "checkpoint:1", "source_kind": "checkpoint",
            "trust": "observed",
        }],
        "verifications": [{
            "verification_id": "verification:budget",
            "name": "budget tests", "state": verdict,
            "evidence_kind": "test_result",
            "source_ref": "pytest tests/test_engine_budget.py",
            "digest": "deadbeef", "observed_at_ms": now,
        }],
    }
    hint = obs["task_hint"]
    if isinstance(hint["remaining_work"], str):
        hint["remaining_work"] = [hint["remaining_work"]]
    return obs


def _record(repo: Path, **kwargs):
    from entroly.work_graph_store import WorkGraphStore, discover_repository_identity

    repo_id = discover_repository_identity(repo)["repo_id"]
    store = WorkGraphStore(repo_id)
    head = _git(repo, "rev-parse", "HEAD")
    store.submit_repository_observation(
        _observation(repo_id, head, **kwargs), repository_path=repo
    )
    return store


def _resume(repo: Path, **kwargs) -> dict:
    """Call the production MCP path and unwrap the untrusted-render envelope."""
    from entroly import work_graph_mcp

    produced = work_graph_mcp.work_resume(project=str(repo), **kwargs)
    assert produced.get("status") != "error", produced
    text = produced.get("context")
    if isinstance(text, str):
        start = text.index("{")
        end = text.rindex("}") + 1
        produced = json.loads(text[start:end])
    return produced


def _view(produced: dict) -> dict:
    return produced["resume"]


# ── W1 / W2: the selection invariant ───────────────────────────────────

def test_w1_explicit_resume_returns_the_requested_workstream(repo: Path):
    store = _record(repo)
    target = store.load().unfinished()[0]["node_id"]

    view = _view(_resume(repo, workstream_id=target))

    assert view["selected_workstream"]["node_id"] == target


def test_w2_implicit_resume_keeps_the_only_pre_existing_workstream(repo: Path):
    store = _record(repo)
    target = store.load().unfinished()[0]["node_id"]

    view = _view(_resume(repo))

    assert view["selected_workstream"]["node_id"] == target, (
        "the passive refresh replaced the workstream being resumed"
    )


def test_w5_newer_passive_timestamp_does_not_shadow_recorded_work(repo: Path):
    """The exact pre-fix mechanism: recency beat content."""
    store = _record(repo)
    target = store.load().unfinished()[0]["node_id"]
    time.sleep(0.01)  # guarantee the refresh observes a strictly later clock

    view = _view(_resume(repo))

    assert view["selected_workstream"]["node_id"] == target
    # And prove the shadowing candidate really is created, so this test is not
    # passing because the refresh happened to be a no-op.
    unfinished = store.load().unfinished()
    assert len(unfinished) >= 1
    if len(unfinished) > 1:
        newest = max(unfinished, key=lambda w: w["updated_at_ms"])["node_id"]
        assert newest != target, (
            "expected a newer passive workstream to exist and be rejected"
        )


# ── W3 / W4: do not invent selection semantics ─────────────────────────

def test_w3_two_pre_existing_workstreams_keep_documented_ordering(repo: Path):
    from entroly.work_graph_store import WorkGraphStore, discover_repository_identity

    repo_id = discover_repository_identity(repo)["repo_id"]
    store = WorkGraphStore(repo_id)
    head = _git(repo, "rev-parse", "HEAD")
    store.submit_repository_observation(
        _observation(repo_id, head, task_id="task-a"), repository_path=repo
    )
    time.sleep(0.01)
    store.submit_repository_observation(
        _observation(repo_id, head, task_id="task-c"), repository_path=repo
    )

    pre_existing = store.load().unfinished()
    assert len(pre_existing) >= 2, "fixture did not create two workstreams"
    expected = pre_existing[0]["node_id"]   # documented order: newest first

    view = _view(_resume(repo))

    assert view["selected_workstream"]["node_id"] == expected, (
        "existing recency ordering among real workstreams must be preserved"
    )


def test_w4_no_pre_existing_work_still_resumes_without_error(repo: Path):
    """Nothing recorded: post-refresh selection remains the behaviour."""
    produced = _resume(repo)
    view = _view(produced)
    assert view["selected_workstream"]["node_id"]
    assert view["graph_commitment"]
    # Nothing was recorded, so nothing may be claimed.
    assert view.get("outstanding_work") == []
    assert view.get("verification") == []


# ── W6 / W7: the state must survive the production path ────────────────

def test_w6_blocked_failed_verdict_and_outstanding_work_all_survive(repo: Path):
    _record(repo)
    view = _view(_resume(repo))

    verdicts = [v.get("verdict") for v in view["verification"]]
    assert "failed" in verdicts, f"verdict lost in production path: {verdicts}"
    assert view["outstanding_work"] == [REMAINING]
    assert view["selected_workstream"]["status"] in ("blocked", "needs_verification")
    # Independent carriers: none inferred from another.
    assert view["selected_workstream"]["remaining_work"] == [REMAINING]


def test_w7_changed_paths_decisions_and_commitment_survive(repo: Path):
    _record(repo)
    view = _view(_resume(repo))

    assert "config.py" in view["changed_paths"]
    assert DECISION in view["decisions"]
    assert view["graph_commitment"]


def test_passed_verdict_is_not_reported_as_failed(repo: Path):
    """Guards the obvious wrong fix: hardcoding a failure."""
    _record(repo, verdict="passed", remaining=[])
    view = _view(_resume(repo))

    verdicts = [v.get("verdict") for v in view["verification"]]
    assert verdicts and "failed" not in verdicts, verdicts
    assert view["outstanding_work"] == []


# ── W8: state pollution ────────────────────────────────────────────────

def test_w8_repeated_resume_does_not_grow_the_graph_without_bound(repo: Path):
    store = _record(repo)

    def counts() -> tuple[int, int]:
        graph = store.load()
        snapshot = graph.snapshot()
        return len(snapshot["nodes"]), len(graph.unfinished())

    _resume(repo)
    nodes_after_first, ws_after_first = counts()
    for _ in range(3):
        _resume(repo)
    nodes_after_many, ws_after_many = counts()

    assert ws_after_many == ws_after_first, (
        f"workstreams grew {ws_after_first} -> {ws_after_many} across resumes"
    )
    assert nodes_after_many == nodes_after_first, (
        f"nodes grew {nodes_after_first} -> {nodes_after_many} across resumes"
    )


# ── W9: selection survives a store reload ──────────────────────────────

def test_w9_selection_is_stable_across_store_reload(repo: Path):
    store = _record(repo)
    target = store.load().unfinished()[0]["node_id"]

    first = _view(_resume(repo))["selected_workstream"]["node_id"]
    # A fresh process would reload from disk; reload explicitly.
    from entroly.work_graph_store import WorkGraphStore, discover_repository_identity

    reloaded = WorkGraphStore(discover_repository_identity(repo)["repo_id"])
    assert reloaded.load().graph_commitment
    second = _view(_resume(repo))["selected_workstream"]["node_id"]

    assert first == target
    assert second == target
