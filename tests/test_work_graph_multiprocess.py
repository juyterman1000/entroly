from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
from pathlib import Path
from queue import Empty

import pytest

from entroly.work_graph import WorkGraph, WorkGraphUnavailableError
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


def _claim_worker(
    repo: str,
    state_root: str,
    agent: str,
    task: str,
    scope: str,
    observed_at_ms: int,
    start: mp.synchronize.Event,
    results: mp.Queue,
) -> None:
    try:
        start.wait(timeout=10)
        store = WorkGraphStore.for_repository(
            repo,
            root=state_root,
            lock_timeout_seconds=10.0,
            stale_lock_seconds=30.0,
        )
        graph, lease_id = store.claim_work(
            repo,
            agent_id=agent,
            task_title=task,
            task_id=task,
            scope_paths=[scope],
            observed_at_ms=observed_at_ms,
        )
        results.put(("ok", agent, lease_id, graph.event_count, graph.graph_commitment))
    except BaseException as exc:  # pragma: no cover - only visible on child failure
        results.put(("error", agent, type(exc).__name__, str(exc)))


def test_concurrent_agent_processes_merge_without_lost_work(
    tmp_path: Path,
) -> None:
    """Independent Claude/Codex-like processes must converge on one graph.

    This is deliberately process-level rather than two in-process Store objects:
    it exercises the exclusive-create lock, latest-state reload, Rust merge,
    atomic replace, and conflict materialization under a real race.
    """

    _require_native()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    # `src/auth` is a package here, not a module, and that matters. The
    # engine's `paths_overlap` treats two scopes as overlapping only when they
    # are equal or one is a parent of the other at a `/` boundary -- so
    # `src/auth` does NOT overlap `src/auth.py`. That is correct: they are
    # different paths, and a bare string prefix would also make `src/auth`
    # collide with `src/authorization.py`. An earlier version of this test
    # claimed `src/auth` against a fixture that only ever contained
    # `src/auth.py`, so no conflict could be materialized and the assertion
    # further down failed. The engine was right; the fixture was not.
    (repo / "src" / "auth").mkdir(parents=True)
    (repo / "src" / "auth" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "src" / "auth" / "tokens.py").write_text("TOKEN = 1\n", encoding="utf-8")
    _git(repo, "add", "src/auth")
    _git(repo, "commit", "-m", "baseline")
    _git(repo, "checkout", "-b", "feature/shared-work")
    (repo / "src" / "auth" / "tokens.py").write_text("TOKEN = 2\n", encoding="utf-8")

    state_root = tmp_path / "shared-state"
    ctx = mp.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_claim_worker,
            args=(
                str(repo),
                str(state_root),
                "claude",
                "auth-implementation",
                "src/auth",
                1_000,
                start,
                results,
            ),
        ),
        ctx.Process(
            target=_claim_worker,
            args=(
                str(repo),
                str(state_root),
                "codex",
                "auth-tests",
                "src/auth/tokens.py",
                1_001,
                start,
                results,
            ),
        ),
    ]

    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("Work Graph child process hung while contending for shared state")
        assert process.exitcode == 0

    child_results = []
    for _ in processes:
        try:
            child_results.append(results.get(timeout=5))
        except Empty:
            pytest.fail("Work Graph child process exited without reporting a result")
    errors = [item for item in child_results if item[0] != "ok"]
    assert not errors, errors
    assert {item[1] for item in child_results} == {"claude", "codex"}
    assert len({item[2] for item in child_results}) == 2

    # A third process/agent view must see both events from the durable graph.
    store = WorkGraphStore.for_repository(
        repo,
        root=state_root,
        lock_timeout_seconds=5.0,
        stale_lock_seconds=30.0,
    )
    graph = store.load()
    assert graph.event_count >= 2
    assert graph.graph_commitment

    report = graph.coordination(2_000)
    assert report["active_leases"] == 2
    assert len(report["conflicts"]) == 1
    conflict = report["conflicts"][0]
    assert {conflict["agent_a"], conflict["agent_b"]} == {"claude", "codex"}
    assert conflict["overlapping_paths"]

    # Atomic persistence should leave no partial-write debris behind.
    assert not list(state_root.rglob(".state-*.tmp"))
    assert not list(state_root.rglob(".lock"))
    if os.name == "posix":
        state_files = list(state_root.rglob("state.json"))
        assert len(state_files) == 1
        assert state_files[0].stat().st_mode & 0o777 == 0o600
