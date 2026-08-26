from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from entroly.work_graph_content_digest import enrich_worktree_content_digests


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_bytes(b"VALUE = 1\n")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    return repo


def _python_digest(repo: Path, change: dict[str, object]) -> str:
    observation = {"changes": [dict(change)]}
    enrich_worktree_content_digests(repo, observation)
    return str(observation["changes"][0].get("content_digest", ""))


def _node_digest(repo: Path, change: dict[str, object]) -> str:
    if NODE is None:
        pytest.skip("node is unavailable")
    script = r"""
const { enrichWorktreeContentDigests } = require('./entroly-wasm/js/work_graph_content_digest');
const change = JSON.parse(process.env.ENTROLY_TEST_CHANGE);
const observation = { changes: [change] };
enrichWorktreeContentDigests(process.env.ENTROLY_TEST_REPO, observation);
process.stdout.write(JSON.stringify(observation.changes[0].content_digest || ''));
"""
    env = os.environ.copy()
    env["ENTROLY_TEST_REPO"] = str(repo)
    env["ENTROLY_TEST_CHANGE"] = json.dumps(change, separators=(",", ":"))
    result = subprocess.run(
        [NODE, "-e", script],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )
    return str(json.loads(result.stdout))


def test_python_and_node_hash_identical_worktree_bytes_the_same(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    # Include non-ASCII and no trailing newline so parity is about exact bytes,
    # not text-normalization behavior in either adapter.
    (repo / "app.py").write_bytes("VALUE = 'café'".encode("utf-8"))
    change = {
        "path": "app.py",
        "kind": "modified",
        "staged": False,
        "conflicted": False,
        "content_digest": "",
    }

    python_digest = _python_digest(repo, change)
    node_digest = _node_digest(repo, change)

    assert python_digest.startswith("git-blob:")
    assert python_digest == node_digest


def test_python_and_node_share_fail_closed_and_deletion_markers(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "app.py").write_bytes(b"VALUE = 2\n")

    staged = {
        "path": "app.py",
        "kind": "modified",
        "staged": True,
        "conflicted": False,
        "content_digest": "",
    }
    assert _python_digest(repo, staged) == ""
    assert _node_digest(repo, staged) == ""

    (repo / "app.py").unlink()
    deleted = {
        "path": "app.py",
        "kind": "deleted",
        "staged": False,
        "conflicted": False,
        "content_digest": "",
    }
    assert _python_digest(repo, deleted) == "worktree:deleted"
    assert _node_digest(repo, deleted) == "worktree:deleted"
