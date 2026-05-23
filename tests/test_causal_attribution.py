"""Regression tests for PRISM credit-assignment fix.

The user-reported bug: when ``record_outcome(success=True)`` is called
with fragment IDs the caller passes in, those fragments get reinforced
EVEN IF the user actually solved the task by editing a different file
that retrieval never surfaced. This test pins the corrected behaviour:

  • verified hits  — fragments whose source files were modified between
                     optimize and outcome → strong positive credit.
  • unverified     — fragments retrieved but whose source was untouched
                     → ABSTAIN (no PRISM update at all).
  • should_have    — files modified but never retrieved → emitted as a
                     learning event so PRISM can boost those fragments.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from entroly.causal_attribution import (
    CausalCredit,
    RetrievalSnapshot,
    RetrievedFragment,
    SnapshotStore,
    attribute,
    build_snapshot,
    causal_credit_enabled,
    reset_global_store_for_tests,
)


# ── git fixture (real init, no network) ───────────────────────────────


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(cwd), text=True
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway git repo with three committed source files."""
    _git("init", "-q", "-b", "main", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "test", cwd=tmp_path)
    _git("config", "commit.gpgsign", "false", cwd=tmp_path)

    for name in ("retrieved_a.py", "retrieved_b.py", "verification_engine.py"):
        (tmp_path / name).write_text(f"# initial content of {name}\n")
    _git("add", "-A", cwd=tmp_path)
    _git("commit", "-q", "-m", "init", cwd=tmp_path)
    return tmp_path


# ── core scenarios ────────────────────────────────────────────────────


def test_off_target_retrieval_is_not_reinforced(repo: Path):
    """The user's exact bug.

    Retrieval surfaces fragments from files A and B. The user edits
    file C (verification_engine.py). The caller naively passes the
    A/B fragment IDs into record_outcome(success=True).

    Before fix: A and B both get +1 reinforcement.
    After fix:  A and B land in `unverified` (abstain). C-derived
                fragments are reported via `should_have_retrieved`.
    """
    snap = RetrievalSnapshot(
        request_id="req-1",
        repo_root=str(repo),
        git_head=_git("rev-parse", "HEAD", cwd=repo).strip(),
        dirty_at_start=frozenset(),
        retrieved=(
            RetrievedFragment("frag_A1", "retrieved_a.py"),
            RetrievedFragment("frag_A2", "retrieved_a.py"),
            RetrievedFragment("frag_B1", "retrieved_b.py"),
        ),
    )

    # The user actually edits a totally different file.
    (repo / "verification_engine.py").write_text("# fixed!\n")

    credit = attribute(snap, ["frag_A1", "frag_A2", "frag_B1"])

    assert credit.is_causal, f"should not have fallen back: {credit.fallback_reason}"
    # No fragment from the off-target retrieval should be marked verified.
    assert credit.verified_hits == [], \
        f"off-target ids must not be verified, got {credit.verified_hits}"
    # All three caller-passed ids should land in unverified (abstain).
    assert sorted(credit.unverified) == ["frag_A1", "frag_A2", "frag_B1"]
    # The actually-edited file shows up as a retrieval miss.
    assert "verification_engine.py" in credit.should_have_retrieved


def test_on_target_retrieval_is_credited(repo: Path):
    """Inverse case: retrieval surfaced the right file, user edited it.

    The fragment whose source was actually modified MUST get verified
    credit. The other one stays unverified.
    """
    snap = RetrievalSnapshot(
        request_id="req-2",
        repo_root=str(repo),
        git_head=_git("rev-parse", "HEAD", cwd=repo).strip(),
        dirty_at_start=frozenset(),
        retrieved=(
            RetrievedFragment("frag_A1", "retrieved_a.py"),
            RetrievedFragment("frag_B1", "retrieved_b.py"),
        ),
    )
    # Edit one of the retrieved files
    (repo / "retrieved_a.py").write_text("# real fix landed here\n")

    credit = attribute(snap, ["frag_A1", "frag_B1"])

    assert credit.verified_hits == ["frag_A1"]
    assert credit.unverified == ["frag_B1"]
    assert credit.should_have_retrieved == []  # no blind spot this time


def test_read_only_task_emits_no_signal(repo: Path):
    """Success without any file modification → ABSTAIN entirely.

    A "answer my question" task creates no diff. Reinforcing retrieval
    in this case would inject pure noise (the legacy bug). The
    correct behaviour is to record nothing.
    """
    snap = RetrievalSnapshot(
        request_id="req-3",
        repo_root=str(repo),
        git_head=_git("rev-parse", "HEAD", cwd=repo).strip(),
        dirty_at_start=frozenset(),
        retrieved=(
            RetrievedFragment("frag_A1", "retrieved_a.py"),
            RetrievedFragment("frag_B1", "retrieved_b.py"),
        ),
    )
    # No edits at all.
    credit = attribute(snap, ["frag_A1", "frag_B1"])

    assert credit.verified_hits == []
    assert sorted(credit.unverified) == ["frag_A1", "frag_B1"]
    assert credit.should_have_retrieved == []
    assert credit.modified_files == []


def test_dirty_at_start_files_are_excluded(repo: Path):
    """Files already dirty before retrieval don't get credit.

    If a file was modified before optimize_context() was ever called,
    its modification can't possibly be caused by this retrieval.
    """
    # Pre-existing dirty file
    (repo / "retrieved_a.py").write_text("# modified before snapshot\n")
    head = _git("rev-parse", "HEAD", cwd=repo).strip()

    snap = RetrievalSnapshot(
        request_id="req-4",
        repo_root=str(repo),
        git_head=head,
        dirty_at_start=frozenset({"retrieved_a.py"}),
        retrieved=(
            RetrievedFragment("frag_A1", "retrieved_a.py"),
            RetrievedFragment("frag_B1", "retrieved_b.py"),
        ),
    )
    # Edit a different file AFTER the snapshot
    (repo / "retrieved_b.py").write_text("# the real fix\n")

    credit = attribute(snap, ["frag_A1", "frag_B1"])

    # frag_A1 is unverified — its file was already dirty.
    assert "frag_A1" not in credit.verified_hits
    # frag_B1 should be verified — its file was edited post-snapshot.
    assert credit.verified_hits == ["frag_B1"]


def test_fallback_when_git_unavailable(monkeypatch, repo: Path):
    """Without git HEAD we MUST fall back to legacy (no silent abstain)."""
    snap = RetrievalSnapshot(
        request_id="req-5",
        repo_root=str(repo),
        git_head=None,  # simulates: not a git repo
        dirty_at_start=frozenset(),
        retrieved=(RetrievedFragment("frag_A1", "retrieved_a.py"),),
    )
    credit = attribute(snap, ["frag_A1"])
    assert not credit.is_causal
    # Legacy semantics: every passed id treated as verified.
    assert credit.verified_hits == ["frag_A1"]


def test_miss_cap_prevents_megapr_noise(repo: Path):
    """Big refactors (1000s of files) must not flood the learning signal."""
    head = _git("rev-parse", "HEAD", cwd=repo).strip()
    # Touch 100 files
    for i in range(100):
        (repo / f"mass_{i}.py").write_text(f"# {i}\n")

    snap = RetrievalSnapshot(
        request_id="req-6",
        repo_root=str(repo),
        git_head=head,
        dirty_at_start=frozenset(),
        retrieved=(RetrievedFragment("frag_A1", "retrieved_a.py"),),
    )
    credit = attribute(snap, ["frag_A1"])

    # Cap (default 20) prevents learning-signal flooding.
    from entroly.causal_attribution import MAX_MISS_FILES_PER_OUTCOME
    assert len(credit.should_have_retrieved) <= MAX_MISS_FILES_PER_OUTCOME


# ── store / lifecycle tests ───────────────────────────────────────────


def test_snapshot_store_lru_evicts_oldest():
    store = SnapshotStore(max_snapshots=3, ttl_seconds=3600)
    for i in range(5):
        store.put(RetrievalSnapshot(
            request_id=f"r{i}",
            repo_root=".",
            git_head=None,
            dirty_at_start=frozenset(),
            retrieved=(),
        ))
    assert len(store) == 3
    assert store.get("r0") is None
    assert store.get("r4") is not None


def test_snapshot_store_ttl_expires():
    store = SnapshotStore(max_snapshots=10, ttl_seconds=0.0001)
    store.put(RetrievalSnapshot(
        request_id="r1",
        repo_root=".",
        git_head=None,
        dirty_at_start=frozenset(),
        retrieved=(),
    ))
    import time as _t
    _t.sleep(0.01)
    assert store.get("r1") is None
    assert store.latest() is None


def test_snapshot_store_latest_skips_stale():
    store = SnapshotStore(max_snapshots=10, ttl_seconds=0.0001)
    store.put(RetrievalSnapshot(
        request_id="old",
        repo_root=".",
        git_head=None,
        dirty_at_start=frozenset(),
        retrieved=(),
    ))
    import time as _t
    _t.sleep(0.01)
    store._ttl = 3600  # extend so the next insert lives
    store.put(RetrievalSnapshot(
        request_id="new",
        repo_root=".",
        git_head=None,
        dirty_at_start=frozenset(),
        retrieved=(),
    ))
    assert store.latest().request_id == "new"


def test_env_flag_disables_causal_path(monkeypatch):
    monkeypatch.setenv("ENTROLY_CAUSAL_CREDIT", "0")
    assert causal_credit_enabled() is False
    monkeypatch.setenv("ENTROLY_CAUSAL_CREDIT", "1")
    assert causal_credit_enabled() is True


def test_build_snapshot_normalizes_fragment_keys(repo: Path):
    """Tolerates id/fragment_id and source/source_path/path key spellings."""
    reset_global_store_for_tests()
    selected = [
        {"id": "f1", "source": "a.py"},
        {"fragment_id": "f2", "source_path": "b.py"},
        {"id": "f3", "path": "c.py"},
        {"no_id_at_all": True},  # silently dropped
    ]
    snap = build_snapshot("req", repo, selected)
    ids = [f.fragment_id for f in snap.retrieved]
    paths = [f.source_path for f in snap.retrieved]
    assert ids == ["f1", "f2", "f3"]
    assert paths == ["a.py", "b.py", "c.py"]


# ── property-style sanity check ───────────────────────────────────────


def test_passed_ids_partition_into_verified_and_unverified(repo: Path):
    """Every caller-passed id must land in exactly one of {verified, unverified}.

    This is the core invariant: the fix never silently drops a passed
    id — it just demotes it from a strong signal to abstention when the
    causal evidence is missing.
    """
    snap = RetrievalSnapshot(
        request_id="part",
        repo_root=str(repo),
        git_head=_git("rev-parse", "HEAD", cwd=repo).strip(),
        dirty_at_start=frozenset(),
        retrieved=(
            RetrievedFragment("a", "retrieved_a.py"),
            RetrievedFragment("b", "retrieved_b.py"),
            RetrievedFragment("c", "verification_engine.py"),
        ),
    )
    (repo / "retrieved_a.py").write_text("touched\n")

    passed = ["a", "b", "c", "ghost_id_caller_invented"]
    credit = attribute(snap, passed)

    union = set(credit.verified_hits) | set(credit.unverified)
    assert union == set(passed), (
        f"passed ids must partition into verified∪unverified, "
        f"missing={set(passed)-union}, extra={union-set(passed)}"
    )
    assert set(credit.verified_hits).isdisjoint(credit.unverified)
