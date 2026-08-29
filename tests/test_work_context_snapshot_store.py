from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from entroly.repository_intelligence.verified_context import _seal_context
from entroly.work_context_snapshot_store import (
    CONTEXT_SNAPSHOT_TOKEN_PREFIX,
    WorkContextSnapshotError,
    WorkContextSnapshotStore,
)
from entroly.work_graph_store import WorkGraphStore


def _native_available() -> bool:
    try:
        import entroly_core  # noqa: F401
    except ImportError:
        return False
    return True


# The Work Graph fails closed on a missing native engine rather than serving a
# degraded timeline. CLAUDE.md permits that ("provide a semantically compatible
# fallback OR fail closed behind the shared native capability gate"), and
# entroly-core is a base dependency, so every real install has the engine. The
# pure-Python fallback CI job removes it deliberately, so a test exercising a
# native-only capability has to declare that rather than fail there.
requires_native = pytest.mark.skipif(
    not _native_available(), reason="native entroly_core not installed"
)

# Every assertion here is about the native context snapshot verifier.
pytestmark = requires_native



def _context(marker: str = "alpha") -> dict:
    """Build a minimal real commitment using the production sealing algorithm."""
    return _seal_context(
        {
            "schema_version": "entroly.verified-code-context.v1",
            "query": f"find {marker}",
            "index_digest": "idx-test",
            "retrieval": {
                "policy": "test",
                "token_budget": 512,
                "estimated_tokens": 1,
            },
            "fragments": [],
            "recoverable_fragments": [],
            "relations": [],
            "unresolved_calls": [],
            "history": {"available": False, "commits": []},
            "proposal_overlay": {"provider": "test"},
            "receipt": {
                "commitment_scope": (
                    "payload-excluding-generation-command-and-context-sha256"
                )
            },
        }
    )


def _store(tmp_path: Path, repo_id: str = "repo:test", **kwargs) -> WorkContextSnapshotStore:
    graph = WorkGraphStore(repo_id, root=tmp_path / "state")
    return WorkContextSnapshotStore(graph, **kwargs)


def test_snapshot_token_is_verified_context_commitment_and_round_trips(tmp_path: Path) -> None:
    store = _store(tmp_path)
    context = _context()
    commitment = context["receipt"]["context_sha256"]

    token = store.put_json(context)

    assert token == CONTEXT_SNAPSHOT_TOKEN_PREFIX + commitment
    assert WorkContextSnapshotStore.token_for_commitment(commitment) == token
    assert store.get_json(token) == context
    assert store.put_json(context) == token  # idempotent content-addressed write
    if os.name == "posix":
        target = store.context_dir / f"{commitment}.json"
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.context_dir.stat().st_mode) == 0o700


def test_snapshot_fails_closed_on_semantic_tamper(tmp_path: Path) -> None:
    store = _store(tmp_path)
    context = _context()
    token = store.put_json(context)
    digest = token.removeprefix(CONTEXT_SNAPSHOT_TOKEN_PREFIX)
    target = store.context_dir / f"{digest}.json"

    tampered = json.loads(target.read_text(encoding="utf-8"))
    tampered["query"] = "different query"
    target.write_text(
        json.dumps(tampered, sort_keys=True, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(WorkContextSnapshotError, match="commitment"):
        store.get_json(token)


def test_snapshot_fails_closed_on_noncanonical_byte_rewrite(tmp_path: Path) -> None:
    store = _store(tmp_path)
    context = _context()
    token = store.put_json(context)
    digest = token.removeprefix(CONTEXT_SNAPSHOT_TOKEN_PREFIX)
    target = store.context_dir / f"{digest}.json"

    # Same JSON value and therefore same semantic commitment; different storage
    # bytes are still a mutation and must not be silently normalized on read.
    target.write_text(json.dumps(context, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(WorkContextSnapshotError, match="not canonical"):
        store.get_json(token)


def test_snapshot_token_cannot_cross_repository_scope(tmp_path: Path) -> None:
    first = _store(tmp_path, "repo:first")
    second = _store(tmp_path, "repo:second")
    token = first.put_json(_context())

    with pytest.raises(WorkContextSnapshotError, match="unavailable"):
        second.get_json(token)


def test_snapshot_rejects_symlink_target(tmp_path: Path) -> None:
    store = _store(tmp_path)
    context = _context()
    commitment = context["receipt"]["context_sha256"]
    token = WorkContextSnapshotStore.token_for_commitment(commitment)
    target = store.context_dir / f"{commitment}.json"
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable in this environment")

    with pytest.raises(WorkContextSnapshotError, match="unsafe"):
        store.get_json(token)
    with pytest.raises(WorkContextSnapshotError, match="unsafe"):
        store.put_json(context)


def test_snapshot_store_capacity_fails_closed_without_deleting_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path, max_snapshots=1)
    first = _context("first")
    second = _context("second")
    first_token = store.put_json(first)

    with pytest.raises(WorkContextSnapshotError, match="entry limit"):
        store.put_json(second)

    # Hitting a bound must not make an older receipt unrecoverable.
    assert store.get_json(first_token) == first


def test_snapshot_rejects_malformed_tokens_before_filesystem_lookup(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for token in (
        "",
        "vctx1.deadbeef",
        CONTEXT_SNAPSHOT_TOKEN_PREFIX + "0" * 63,
        CONTEXT_SNAPSHOT_TOKEN_PREFIX + "g" * 64,
        CONTEXT_SNAPSHOT_TOKEN_PREFIX + "0" * 65,
    ):
        with pytest.raises(WorkContextSnapshotError):
            store.get_json(token)


def test_snapshot_fails_closed_on_surrogateescaped_source_bytes(tmp_path: Path) -> None:
    """Lone surrogate escapes cannot be represented by cross-runtime JSON.

    Repository readers may use ``surrogateescape`` to retain undecodable source
    bytes.  Persisting that host-only representation would make Python and Node
    disagree, so the shared Rust verifier must reject it before any snapshot is
    written.
    """
    store = _store(tmp_path)
    marker = b"source-\xff-byte".decode("utf-8", errors="surrogateescape")
    context = _context(marker)

    with pytest.raises(WorkContextSnapshotError, match="not valid JSON"):
        store.put_json(context)

    assert list(store.context_dir.glob("*.json")) == []
