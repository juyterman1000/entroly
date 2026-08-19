"""One artifact, one id, in every runtime.

`stable_node_id` is how the Work Graph names a repository artifact. Until the
engine exposed it, no runtime outside Rust could call it, so
`entroly/repository_intelligence` — the highest fan-in module set in the package
— grew its own free-form ``symbol_id``. The consequence was two graphs of the
same repository that could not be joined: a ``NodeKind::File`` in the work graph
and a ``FileRecord`` in repository intelligence describe the same file and had
no id in common.

These tests pin the construction itself rather than "some string comes back",
because a lookalike id that is merely stable would pass a weaker assertion while
still failing to match the graph.
"""

from __future__ import annotations

import hashlib

import pytest

from entroly.work_graph import WorkGraphUnavailableError, stable_edge_id, stable_node_id


def _skip_without_native() -> None:
    try:
        stable_node_id("file", "repo:probe", "probe.py")
    except WorkGraphUnavailableError as exc:  # pragma: no cover - environment
        pytest.skip(f"native work graph unavailable: {exc}")


def test_node_id_matches_the_engine_construction() -> None:
    """The id must be the engine's, not merely deterministic.

    `stable_node_id(kind, repo, key)` is
    ``{token}:{sha256("node|{token}|{repo}|{key}")[:24]}``. Recomputing it here
    means a future change to either side breaks this test instead of silently
    producing two id spaces again.
    """
    _skip_without_native()

    node_id = stable_node_id("file", "repo:demo", "src/app.py")
    digest = hashlib.sha256(b"node|file|repo:demo|src/app.py").hexdigest()

    assert node_id == f"file:{digest[:24]}"


def test_edge_id_matches_the_engine_construction() -> None:
    _skip_without_native()

    edge_id = stable_edge_id("file:aaa", "defines", "symbol:bbb")
    digest = hashlib.sha256(b"edge|file:aaa|defines|symbol:bbb").hexdigest()

    assert edge_id == f"edge:{digest[:24]}"


def test_identity_is_stable_across_calls() -> None:
    _skip_without_native()

    first = stable_node_id("symbol", "repo:demo", "src/app.py::handler")
    second = stable_node_id("symbol", "repo:demo", "src/app.py::handler")

    assert first == second


def test_kind_is_part_of_identity() -> None:
    """A file and a symbol with the same key are different artifacts."""
    _skip_without_native()

    assert stable_node_id("file", "repo:demo", "x") != stable_node_id(
        "symbol", "repo:demo", "x"
    )


def test_repository_is_part_of_identity() -> None:
    """The same path in two repositories is not the same node."""
    _skip_without_native()

    assert stable_node_id("file", "repo:a", "x.py") != stable_node_id(
        "file", "repo:b", "x.py"
    )


def test_unknown_kind_is_rejected_not_hashed() -> None:
    """Fail closed.

    Hashing an unrecognised token would mint a plausible id in a namespace no
    reader of the graph knows about — worse than an error, because nothing
    downstream could tell it apart from a real node.
    """
    _skip_without_native()

    with pytest.raises(Exception) as exc_info:
        stable_node_id("not_a_kind", "repo:demo", "x")

    assert "unknown node kind" in str(exc_info.value)


def test_unknown_edge_kind_is_rejected_not_hashed() -> None:
    _skip_without_native()

    with pytest.raises(Exception) as exc_info:
        stable_edge_id("file:a", "not_an_edge", "file:b")

    assert "unknown edge kind" in str(exc_info.value)
