"""Two id spaces named `file:` and `symbol:` must not be confusable.

`entroly/repository_intelligence/graph_query.py` mints `file:{path}` and
`symbol:{symbol_id}`. `entroly_engine::work_graph::stable_node_id` mints
`file:{sha256(...)[:24]}` and `symbol:{sha256(...)[:24]}`. Same two namespace
tokens, different content, and `graph_query._node_path` dispatches on exactly
that prefix and slices it off positionally.

The consequence is not a crash. A genuine Work Graph node id fed to the
repository-intelligence resolver slices to a hex string, fails an `in
index.files` test, and returns `None` -- a node that exists, reported absent.
That is the fabricated-completeness failure the handoff forbids, so it is pinned
here rather than left to be rediscovered.

These tests reach into `_node_path` deliberately. It is the dispatch that carries
the hazard; asserting only against the public wrapper would leave the actual
mechanism untested.
"""

from __future__ import annotations

import pytest

from entroly.repository_intelligence.graph_query import _file_node, _node_path, _symbol_node
from entroly.repository_intelligence.models import FileRecord, RepositoryIndex, Symbol
from entroly.work_graph import WorkGraphUnavailableError, stable_node_id

REPO = "repo:demo"
PATH = "src/app.py"
SYMBOL_ID = "src/app.py::App.handler::function"


def _skip_without_native() -> None:
    try:
        stable_node_id("file", "repo:probe", "probe.py")
    except WorkGraphUnavailableError as exc:  # pragma: no cover - environment
        pytest.skip(f"native work graph unavailable: {exc}")


def _index() -> RepositoryIndex:
    return RepositoryIndex(
        root="/repo",
        files={
            PATH: FileRecord(
                path=PATH,
                language="python",
                sha256="0" * 64,
                byte_length=120,
                line_count=8,
                is_test=False,
            )
        },
        symbols={
            SYMBOL_ID: Symbol(
                symbol_id=SYMBOL_ID,
                path=PATH,
                name="handler",
                qualified_name="App.handler",
                kind="function",
                line_start=3,
                line_end=6,
                language="python",
            )
        },
    )


def test_the_two_id_spaces_share_a_namespace_token() -> None:
    """This is the hazard, stated as an assertion.

    If these prefixes ever diverge the collision is gone and the rest of this
    file becomes unnecessary -- which is the outcome to want.
    """
    _skip_without_native()

    local = _file_node(PATH)
    canonical = stable_node_id("file", REPO, PATH)

    assert local.split(":", 1)[0] == canonical.split(":", 1)[0] == "file"
    assert local != canonical


def test_a_work_graph_id_resolves_to_nothing_rather_than_erroring() -> None:
    """Current behaviour, pinned because it is silent.

    `None` here is indistinguishable from "this path is not in the index". A
    caller cannot tell a genuinely unknown file from a correctly-formed id in the
    wrong namespace, and neither can a reader of the return value.
    """
    _skip_without_native()

    index = _index()
    canonical = stable_node_id("file", REPO, PATH)

    assert _node_path(_file_node(PATH), index) == PATH
    assert _node_path(canonical, index) is None


def test_the_symbol_namespace_collides_the_same_way() -> None:
    _skip_without_native()

    index = _index()
    canonical = stable_node_id("symbol", REPO, SYMBOL_ID)

    assert _node_path(_symbol_node(SYMBOL_ID), index) == PATH
    assert _node_path(canonical, index) is None


def test_the_canonical_id_is_length_distinguishable() -> None:
    """The one property that makes the two spaces separable in practice.

    `stable_node_id` always emits exactly 24 hex characters after the token, so a
    caller that must accept both can discriminate without guessing. A local id
    equal to a 24-char hex path would defeat this, which is why the assertion
    names the digest shape rather than just a length.
    """
    _skip_without_native()

    canonical = stable_node_id("file", REPO, PATH)
    _, _, digest = canonical.partition(":")

    assert len(digest) == 24
    assert all(character in "0123456789abcdef" for character in digest)
    assert "/" not in digest and "." not in digest


def test_the_join_module_produces_the_canonical_space_not_the_local_one() -> None:
    """`graph_identity` must not have quietly adopted the local convention."""
    _skip_without_native()

    from entroly.repository_intelligence.graph_identity import file_node_id, symbol_node_id

    assert file_node_id(REPO, PATH) == stable_node_id("file", REPO, PATH)
    assert file_node_id(REPO, PATH) != _file_node(PATH)
    assert symbol_node_id(REPO, SYMBOL_ID) != _symbol_node(SYMBOL_ID)
