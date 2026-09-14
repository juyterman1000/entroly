"""The build path must not be observable in the repository index.

An index may be produced three ways: by the plain builder, by a cold
incremental build, or by evolving a warm cache through a sequence of edits.
All three are schedules for the same computation, so for a given final tree
they must produce the same index.

Nothing else asserts this. The existing incremental tests check that the cache
hits, invalidates, and survives corruption -- they compare a warm build against
a cold build of the *same* cache, so a defect that biases both identically stays
invisible. These tests instead compare across independent build paths, which is
where such a defect becomes observable.

The risk is concrete: `file_dependencies` is computed by two separate copies of
the same merge logic (`_merged_dependencies` in `incremental.py` and
`_merge_file_dependencies` in `__init__.py`). They agree today. Nothing keeps
them agreeing.
"""

from __future__ import annotations

from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.incremental import (
    build_repository_index_incremental,
)

# Emitted only by the incremental builder, and only to report cache behavior.
# These describe how the index was produced, not what it contains.
_CACHE_TELEMETRY_PREFIXES = (
    "incremental-parse-cache",
    "persistent-index-snapshot",
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _semantic_core(index) -> dict[str, object]:
    """Index content that must not depend on how the index was built.

    Drops ``root`` (an absolute path naming the build site) and ``diagnostics``
    (build-path telemetry). Everything else -- files, symbols, call edges,
    unresolved calls, dependencies -- is compared exactly, including the order
    of the edge tuples, since ``to_dict`` serializes those in stored order.
    """
    payload = index.to_dict()
    payload.pop("root", None)
    payload.pop("diagnostics", None)
    return payload


def _substantive_diagnostics(index) -> tuple[str, ...]:
    """Diagnostics with cache telemetry removed."""
    return tuple(
        item
        for item in index.diagnostics
        if not item.startswith(_CACHE_TELEMETRY_PREFIXES)
    )


def _assert_same_index(actual, expected, label: str) -> None:
    """Compare component by component so a failure names the divergence."""
    left = _semantic_core(actual)
    right = _semantic_core(expected)
    for key in sorted(set(left) | set(right)):
        assert left.get(key) == right.get(key), (
            f"{label}: {key!r} differs between build paths"
        )
    assert left == right, label


def test_incremental_edit_sequence_equals_cold_rebuild(tmp_path: Path) -> None:
    """A cache evolved through edits must match a cache that never saw them."""
    warm_root = tmp_path / "warm"
    warm_cache = tmp_path / "warm-cache"

    _write(warm_root, "core.py", "def alpha():\n    return 1\n")
    _write(warm_root, "util.py", "def helper():\n    return 2\n")
    build_repository_index_incremental(warm_root, cache_dir=warm_cache)

    # Modify a file so it gains a dependency edge.
    _write(
        warm_root,
        "core.py",
        "from util import helper\ndef alpha():\n    return helper()\n",
    )
    build_repository_index_incremental(warm_root, cache_dir=warm_cache)

    # Add a file that depends on the modified one.
    _write(
        warm_root,
        "app.py",
        "from core import alpha\ndef main():\n    return alpha()\n",
    )
    build_repository_index_incremental(warm_root, cache_dir=warm_cache)

    # Delete a file and revert the edit that referenced it. Reverting restores
    # content the cache already holds under its original digest, so the final
    # build resurrects a cache entry written three builds ago while a deleted
    # file's entry must not survive into the graph.
    (warm_root / "util.py").unlink()
    _write(warm_root, "core.py", "def alpha():\n    return 1\n")
    warm = build_repository_index_incremental(warm_root, cache_dir=warm_cache)

    # The same final tree, built by a cache with no history.
    cold_root = tmp_path / "cold"
    cold_cache = tmp_path / "cold-cache"
    _write(cold_root, "core.py", "def alpha():\n    return 1\n")
    _write(
        cold_root,
        "app.py",
        "from core import alpha\ndef main():\n    return alpha()\n",
    )
    cold = build_repository_index_incremental(cold_root, cache_dir=cold_cache)

    assert set(warm.files) == {"app.py", "core.py"}
    _assert_same_index(warm, cold, "warm edit sequence vs cold rebuild")
    assert _substantive_diagnostics(warm) == _substantive_diagnostics(cold)


def test_incremental_equals_non_incremental_reference_builder(
    tmp_path: Path,
) -> None:
    """The two index implementations must agree on the same tree.

    ``build_repository_index`` and ``build_repository_index_incremental`` derive
    ``file_dependencies`` through separate copies of the same merge. This is the
    test that fails if one copy is changed and the other is not.
    """
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write(root, "util.py", "def helper():\n    return 2\n")
    _write(
        root,
        "core.py",
        "from util import helper\ndef alpha():\n    return helper()\n",
    )
    _write(
        root,
        "app.py",
        "from core import alpha\nfrom util import helper\n"
        "def main():\n    return alpha() + helper()\n",
    )

    incremental = build_repository_index_incremental(root, cache_dir=cache)
    plain = build_repository_index(root)

    _assert_same_index(incremental, plain, "incremental vs plain builder")

    # The dependency merge is the duplicated surface; pin it explicitly so a
    # regression there is legible without decoding a whole-payload diff.
    assert incremental.file_dependencies == plain.file_dependencies
    assert incremental.file_dependencies["app.py"] == ("core.py", "util.py")


def test_edit_order_does_not_affect_the_index(tmp_path: Path) -> None:
    """Two orders reaching one final tree must produce one index."""
    final_files = {
        "util.py": "def helper():\n    return 2\n",
        "core.py": "from util import helper\ndef alpha():\n    return helper()\n",
        "app.py": "from core import alpha\ndef main():\n    return alpha()\n",
    }

    forward_root = tmp_path / "forward"
    forward_cache = tmp_path / "forward-cache"
    for name in ("util.py", "core.py", "app.py"):
        _write(forward_root, name, final_files[name])
        build_repository_index_incremental(forward_root, cache_dir=forward_cache)
    forward = build_repository_index_incremental(
        forward_root, cache_dir=forward_cache
    )

    reverse_root = tmp_path / "reverse"
    reverse_cache = tmp_path / "reverse-cache"
    for name in ("app.py", "core.py", "util.py"):
        _write(reverse_root, name, final_files[name])
        build_repository_index_incremental(reverse_root, cache_dir=reverse_cache)
    reverse = build_repository_index_incremental(
        reverse_root, cache_dir=reverse_cache
    )

    _assert_same_index(forward, reverse, "forward vs reverse edit order")


def test_content_edit_without_path_set_change_is_reflected(tmp_path: Path) -> None:
    """Rewriting a file in place must not serve the previous index.

    This is the case a snapshot key can silently miss: the set of paths is
    unchanged, so a key that covers paths but not content collides with the
    previous build and returns it. Every other test here varies the path set
    between builds and would not detect that.
    """
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write(root, "mod.py", "def before():\n    return 1\n")
    build_repository_index_incremental(root, cache_dir=cache)

    _write(root, "mod.py", "def after():\n    return 2\n")
    edited = build_repository_index_incremental(root, cache_dir=cache)

    assert {symbol.name for symbol in edited.symbols.values()} == {"after"}

    reference_root = tmp_path / "reference"
    reference_cache = tmp_path / "reference-cache"
    _write(reference_root, "mod.py", "def after():\n    return 2\n")
    reference = build_repository_index_incremental(
        reference_root, cache_dir=reference_cache
    )
    _assert_same_index(edited, reference, "in-place edit vs fresh build")


def test_deleted_file_leaves_no_residue_in_the_index(tmp_path: Path) -> None:
    """A deleted file must vanish from files, symbols, and dependency edges."""
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    _write(root, "keep.py", "def kept():\n    return 1\n")
    _write(root, "drop.py", "def dropped():\n    return 2\n")
    build_repository_index_incremental(root, cache_dir=cache)

    (root / "drop.py").unlink()
    after = build_repository_index_incremental(root, cache_dir=cache)

    assert set(after.files) == {"keep.py"}
    assert {symbol.name for symbol in after.symbols.values()} == {"kept"}
    assert "drop.py" not in after.file_dependencies
    assert all(
        "drop.py" not in dependencies
        for dependencies in after.file_dependencies.values()
    )

    reference_root = tmp_path / "reference"
    reference_cache = tmp_path / "reference-cache"
    _write(reference_root, "keep.py", "def kept():\n    return 1\n")
    reference = build_repository_index_incremental(
        reference_root, cache_dir=reference_cache
    )
    _assert_same_index(after, reference, "post-deletion vs never-had-the-file")
