"""Re-indexing an unchanged repository must not destroy a large file's evidence.

An oversized source is indexed as an ordered chunk group -- `path`, `path#1`,
`path#2`, ... -- so that a 266 KB file contributes evidence instead of being
skipped. Reconcile then asked, for every indexed source, "is this still on
disk?" using the raw source name. No suffixed chunk exists on disk, so the
second pass over an *unchanged* repository deleted every chunk of every large
file and never re-added them.

Measured on this repository before the fix: pass 1 indexed cli.py as 33 chunks,
proxy.py as 32, auto_index.py as 9; pass 2 dropped all 261 chunk fragments and
the corpus fell from 1221 to 960. The three files silently missing from the
index were exactly the three the retention benchmark could never retrieve.
"""

from __future__ import annotations

from entroly.auto_index import _logical_rel_path


def test_chunk_suffix_resolves_to_the_path_on_disk():
    assert _logical_rel_path("entroly/cli.py#7") == "entroly/cli.py"
    assert _logical_rel_path("entroly/cli.py#0") == "entroly/cli.py"
    assert _logical_rel_path("entroly/cli.py") == "entroly/cli.py"


def test_hash_in_a_real_filename_is_not_mistaken_for_a_chunk_marker():
    # Only a trailing all-digit suffix is a chunk marker. Stripping a literal
    # '#' would invent a path that does not exist and delete a live file.
    assert _logical_rel_path("docs/c#/guide.md") == "docs/c#/guide.md"
    assert _logical_rel_path("a/b#c.py") == "a/b#c.py"
    assert _logical_rel_path("notes#draft") == "notes#draft"
    assert _logical_rel_path("weird#12ab") == "weird#12ab"


def test_deletion_check_keeps_chunks_of_a_file_that_still_exists():
    """The exact predicate reconcile uses, exercised over a chunk group.

    Reverting `_logical_rel_path` to the previous `source[5:]` makes every
    suffixed chunk look deleted and fails this test.
    """
    current_paths = {"entroly/cli.py", "entroly/checkpoint.py"}
    existing = [
        "file:entroly/cli.py",
        "file:entroly/cli.py#1",
        "file:entroly/cli.py#32",
        "file:entroly/checkpoint.py",
    ]
    deleted = [
        source for source in existing
        if _logical_rel_path(source[len("file:"):]) not in current_paths
    ]
    assert deleted == []


def test_chunks_of_a_genuinely_removed_file_are_still_deleted():
    # The fix must not make deletion impossible -- a real removal still reaps
    # the whole chunk group, or the index would keep serving a deleted file.
    current_paths = {"entroly/checkpoint.py"}
    existing = [
        "file:entroly/gone.py",
        "file:entroly/gone.py#1",
        "file:entroly/checkpoint.py",
    ]
    deleted = [
        source for source in existing
        if _logical_rel_path(source[len("file:"):]) not in current_paths
    ]
    assert deleted == ["file:entroly/gone.py", "file:entroly/gone.py#1"]
