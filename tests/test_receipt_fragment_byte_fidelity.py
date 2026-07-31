"""Every ingested fragment must be an exact slice of its source.

This is the invariant behind the product's exact-recovery claim:

    source_text[chunk.byte_start:chunk.byte_end] == chunk.text

The oracle is always the source passed in, never the chunker's own output, and
the fixtures carry the structures that previously broke it — indentation,
comment syntax that looks like a Markdown heading, blank lines, CRLF, tabs and
a missing final newline.

Regression guard for docs/investigations/P0-receipt-chunk-fidelity.md, where
this held for only 20.9% of fragments across the tracked corpus.
"""
from __future__ import annotations

import subprocess

import pytest

from entroly.context_receipts.ingest import ingest_documents

BASELINE_REF = "1ecf1e093348068539f9e1463826209c966ed535"


def assert_fragments_are_exact(path: str, text: str) -> int:
    """Assert every chunk is byte-addressable in `text`; return the chunk count."""
    raw = text.encode("utf-8")
    index = ingest_documents([(path, text)])
    assert index.chunks, f"{path} produced no chunks"

    for chunk in index.chunks:
        assert chunk.text in text, f"{path}:{chunk.chunk_id} text absent from source"
        assert raw[chunk.byte_start:chunk.byte_end] == chunk.text.encode("utf-8"), (
            f"{path}:{chunk.chunk_id} byte range "
            f"[{chunk.byte_start}:{chunk.byte_end}] does not slice back to its text"
        )
    return len(index.chunks)


# ── the structures that actually broke ───────────────────────────────────────


def test_indented_python_comments_are_not_treated_as_headings():
    """`    # comment` matches the Markdown ATX heading pattern; it must not split."""
    source = (
        "def scoring(claim, evidence):\n"
        "    weight = 1.0\n"
        "    # IDF-weighted Jaccard over content words\n"
        "    # Penalty -0.35 / bonus +0.15 from calibration\n"
        "    shared = claim & evidence\n"
        "    return weight * len(shared)\n"
    )
    assert_fragments_are_exact("scoring.py", source)


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("tabs.py", "def f():\n\tif x:\n\t\t# tabbed comment\n\t\treturn 1\n"),
        ("no_final_newline.py", "x = 1\n\n# trailing comment, no newline at EOF"),
        ("crlf.py", "def f():\r\n    # crlf comment\r\n    return 1\r\n"),
        ("mixed_newlines.py", "a = 1\n\n# lf then crlf\r\nb = 2\r\n"),
        ("blank_lines.py", "a = 1\n\n\n\n# many blanks above\n\n\nb = 2\n"),
        ("trailing_ws.py", "def f():   \n    return 1   \n\n# comment   \n"),
        ("yaml_comments.yml", "key: value\n\n# a yaml comment\nnested:\n  # indented\n  a: 1\n"),
        ("shell.sh", "#!/bin/sh\nset -e\n\n# a shell comment\necho hi\n"),
        ("heading_in_string.py", 'DOC = """\n# Not A Heading\n"""\nx = 1\n'),
        ("unicode.py", "def f():\n    # café ☕ naïve — em dash\n    s = 'ünïcödé'\n    return s\n"),
        ("markdown.md", "# Real Heading\n\nProse paragraph.\n\n## Another\n\n- a\n- b\n"),
        ("json_config.json", '{\n  "a": 1,\n  "b": [1, 2, 3]\n}\n'),
    ],
)
def test_structures_that_previously_corrupted_fragments(name, source):
    assert_fragments_are_exact(name, source)


def test_crlf_and_lf_of_the_same_file_are_both_exact():
    """Newline form must not affect the invariant; only the bytes differ."""
    lf = "def f():\n    # comment one\n    # comment two\n    return 1\n"
    crlf = lf.replace("\n", "\r\n")
    assert_fragments_are_exact("lf.py", lf)
    assert_fragments_are_exact("crlf.py", crlf)


def test_empty_and_whitespace_only_documents_are_safe():
    for text in ("", "   ", "\n\n\n", "\t\n \n"):
        index = ingest_documents([("blank.py", text)])
        for chunk in index.chunks:
            assert text.encode("utf-8")[chunk.byte_start:chunk.byte_end] == chunk.text.encode("utf-8")


def test_a_large_block_split_across_chunks_stays_exact():
    """Exercises _split_large_block, whose offsets were relative to stripped text."""
    source = "def f():\n" + "".join(f"    value_{i} = {i}\n" for i in range(400))
    assert_fragments_are_exact("large.py", source)


def test_duplicate_identical_blocks_keep_distinct_correct_ranges():
    body = "def f():\n    # same comment\n    return 1\n"
    source = body + "\n" + body
    index = ingest_documents([("dupes.py", source)])
    raw = source.encode("utf-8")
    ranges = {(c.byte_start, c.byte_end) for c in index.chunks}
    assert len(ranges) == len(index.chunks), "identical text collapsed onto one range"
    for chunk in index.chunks:
        assert raw[chunk.byte_start:chunk.byte_end] == chunk.text.encode("utf-8")


# ── the real corpus, at the pinned baseline ──────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "entroly/esg.py",
        "entroly/evidence_locked_compression.py",
        "entroly/context_receipts/ingest.py",
        "README.md",
        "pyproject.toml",
    ],
)
def test_real_repository_files_are_exact(path):
    """Real files at the pinned baseline, read as bytes so newlines are preserved."""
    raw = subprocess.run(
        ["git", "cat-file", "blob", f"{BASELINE_REF}:{path}"],
        capture_output=True, check=True,
    ).stdout
    assert_fragments_are_exact(path, raw.decode("utf-8"))
