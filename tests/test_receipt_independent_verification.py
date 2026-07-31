"""A receipt holder must be able to verify a fragment without trusting Entroly.

The verifier below deliberately does not import any Entroly helper. It uses only
hashlib and the receipt's own fields, which is the whole point: if verification
required calling the code that produced the receipt, it would prove internal
consistency rather than fidelity to the source.

Covers the second half of docs/investigations/P0-receipt-chunk-fidelity.md,
where the recorded fingerprint matched neither the recovered bytes nor the
original span, leaving a receipt holder with nothing to check.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from entroly.context_receipts.ingest import ingest_documents, read_documents_from_path


def independently_verify(source_bytes: bytes, chunk: dict[str, object]) -> bool:
    """Reference verifier: receipt + source bytes + a documented hash. No Entroly."""
    start, end = int(chunk["byte_start"]), int(chunk["byte_end"])
    if not 0 <= start <= end <= len(source_bytes):
        return False
    if "sha256:" + hashlib.sha256(source_bytes).hexdigest() != chunk["source_sha256"]:
        return False
    fragment = source_bytes[start:end]
    return "sha256:" + hashlib.sha256(fragment).hexdigest() == chunk["fragment_sha256"]


def _as_dict(chunk) -> dict[str, object]:
    return {
        "byte_start": chunk.byte_start,
        "byte_end": chunk.byte_end,
        "fragment_sha256": chunk.fragment_sha256,
        "source_sha256": chunk.source_sha256,
    }


CASES = [
    ("comments.py", "def f():\n    x = 1\n    # one\n    # two\n    return x\n"),
    ("crlf.py", "def f():\r\n    # comment\r\n    return 1\r\n"),
    ("tabs.py", "def f():\n\t# tabbed\n\treturn 1\n"),
    ("no_final_newline.py", "a = 1\n# end"),
    ("unicode.py", "# café ☕ — em dash\ns = 'ünïcödé'\n"),
    ("prose.md", "# Heading\n\nParagraph.\n\n## Sub\n\nMore.\n"),
]


@pytest.mark.parametrize(("name", "text"), CASES)
def test_every_fragment_verifies_independently(name, text):
    source_bytes = text.encode("utf-8")
    index = ingest_documents([(name, text)])
    assert index.chunks
    for chunk in index.chunks:
        assert independently_verify(source_bytes, _as_dict(chunk)), (
            f"{name}:{chunk.chunk_id} could not be verified from the receipt alone"
        )


# ── the verifier must reject tampering, not just accept the happy path ───────


def _one_chunk(text: str) -> dict[str, object]:
    index = ingest_documents([("m.py", text)])
    # pick a chunk that is not the whole file so mutations are observable
    return _as_dict(index.chunks[0])


MUTATION_TEXT = "def f():\n    # comment\n    return 1\n\n\ndef g():\n    return 2\n"


def test_a_single_changed_source_byte_is_detected():
    chunk = _one_chunk(MUTATION_TEXT)
    tampered = MUTATION_TEXT.replace("return 1", "return 2").encode("utf-8")
    assert not independently_verify(tampered, chunk)


def test_a_changed_fragment_hash_is_detected():
    chunk = _one_chunk(MUTATION_TEXT)
    chunk["fragment_sha256"] = "sha256:" + "0" * 64
    assert not independently_verify(MUTATION_TEXT.encode("utf-8"), chunk)


def test_a_changed_source_hash_is_detected():
    chunk = _one_chunk(MUTATION_TEXT)
    chunk["source_sha256"] = "sha256:" + "0" * 64
    assert not independently_verify(MUTATION_TEXT.encode("utf-8"), chunk)


@pytest.mark.parametrize("shift", [-1, 1])
def test_a_shifted_byte_range_is_detected(shift):
    chunk = _one_chunk(MUTATION_TEXT)
    chunk["byte_end"] = int(chunk["byte_end"]) + shift
    assert not independently_verify(MUTATION_TEXT.encode("utf-8"), chunk)


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 10), (0, 10**9), (10, 5)],
)
def test_out_of_bounds_and_inverted_ranges_are_rejected(start, end):
    chunk = _one_chunk(MUTATION_TEXT)
    chunk["byte_start"], chunk["byte_end"] = start, end
    assert not independently_verify(MUTATION_TEXT.encode("utf-8"), chunk)


# ── the path-based API must not normalise newlines ──────────────────────────


def test_path_based_ingestion_preserves_crlf(tmp_path):
    """Path.read_text() folds CRLF to LF; recovery is a byte contract."""
    target = tmp_path / "crlf_module.py"
    raw = b"def f():\r\n    # comment\r\n    return 1\r\n"
    target.write_bytes(raw)

    documents = read_documents_from_path(target)
    assert documents, "file was not discovered"
    _, text = documents[0]
    assert text.encode("utf-8") == raw, "newlines were normalised during ingestion"

    index = ingest_documents([(Path(target).as_posix(), text)])
    for chunk in index.chunks:
        assert independently_verify(raw, _as_dict(chunk))
