"""The native and pure-Python receipt backends must agree.

`entroly.context_receipts.ingest_documents` dispatches to Rust when a native
wheel is present and falls back to Python otherwise, so the two must produce the
same chunk boundaries. When they disagree, a user's receipts depend on whether
their wheel happened to build — and a fix applied to one backend silently leaves
the other broken, which is exactly what happened in
docs/investigations/P0-receipt-chunk-fidelity.md.

Every case here asserts the byte-exactness invariant on *both* backends, so
neither can regress independently.
"""
from __future__ import annotations

import subprocess

import pytest

from entroly.context_receipts import ingest_documents as ingest_dispatch
from entroly.context_receipts import select_from_index
from entroly.context_receipts.ingest import ingest_documents as ingest_python

BASELINE_REF = "1ecf1e093348068539f9e1463826209c966ed535"




def _native_available() -> bool:
    try:
        import entroly_core  # noqa: F401
    except ImportError:
        return False
    return True


requires_native = pytest.mark.skipif(
    not _native_available(), reason="native entroly_core not installed"
)


def _dispatch_chunks(path: str, text: str) -> list[dict[str, object]]:
    index = ingest_dispatch([(path, text)])
    chunks = index["chunks"] if isinstance(index, dict) else index.chunks
    out = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            out.append(
                {
                    "text": chunk["text"],
                    "byte_start": chunk["byte_start"],
                    "byte_end": chunk["byte_end"],
                }
            )
        else:
            out.append(
                {
                    "text": chunk.text,
                    "byte_start": chunk.byte_start,
                    "byte_end": chunk.byte_end,
                }
            )
    return out


def _python_chunks(path: str, text: str) -> list[dict[str, object]]:
    return [
        {"text": c.text, "byte_start": c.byte_start, "byte_end": c.byte_end}
        for c in ingest_python([(path, text)]).chunks
    ]


CASES = [
    ("comments.py", "def f():\n    x = 1\n    # one\n    # two\n    return x\n"),
    ("shebang.sh", "#!/bin/sh\nset -e\n\n# comment\necho hi\n"),
    ("yaml.yml", "a: 1\n\n# comment\nb:\n  # nested\n  c: 2\n"),
    ("prose.md", "# Heading\n\nParagraph one.\n\n## Sub\n\nParagraph two.\n"),
    ("crlf.py", "def f():\r\n    # comment\r\n    return 1\r\n"),
    ("tabs.py", "def f():\n\t# tabbed\n\treturn 1\n"),
    ("unicode_box.py", "x = 1\n\n# ── section ──────\ny = 2\n"),
    ("no_trailing_nl.py", "a = 1\n# end"),
]


@requires_native
@pytest.mark.parametrize(("name", "text"), CASES)
def test_both_backends_produce_byte_exact_fragments(name, text):
    raw = text.encode("utf-8")
    for label, chunks in (
        ("python", _python_chunks(name, text)),
        ("default", _dispatch_chunks(name, text)),
    ):
        for chunk in chunks:
            assert chunk["text"] in text, f"{label}: {name} fragment absent from source"
            assert raw[chunk["byte_start"]:chunk["byte_end"]] == chunk["text"].encode("utf-8"), (
                f"{label}: {name} byte range does not slice back to its text"
            )


@requires_native
@pytest.mark.parametrize(("name", "text"), CASES)
def test_backends_agree_on_chunk_boundaries(name, text):
    python_ranges = [(c["byte_start"], c["byte_end"]) for c in _python_chunks(name, text)]
    default_ranges = [(c["byte_start"], c["byte_end"]) for c in _dispatch_chunks(name, text)]
    assert python_ranges == default_ranges, (
        f"{name}: backends disagree on boundaries\n"
        f"  python : {python_ranges}\n"
        f"  default: {default_ranges}"
    )


@requires_native
def test_multibyte_characters_do_not_crash_the_native_backend():
    """A chunk boundary inside a multi-byte character used to panic across PyO3.

    A panic is not a fail-closed degradation; it takes down the caller.
    """
    text = "x = 1\n\n# " + "─" * 400 + " section\ny = 2\n"
    chunks = _dispatch_chunks("boxdrawing.py", text)
    raw = text.encode("utf-8")
    for chunk in chunks:
        assert raw[chunk["byte_start"]:chunk["byte_end"]] == chunk["text"].encode("utf-8")


@requires_native
@pytest.mark.parametrize("path", ["entroly/esg.py", "README.md"])
def test_real_files_are_byte_exact_on_both_backends(path):
    """The invariant must hold on real files regardless of how they are split."""
    raw = subprocess.run(
        ["git", "cat-file", "blob", f"{BASELINE_REF}:{path}"],
        capture_output=True, check=True,
    ).stdout
    text = raw.decode("utf-8")
    for label, chunks in (
        ("python", _python_chunks(path, text)),
        ("default", _dispatch_chunks(path, text)),
    ):
        assert chunks, f"{label}: {path} produced no chunks"
        for chunk in chunks:
            assert raw[chunk["byte_start"]:chunk["byte_end"]] == chunk["text"].encode("utf-8"), (
                f"{label}: {path} byte range does not slice back to its text"
            )


@requires_native
@pytest.mark.xfail(
    reason=(
        "Pre-existing token-estimator divergence, not a fidelity defect. Python's "
        "TOKEN_RE is r\"[A-Za-z0-9][A-Za-z0-9_\\-']*|[^\\w\\s]\" and counts punctuation "
        "as tokens; Rust's token_re is r\"[A-Za-z0-9][A-Za-z0-9_']*\" and does not. "
        "Python therefore estimates more tokens per block and packs smaller chunks "
        "(esg.py: 18 chunks vs 9). Both backends are byte-exact, so recovery is "
        "correct either way, but a user's chunk boundaries still depend on whether "
        "a native wheel is installed. Aligning the estimators changes chunk sizes "
        "product-wide and is out of scope for the fidelity repair."
    ),
    strict=True,
)
@pytest.mark.parametrize("path", ["entroly/esg.py", "README.md"])
def test_real_files_agree_on_boundaries_across_backends(path):
    raw = subprocess.run(
        ["git", "cat-file", "blob", f"{BASELINE_REF}:{path}"],
        capture_output=True, check=True,
    ).stdout
    text = raw.decode("utf-8")
    python_ranges = [(c["byte_start"], c["byte_end"]) for c in _python_chunks(path, text)]
    default_ranges = [(c["byte_start"], c["byte_end"]) for c in _dispatch_chunks(path, text)]
    assert python_ranges == default_ranges, f"{path}: backends disagree on boundaries"


@requires_native
def test_selection_certificate_is_identical_across_backends():
    documents = [
        ("alpha.md", "alpha evidence\n"),
        ("banana.md", "banana evidence\n"),
        ("carrot.md", "carrot evidence\n"),
    ]
    index = ingest_dispatch(documents, chunk_tokens=20, prefer_rust=False)

    python_receipt = select_from_index(
        index, query="evidence", token_budget=10, prefer_rust=False
    )
    rust_receipt = select_from_index(
        index, query="evidence", token_budget=10, prefer_rust=True
    )

    python_certificate = python_receipt["risk_summary"]["selection_certificate"]
    rust_certificate = rust_receipt["risk_summary"]["selection_certificate"]
    assert python_certificate == rust_certificate
    assert python_certificate["optimizer"] == "exact_dependency_closed_enumeration"
    assert (
        python_certificate["optimality"]
        == "exact_for_internal_relevance_objective"
    )
    assert (
        python_certificate["objective"]["certified_regret_upper_bound"] == 0.0
    )
