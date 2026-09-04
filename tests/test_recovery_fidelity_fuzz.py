"""Byte-fidelity fuzz for the compress -> receipt -> recover round trip.

Everything here asserts on BYTES. Text equality is not the contract: the
promise is that what comes back out of the store is the same octets that went
in, for content nobody would hand-write into a fixture.

Three defects motivated this file, all on the path a file with one non-UTF-8
byte takes through `entroly compress`:

1. `entroly/codec.py` measured the stored length with a bare
   ``content.encode("utf-8")``. `cli_recover.cmd_compress` reads every file
   with ``errors="surrogateescape"`` on purpose, so a byte that is not valid
   UTF-8 arrives as a lone surrogate and the bare encode raised -- turning a
   10 KB tool-output capture into ``Error: 'utf-8' codec can't encode character
   '\\udcff'``. `_to_bytes` in that same module exists precisely to stop this,
   and its docstring says so.

2. `compression_retrieval_store._estimate_tokens` had the same bare encode, and
   it runs from `as_dict` during `_persist`, so the store could not even write
   the receipt it had just built.

3. `compression_retrieval_store._sha256_text` hashed with ``errors="ignore"``,
   which DELETES unencodable characters before hashing. That made the store's
   content address non-injective: a file holding a raw non-UTF-8 byte addressed
   identically to the same file with that byte deleted. `put` returns the
   already-stored item for a known ``receipt_id``, so the collision silently
   handed back the wrong original.

The existing suites did not catch any of these. `test_codec_abuse` does assert
that no codec raises on a lone surrogate, but its payloads are a few dozen
characters -- far too small for any codec to reach `RecoveryStore.put`, which
is where all three defects live.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from entroly.codec import RecoveryStore, content_digest
from entroly.compression_retrieval_store import _sha256_text
from entroly.compression_retrieval_store_secure import CompressionRetrievalStore


# --------------------------------------------------------------------------
# Content classes. Every one varies line to line: identical repeated lines are
# collapsed by dedup elsewhere in the stack, and a fixture that dedups away
# certifies nothing.
# --------------------------------------------------------------------------

def _vary(template: str, n: int = 40) -> str:
    return "".join(template.format(i=i) for i in range(n))


CONTENT_CASES: dict[str, str] = {
    # line terminators -- source bytes, not presentation
    "lf": _vary("line {i} alpha bravo\n"),
    "crlf": _vary("line {i} alpha bravo\r\n"),
    "lone_cr": _vary("line {i} alpha bravo\r"),
    "mixed_eol": _vary("line {i} a\r\nline {i} b\nline {i} c\r"),
    "no_trailing_newline": _vary("line {i} alpha\n").rstrip("\n"),
    "crlf_no_trailing": _vary("line {i} alpha\r\n").rstrip("\r\n"),
    # characters str.splitlines() ALSO treats as line breaks. If any layer
    # rebuilds content by splitting and rejoining on "\n" alone, these are the
    # ones that get silently re-terminated.
    "vertical_tab": _vary("line {i}\x0bcontinued {i}\n"),
    "form_feed": _vary("line {i}\x0ccontinued {i}\n"),
    "file_separators": _vary("line {i}\x1ca\x1db\x1ec\n"),
    "next_line_u0085": _vary("line {i}continued {i}\n"),
    "line_separator_u2028": _vary("line {i} continued {i}\n"),
    "paragraph_sep_u2029": _vary("line {i} continued {i}\n"),
    # degenerate sizes
    "empty": "",
    "one_char": "x",
    "one_newline": "\n",
    "one_cr": "\r",
    "bom_only": "﻿",
    # unicode where byte length != character count. Entropy in this codebase is
    # measured on bytes; a char-vs-byte confusion shows up here first.
    "bom_prefixed": "﻿" + _vary("line {i} after a byte order mark\n"),
    "astral_emoji": _vary("line {i} \U0001f600\U0001f680\U0001f9ea tail\n"),
    "combining_marks": _vary("line {i} é à̧ ȫ ñ\n"),
    "rtl_marks": _vary("line {i} ‏‎‮ rev ‬ ؜ العربية\n"),
    "zero_width_joiners": _vary(
        "line {i} \U0001f468‍\U0001f469‍\U0001f467 "
        "\U0001f3f3️‍\U0001f308\n"
    ),
    "cjk": _vary("line {i} 中文测试 日本語\n"),
    "latin1_lookalike": _vary("line {i} " + "".join(chr(c) for c in range(0x80, 0x100)) + "\n"),
    "nul_bytes": _vary("line {i}\x00middle\x00tail\n"),
    # content shaped like the receipt's own machinery
    "delimiter_lookalike": _vary(
        'line {i} {{"omitted_spans": [{{"start_line": 1, "end_line": {i}}}]}} '
        'sha256:deadbeef{i} "_entroly_scope_sha256" ccr:abc{i} '
        "[... exact excerpt gap; retrieve full source by handle ...]\n"
    ),
    # one very long line
    "megaline": "x" * (1024 * 1024 + 7),
    # bytes that are not valid UTF-8 at all, decoded the way the CLI decodes
    # them. This is the class every defect above lived in.
    "invalid_utf8_single": (
        _vary("line {i} alpha\n") + b"\xff".decode("utf-8", "surrogateescape") + "\n"
    ),
    "invalid_utf8_block": (
        _vary("line {i} alpha\n")
        + bytes(range(0xC0, 0xD0)).decode("utf-8", "surrogateescape")
        + "\n"
    ),
    "invalid_utf8_interleaved": _vary(
        "line {i} " + b"\x80\xfe".decode("utf-8", "surrogateescape") + " tail\n"
    ),
    # a genuine lone surrogate that surrogateescape cannot encode -- it is
    # outside U+DC80..U+DCFF, so any fix must not simply swap one error handler
    # for another.
    "lone_high_surrogate": _vary("line {i} \ud800 tail\n"),
}


def _exact(text: str) -> bytes:
    return text.encode("utf-8", "surrogatepass")


@pytest.mark.parametrize("label", sorted(CONTENT_CASES))
def test_recovery_store_round_trip_is_byte_exact(label, tmp_path: Path):
    """put -> (fresh store object) reference_for -> recover returns the bytes.

    The second store is deliberately a new instance over the same path: that is
    the code path `entroly recover` takes in a separate process, and it
    reconstructs the reference from the persisted receipt rather than from
    in-process state.
    """
    text = CONTENT_CASES[label]
    expected = _exact(text)

    written = RecoveryStore(tmp_path / "recovery.json", scope_id="fuzz")
    ref = written.put(text, item_count=1, note="fuzz")

    reread = RecoveryStore(tmp_path / "recovery.json", scope_id="fuzz")
    rebuilt = reread.reference_for(ref.digest)
    assert rebuilt is not None, f"{label}: digest did not survive persistence"

    recovered = reread.recover(rebuilt)
    assert _exact(recovered) == expected, f"{label}: recovered bytes differ"


@pytest.mark.parametrize("label", sorted(CONTENT_CASES))
def test_recovery_reference_byte_length_matches_the_content(label, tmp_path: Path):
    """`byte_length` is half of what `verify` authenticates -- it must be true.

    Checking the digest alone would leave the length decorative, which is the
    reason `RecoveryReference.verify` compares both.
    """
    text = CONTENT_CASES[label]
    store = RecoveryStore(tmp_path / "recovery.json", scope_id="fuzz")
    ref = store.put(text)
    assert ref.byte_length == len(_exact(text))

    reread = RecoveryStore(tmp_path / "recovery.json", scope_id="fuzz")
    rebuilt = reread.reference_for(ref.digest)
    assert rebuilt is not None
    # Rebuilt from persisted metadata, so this is a different code path than
    # the one above and has independently been wrong.
    assert rebuilt.byte_length == len(_exact(text)), (
        f"{label}: persisted byte_length {rebuilt.byte_length} "
        f"!= {len(_exact(text))}"
    )
    assert rebuilt.verify(text)


def test_content_digest_is_sha256_of_the_exact_bytes():
    """The digest a receipt publishes has to be a digest of something real."""
    for label, text in CONTENT_CASES.items():
        expected = "sha256:" + hashlib.sha256(_exact(text)).hexdigest()
        assert content_digest(text) == expected, label


# --------------------------------------------------------------------------
# The store's content address must be injective over its own input domain.
# --------------------------------------------------------------------------

_COLLIDING_PAIRS = [
    # a raw non-UTF-8 byte vs. that byte simply not being there
    ("alpha\n\udcff SECRET\nomega\n", "alpha\n SECRET\nomega\n"),
    ("\udc80head\ntail\n", "head\ntail\n"),
    # two DIFFERENT non-UTF-8 bytes
    ("alpha\n\udcff\nomega\n", "alpha\n\udcfe\nomega\n"),
    # a genuine lone surrogate vs. nothing
    ("alpha\n\ud800\nomega\n", "alpha\n\nomega\n"),
]


@pytest.mark.parametrize("left,right", _COLLIDING_PAIRS)
def test_store_content_address_separates_different_content(left, right):
    """Different bytes must not share a content address.

    ``errors="ignore"`` deleted every unencodable character before hashing, so
    each of these pairs hashed identically. `put` returns the item already
    stored for a known ``receipt_id``, and ``receipt_id`` is derived from this
    hash -- so the collision was not cosmetic, it returned the wrong original.
    """
    assert left != right
    assert _sha256_text(left) != _sha256_text(right)


def test_put_does_not_alias_two_originals_that_differ_only_in_raw_bytes(tmp_path: Path):
    """The end-to-end consequence of the address collision, on the real store."""
    receipt = {
        "original_tokens": 5,
        "compressed_tokens": 0,
        "omitted_spans": [
            {"start_line": 1, "end_line": 3, "reason": "codec_original_source"}
        ],
    }
    first = "alpha\n\udcff SECRET-A\nomega\n"
    second = "alpha\n SECRET-A\nomega\n"

    store = CompressionRetrievalStore(
        tmp_path / "recovery.json", scope_id="fuzz", require_scope=True
    )
    stored_first = store.put(
        original_text=first, compressed_text="", receipt=dict(receipt)
    )
    stored_second = store.put(
        original_text=second, compressed_text="", receipt=dict(receipt)
    )

    assert stored_first.receipt_id != stored_second.receipt_id

    back_first = store.get_span(stored_first.receipt_id, stored_first.spans[0].span_id)
    back_second = store.get_span(
        stored_second.receipt_id, stored_second.spans[0].span_id
    )
    assert back_first is not None and back_second is not None
    assert _exact(back_first.content) == _exact(first)
    assert _exact(back_second.content) == _exact(second)


# --------------------------------------------------------------------------
# The compress path proper: a codec has to be able to reach `put` at all.
# --------------------------------------------------------------------------

def _shell_capture(mutate_bytes) -> str:
    """Tool output big enough for a codec to claim AND compress."""
    lines = ["$ ./run-suite.sh --verbose"]
    for i in range(200):
        lines.append(
            f"[{i:04d}] INFO worker-{i} handled rid={1000 + i} in {i % 97}ms ok"
        )
    lines += [
        "FAILED tests/test_alpha.py::test_one - AssertionError: mismatch",
        "3 passed",
        "exit code 1",
    ]
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    return mutate_bytes(raw).decode("utf-8", "surrogateescape")


@pytest.mark.parametrize(
    "label,mutate",
    [
        ("clean", lambda b: b),
        ("single_invalid_byte", lambda b: b.replace(b"worker-7 ", b"worker-\xff ", 1)),
        (
            "truncated_multibyte",
            lambda b: b.replace(b"worker-9 ", b"worker-\xe4\xb8 ", 1),
        ),
        (
            "latin1_block",
            lambda b: b.replace(b"worker-11 ", bytes(range(0xC0, 0xD0)) + b" ", 1),
        ),
        ("crlf_and_invalid", lambda b: b.replace(b"\n", b"\r\n").replace(
            b"worker-7 ", b"worker-\xff ", 1)),
    ],
)
def test_compress_path_recovers_the_original_bytes(label, mutate, tmp_path: Path):
    """Route real content through the registry and recover it byte-for-byte.

    Not a unit test of one codec: `default_registry` picks the winner, and the
    point is that whichever codec wins can still store and return the source.
    Before the fix, `single_invalid_byte` raised UnicodeEncodeError here.
    """
    from entroly.codecs_builtin import default_registry

    text = _shell_capture(mutate)
    store = RecoveryStore(tmp_path / "recovery.json", scope_id="fuzz")
    reps = default_registry(store).representations(text, source_id=f"capture-{label}")
    assert reps, f"{label}: no codec claimed shell-shaped output"

    recoverable = [r for r in reps if r.recovery is not None]
    assert recoverable, f"{label}: no representation carried a recovery reference"

    reread = RecoveryStore(tmp_path / "recovery.json", scope_id="fuzz")
    for rep in recoverable:
        rebuilt = reread.reference_for(rep.recovery.digest)
        assert rebuilt is not None, f"{label}: {rep.representation_id} lost its digest"
        recovered = reread.recover(rebuilt)
        assert _exact(recovered) == _exact(text), (
            f"{label}: {rep.representation_id} did not recover the source"
        )
        assert rep.source_sha256 == content_digest(text)
