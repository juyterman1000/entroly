"""One recovery verdict, whichever implementation answers.

Entroly has two recovery implementations that a caller can reach. The Rust
engine contract (``RecoveryHandle``, via PyO3 and WASM) recomputes the digest
and answers ``verified`` / ``commitment_mismatch`` / ``not_recoverable``. The
pure-Python content-addressed store answers from ``ccr.retrieve``. Until
recently they disagreed on the first question that matters -- Rust recomputed,
Python reported a carried field -- so the same guarantee behaved differently
depending on which side of the binding a caller was on.

``tests/test_context_receipt_parity.py`` already pins PyO3 against WASM. Both
of those are the *same Rust engine*, so that test cannot catch a Python/Rust
divergence; it was never meant to. This file holds the comparison nothing else
makes: for an identical state, the Python path and the Rust contract must reach
the same verdict, and neither may promise recovery when the other refuses it.

Encodings are exercised because a digest over text is where this diverges
quietly. ``governance/identity.py`` had already shipped a Python/Rust
divergence on non-ASCII tokens, so agreeing on ASCII proves very little. CJK,
emoji, combining sequences, CRLF and an embedded NUL are each checked.

One asymmetry is recorded rather than hidden: the Python store has no
disposition concept, so it cannot express ``not_recoverable`` -- the state
meaning "this material was dropped and nothing promises it back". That is a
real gap in the Python vocabulary, not a test problem, and it is asserted here
so the gap stays visible until the vocabularies are unified.
"""
from __future__ import annotations

import hashlib

import pytest

from entroly.ccr import CompressedContextStore

entroly_core = pytest.importorskip("entroly_core")

pytestmark = pytest.mark.skipif(
    not hasattr(entroly_core, "recovery_handle_verify_bytes"),
    reason="native entroly_core does not expose the recovery handle contract",
)

# Each payload is a shape where a text digest can diverge between runtimes.
PAYLOADS = {
    "ascii": "def login():\n    return True\n",
    "cjk": "def 登录():\n    返回 真\n",
    "emoji": "msg = '\U0001f680 ship it'\n",
    "combining": "café vs café\n",
    "crlf": "line1\r\nline2\r\n",
    "embedded_nul": "a\x00b\n",
    "lone_surrogate_escaped": "text with \\ud800 as literal text\n",
    "trailing_whitespace": "value = 1   \n\n   \n",
}


def _rust_verdict(content: str, payload: str, *, promises: bool = True) -> str:
    """Build a handle committing to ``content``, then verify ``payload``."""
    digest = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
    handle = entroly_core.recovery_handle_build_json(
        "repo:conformance",
        "cr_conformance",
        "omitted_but_recoverable" if promises else "omitted_and_unavailable",
        "src/subject.py",
        "sha256:source",
        digest if promises else "",
        0,
        len(content.encode("utf-8")),
        "commit:conformance",
        "",
        1,
    )
    return entroly_core.recovery_handle_verify_bytes(
        handle, payload.encode("utf-8")
    )


def _python_verdict(content: str, payload: str) -> str:
    """Store ``content``, substitute ``payload`` on disk, then retrieve."""
    store = CompressedContextStore(max_entries=4)
    handle = store.store(
        source="src/subject.py",
        original_content=content,
        compressed_content="[skeleton]",
    )
    if payload != content:
        store._store[handle]["original"] = payload
    entry = store.retrieve(handle)
    assert entry is not None
    return str(entry["integrity"])


@pytest.mark.parametrize("name", sorted(PAYLOADS))
def test_intact_bytes_verify_in_both_implementations(name: str) -> None:
    content = PAYLOADS[name]

    assert _rust_verdict(content, content) == "verified"
    assert _python_verdict(content, content) == "verified"


@pytest.mark.parametrize("name", sorted(PAYLOADS))
def test_substituted_bytes_are_refused_by_both(name: str) -> None:
    content = PAYLOADS[name]
    forged = content + "// appended\n"

    rust = _rust_verdict(content, forged)
    python = _python_verdict(content, forged)

    assert rust == "commitment_mismatch"
    assert python == "commitment_mismatch", (
        "the Python store must use the engine's token for this state; a "
        "different string for the same verdict forces every caller to "
        "translate, which is how one guarantee becomes two behaviours"
    )
    assert rust == python


@pytest.mark.parametrize("name", sorted(PAYLOADS))
def test_neither_promises_recovery_the_other_refuses(name: str) -> None:
    """The invariant, stated directly."""
    content = PAYLOADS[name]
    for payload in (content, content + "drift"):
        rust_ok = _rust_verdict(content, payload) == "verified"
        python_ok = _python_verdict(content, payload) == "verified"
        assert rust_ok == python_ok, (
            f"{name}: rust verified={rust_ok}, python verified={python_ok}"
        )


def test_a_single_byte_difference_is_caught_by_both() -> None:
    """Not a near-match metric: one flipped byte must refuse."""
    content = "balance = 1000\n"
    forged = "balance = 1001\n"

    assert _rust_verdict(content, forged) == "commitment_mismatch"
    assert _python_verdict(content, forged) == "commitment_mismatch"


def test_whitespace_is_not_normalized_away_by_either() -> None:
    """A normalizing digest would make recovery inexact without saying so."""
    content = "value = 1\n"
    for forged in ("value = 1", "value  = 1\n", "value = 1\r\n", " value = 1\n"):
        assert _rust_verdict(content, forged) == "commitment_mismatch", forged
        assert _python_verdict(content, forged) == "commitment_mismatch", forged


def test_only_the_rust_contract_can_express_not_recoverable() -> None:
    """A recorded asymmetry, so it stays visible until the vocabularies merge.

    ``not_recoverable`` means the material was dropped and nothing promises it
    back. The Rust contract reaches it from a disposition; the Python store has
    no disposition concept and so cannot express it at all. A Python caller
    sees only verified or commitment_mismatch, and must infer "never promised"
    from the absence of an entry -- which is the same answer it gives for an
    eviction.
    """
    assert _rust_verdict("bytes", "bytes", promises=False) == "not_recoverable"

    store = CompressedContextStore(max_entries=4)
    handle = store.store(
        source="src/subject.py",
        original_content="bytes",
        compressed_content="[skeleton]",
    )
    verdicts = {str(store.retrieve(handle)["integrity"])}
    store._store[handle]["original"] = "other"
    verdicts.add(str(store.retrieve(handle)["integrity"]))

    assert verdicts == {"verified", "commitment_mismatch"}
    assert "not_recoverable" not in verdicts


def test_the_engine_tokens_are_the_canonical_set() -> None:
    """If the engine adds a state, this fails until Python accounts for it."""
    content = "subject\n"
    observed = {
        _rust_verdict(content, content),
        _rust_verdict(content, "drift"),
        _rust_verdict(content, content, promises=False),
    }

    assert observed == {"verified", "commitment_mismatch", "not_recoverable"}, (
        "the Rust contract's verdict vocabulary changed; the Python store and "
        "this conformance table must be updated together"
    )
