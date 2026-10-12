"""A content-addressed store must answer the question it exists to answer.

``retrieve`` carried the stored ``content_sha256`` out of the store untouched.
The commitment was *reported*, never *checked*, so the one question a
content-addressed store exists to answer -- are these the bytes the handle
names -- went unanswered on the Python path. The Rust path already recomputed
and could return ``not_recoverable``, so one nominal guarantee had two
behaviours depending on which side of the binding you were on.

``store`` had the matching gap: an existing entry was replaced unconditionally.
A handle is ``sha256(source + NUL + content)`` truncated to 24 hex characters,
so it carries 96 bits; a collision would substitute one payload for another
under an identifier already handed out, and every digest in the new entry would
agree with the substitute. Internally consistent, and wrong.

These tests keep four outcomes apart, because they call for different actions:

  verified      -- the bytes hash to their commitment
  absent        -- nothing stored; for an exact handle, that version was evicted
  corrupt       -- stored bytes no longer hash to their commitment
  conflict      -- a second, different payload claims an issued handle

Collapsing corrupt into absent would send a caller to retry a path that cannot
succeed. Collapsing conflict into success would hand back bytes that were never
compressed.
"""
from __future__ import annotations

import pytest

from entroly.ccr import CcrHandleCollision, CompressedContextStore, _content_hash
from entroly.integrations.exact_recovery import ExactRecoveryError, retrieve_exact


def _store_one(store: CompressedContextStore, source: str, content: str) -> str:
    return store.store(
        source=source,
        original_content=content,
        compressed_content="[skeleton]",
        resolution="skeleton",
        original_tokens=100,
        compressed_tokens=10,
    )


@pytest.fixture()
def store() -> CompressedContextStore:
    return CompressedContextStore(max_entries=16)


# ── verified ──────────────────────────────────────────────────────────


def test_an_intact_entry_reports_verified(store) -> None:
    handle = _store_one(store, "file:auth.py", "def login():\n    return True\n")

    entry = store.retrieve(handle)

    assert entry is not None
    assert entry["integrity"] == "verified"
    assert entry["original"] == "def login():\n    return True\n"
    assert entry["content_sha256"] == _content_hash(entry["original"])


def test_verification_is_recomputed_not_trusted(store) -> None:
    """Editing the commitment to match forged bytes must not certify them."""
    handle = _store_one(store, "file:auth.py", "original bytes")
    forged = "forged bytes"
    store._store[handle]["original"] = forged
    store._store[handle]["content_sha256"] = _content_hash(forged)

    entry = store.retrieve(handle)

    # Both fields now agree with each other, so this entry is internally
    # consistent. That is exactly why a self-hash is not authenticity: the
    # store cannot distinguish this from a legitimate re-store, and says
    # verified. The protection against it is store-side refusal, not retrieval.
    assert entry["integrity"] == "verified"
    assert entry["original"] == forged


# ── corrupt, and distinct from absent ─────────────────────────────────


def test_tampered_content_is_reported_and_withheld(store) -> None:
    handle = _store_one(store, "file:auth.py", "original bytes")
    store._store[handle]["original"] = "tampered bytes"

    entry = store.retrieve(handle)

    assert entry is not None, "corrupt is not absent"
    assert entry["integrity"] == "content_mismatch"
    assert entry["original"] is None, (
        "unverified bytes must not be returned; a caller that ignores the "
        "integrity field would otherwise use them"
    )
    detail = entry["integrity_detail"]
    assert detail["expected_content_sha256"] == _content_hash("original bytes")
    assert detail["actual_content_sha256"] == _content_hash("tampered bytes")


def test_an_entry_with_no_commitment_cannot_be_called_verified(store) -> None:
    handle = _store_one(store, "file:auth.py", "bytes")
    store._store[handle].pop("content_sha256")

    entry = store.retrieve(handle)

    assert entry["integrity"] == "content_mismatch"
    assert "no content commitment" in entry["integrity_detail"]["reason"]
    assert entry["original"] is None


def test_absent_stays_absent(store) -> None:
    assert store.retrieve("ccr:" + "0" * 24) is None
    assert store.retrieve("file:never-stored.py") is None


def test_exact_recovery_distinguishes_evicted_from_corrupt(store) -> None:
    """The two failures call for different actions, so not one message."""
    missing = "ccr:" + "a" * 24
    with pytest.raises(ExactRecoveryError) as absent:
        retrieve_exact(missing, store=store)
    assert "unavailable or was evicted" in str(absent.value)

    handle = _store_one(store, "file:auth.py", "original bytes")
    store._store[handle]["original"] = "tampered bytes"
    with pytest.raises(ExactRecoveryError) as corrupt:
        retrieve_exact(handle, store=store)

    message = str(corrupt.value)
    assert "integrity check" in message
    assert "content_mismatch" in message
    assert "evicted" not in message, (
        "a damaged store must not be reported as an eviction; re-requesting "
        "cannot fix it and the caller would retry forever"
    )


def test_a_verified_exact_recovery_says_so(store) -> None:
    handle = _store_one(store, "file:auth.py", "exact original bytes")

    result = retrieve_exact(handle, store=store)

    assert result["status"] == "recovered"
    assert result["integrity"] == "verified"
    assert result["original_content"] == "exact original bytes"


# ── conflict ──────────────────────────────────────────────────────────


def test_identical_content_restores_idempotently(store) -> None:
    """Re-storing the same payload is not a collision."""
    first = _store_one(store, "file:auth.py", "same bytes")
    second = _store_one(store, "file:auth.py", "same bytes")

    assert first == second
    assert store.retrieve(first)["integrity"] == "verified"


def test_a_colliding_handle_is_refused_without_overwriting(store, monkeypatch) -> None:
    """96 bits can collide; the stored payload is what a handle already promises."""
    import entroly.ccr as ccr

    fixed = "ccr:" + "f" * 24
    monkeypatch.setattr(ccr, "_retrieval_handle", lambda source, content: fixed)

    _store_one(store, "file:first.py", "first payload")

    with pytest.raises(CcrHandleCollision) as collision:
        _store_one(store, "file:second.py", "second payload")

    assert fixed in str(collision.value)
    assert "refusing to overwrite" in str(collision.value)

    survivor = store.retrieve(fixed)
    assert survivor["original"] == "first payload", (
        "the first payload is the one the issued handle promises; replacing it "
        "would make recovery return bytes that were never compressed"
    )
    assert survivor["integrity"] == "verified"


def test_a_refused_collision_does_not_repoint_the_source_key(store, monkeypatch) -> None:
    """A rejected write must leave no trace, including the latest-by-source map."""
    import entroly.ccr as ccr

    fixed = "ccr:" + "e" * 24
    monkeypatch.setattr(ccr, "_retrieval_handle", lambda source, content: fixed)
    _store_one(store, "file:first.py", "first payload")

    with pytest.raises(CcrHandleCollision):
        _store_one(store, "file:second.py", "second payload")

    assert store.retrieve("file:second.py") is None, (
        "the refused source must not resolve to the surviving payload"
    )
    assert store.retrieve("file:first.py")["original"] == "first payload"
