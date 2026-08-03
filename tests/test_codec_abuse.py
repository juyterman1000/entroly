"""Abuse suite: try to break the codec and recovery contracts.

Every test here is written to FALSIFY a claim the codecs make, not to confirm
one. The recovery contract in particular is a security property -- a caller
that trusts `recover()` and gets back something other than the original bytes
has been silently lied to -- so it is attacked from several directions:
forged references, corrupt stores, hostile encodings, and inputs designed to
make a codec inflate, hang or recurse.

Threat model covered here
-------------------------

* **Forged receipts** -- an attacker (or a corrupt store) returns content that
  is not what was removed. Content addressing must catch it.
* **Compression bombs** -- deeply nested or hugely repetitive structures that
  make a codec allocate or recurse without bound.
* **Malformed input** -- truncated JSON, mixed encodings, lone surrogates,
  NUL bytes. A codec that "mostly parses" these can corrupt data silently.
* **Inflation** -- a "compressed" form larger than the original wastes tokens
  and is never a valid choice.
* **Cross-scope leakage** -- one store's references resolving in another.

Not covered here, and not claimed: prompt injection in tool output, symlink
and path traversal, proxy authentication, and concurrent-writer durability.
Those need the persistence and proxy layers, which this suite does not touch.
"""

from __future__ import annotations

import json

import pytest

from entroly.codec import (
    RecoveryReference,
    RecoveryStore,
    content_digest,
)
from entroly.codecs_builtin import JsonCodec, LogCodec, ShellCodec, default_registry
from entroly.codecs_content import CodeCodec, ConversationCodec, DocumentCodec, SchemaCodec

ALL_CODECS = [JsonCodec, LogCodec, ShellCodec, CodeCodec, DocumentCodec,
              ConversationCodec, SchemaCodec]


# ── Forged and corrupted recovery ───────────────────────────────────────────


def test_forged_digest_is_rejected():
    store = RecoveryStore()
    ref = store.put("the real omitted content", item_count=1)
    forged = RecoveryReference(
        digest=content_digest("attacker supplied"),
        byte_length=ref.byte_length,
    )
    assert not forged.verify("the real omitted content")


def test_forged_byte_length_is_rejected():
    """Length is authenticated, not decorative."""
    ref = RecoveryReference(digest=content_digest("abc"), byte_length=9999)
    assert not ref.verify("abc")


def test_corrupt_store_entry_raises_rather_than_returning_wrong_bytes(tmp_path):
    """Swap the persisted payload; content addressing must catch it.

    The store is backed by the hardened scoped retrieval store, so this
    tampers with the real persisted record rather than an in-memory dict.
    """
    path = tmp_path / "rec.json"
    store = RecoveryStore(path)
    ref = store.put("original", item_count=1)

    raw = json.loads(path.read_text(encoding="utf-8"))
    swapped = False
    for item in raw.get("items", []):
        for span in item.get("spans", []):
            if isinstance(span.get("content"), str) and "original" in span["content"]:
                span["content"] = span["content"].replace("original", "tampered")
                swapped = True
    assert swapped, "fixture did not find the stored span content to corrupt"
    path.write_text(json.dumps(raw), encoding="utf-8")

    reloaded = RecoveryStore(path)
    with pytest.raises((ValueError, KeyError)):
        reloaded.recover(ref)


def test_missing_entry_raises_rather_than_returning_empty():
    """Silently returning "" would look like successful recovery of nothing."""
    store = RecoveryStore()
    ref = RecoveryReference(digest=content_digest("never stored"), byte_length=12)
    with pytest.raises(KeyError):
        store.recover(ref)


def test_reference_from_one_store_does_not_resolve_in_another():
    a, b = RecoveryStore(), RecoveryStore()
    ref = a.put("scope A secret", item_count=1)
    with pytest.raises(KeyError):
        b.recover(ref)


def test_wholesale_sidecar_replacement_does_not_yield_false_recovery(tmp_path):
    """Replacing the whole file must fail, never return attacker content."""
    path = tmp_path / "recovery.json"
    ref = RecoveryStore(path).put("original bytes", item_count=1)
    path.write_text(json.dumps({"items": []}), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        RecoveryStore(path).recover(ref)


# ── Compression bombs and resource bounds ───────────────────────────────────


@pytest.mark.parametrize("depth", [1000, 4000, 20000])
def test_deeply_nested_json_is_declined_deterministically(depth):
    """A nesting bomb must be refused by structure, not by stack luck.

    Catching RecursionError was not enough: where it fires depends on the
    platform's stack, so the guard held on Windows and the same input still
    recursed on Linux CI. The depth is now counted before parsing, which is
    the same answer everywhere.
    """
    payload = "[" * depth + "1" + "]" * depth
    codec = JsonCodec()
    decision = codec.supports(payload)
    assert not decision, f"depth {depth} should be declined without parsing"
    assert "nesting" in decision.reason


def test_realistic_nesting_is_still_accepted():
    """The guard must not reject ordinary payloads."""
    nested = json.dumps({"a": [{"b": [{"c": [1, 2, 3]}]}]})
    assert JsonCodec().supports(nested)


def test_brackets_inside_strings_do_not_count_as_nesting():
    from entroly.codecs_builtin import _exceeds_nesting_limit

    assert not _exceeds_nesting_limit(json.dumps({"s": "[" * 500}))


def test_hugely_repetitive_log_stays_bounded():
    text = "\n".join("2026-08-02T10:00:00Z ERROR pool exhausted" for _ in range(50_000))
    reps = LogCodec().representations(text, source_id="flood")
    smallest = min(reps, key=lambda r: r.token_cost)
    assert smallest.token_cost < len(text) // 400, (
        "50k identical lines should collapse hard; got "
        f"{smallest.token_cost} tokens"
    )


def test_wide_json_object_does_not_explode():
    payload = json.dumps({f"k{i}": i for i in range(20_000)})
    reps = JsonCodec().representations(payload, source_id="wide")
    assert reps and all(len(r.text) <= max(len(payload), 1) * 4 for r in reps)


# ── Malformed and hostile input ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "label,payload",
    [
        ("truncated object", '{"a": 1, "b": '),
        ("trailing commas", '{"a": 1,,}'),
        ("bare word", "not json at all"),
        ("nul bytes", '{"a": "\x00\x00"}'),
        ("lone high surrogate", '{"a": "\ud800"}'),
        ("empty", ""),
        ("whitespace only", "   \n\t  "),
    ],
)
def test_malformed_input_never_crashes_any_codec(label, payload):
    """A codec may decline, but it must not raise on hostile input."""
    for cls in ALL_CODECS:
        codec = cls()
        try:
            decision = codec.supports(payload)
            if decision:
                codec.representations(payload, source_id="hostile", query="q")
        except Exception as exc:  # noqa: BLE001 - the point is to catch everything
            pytest.fail(f"{cls.__name__} raised on {label}: {exc!r}")


def test_registry_declines_malformed_json_rather_than_rewriting_it():
    assert default_registry().select('{"a": 1, "b": [1,2,,]}') is None


@pytest.mark.parametrize("payload", ["", "   ", "\n\n"])
def test_empty_input_selects_no_codec(payload):
    assert default_registry().select(payload) is None


def test_unicode_survives_recovery_byte_exactly():
    store = RecoveryStore()
    payload = json.dumps(
        {"note": "naïve café — 日本語 🚀", "items": [{"x": i} for i in range(30)]},
        indent=2,
        ensure_ascii=False,
    )
    rep = JsonCodec(store).representations(payload, source_id="u")[-1]
    if rep.recovery is not None:
        assert store.recover(rep.recovery) == payload


@pytest.mark.parametrize(
    "label,text",
    [
        ("LF", "2026-01-01 INFO a\n2026-01-01 INFO a\n2026-01-01 INFO a\n"),
        ("CRLF", "2026-01-01 INFO a\r\n2026-01-01 INFO a\r\n2026-01-01 INFO a\r\n"),
        ("no trailing newline", "2026-01-01 INFO a\n2026-01-01 INFO a\n2026-01-01 INFO a"),
        ("mixed", "2026-01-01 INFO a\r\n2026-01-01 INFO a\n2026-01-01 INFO a\r\n"),
    ],
)
def test_line_ending_variants_recover_byte_exactly(label, text):
    store = RecoveryStore()
    rep = LogCodec(store).representations(text, source_id="le")[-1]
    if rep.recovery is None:
        assert rep.text == text
        return
    assert store.recover(rep.recovery) == text, f"{label}: recovery altered bytes"


# ── Anti-inflation ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "label,payload",
    [
        ("tiny json", '{"a":1}'),
        ("one log line", "2026-08-02T10:00:00Z INFO started"),
        ("two code lines", "import os\n\n\ndef f():\n    return 1\n"),
        ("short schema", '{"$schema":"x","type":"object"}'),
    ],
)
def test_no_codec_offers_a_representation_larger_than_its_input(label, payload):
    for cls in ALL_CODECS:
        codec = cls()
        if not codec.supports(payload):
            continue
        for rep in codec.representations(payload, source_id="tiny", query="q"):
            assert len(rep.text) <= len(payload), (
                f"{cls.__name__} inflated {label}: {len(payload)} -> {len(rep.text)}"
            )


# ── Protected-evidence honesty ──────────────────────────────────────────────


def test_protected_evidence_claims_are_true_of_the_emitted_text():
    """A codec claiming evidence it did not keep is worse than claiming none."""
    samples = {
        JsonCodec: json.dumps({"id": "x1", "items": [{"a": i} for i in range(30)]}, indent=2),
        LogCodec: "\n".join(
            [f"2026-08-02T10:00:0{i%10}Z ERROR pool exhausted (retry {i})" for i in range(40)]
        ),
        ShellCodec: "$ pytest\n" + "ok\n" * 200 + "1 failed, 2 passed\nexit code 1\n",
        CodeCodec: "import os\n\n\ndef a():\n    x = 1\n    return x\n\n\ndef b():\n    return 2\n",
        SchemaCodec: json.dumps(
            {"$schema": "s", "type": "object", "required": ["id"],
             "properties": {"id": {"type": "string", "description": "d" * 200}}}, indent=2
        ),
    }
    for cls, text in samples.items():
        for rep in cls().representations(text, source_id="p"):
            missing = rep.verify_protected_evidence()
            assert not missing, f"{cls.__name__} claimed but dropped: {missing}"
