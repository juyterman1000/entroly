"""Identifier-bearing JSON must compress without destroying identifiers.

The JSON codec protects load-bearing values -- ``sku``, ``order_id``,
``email``, ``price`` -- by refusing any lossy form that would drop one. That is
the right instinct, but it left only two outcomes: summarise everything, or
summarise nothing. Real API payloads are almost entirely identifiers, so they
took the second branch and compressed by 0%.

Measured on 200 records before this change, the only difference being one key
name::

    {"id", "name"}  ->  53%
    {"id", "sku"}   ->   0%

Two separate limits caused it: the schema rendering is *larger* than the source
for identifier-dense records, and enumerating every protected value overflows a
node budget. Both made the codec decline entirely.

The columnar form keeps whole load-bearing columns verbatim and elides the
repeated keys and non-identifying columns around them, so preservation is
structural rather than a substring scan. These tests fix the property that
makes it safe: **every load-bearing value survives byte-exact, or the
representation is not offered.**
"""

from __future__ import annotations

import json

import pytest

from entroly.codec import RecoveryStore
from entroly.codecs_builtin import default_registry
from entroly.universal_compress import _is_load_bearing_key

IDENTIFIER_ROWS = [
    {
        "order_id": f"O-{i:05d}",
        "sku": f"SKU-{i}-{i * 7}",
        "email": f"user{i}@example.com",
        "price": round(i * 1.37, 2),
        "qty": i % 9,
        "note": "standard shipping",
    }
    for i in range(120)
]


def _columnar_for(payload: str):
    registry = default_registry(RecoveryStore())
    reps = registry.representations(
        payload, source_id="p", content_type="", query=""
    )
    return next(
        (r for r in reps if r.representation_id.endswith("#json.columnar")), None
    )


@pytest.mark.parametrize("wrapped", [False, True], ids=["bare", "wrapped"])
def test_every_load_bearing_value_survives_byte_exact(wrapped: bool) -> None:
    """The whole point. One dropped identifier invalidates the form."""
    payload = json.dumps({"data": IDENTIFIER_ROWS} if wrapped else IDENTIFIER_ROWS)
    rep = _columnar_for(payload)
    assert rep is not None, "no columnar representation offered"

    rendered = json.loads(rep.text)
    for key in IDENTIFIER_ROWS[0]:
        if not _is_load_bearing_key(key):
            continue
        expected = [row[key] for row in IDENTIFIER_ROWS]
        assert rendered.get(key) == expected, (
            f"load-bearing column {key!r} was altered or dropped"
        )


def test_identifier_dense_payloads_actually_compress() -> None:
    """Regression on the measured defect: these used to yield exactly 0%."""
    payload = json.dumps(IDENTIFIER_ROWS)
    rep = _columnar_for(payload)
    assert rep is not None
    original_tokens = len(payload) // 4
    assert rep.token_cost < original_tokens * 0.9, (
        f"columnar form saved almost nothing: {rep.token_cost} vs "
        f"{original_tokens}"
    )


def test_the_original_is_recoverable() -> None:
    """Compression here is a view, not a replacement."""
    store = RecoveryStore()
    payload = json.dumps(IDENTIFIER_ROWS)
    reps = default_registry(store).representations(
        payload, source_id="p", content_type="", query=""
    )
    rep = next(r for r in reps if r.representation_id.endswith("#json.columnar"))
    assert rep.recovery is not None, "columnar form must carry a recovery handle"
    assert store.recover(rep.recovery) == payload


def test_non_uniform_records_are_declined() -> None:
    """Ragged keys have no columnar form; guessing one would lose data."""
    ragged = [{"sku": "A", "qty": 1}, {"sku": "B"}, {"sku": "C", "qty": 3}]
    assert _columnar_for(json.dumps(ragged)) is None


def test_nested_values_are_declined() -> None:
    """A nested object under a record key cannot be preserved columnwise."""
    nested = [{"sku": f"S{i}", "meta": {"a": i}} for i in range(20)]
    assert _columnar_for(json.dumps(nested)) is None


def test_payload_without_identifiers_is_left_to_the_schema_form() -> None:
    """Nothing to protect means the existing schema form compresses harder.

    `note` is deliberately low-cardinality here. Protection is now decided by
    the data as well as the key name, so a near-unique column counts as an
    identifier whatever it is called -- which is the point -- and a payload
    where every column repeats has nothing for the columnar form to hold.
    """
    rows = [
        {"note": f"n{i % 3}", "label": "x", "flag": True} for i in range(50)
    ]
    assert _columnar_for(json.dumps(rows)) is None


def test_a_column_that_is_all_identifiers_still_compresses_a_little() -> None:
    """Even then the repeated key names are removable, and values must stay."""
    rows = [{"id": i, "sku": f"S{i}"} for i in range(200)]
    payload = json.dumps(rows)
    rep = _columnar_for(payload)
    assert rep is not None
    rendered = json.loads(rep.text)
    assert rendered["id"] == [r["id"] for r in rows]
    assert rendered["sku"] == [r["sku"] for r in rows]
    assert rep.token_cost < len(payload) // 4


def test_evidence_sample_is_bounded() -> None:
    """The receipt must not carry thousands of identifiers."""
    rep = _columnar_for(json.dumps(IDENTIFIER_ROWS))
    assert rep is not None
    assert 0 < len(rep.protected_evidence) <= 8
    # And what it does carry must really be present.
    for value in rep.protected_evidence:
        assert value.strip('"') in rep.text or value in rep.text


def test_declines_when_a_column_would_be_altered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The structural check must be load-bearing, not decorative."""
    import entroly.codecs_builtin as builtin

    real = builtin._columnar_json

    def truncating(data, is_load_bearing_key):  # noqa: ANN001
        rendered = real(data, is_load_bearing_key)
        if rendered is None:
            return None
        payload = json.loads(rendered)
        # Target a real identifier column, not the metadata lists that sit
        # beside it -- truncating `_verbatim_columns` proves nothing.
        assert isinstance(payload.get("sku"), list) and len(payload["sku"]) > 2
        payload["sku"] = payload["sku"][:-1]  # silently lose one identifier
        return json.dumps(payload)

    monkeypatch.setattr(builtin, "_columnar_json", truncating)
    assert _columnar_for(json.dumps(IDENTIFIER_ROWS)) is None, (
        "a truncated identifier column was accepted; the preservation check "
        "does not actually gate the representation"
    )
