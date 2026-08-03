"""CSV/TSV codec (6.2).

A table is read for two things, and losing either makes the rest useless:

* the **contract** -- which columns exist and what each holds. A summary
  without column names cannot be joined, filtered or reasoned about.
* the **shape** -- size, sparsity, range, extremes. Someone asking "did this
  export go wrong" needs the missingness and the bounds, not row 4,000.

So these tests check the header survives verbatim, the per-column facts are
present and correct, and the complete original is recoverable byte-for-byte.
They also check the codec DECLINES prose that merely contains commas, which is
the failure that would corrupt a document by reformatting it as a table.
"""

from __future__ import annotations

import random

import pytest

from entroly.codec import RecoveryStore
from entroly.codecs_builtin import default_registry
from entroly.codecs_table import TableCodec, _column_type


def _orders_csv(rows: int = 500) -> str:
    random.seed(3)
    out = ["order_id,customer,amount_cents,status,notes"]
    for i in range(rows):
        out.append(
            f"ord-{i:05d},cust{i % 40},{random.randint(100, 900000)},"
            f"{'paid' if i % 7 else 'failed'},{'' if i % 3 else 'note'}"
        )
    return "\n".join(out)


def _summary(text: str, source_id: str = "orders.csv"):
    return TableCodec().representations(text, source_id=source_id)[-1]


# ── Routing ─────────────────────────────────────────────────────────────────


def test_registry_routes_csv_to_the_table_codec():
    codec = default_registry().select(_orders_csv())
    assert codec is not None and codec.name == "table"


def test_tsv_is_recognised():
    tsv = "\n".join(["a\tb\tc"] + [f"{i}\t{i * 2}\tx" for i in range(20)])
    assert TableCodec().supports(tsv)


@pytest.mark.parametrize(
    "label,text",
    [
        ("prose with commas", "One, two, and three. " * 40),
        ("json array", '[{"a": 1}, {"a": 2}, {"a": 3}]'),
        ("single column", "\n".join(["header"] + [str(i) for i in range(30)])),
        ("too few rows", "a,b\n1,2\n3,4"),
        ("ragged", "\n".join(["a,b,c"] + [",".join(["x"] * (i % 5 + 1)) for i in range(30)])),
    ],
)
def test_declines_content_that_is_not_a_table(label, text):
    assert not TableCodec().supports(text), f"{label} should not be claimed"


# ── Contract preservation ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "column", ["order_id", "customer", "amount_cents", "status", "notes"]
)
def test_every_column_name_survives(column):
    assert column in _summary(_orders_csv()).text


def test_header_row_is_kept_verbatim():
    assert _summary(_orders_csv()).text.startswith(
        "order_id,customer,amount_cents,status,notes"
    )


def test_column_names_are_declared_as_protected_and_are_present():
    rep = _summary(_orders_csv())
    assert rep.protected_evidence
    assert rep.verify_protected_evidence() == ()


# ── Shape ───────────────────────────────────────────────────────────────────


def test_row_and_column_counts_are_reported():
    text = _summary(_orders_csv(500)).text
    assert "500 data rows" in text and "5 columns" in text


def test_numeric_column_reports_bounds_and_median():
    text = _summary(_orders_csv()).text
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("amount_cents"))
    assert "number" in line
    for stat in ("min=", "p50=", "max="):
        assert stat in line, f"{stat} missing from {line!r}"


def test_missingness_is_reported():
    text = _summary(_orders_csv()).text
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("notes"))
    assert "missing" in line, f"sparse column reported no missingness: {line!r}"


def test_identifier_column_is_typed_as_unique():
    assert _column_type([f"id-{i}" for i in range(50)]) == "unique"


def test_low_cardinality_column_reports_distinct_count():
    text = _summary(_orders_csv()).text
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("status"))
    assert "2 distinct" in line


def test_edge_rows_are_kept_so_a_record_shape_is_visible():
    text = _summary(_orders_csv()).text
    assert "ord-00000" in text, "first data row should be shown"
    assert "ord-00499" in text, "last data row should be shown"
    assert "rows elided" in text, "the gap must be declared, not hidden"


# ── Recovery and cost ───────────────────────────────────────────────────────


def test_original_table_is_recoverable_byte_exactly():
    store = RecoveryStore()
    csv_text = _orders_csv()
    rep = TableCodec(store).representations(csv_text, source_id="orders.csv")[-1]
    assert rep.recovery is not None
    assert store.recover(rep.recovery) == csv_text


def test_summary_is_far_smaller_than_the_table():
    csv_text = _orders_csv()
    reps = TableCodec().representations(csv_text, source_id="orders.csv")
    assert reps[-1].token_cost < reps[0].token_cost // 10


def test_small_table_is_left_alone():
    """Below the edge-row threshold there is nothing to elide."""
    small = "\n".join(["a,b,c"] + [f"{i},{i},{i}" for i in range(6)])
    reps = TableCodec().representations(small, source_id="small.csv")
    assert all(len(r.text) <= len(small) for r in reps)
