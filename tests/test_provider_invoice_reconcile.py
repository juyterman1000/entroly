"""Reconcile only complete, exact request-level provider exports."""
import csv
import io

import pytest

from benchmarks.provider_invoice_reconcile import reconcile


def rows(headers, entries):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=headers)
    writer.writeheader()
    writer.writerows(entries)
    return out.getvalue().encode()


LEDGER_HEADERS = ["request_id", "provider", "model", "cost_micro_usd", "pricing_source"]
BILL_HEADERS = ["request_id", "provider", "model", "billed_usd", "currency"]


def ledger(*items):
    return rows(LEDGER_HEADERS, list(items))


def invoice(*items):
    return rows(BILL_HEADERS, list(items))


def l(rid="a", price="100", pricing_source="explicit"):
    return dict(request_id=rid, provider="openai", model="example", cost_micro_usd=price,
                pricing_source=pricing_source)


def b(rid="a", price="0.000100", currency="USD"):
    return dict(request_id=rid, provider="openai", model="example", billed_usd=price,
                currency=currency)


def test_reconciles_exact_invoice_and_ledger_without_claiming_authenticity():
    x = reconcile(ledger(l("a", "100"), l("b", "200")),
                  invoice(b("a", "0.000100"), b("b", "0.000200")))
    assert x["status"] == "reconciled"
    assert x["matched_request_count"] == 2
    assert x["matched_invoice_cost_usd"] == "0.000300"
    assert x["provenance"]["invoice_authenticity"] == "NOT_INDEPENDENTLY_VERIFIED"
    assert len(x["provenance"]["invoice_export_sha256"]) == 64


@pytest.mark.parametrize("bad", ["NaN", "sNaN", "Infinity", "-Infinity", "-0.1", "not-money"])
def test_rejects_invalid_provider_amounts(bad):
    with pytest.raises(ValueError, match="finite nonnegative"):
        reconcile(ledger(l()), invoice(b(price=bad)))


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-1", "0.5", "not-money"])
def test_rejects_invalid_ledger_microdollars(bad):
    with pytest.raises(ValueError):
        reconcile(ledger(l(price=bad)), invoice(b()))


def test_mismatch_marks_unverified_even_when_aggregate_looks_equal():
    x = reconcile(ledger(l("a", "100"), l("b", "200")),
                  invoice(b("a", "0.000200"), b("b", "0.000100")))
    assert x["status"] == "unverified"
    assert len(x["mismatches"]) == 2
    assert x["aggregate_difference_usd"] == "0.000000"


def test_missing_invoice_request_id_fails_closed():
    x = reconcile(ledger(l("a"), l("b")), invoice(b("a")))
    assert x["status"] == "unverified"
    assert x["missing_invoice_request_ids"] == ["b"]


def test_extra_invoice_request_id_fails_closed():
    x = reconcile(ledger(l("a")), invoice(b("a"), b("b")))
    assert x["status"] == "unverified"
    assert x["missing_ledger_request_ids"] == ["b"]


def test_rejects_price_while_ledger_flags_unpriced():
    x = reconcile(ledger(l(pricing_source="unpriced:provider")), invoice(b()))
    assert x["status"] == "unverified"
    assert x["mismatches"][0]["reason"] == "unpriced_ledger_event"


def test_rejects_foreign_currency_and_wrong_provider():
    x = reconcile(ledger(l()), invoice(b(currency="EUR")))
    assert x["status"] == "unverified"
    assert x["mismatches"][0]["reason"] == "non_usd_currency"
    wrong = b()
    wrong["provider"] = "anthropic"
    y = reconcile(ledger(l()), invoice(wrong))
    assert y["status"] == "unverified"
    assert y["mismatches"][0]["reason"] == "provider_or_model_mismatch"


def test_one_microdollar_tolerance_is_explicit():
    x = reconcile(ledger(l(price="101")), invoice(b(price="0.000100")))
    assert x["status"] == "reconciled"
    y = reconcile(ledger(l(price="101")), invoice(b(price="0.000100")),
                  tolerance_micro_usd=0)
    assert y["status"] == "unverified"


def test_duplicate_ids_rejected_without_overwrite():
    with pytest.raises(ValueError, match="duplicated"):
        reconcile(ledger(l("a"), l("a")), invoice(b()))


def test_missing_columns_and_aggregate_only_invoice_not_sufficient():
    aggregate = rows(["provider", "billed_usd"], [{"provider": "openai", "billed_usd": "1"}])
    with pytest.raises(ValueError, match="missing columns"):
        reconcile(ledger(l()), aggregate)


def test_empty_exports_are_not_real_evidence():
    with pytest.raises(ValueError, match="empty"):
        reconcile(b"", invoice(b()))
    with pytest.raises(ValueError, match="no rows"):
        reconcile(ledger(), invoice(b()))


def test_ledger_and_invoice_exports_remain_private_not_in_report():
    x = reconcile(ledger(l(rid="sensitive-id")), invoice(b(rid="sensitive-id")))
    assert x["status"] == "reconciled"
    assert "sensitive-id" not in str(x)
