#!/usr/bin/env python3
"""Strict request-level provider invoice vs Entroly usage ledger reconciliation.

Inputs:
- Entroly UsageLedger.export_csv() complete (all rows in the period; note the
  existing API page limit of 10k).
- A *normalized* per-request provider billing export with fields
  request_id, provider, model, billed_usd, currency.

This script NEVER claims an export is genuine, never synthesizes provider
charges, and never interprets local Ollama costs as commercial invoices.
Aggregate-only invoices cannot support exact request-level reconciliation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

MICRO = Decimal("1000000")
REQUIRED_LEDGER = {"request_id", "provider", "model", "cost_micro_usd", "pricing_source"}
REQUIRED_BILL = {"request_id", "provider", "model", "billed_usd", "currency"}


def money(value: object, *, name: str) -> Decimal:
    try:
        item = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid finite nonnegative amount") from exc
    if not item.is_finite() or item < 0:
        raise ValueError(f"{name} is not a valid finite nonnegative amount")
    return item


def _csv_rows(payload: bytes, required: set[str], label: str) -> list[dict[str, str]]:
    if not payload:
        raise ValueError(f"{label}: export is empty")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: expected UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError(f"{label}: missing or duplicate CSV headers")
    if not required.issubset(reader.fieldnames):
        raise ValueError(f"{label}: missing columns {sorted(required - set(reader.fieldnames))}")
    records = list(reader)
    if not records:
        raise ValueError(f"{label}: no rows to verify")
    if any(None in row for row in records):
        raise ValueError(f"{label}: row has unexpected extra columns")
    return records


def _read_set(payload: bytes, required: set[str], label: str) -> dict[str, dict[str, str]]:
    rows = _csv_rows(payload, required, label)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        rid = row["request_id"].strip()
        if not rid or rid in result:
            raise ValueError(f"{label}: empty or duplicated request ID")
        result[rid] = row
    return result


def reconcile(ledger_csv: bytes, invoice_csv: bytes, *, tolerance_micro_usd: int = 1) -> dict:
    if type(tolerance_micro_usd) is not int or tolerance_micro_usd < 0:
        raise ValueError("tolerance must be a nonnegative integer number of microdollars")
    ledger = _read_set(ledger_csv, REQUIRED_LEDGER, "ledger")
    invoice = _read_set(invoice_csv, REQUIRED_BILL, "invoice")
    ids = sorted(set(ledger) & set(invoice))
    missing_invoice = sorted(set(ledger) - set(invoice))
    missing_ledger = sorted(set(invoice) - set(ledger))
    pairs = []
    total_ledger = Decimal(0)
    total_invoice = Decimal(0)
    failures = []
    for rid in ids:
        left, right = ledger[rid], invoice[rid]
        if right["currency"].upper().strip() != "USD":
            failures.append({"request_id": rid, "reason": "non_usd_currency"})
            continue
        if left["provider"].lower().strip() != right["provider"].lower().strip() or (
                left["model"].strip() != right["model"].strip()):
            failures.append({"request_id": rid, "reason": "provider_or_model_mismatch"})
            continue
        if left["pricing_source"].strip().lower().startswith("unpriced:"):
            failures.append({"request_id": rid, "reason": "unpriced_ledger_event"})
            continue
        ledger_micro = money(left["cost_micro_usd"], name="ledger_micro")
        if ledger_micro != ledger_micro.to_integral_value():
            raise ValueError("ledger microdollars must be integral")
        billed = money(right["billed_usd"], name="provider_billed_usd")
        delta = ledger_micro / MICRO - billed
        total_ledger += ledger_micro / MICRO
        total_invoice += billed
        if abs(delta) > Decimal(tolerance_micro_usd) / MICRO:
            failures.append({
                "request_id": rid,
                "reason": "billed_cost_mismatch",
                "difference_usd": str(delta),
            })
        pairs.append(rid)
    reconciled = bool(pairs) and not missing_invoice and not missing_ledger and not failures
    return {
        "schema": "entroly.provider-invoice-reconciliation.v1",
        "status": "reconciled" if reconciled else "unverified",
        "provenance": {
            "ledger_sha256": hashlib.sha256(ledger_csv).hexdigest(),
            "invoice_export_sha256": hashlib.sha256(invoice_csv).hexdigest(),
            "invoice_authenticity": "NOT_INDEPENDENTLY_VERIFIED",
            "meaning": "Matches the two user-provided CSV exports, not an attested provider bill",
        },
        "tolerance_micro_usd": tolerance_micro_usd,
        "matched_request_count": len(pairs),
        "ledger_request_count": len(ledger),
        "invoice_request_count": len(invoice),
        "missing_invoice_request_ids": missing_invoice,
        "missing_ledger_request_ids": missing_ledger,
        "mismatches": failures,
        "matched_ledger_cost_usd": str(total_ledger),
        "matched_invoice_cost_usd": str(total_invoice),
        "aggregate_difference_usd": str(total_ledger - total_invoice),
        "limitations": [
            "Provider-specific billing must be normalized without inventing per-request detail",
            "Invoice authenticity and completeness cannot be established from a CSV alone",
            "Cost equality does not prove RAW/COMPRESS paired task quality or ROI",
            "Taxes, credits, prepaid commitments, and compute hosting are not covered",
            "UsageLedger.export_csv() has a 10,000-row query limit; ensure full ledger coverage",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--invoice", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tolerance-micro-usd", type=int, default=1)
    args = parser.parse_args()
    if args.out.resolve() in {args.ledger.resolve(), args.invoice.resolve()}:
        raise ValueError("report path must not overwrite the source exports")
    result = reconcile(args.ledger.read_bytes(), args.invoice.read_bytes(),
                       tolerance_micro_usd=args.tolerance_micro_usd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "matched": result["matched_request_count"],
                      "mismatches": len(result["mismatches"])}))
    return 0 if result["status"] == "reconciled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
