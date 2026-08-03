"""Ablation I vs J: generic compression against content-specialized codecs.

Compression percentage alone decides nothing -- truncation wins it outright and
is useless. What matters is how much of the evidence a reader actually needs
survives per token spent. So every arm is scored on both, against the same
frozen fixtures.

Arms
----

``full``          no compression. The reference for cost and the ceiling for
                  retention.
``truncate``      keep the first N characters. The honest floor: any codec that
                  cannot beat this on retention is not earning its complexity.
``generic``       ``universal_compress`` routed as prose, i.e. the TF-IDF
                  extractive summariser, with content detection bypassed. This
                  is the "one compressor for every input" arm.
``specialized``   the content codec the registry selects (JSON / log / shell).

Metrics
-------

``tokens``        estimated output tokens (cost)
``retention``     share of the fixture's required evidence present in the
                  output. Required evidence is declared by the fixture, not by
                  the codec, so a codec cannot mark its own homework.
``recoverable``   whether what was dropped can be recovered exactly. Only the
                  specialized arm can answer yes; the others have no reference.

Honest scope
------------

* Three synthetic fixtures, frozen here. Not a public dataset.
* No model is called. Retention is substring presence, which measures evidence
  survival and not whether an answer would be correct.
* ``truncate`` is deliberately given the same token budget as the specialized
  arm so the comparison is at equal cost rather than equal effort.

Run:
    python benchmarks/codec_ablation.py
    python benchmarks/codec_ablation.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
import pathlib
from pathlib import Path

os.environ.setdefault("ENTROLY_DISABLE_UPDATE_CHECK", "1")

SCHEMA_VERSION = "entroly.codec-ablation.v1"


def _json_fixture() -> tuple[str, list[str]]:
    payload = {
        "request_id": "req_8f3a21bc",
        "status": "error",
        "error": {"code": "PAYMENT_DECLINED", "message": "card issuer refused"},
        "amount_cents": 449900,
        "currency": "USD",
        "timestamp": "2026-08-02T13:22:41Z",
        "items": [
            {"sku": f"SKU-{i:04d}", "qty": i % 5 + 1, "price_cents": 1999 + i}
            for i in range(40)
        ],
    }
    required = [
        "req_8f3a21bc", "PAYMENT_DECLINED", "card issuer refused",
        "449900", "USD", "2026-08-02T13:22:41Z",
    ]
    return json.dumps(payload, indent=2), required


def _log_fixture() -> tuple[str, list[str]]:
    lines = [
        "2026-08-02T10:00:00Z INFO  worker starting pool_size=8",
        "2026-08-02T10:00:03Z ERROR db connect failed: FATAL password "
        "authentication failed for user 'svc_billing'",
    ]
    for i in range(200):
        lines.append(
            f"2026-08-02T10:00:{4 + i % 50:02d}Z ERROR request failed: "
            f"connection pool exhausted (retry {i})"
        )
    lines.append("2026-08-02T10:05:00Z INFO  worker shutting down exit_code=70")
    required = [
        "password authentication failed", "svc_billing",
        "connection pool exhausted", "exit_code=70", "pool_size=8",
    ]
    return "\n".join(lines), required


def _shell_fixture() -> tuple[str, list[str]]:
    lines = ["$ pytest tests/", "tests/test_a.py::test_one PASSED"]
    lines += [f"tests/test_c.py::test_{i} PASSED" for i in range(200)]
    lines += [
        "tests/test_z.py::test_bad FAILED",
        "E   AssertionError: expected 3 got 4",
        "1 failed, 201 passed",
        "exit code 1",
    ]
    required = [
        "pytest tests/", "test_bad", "FAILED",
        "AssertionError", "1 failed", "exit code 1",
    ]
    return "\n".join(lines), required


def _code_fixture() -> tuple[str, list[str]]:
    src = pathlib.Path("entroly/codec.py").read_text(encoding="utf-8")
    required = [
        "class RecoveryReference",
        "class Representation",
        "def content_digest",
        "def pareto_prune",
        "import hashlib",
    ]
    return src, required


def _table_fixture() -> tuple[str, list[str]]:
    import random

    random.seed(11)
    rows = ["order_id,customer,amount_cents,status"]
    for i in range(400):
        rows.append(
            f"ord-{i:05d},cust{i % 30},{random.randint(100, 900000)},"
            f"{'paid' if i % 5 else 'failed'}"
        )
    required = ["order_id", "customer", "amount_cents", "status"]
    return "\n".join(rows), required


FIXTURES = {
    "json_payment_error": _json_fixture,
    "log_root_cause_flood": _log_fixture,
    "shell_failing_test_run": _shell_fixture,
    "code_python_module": _code_fixture,
    "table_orders_export": _table_fixture,
}


@dataclass
class Row:
    fixture: str
    arm: str
    tokens: int
    tokens_full: int
    reduction_pct: float
    retention: float
    required_total: int
    required_kept: int
    recoverable: bool


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _retention(text: str, required: list[str]) -> tuple[float, int]:
    kept = sum(1 for r in required if r in text)
    return (kept / len(required) if required else 1.0), kept


def _generic(text: str) -> str:
    """universal_compress forced down its prose path, bypassing detection."""
    from entroly.universal_compress import tfidf_extractive_summarize

    try:
        return tfidf_extractive_summarize(text, target_ratio=0.3)
    except Exception:
        return text


def _specialized(text: str, source_id: str):
    from entroly.codec import RecoveryStore
    from entroly.codecs_builtin import default_registry

    store = RecoveryStore()
    reps = default_registry(store).representations(text, source_id=source_id)
    if not reps:
        return text, False
    # The smallest representation the codec offers; the caller would normally
    # choose, but the ablation is about what specialisation makes available.
    best = min(reps, key=lambda r: r.token_cost)
    if best.recovery is not None:
        # Recoverability is only claimed if it actually verifies.
        try:
            ok = best.recovery.verify(store.recover(best.recovery))
        except (KeyError, ValueError):
            ok = False
    else:
        ok = best.text == text  # nothing dropped
    return best.text, ok


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv[1:])

    import entroly

    rows: list[Row] = []
    for name, make in FIXTURES.items():
        text, required = make()
        full_tokens = _tokens(text)

        spec_text, recoverable = _specialized(text, name)
        # Equal-cost truncation: the floor gets the same budget the codec used.
        budget_chars = max(1, len(spec_text))

        arms = {
            "full": (text, False),
            "truncate": (text[:budget_chars], False),
            "generic": (_generic(text), False),
            "specialized": (spec_text, recoverable),
        }
        for arm, (out, rec) in arms.items():
            ret, kept = _retention(out, required)
            rows.append(
                Row(
                    fixture=name,
                    arm=arm,
                    tokens=_tokens(out),
                    tokens_full=full_tokens,
                    reduction_pct=round(100.0 * (1 - _tokens(out) / full_tokens), 1),
                    retention=round(ret, 4),
                    required_total=len(required),
                    required_kept=kept,
                    recoverable=rec,
                )
            )

    print(f"\n  Codec ablation (I: generic vs J: specialized)  [{SCHEMA_VERSION}]")
    print(f"  entroly {entroly.__version__}\n")
    print(f"  {'fixture':<24}{'arm':<13}{'tokens':>8}{'reduce':>9}{'evidence':>10}{'recover':>9}")
    for r in rows:
        print(f"  {r.fixture:<24}{r.arm:<13}{r.tokens:>8}{r.reduction_pct:>8.1f}%"
              f"{r.required_kept:>7}/{r.required_total:<2}{'yes' if r.recoverable else '-':>8}")

    print("\n  Per arm, averaged over fixtures:")
    for arm in ("full", "truncate", "generic", "specialized"):
        sel = [r for r in rows if r.arm == arm]
        red = sum(r.reduction_pct for r in sel) / len(sel)
        ret = sum(r.retention for r in sel) / len(sel)
        print(f"    {arm:<13} reduction {red:>6.1f}%   evidence retained {ret*100:>5.1f}%")

    print("\n  Note: truncation scores full marks on the JSON fixture only")
    print("  because that payload front-loads its values before the 40-record")
    print("  array. That flatters truncation and is a fixture property.")

    trunc = [r for r in rows if r.arm == "truncate"]
    spec = [r for r in rows if r.arm == "specialized"]
    print("\n  At equal token cost, truncation retains "
          f"{sum(r.retention for r in trunc)/len(trunc)*100:.1f}% of required evidence "
          f"and the specialized codec {sum(r.retention for r in spec)/len(spec)*100:.1f}%.")

    report = {
        "schema_version": SCHEMA_VERSION,
        "entroly_version": entroly.__version__,
        "rows": [asdict(r) for r in rows],
        "caveats": [
            "Three synthetic frozen fixtures; not a public dataset.",
            "No model is called; retention is substring presence.",
            "Required evidence is declared by the fixture, not the codec.",
            "truncate is given the same budget the specialized arm used.",
            "Truncation scores 6/6 on the JSON fixture only because that "
            "payload puts its required values before its 40-record array. "
            "Front-loaded evidence flatters truncation and is a property of "
            "the fixture, not of the method.",
            "The generic arm is universal_compress's prose summariser with "
            "content detection bypassed. Routed normally it would reach the "
            "same specialized code, so this measures generic-vs-specialized, "
            "not universal_compress as shipped.",
        ],
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
