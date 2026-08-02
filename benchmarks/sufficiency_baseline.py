"""Baseline: does selection STOP at sufficiency, or fill the budget?

Why this exists
---------------

Entroly's selector maximises utility subject to ``tokens <= budget``. Such an
objective can add context whenever estimated marginal utility stays positive,
which would push selection toward the ceiling. Chroma's Context Rot study
measured the cost of that on 18 frontier models: focused prompts outperform
full prompts, and "even a single distractor reduces performance". Under that
finding, tokens added past sufficiency have negative expected value, not zero.

What this harness actually measured (v1.0.72, native engine)
------------------------------------------------------------

The budget-filling hypothesis was **not** confirmed at this scale. On two of
three fixtures the selector returned exactly one fragment -- the needle, and
nothing else -- at every budget from 64 to 4096, holding at 24 and 28 tokens
while the ceiling grew 64x. The whole 6-fragment pool is only ~164 tokens, so
at budget 4096 a fill-to-budget selector would have taken all of it. It did not.

The failure that did appear is ranking, not stopping. On ``retry_needle``:

* budget 64 -- selects 2 fragments / 44 tokens and *drops the needle*, which is
  39 tokens and would have fit on its own. Distractors outranked the answer.
* budget 128 -- 4 fragments / 91 tokens, needle still absent.
* budget 256+ -- 5 of 6 fragments / 130 tokens, needle present but carrying 91
  tokens of distractors.

That 130-token plateau is *running out of candidates*, not a sufficiency
decision: it is 5 of the 6 fragments that exist.

Limitation this exposes
-----------------------

A 6-fragment pool cannot distinguish "stopped because the evidence was
sufficient" from "stopped because nothing was left". Testing fill-to-budget
behaviour honestly needs a candidate pool far larger than the budget. An
earlier run appeared to show +3,629 tokens of budget-driven growth; that was an
artifact of the engine warm-starting 1,348 unrelated repo fragments (see the
ENTROLY_DIR note below), not a property of the selector, and it is not
reproduced here.

This harness measures the gap directly. Each fixture has exactly one
answer-bearing needle plus known distractors, so "sufficient" is ground truth
rather than an estimate: the smallest correct answer is the needle alone.

    waste = selected_tokens - needle_tokens        (tokens past sufficiency)
    recall = needle retained?                      (did we keep the answer)

Budget utilisation is reported as a COST, not an achievement. A run that keeps
the needle in 40 tokens beats one that keeps it in 400.

Honest scope
------------

* Fixtures are synthetic and frozen here, chosen so sufficiency is decidable.
  They are a regression baseline, not a public task dataset, and no claim about
  downstream answer quality is made -- nothing here calls a model.
* Recall is exact-substring on the needle. That measures evidence retention,
  not whether a model would answer correctly.
* Measures the installed engine, whichever it is. Run it twice (with and
  without ``entroly_core``) to compare paths; the report records which ran.

Run:
    python benchmarks/sufficiency_baseline.py
    python benchmarks/sufficiency_baseline.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("ENTROLY_DISABLE_UPDATE_CHECK", "1")

# Isolate the engine BEFORE importing entroly.
#
# EntrolyConfig.checkpoint_dir defaults to ~/.entroly/checkpoints/<sha256(cwd)>,
# and EntrolyEngine lazily warm-starts the shared index from it. Run from a repo
# checkout, that silently loads the whole indexed repo: the first version of this
# harness reported "fragments_total: 6" while the engine actually held 1,348
# restored fragments plus the 6 fixtures. Every number it produced described the
# repo index competing with the needle, not one needle against five distractors.
#
# ENTROLY_DIR overrides the whole path (config.py:_project_checkpoint_dir), so a
# fresh temp dir per process gives a guaranteed-empty index and keeps fixtures out
# of the user's real checkpoint store. Must be set before the import, because
# checkpoint_dir is resolved by a dataclass default_factory at construction.
_ISOLATED_DIR = tempfile.mkdtemp(prefix="entroly-sufficiency-")
os.environ["ENTROLY_DIR"] = _ISOLATED_DIR

SCHEMA_VERSION = "entroly.sufficiency-baseline.v1"

# ── Frozen fixtures ──────────────────────────────────────────────────────
# Each: a query, ONE needle that answers it, and distractors that are
# topically adjacent but do not answer it -- the shape Chroma isolates as
# "distractor interference".

DISTRACTOR_POOL = [
    ("orders.py", "def place_order(cart, user):\n    total = calculate_total(cart)\n    return submit_order(user, total)"),
    ("search.py", "def query_index(term, limit=10):\n    hits = index.lookup(term)\n    return rank_results(hits)[:limit]"),
    ("cache.py", "def get_cached(key):\n    entry = store.get(key)\n    return entry.value if entry else None"),
    ("users.py", "def update_profile(user, fields):\n    validate(fields)\n    return save_user(user, fields)"),
    ("report.py", "def build_report(rows):\n    grouped = group_by_day(rows)\n    return render_table(grouped)"),
]

FIXTURES = [
    {
        "id": "auth_needle",
        "query": "how is the session token issued after login",
        "needle_source": "auth.py",
        "needle": "def login(user, pw):\n    token = verify_password(user, pw)\n    return issue_session_token(token)",
        "must_contain": "issue_session_token",
    },
    {
        "id": "billing_needle",
        "query": "how is a credit card charged through the payment gateway",
        "needle_source": "billing.py",
        "needle": "def charge_card(customer, amount):\n    gateway = StripeGateway()\n    return gateway.charge(customer.card, amount)",
        "must_contain": "StripeGateway",
    },
    {
        "id": "retry_needle",
        "query": "what happens when a request times out and needs retrying",
        "needle_source": "retry.py",
        "needle": "def retry_request(req, attempts=3):\n    for i in range(attempts):\n        if send(req).ok:\n            return True\n    raise TimeoutError('exhausted retries')",
        "must_contain": "TimeoutError",
    },
]

BUDGETS = [64, 128, 256, 512, 1024, 2048, 4096]


@dataclass
class Row:
    fixture: str
    budget: int
    selected_tokens: int
    needle_tokens: int
    waste_tokens: int          # tokens past the sufficient set
    needle_retained: bool
    fragments_selected: int
    fragments_total: int
    budget_utilization: float  # reported as a COST
    latency_ms: float


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _build_engine(fixture: dict):
    """One engine per fixture, reused across budgets.

    Engine construction dominates runtime (warm-start plus index load), so
    building one per (fixture, budget) pair made the sweep exceed ten minutes.
    Budget is a per-call argument, so a single engine serves the whole sweep
    and the measurement is unaffected.
    """
    from entroly.server import EntrolyEngine

    # A fresh dir per ENGINE, not per process. The engine persists its index to
    # checkpoint_dir, so with one shared dir the second fixture warm-started
    # from the first fixture's fragments -- the guard below caught exactly that
    # ("engine holds 7 fragments, expected 6"). checkpoint_dir is resolved by a
    # dataclass default_factory at construction, so setting the env var here,
    # before EntrolyEngine(), is what takes effect.
    os.environ["ENTROLY_DIR"] = tempfile.mkdtemp(prefix="entroly-sufficiency-")
    engine = EntrolyEngine()

    # Fail loudly if isolation did not hold. The engine warm-starts lazily on
    # first mutation, so this is checked AFTER the ingests, not before.
    expected = 1 + len(DISTRACTOR_POOL)
    engine.ingest_fragment(fixture["needle"], f"file:{fixture['needle_source']}",
                           _estimate_tokens(fixture["needle"]))
    for src, body in DISTRACTOR_POOL:
        engine.ingest_fragment(body, f"file:{src}", _estimate_tokens(body))

    # get_stats()["session"]["total_fragments"] is the portable count: the Rust
    # path keeps state in self._rust and has no ._fragments at all, so reaching
    # for that attribute worked only on the pure-Python fallback.
    actual = int(engine.get_stats()["session"]["total_fragments"])
    if actual != expected:
        raise SystemExit(
            f"REFUSING TO REPORT: engine holds {actual} fragments, expected "
            f"{expected}. Warm-start restored a foreign index, so 'one needle vs "
            f"{len(DISTRACTOR_POOL)} distractors' is not what would be measured. "
            f"ENTROLY_DIR={os.environ.get('ENTROLY_DIR')!r}"
        )
    return engine


def _run_fixture(fixture: dict, budget: int, engine) -> Row:
    t0 = time.perf_counter()
    result = engine.optimize_context(token_budget=budget, query=fixture["query"])
    latency_ms = (time.perf_counter() - t0) * 1000

    selected = result.get("selected_fragments") or result.get("selected") or []
    text = "\n".join(f.get("content", "") for f in selected if isinstance(f, dict))
    selected_tokens = sum(int(f.get("token_count", 0) or 0) for f in selected if isinstance(f, dict))
    needle_tokens = _estimate_tokens(fixture["needle"])

    return Row(
        fixture=fixture["id"],
        budget=budget,
        selected_tokens=selected_tokens,
        needle_tokens=needle_tokens,
        waste_tokens=max(0, selected_tokens - needle_tokens),
        needle_retained=fixture["must_contain"] in text,
        fragments_selected=len(selected),
        fragments_total=1 + len(DISTRACTOR_POOL),
        budget_utilization=round(selected_tokens / budget, 4) if budget else 0.0,
        latency_ms=round(latency_ms, 2),
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", dest="json_out", help="write the full report here")
    args = ap.parse_args(argv[1:])

    try:
        import entroly_core  # noqa: F401
        engine_mode = "native (entroly_core present)"
    except ImportError:
        engine_mode = "pure-python (entroly_core absent)"

    import entroly

    rows = []
    for fixture in FIXTURES:
        engine = _build_engine(fixture)
        for budget in BUDGETS:
            rows.append(_run_fixture(fixture, budget, engine))

    print(f"\n  Entroly sufficiency baseline  [{SCHEMA_VERSION}]")
    print(f"  version: {entroly.__version__}   engine: {engine_mode}")
    print(f"  fixtures: {len(FIXTURES)}  budgets: {BUDGETS}\n")
    print(f"  {'fixture':<16}{'budget':>7}{'sel':>7}{'needle':>8}{'waste':>7}{'keep':>6}{'util':>8}{'ms':>8}")
    for r in rows:
        print(f"  {r.fixture:<16}{r.budget:>7}{r.selected_tokens:>7}{r.needle_tokens:>8}"
              f"{r.waste_tokens:>7}{'yes' if r.needle_retained else 'NO':>6}"
              f"{r.budget_utilization:>8.2f}{r.latency_ms:>8.1f}")

    retained = [r for r in rows if r.needle_retained]
    recall = len(retained) / len(rows) if rows else 0.0
    waste = [r.waste_tokens for r in retained]
    # How much does selection grow purely because the ceiling rose?
    by_budget = {b: statistics.mean([r.selected_tokens for r in rows if r.budget == b]) for b in BUDGETS}
    growth = by_budget[BUDGETS[-1]] - by_budget[BUDGETS[0]]

    print(f"\n  evidence recall            : {recall*100:.1f}%  ({len(retained)}/{len(rows)})")
    if waste:
        print(f"  median tokens past needle  : {statistics.median(waste):.0f}")
        print(f"  max tokens past needle     : {max(waste)}")
    print(f"  mean selected @ budget {BUDGETS[0]:<5}: {by_budget[BUDGETS[0]]:.0f} tokens")
    print(f"  mean selected @ budget {BUDGETS[-1]:<5}: {by_budget[BUDGETS[-1]]:.0f} tokens")
    print(f"  growth from a larger ceiling: +{growth:.0f} tokens")
    print("\n  A sufficiency-first selector would hold the last two roughly equal:")
    print("  the answer does not get larger because the budget did.\n")

    report = {
        "schema_version": SCHEMA_VERSION,
        "entroly_version": entroly.__version__,
        "engine_mode": engine_mode,
        "budgets": BUDGETS,
        "rows": [asdict(r) for r in rows],
        "summary": {
            "evidence_recall": round(recall, 4),
            "median_waste_tokens": statistics.median(waste) if waste else None,
            "max_waste_tokens": max(waste) if waste else None,
            "mean_selected_smallest_budget": round(by_budget[BUDGETS[0]], 1),
            "mean_selected_largest_budget": round(by_budget[BUDGETS[-1]], 1),
            "growth_from_larger_ceiling": round(growth, 1),
        },
        "caveats": [
            "Synthetic frozen fixtures; not a public task dataset.",
            "No model is called; this measures evidence retention, not answer quality.",
            "Recall is exact-substring on the needle.",
            "Measures whichever engine is installed; see engine_mode.",
        ],
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  wrote {args.json_out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
