"""Does ranking the journal beat restoring the last N events?

The null hypothesis for session continuity is "show the most recent events".
It is free, it needs no index, and it is what every harness already does
implicitly. A ranker that does not beat it is complexity without benefit, so
this measures the comparison rather than asserting it.

Protocol, fixed before running:

* A journal is generated with the shape real sessions have -- a few files
  touched repeatedly, a long tail touched once, commands, and failures -- and
  one **needle** event placed at a known age.
* The query is derived from the needle's own content, which is the realistic
  case: an agent resuming asks about the thing it was working on.
* Both strategies return ``k`` events. Recency returns the last ``k``. Ranking
  returns the BM25 top ``k`` over the same journal.
* Success is whether the needle is in the returned set. Recall at k, nothing
  else -- not a score, not a judgement.

Why the needle's age is swept rather than fixed: recency wins trivially when
the needle is recent and loses trivially when it is old. A single age would
let the author pick the answer. The honest statement is the curve, and the
number that matters is where the two cross.

Nothing here calls a model. It measures retrieval, which is the only part of
session continuity that can be measured without a task suite.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

HOT_FILES = [
    "entroly/ccr.py",
    "entroly/server.py",
    "entroly/tokens.py",
    "entroly/proxy.py",
]
COMMANDS = [
    "pytest tests/test_ccr.py",
    "ruff check entroly/",
    "cargo test --lib",
    "pytest tests/test_server.py",
    "npm test",
]
COLD_PREFIXES = ["src/legacy", "docs/notes", "scripts/oneoff", "tests/fixtures"]

# The needle: a distinctive file and a failure, which is what an agent resuming
# a broken task actually asks about.
NEEDLE_TARGET = "entroly/governance/identity_rotation.py"
NEEDLE_DETAIL = "FAILED test_rotation_preserves_tokens - AssertionError: digest drift"
NEEDLE_QUERY = "identity rotation digest drift"


def build_journal(
    path: Path, *, total: int, needle_age: int, rng: random.Random
) -> None:
    """Write ``total`` events, with the needle ``needle_age`` events from the end."""
    lines = []
    for index in range(total):
        roll = rng.random()
        if roll < 0.45:
            kind, tool = "edit", "Edit"
            target = rng.choice(HOT_FILES)
            status, detail = "ok", ""
        elif roll < 0.70:
            kind, tool = "command", "Bash"
            target = rng.choice(COMMANDS)
            status, detail = "ok", ""
        elif roll < 0.80:
            kind, tool = "error", "Bash"
            target = rng.choice(COMMANDS)
            status = "exit:1"
            detail = "FAILED some_unrelated_test - AssertionError"
        else:
            kind, tool = "read", "Read"
            target = f"{rng.choice(COLD_PREFIXES)}/module_{index}.py"
            status, detail = "ok", ""
        lines.append(
            {
                "schema": "entroly.session-journal.v1",
                "ts": 1_700_000_000.0 + index,
                "kind": kind,
                "tool": tool,
                "target": target,
                "status": status,
                "detail": detail,
                "session": "bench",
            }
        )

    needle_index = max(0, total - 1 - needle_age)
    lines[needle_index] = {
        "schema": "entroly.session-journal.v1",
        "ts": 1_700_000_000.0 + needle_index,
        "kind": "error",
        "tool": "Bash",
        "target": NEEDLE_TARGET,
        "status": "exit:1",
        "detail": NEEDLE_DETAIL,
        "session": "bench",
    }
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in lines) + "\n",
        encoding="utf-8",
    )


# The adversarial arm. The needle above is lexically distinctive against
# generic noise, which flatters a lexical ranker. Here the needle is one of the
# files the session touches constantly, so hundreds of events share its
# vocabulary and BM25 has to discriminate rather than merely match. A ranker
# that only wins the easy arm should be described as winning the easy arm.
AMBIGUOUS_TARGET = HOT_FILES[0]
AMBIGUOUS_QUERY = "ccr"


def trial(
    *, total: int, needle_age: int, k: int, seed: int, ambiguous: bool = False
) -> tuple[bool, bool]:
    """Return (recency_found, ranked_found) for one journal."""
    from entroly.session_journal import recall

    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        build_journal(root / "session_journal.jsonl", total=total,
                      needle_age=needle_age, rng=rng)

        if ambiguous:
            # Re-point the needle at a file the journal is saturated with, and
            # mark it so a hit can be told from the hundreds of look-alikes.
            rows = (root / "session_journal.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            needle_index = max(0, total - 1 - needle_age)
            payload = json.loads(rows[needle_index])
            payload["target"] = AMBIGUOUS_TARGET
            payload["detail"] = "needle"
            rows[needle_index] = json.dumps(payload, ensure_ascii=False)
            (root / "session_journal.jsonl").write_text(
                "\n".join(rows) + "\n", encoding="utf-8"
            )

        query = AMBIGUOUS_QUERY if ambiguous else NEEDLE_QUERY
        recency = recall("", limit=k, entroly_dir=root)
        ranked = recall(query, limit=k, entroly_dir=root)

    if ambiguous:
        def found(result):
            return any(event.detail == "needle" for event in result.events)
    else:
        def found(result):
            return any(event.target == NEEDLE_TARGET for event in result.events)
    return found(recency), found(ranked)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=400)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--ambiguous",
        action="store_true",
        help="place the needle in a file the journal is saturated with",
    )
    args = parser.parse_args()

    os.environ.setdefault("ENTROLY_NO_SELF_HEAL", "1")

    ages = [0, 2, 5, 10, 25, 50, 100, 200, 350]
    rows = []
    arm = "ambiguous" if args.ambiguous else "distinctive"
    print(
        f"arm={arm}  journal={args.events} events  k={args.k}  "
        f"trials={args.trials} per age\n"
        f"{'needle age':>11}{'recency':>10}{'ranked':>9}  verdict"
    )
    for age in ages:
        if age >= args.events:
            continue
        recency_hits = ranked_hits = 0
        for seed in range(args.trials):
            was_recent, was_ranked = trial(
                total=args.events,
                needle_age=age,
                k=args.k,
                seed=seed,
                ambiguous=args.ambiguous,
            )
            recency_hits += was_recent
            ranked_hits += was_ranked
        r_rate = recency_hits / args.trials
        k_rate = ranked_hits / args.trials
        verdict = (
            "ranking wins" if k_rate > r_rate + 1e-9
            else "tie" if abs(k_rate - r_rate) <= 1e-9
            else "RECENCY WINS"
        )
        print(f"{age:>11}{r_rate:>10.2f}{k_rate:>9.2f}  {verdict}")
        rows.append({"needle_age": age, "recency_recall": r_rate,
                     "ranked_recall": k_rate})

    wins = sum(1 for row in rows if row["ranked_recall"] > row["recency_recall"])
    losses = sum(1 for row in rows if row["ranked_recall"] < row["recency_recall"])
    print(
        f"\nranking better at {wins}/{len(rows)} ages, worse at {losses}. "
        "Recency can only win where the needle is inside the last k, which is "
        "exactly the case a journal is not needed for."
    )
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "protocol": "session-journal retrieval vs recency, recall@k",
                    "events": args.events,
                    "k": args.k,
                    "trials_per_age": args.trials,
                    "arm": arm,
                    "query": AMBIGUOUS_QUERY if args.ambiguous else NEEDLE_QUERY,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
