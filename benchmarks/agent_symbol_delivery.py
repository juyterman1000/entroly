#!/usr/bin/env python3
"""When the system hands an agent a function, does the agent get that function?

Protocol is fixed in AGENT_SYMBOL_DELIVERY_PREREGISTRATION.md and was written
before this was run. Summary:

  * corpus  : Python files tracked at a pinned ref, parseable, <=400 KB
  * tasks   : top-level def/async def/class with an exact source segment
  * query   : the symbol's name and docstring first line only, never its body
  * budget  : 400 tokens, fixed
  * oracle  : the original file bytes, never the system's own output

Each task is classified against the original file bytes:

  complete  - the symbol's entire source was delivered, unaltered
  partial   - delivered fragments are all real, but do not cover the whole
              symbol; unavoidable when the symbol exceeds the token budget
  corrupted - a delivered fragment appears nowhere in the file, so the agent was
              shown code that does not exist in the repository
  absent    - retrieval did not reach the symbol

Neither `absent` nor `partial` is a fidelity failure. A backend may honestly
find a symbol irrelevant, and a 400-token budget cannot deliver an 839-token
function whole. Only `corrupted` is the defect this repair addressed:

    uncorrupted_delivery_rate = (complete + partial) / delivered

Usage::

    python -m benchmarks.agent_symbol_delivery run --out results.json
    python -m benchmarks.agent_symbol_delivery verify results.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from entroly.context_receipts import select_from_index
from entroly.context_receipts.ingest import estimate_tokens, ingest_documents

SCHEMA_VERSION = "agent-symbol-delivery.v2"
BASELINE_REF = "1ecf1e093348068539f9e1463826209c966ed535"
REPO_ROOT = Path(__file__).resolve().parent.parent

MAX_BYTES = 400_000
TOKEN_BUDGET = 400
MIN_SYMBOL_TOKENS = 20
MAX_SYMBOLS_PER_FILE = 3


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _blob(ref: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{ref}:{path}"],
        cwd=REPO_ROOT, capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


# ── task construction ────────────────────────────────────────────────────────


def symbol_query(node: ast.AST, name: str) -> str:
    """Query from the symbol's public identity only — never from its body."""
    words = [part for part in name.split("_") if part]
    doc = ast.get_docstring(node) if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ) else None
    if doc:
        words.append(doc.strip().splitlines()[0].strip())
    return " ".join(words)


def tasks_for(path: str, text: str) -> list[dict[str, object]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    found: list[dict[str, object]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        segment = ast.get_source_segment(text, node)
        if not segment or estimate_tokens(segment) < MIN_SYMBOL_TOKENS:
            continue
        found.append(
            {
                "path": path,
                "symbol": node.name,
                "query": symbol_query(node, node.name),
                "source": segment,
            }
        )
        if len(found) >= MAX_SYMBOLS_PER_FILE:
            break
    return found


def build_corpus(ref: str) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    included: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []

    for rel in sorted(_git("ls-tree", "-r", "--name-only", ref).splitlines()):
        if not rel.endswith(".py"):
            continue
        raw = _blob(ref, rel)
        if raw is None:
            excluded.append({"path": rel, "reason": "unreadable_at_ref"})
            continue
        if len(raw) > MAX_BYTES:
            excluded.append({"path": rel, "reason": "over_size_cap"})
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            excluded.append({"path": rel, "reason": "not_strict_utf8"})
            continue
        found = tasks_for(rel, text)
        if not found:
            excluded.append({"path": rel, "reason": "no_qualifying_symbol"})
            continue
        included.append({"path": rel, "text": text, "tasks": found})

    return included, excluded


# ── measurement ──────────────────────────────────────────────────────────────


def classify(text: str, items: list[dict[str, object]], source: str) -> str:
    """Classify one task against the original file bytes.

    corrupted - a delivered fragment does not appear in the file at all, so the
                agent was shown code that does not exist in the repository
    complete  - the symbol's entire source was delivered, unaltered
    partial   - delivered fragments are all real, but do not cover the whole
                symbol; a token budget smaller than the symbol makes this the
                only possible outcome, so it is not a fidelity failure
    absent    - retrieval did not reach the symbol

    The original preregistered classifier called `partial` "altered" and counted
    it as a defect. That was wrong: it conflated budget truncation with
    corruption. See the amendment in the preregistration.
    """
    if not items:
        return "absent"

    if any(str(item["text"]) not in text for item in items):
        return "corrupted"

    delivered = "\n".join(str(item["text"]) for item in items)
    if source in delivered:
        return "complete"

    # Does any delivered fragment overlap the symbol's byte span?
    char_index = text.find(source)
    if char_index < 0:
        return "absent"
    sym_start = len(text[:char_index].encode("utf-8"))
    sym_end = sym_start + len(source.encode("utf-8"))
    for item in items:
        start, end = int(item["byte_start"]), int(item["byte_end"])
        if start < sym_end and end > sym_start:
            return "partial"
    return "absent"


def run_task(path: str, text: str, task: dict[str, object]) -> str:
    index = ingest_documents([(path, text)])
    receipt = select_from_index(
        index, query=str(task["query"]), token_budget=TOKEN_BUDGET, prefer_rust=False
    )
    return classify(text, receipt.get("selected_context", []), str(task["source"]))


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; correct at small n where the normal approx is not."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(max(0.0, centre - margin), 6), round(min(1.0, centre + margin), 6))


def run_measurement(ref: str) -> dict[str, object]:
    included, excluded = build_corpus(ref)
    outcomes: list[dict[str, str]] = []

    for record in included:
        path, text = str(record["path"]), str(record["text"])
        for task in record["tasks"]:  # type: ignore[union-attr]
            try:
                bucket = run_task(path, text, task)
            except BaseException:  # noqa: BLE001 - a crash is a delivery failure
                bucket = "error"
            outcomes.append(
                {"path": path, "symbol": str(task["symbol"]), "outcome": bucket}
            )

    counts = {k: 0 for k in ("complete", "partial", "corrupted", "absent", "error")}
    for row in outcomes:
        counts[row["outcome"]] += 1

    delivered = counts["complete"] + counts["partial"] + counts["corrupted"]
    clean = counts["complete"] + counts["partial"]
    rate = clean / delivered if delivered else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_ref": ref,
        "protocol": {
            "token_budget": TOKEN_BUDGET,
            "min_symbol_tokens": MIN_SYMBOL_TOKENS,
            "max_symbols_per_file": MAX_SYMBOLS_PER_FILE,
            "prefer_rust": False,
            "preregistration": "benchmarks/AGENT_SYMBOL_DELIVERY_PREREGISTRATION.md",
        },
        "environment": {
            "entroly_version": __import__("entroly").__version__,
            "implementation_commit": _git("rev-parse", "HEAD"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "totals": {
            "files": len(included),
            "tasks": len(outcomes),
            **counts,
            "delivered": delivered,
            "uncorrupted_delivery_rate": round(rate, 6),
            "uncorrupted_delivery_ci95": wilson(clean, delivered),
            "delivery_rate": round(delivered / len(outcomes), 6) if outcomes else 0.0,
        },
        "excluded": excluded,
        "outcomes": outcomes,
    }


def render(report: dict[str, object]) -> str:
    t = report["totals"]  # type: ignore[index]
    low, high = t["uncorrupted_delivery_ci95"]
    return "\n".join(
        [
            f"commit {report['environment']['implementation_commit'][:12]}  "  # type: ignore[index]
            f"budget {report['protocol']['token_budget']}",  # type: ignore[index]
            "",
            f"  files                    : {t['files']}",
            f"  symbol tasks             : {t['tasks']}",
            f"  delivered                : {t['delivered']}",
            f"    complete symbol        : {t['complete']}",
            f"    partial (budget-capped): {t['partial']}",
            f"    CORRUPTED              : {t['corrupted']}",
            f"  not selected (absent)    : {t['absent']}",
            f"  errors                   : {t['error']}",
            "",
            f"  UNCORRUPTED DELIVERY     : {t['uncorrupted_delivery_rate']:.1%}  "
            f"95% CI [{low:.1%}, {high:.1%}]",
            "  (of what the agent received, how much was real code from the repo)",
        ]
    )


def compare(before_path: Path, after_path: Path) -> int:
    """Paired comparison of two artifacts over the identical task set.

    Reported on the *task* denominator, which is genuinely paired. The
    `uncorrupted_delivery_rate` in each artifact is conditioned on delivery, and
    delivery itself shifts between conditions, so it is not a like-for-like
    denominator across conditions and is shown only for context.
    """
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))

    b_map = {(r["path"], r["symbol"]): r["outcome"] for r in before["outcomes"]}
    a_map = {(r["path"], r["symbol"]): r["outcome"] for r in after["outcomes"]}
    shared = sorted(set(b_map) & set(a_map))
    if not shared:
        print("FAIL: the two artifacts share no tasks")
        return 1
    if len(shared) != len(b_map) or len(shared) != len(a_map):
        print(
            f"WARNING: task sets differ; comparing the {len(shared)} shared tasks "
            f"(before={len(b_map)}, after={len(a_map)})"
        )

    b_bad = sum(b_map[k] == "corrupted" for k in shared)
    a_bad = sum(a_map[k] == "corrupted" for k in shared)
    fixed = sum(b_map[k] == "corrupted" and a_map[k] != "corrupted" for k in shared)
    broke = sum(b_map[k] != "corrupted" and a_map[k] == "corrupted" for k in shared)

    n = len(shared)
    # Exact McNemar on the discordant pairs.
    discordant = fixed + broke
    if discordant == 0:
        p_text = "1 (no discordant pairs)"
    else:
        # Computed in log space: with ~1500 discordant pairs, 2**discordant
        # overflows a float and the exact p underflows to zero anyway.
        log_tail = math.log(
            sum(math.comb(discordant, i) for i in range(min(fixed, broke) + 1))
        )
        log_p = math.log(2.0) + log_tail - discordant * math.log(2.0)
        p_text = f"{math.exp(log_p):.3g}" if log_p > -700 else f"< 1e-300 (log10 p ~ {log_p / math.log(10):.0f})"

    print(f"paired tasks                     : {n}")
    print(f"  delivered fabricated code, before: {b_bad}/{n} ({b_bad / n:.1%})")
    print(f"  delivered fabricated code, after : {a_bad}/{n} ({a_bad / n:.1%})")
    print("")
    print(f"  fixed  (corrupt -> clean)        : {fixed}")
    print(f"  broken (clean -> corrupt)        : {broke}")
    print(f"  exact McNemar p                  : {p_text}")
    print("")
    print(f"  before uncorrupted delivery      : {before['totals']['uncorrupted_delivery_rate']:.1%}"
          f"  (n={before['totals']['delivered']} delivered)")
    print(f"  after  uncorrupted delivery      : {after['totals']['uncorrupted_delivery_rate']:.1%}"
          f"  (n={after['totals']['delivered']} delivered)")
    return 0


def verify(path: Path) -> int:
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("schema_version") != SCHEMA_VERSION:
        print(f"FAIL: schema {stored.get('schema_version')} != {SCHEMA_VERSION}")
        return 1

    counts = {k: 0 for k in ("complete", "partial", "corrupted", "absent", "error")}
    for row in stored["outcomes"]:
        counts[row["outcome"]] += 1

    failures = [
        f"totals.{k}: stored {stored['totals'][k]} != recount {v}"
        for k, v in counts.items()
        if stored["totals"][k] != v
    ]
    delivered = counts["complete"] + counts["partial"] + counts["corrupted"]
    clean = counts["complete"] + counts["partial"]
    if delivered and abs(stored["totals"]["uncorrupted_delivery_rate"] - clean / delivered) > 1e-6:
        failures.append("uncorrupted_delivery_rate does not match the recorded outcomes")

    if failures:
        print(f"VERIFY FAILED ({len(failures)})")
        for line in failures:
            print(f"  - {line}")
        return 1

    print("VERIFY OK")
    print(render(stored))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--out", type=Path, required=True)
    run_cmd.add_argument("--ref", default=BASELINE_REF)
    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("artifact", type=Path)
    cmp_cmd = sub.add_parser("compare", help="paired before/after comparison")
    cmp_cmd.add_argument("before", type=Path)
    cmp_cmd.add_argument("after", type=Path)
    args = parser.parse_args()

    if args.command == "verify":
        return verify(args.artifact)

    if args.command == "compare":
        return compare(args.before, args.after)

    report = run_measurement(args.ref)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(render(report))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
