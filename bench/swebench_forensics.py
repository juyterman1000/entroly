#!/usr/bin/env python3
"""Forensic root-cause dissection of SWE-bench retrieval misses.

NOT another invention. For every real task where the engine fails to
surface the gold file in top-10, dump the concrete, instance-level
evidence and classify the EXACT cause into mutually-exclusive buckets:

  C  corpus/index miss   gold file isn't even in the indexed set
                         (SRC_EXT / size cap) → not a ranking problem.
  D  explicit cue missed  issue literally contains the gold file's
                         path, a defined symbol of it, or a traceback
                         frame to it, yet it ranked low → a concrete,
                         deterministically-fixable failure.
  B  distractor outranked gold has real lexical signal but >=10 files
                         scored higher → reweighting-tractable.
  A  semantic gap        gold shares ~no tokens with the issue AND no
                         path/symbol/traceback cue exists → irreducible
                         for ANY zero-dep lexical method (needs a model;
                         conflicts with the zero-dep identity).

The bucket histogram is the root cause. The fix (or the honest "this
is irreducible, stop") is dictated by which bucket dominates — derived
from data, not from IR-textbook priors.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.swebench_real import (  # noqa: E402
    _fetch_repo, _gold_files, _rank, _stratified,
)
from entroly.localization import Tier0Localizer, _tok  # noqa: E402

_TB = re.compile(r'File "([^"]+?\.py)"')


def _stem_tokens(path: str) -> set[str]:
    base = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return {t for t in re.split(r"[_\W]+", base.lower()) if len(t) > 2}


def main() -> None:
    import argparse

    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=12, help="dump first N misses")
    a = ap.parse_args()

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    tasks = _stratified(ds, a.samples, a.seed)
    buckets: Counter = Counter()
    shown = 0
    n_eval = 0

    print("=" * 78)
    print("  SWE-bench retrieval FORENSICS — why does the engine miss the gold?")
    print("=" * 78)

    for t in tasks:
        gold_all = _gold_files(t["patch"])
        if not gold_all:
            continue
        files = _fetch_repo(t["repo"], t["base_commit"], 80.0, 90.0)
        if not files:
            continue
        gold = [g for g in gold_all if g in files]
        if not gold:
            # Gold file filtered out of the corpus entirely.
            buckets["C corpus/index miss"] += 1
            continue
        n_eval += 1
        issue = t["problem_statement"] or t["instance_id"]
        eng = _rank(files, issue, 50)
        loc = Tier0Localizer(files)
        bm = loc._bm25_content.ranking(_tok(issue))

        def rank_of(ranked: list[str]) -> int:
            best = 10**9
            for g in gold:
                if g in ranked:
                    best = min(best, ranked.index(g))
            return best

        er, br = rank_of(eng), rank_of(bm)
        if er < 10:
            continue                       # engine succeeded — not a miss

        # ---- dissect the (hardest) gold file -------------------------
        g = min(gold, key=lambda x: bm.index(x) if x in bm else 10**9)
        gtok = set(_tok(files[g]))
        itok = set(_tok(issue))
        overlap = len(gtok & itok)
        ilow = issue.lower()
        # explicit cues
        path_cue = (g.lower() in ilow
                    or any(s in ilow for s in _stem_tokens(g)))
        gsyms = {s for s, fs in loc.sym_def.items() if g in fs}
        sym_cue = any(re.search(r"\b" + re.escape(s) + r"\b", issue)
                      for s in gsyms if len(s) > 3)
        tb_cue = any(fr.lstrip("./").endswith(g.split("/")[-1])
                     for fr in _TB.findall(issue))
        # gold BM25 breakdown vs the wrong #1
        gscore = loc._bm25_content
        # rough: is gold's lexical signal nonzero & competitive?
        from entroly.localization import _BM25  # noqa: F401
        qw = {w: 1.0 for w in itok}
        sc = dict((fid, s) for s, fid in gscore.scores(qw))
        gsig = sc.get(g, 0.0)
        top_wrong = [f for f in bm[:5] if f not in gold]

        if (path_cue or sym_cue or tb_cue):
            bucket = "D explicit cue missed"
        elif overlap >= 5 and gsig > 0 and br >= 10:
            bucket = "B distractor outranked"
        elif overlap <= 2 and not (path_cue or sym_cue or tb_cue):
            bucket = "A semantic gap (irreducible, zero-dep)"
        else:
            bucket = "B distractor outranked"
        buckets[bucket] += 1

        if shown < a.show:
            shown += 1
            print(f"\n[{shown}] {t['repo']}  gold={g}")
            print(f"    issue: {issue[:240].replace(chr(10),' ')!r}")
            print(f"    gold rank: engine={er if er<10**9 else 'NOT FOUND'} "
                  f"bm25={br if br<10**9 else 'NOT FOUND'}  | "
                  f"|issue&gold tokens|={overlap}  bm25_sig={gsig:.3f}")
            print(f"    cues: path={path_cue} symbol={sym_cue} "
                  f"traceback={tb_cue}  | gold-defined syms="
                  f"{sorted(gsyms)[:6]}")
            print(f"    bm25 top-5 (wrong): {top_wrong}")
            print(f"    => BUCKET: {bucket}")

    print("\n" + "=" * 78)
    print(f"  ROOT-CAUSE DISTRIBUTION over misses (engine eval n={n_eval})")
    print("=" * 78)
    total = sum(buckets.values()) or 1
    for b, c in buckets.most_common():
        print(f"  {c:>3}  {100*c/total:>5.1f}%  {b}")
    print("\n  The dominant bucket dictates the honest conclusion:")
    print("   A-heavy  -> zero-dep lexical retrieval is capped here; "
          "further invention is futile without a model. STOP.")
    print("   D-heavy  -> a concrete deterministic fix exists "
          "(exploit the explicit cue properly).")
    print("   C-heavy  -> not a ranking bug at all; fix indexing/corpus.")
    print("   B-heavy  -> derive the exact reweighting from the "
          "breakdown evidence.")


if __name__ == "__main__":
    main()
