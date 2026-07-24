"""Experiment 1 — Entroly reproducibility matrix.

Runs the frozen-corpus QCCR selection under a battery of environment/ordering
perturbations, each in a FRESH subprocess, and measures divergence at three
levels: set overlap (Jaccard), rank agreement (Kendall's tau), byte identity.

Nondeterminism is treated as a HYPOTHESIS. The harness reports what it finds;
targets for strict deterministic mode are Jaccard=1.0, tau=1.0, byte=100%.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE = os.path.join(HERE, "capture_selection.py")
CORPUS = os.path.join(HERE, "frozen_corpus.json")
BUDGET = "2000"
QUERY = "where does the proxy inject compressed context into requests"


def run_capture(corpus_path: str, env_extra: dict[str, str]) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    env.setdefault("ENTROLY_SOURCE", os.getcwd())
    out = subprocess.run(
        [sys.executable, "-u", CAPTURE, corpus_path, BUDGET, QUERY],
        capture_output=True, text=True, env=env, cwd=os.getcwd(),
    )
    line = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("{")]
    if not line:
        raise RuntimeError(f"capture failed: {out.stderr[-500:]}")
    return json.loads(line[-1])


def sources(res: dict) -> list[str]:
    return [o["source"] for o in res["order"]]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def kendall_tau(a: list[str], b: list[str]) -> float | None:
    """Kendall's tau over the ranking of sources common to both selections."""
    common = [s for s in a if s in set(b)]
    if len(common) < 2:
        return None
    ra = {s: i for i, s in enumerate(a)}
    rb = {s: i for i, s in enumerate(b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            si, sj = common[i], common[j]
            s = (ra[si] - ra[sj]) * (rb[si] - rb[sj])
            conc += s > 0
            disc += s < 0
    total = conc + disc
    return (conc - disc) / total if total else 1.0


def permuted_corpus(seed: int) -> str:
    frags = json.load(open(CORPUS, encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(frags)
    path = os.path.join(HERE, f"frozen_corpus_perm{seed}.json")
    json.dump(frags, open(path, "w", encoding="utf-8"))
    return path


def reversed_corpus() -> str:
    frags = json.load(open(CORPUS, encoding="utf-8"))
    frags.reverse()
    path = os.path.join(HERE, "frozen_corpus_rev.json")
    json.dump(frags, open(path, "w", encoding="utf-8"))
    return path


def main() -> int:
    conditions: list[tuple[str, str, dict[str, str]]] = [
        ("baseline",              CORPUS, {}),
        ("baseline-2",            CORPUS, {}),
        ("hashseed=0",            CORPUS, {"PYTHONHASHSEED": "0"}),
        ("hashseed=1",            CORPUS, {"PYTHONHASHSEED": "1"}),
        ("hashseed=42",           CORPUS, {"PYTHONHASHSEED": "42"}),
        ("hashseed=random",       CORPUS, {"PYTHONHASHSEED": "random"}),
        ("threads=1",             CORPUS, {"RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}),
        ("threads=2",             CORPUS, {"RAYON_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}),
        ("threads=8",             CORPUS, {"RAYON_NUM_THREADS": "8", "OMP_NUM_THREADS": "8"}),
        ("insert-perm-1",         permuted_corpus(1), {}),
        ("insert-perm-2",         permuted_corpus(2), {}),
        ("insert-perm-3",         permuted_corpus(7), {}),
        ("insert-reversed",       reversed_corpus(), {}),
    ]

    results: dict[str, dict] = {}
    for label, corpus, env in conditions:
        try:
            results[label] = run_capture(corpus, env)
            r = results[label]
            print(f"  ran {label:18s} n={r['n']:2d} digest={r['digest'][:12]}")
        except Exception as exc:
            print(f"  FAIL {label:18s}: {exc}")

    base = results["baseline"]
    base_src, base_dig = sources(base), base["digest"]
    print("\n=== reproducibility matrix (vs baseline) ===")
    print(f"{'condition':18s} {'Jaccard':>8s} {'tau':>7s} {'byte':>6s}  n")
    all_byte = True
    all_set = True
    for label, r in results.items():
        if label == "baseline":
            continue
        j = jaccard(base_src, sources(r))
        t = kendall_tau(base_src, sources(r))
        byte = (r["digest"] == base_dig)
        all_byte &= byte
        all_set &= (j == 1.0)
        tau_s = "n/a" if t is None else f"{t:+.3f}"
        print(f"{label:18s} {j:8.3f} {tau_s:>7s} {str(byte):>6s}  {r['n']}")

    print("\n=== verdict ===")
    print(f"  set-identical across all conditions:  {all_set}")
    print(f"  byte-identical across all conditions: {all_byte}")
    if all_byte:
        print("  => STRICT DETERMINISTIC on tested axes (Windows, this arch, this engine build)")
    elif all_set:
        print("  => SET-stable but ORDER/BYTE varies on some axis (rank tie-break nondeterminism)")
    else:
        print("  => SET diverges on some axis (selection itself is nondeterministic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
