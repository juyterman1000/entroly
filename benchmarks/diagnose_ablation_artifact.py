"""
Diagnose the Experiment 1 F1 artifact (dev items only).

benchmarks/results/verifier_ablation_recall.json shows AUROC(intact vs random
length-matched ablation) = 0.9334 despite near-identical means — a
rank-consistent artifact. Preregistered interpretation commitment: isolate
evidence-gap-specific signals before any v2. This script runs on the v1 DEV
half (first 100 eligible items) only; the v2 verdict must use fresh items.

Hypotheses tested:
  H_fmt    v1 construction flaw — INTACT used the raw context string while
           CRITICAL/RANDOM used re-joined stripped sentences; formatting alone
           may shift the score. Test: raw ctx vs " ".join(sentences(ctx)).
  H_layer  the artifact lives in specific Phi layers (h_sem_norm /
           nli_bidir_score) while ESG unsupported_fraction is gap-specific.
  H_op     the artifact is operationally irrelevant at the deployed tau
           (random flag rate ~ intact FPR).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from entroly.eicv import EICVAnalyzer  # noqa: E402
from benchmarks.verifier_ablation_recall import (  # noqa: E402
    SEED, N_CAL, _auroc, _sentences, build_item, load_pool,
)

N_DEV = 100
TAU = 0.3338  # v1 dev-selected operating point (results JSON)


def main() -> int:
    pool = load_pool()
    rng = random.Random(SEED)
    items = []
    cursor = 0
    for cursor, (ctx, ans, _q) in enumerate(pool):
        item, gate = build_item(ctx, ans, rng)
        if item is not None:
            items.append(item)
        if len(items) == 200:
            break
    dev = items[:N_DEV]
    cal_pairs = [(c, a) for c, a, _ in pool[cursor + 1: cursor + 1 + N_CAL]]

    ana = EICVAnalyzer()
    ana.fit_calibrators(cal_pairs)

    layers = ("esg_tension", "nli_bidir_score", "h_sem_norm")
    per = {c: {"score": [], "unsup": [], **{k: [] for k in layers}}
           for c in ("intact_raw", "intact_joined", "critical", "random")}

    for item in dev:
        variants = {
            "intact_raw": item["intact"],
            "intact_joined": " ".join(_sentences(item["intact"])),
            "critical": item["critical"],
            "random": item["random"],
        }
        for cond, ev in variants.items():
            cert = ana.verify(ev, item["answer"])
            per[cond]["score"].append(1.0 - cert.phi)
            per[cond]["unsup"].append(cert.unsupported_fraction)
            for k in layers:
                per[cond][k].append(cert.layer_scores.get(k, 0.0))

    n = len(dev)

    def auc(a: str, b: str, key: str) -> float:
        return _auroc(per[a][key] + per[b][key], [0] * n + [1] * n)

    def pct(vals, q):
        s = sorted(vals)
        return s[min(len(s) - 1, int(q * len(s)))]

    report = {
        "n_dev": n,
        "H_fmt_formatting_artifact": {
            "auroc_raw_vs_joined_intact": round(auc("intact_raw", "intact_joined", "score"), 4),
            "mean_raw": round(sum(per["intact_raw"]["score"]) / n, 4),
            "mean_joined": round(sum(per["intact_joined"]["score"]) / n, 4),
        },
        "auroc_joined_intact_vs_random": round(auc("intact_joined", "random", "score"), 4),
        "auroc_joined_intact_vs_critical": round(auc("intact_joined", "critical", "score"), 4),
        "H_layer_per_layer_auroc": {
            key: {
                "intact_joined_vs_critical": round(auc("intact_joined", "critical", key), 4),
                "intact_joined_vs_random": round(auc("intact_joined", "random", key), 4),
            }
            for key in (*layers, "unsup")
        },
        "H_op_flag_rates_at_v1_tau": {
            c: round(sum(s >= TAU for s in per[c]["score"]) / n, 4)
            for c in per
        },
        "score_percentiles": {
            c: {q: round(pct(per[c]["score"], float(q)), 4) for q in ("0.1", "0.5", "0.9")}
            for c in per
        },
    }
    print(json.dumps(report, indent=2))
    out = _THIS / "results" / "diagnose_ablation_artifact.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
