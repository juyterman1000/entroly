"""Is RewardCrystallizer output a function of its input sequence alone?

Why ask: crystallization is a candidate-authorizing signal. A CrystallizationEvent
feeds SkillEngine, so if the same observation sequence produces different events
depending on process-level hash randomization, then "this query family earns a
skill" is partly a coin flip, and no crystallization result is reproducible
across runs.

The suspect found by reading, before measuring, is reward_crystallizer.py:445:

    cl.centroid_tokens |= qtokens
    if len(cl.centroid_tokens) > 32:
        cl.centroid_tokens.pop()      # set.pop() over a set[str]

``set.pop`` removes an arbitrary element determined by the hash table layout,
and CPython salts ``str`` hashing per process (PYTHONHASHSEED). The dropped
token changes the centroid, the centroid decides future cluster membership via
Jaccard, and membership decides which window accumulates and therefore when the
Hoeffding bound is crossed. A single dropped token can propagate to a different
event.

Other candidates were examined and cleared by inspection:
  _top_fragments  sorted() over a dict built from tuple(sorted(...)) -> stable
  _common_terms   sorted by (count, token) -> total order
  _jaccard        set algebra returning a float -> order-independent
  min(clusters)   ties fall back to dict insertion order -> deterministic

Comparison excludes cluster_id (uuid4) and timestamps, which are expected to
vary. It compares semantic output: how queries partition into clusters, each
cluster's surviving centroid, where events fire in the sequence, event payloads,
and final stats.

Run: python research/experiments/crystallizer_determinism.py
Writes research/ledger/crystallizer_determinism.json
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "ledger" / "crystallizer_determinism.json"

SEEDS = ["0", "1", "42", "12345", "7919", "104729", "random"]


# ── The sequence under test (identical in every subprocess) ────────────
# A first attempt capped every centroid at 16 tokens and emitted no event, so it
# measured nothing: with a 4-token core, adding 3 new tokens drops Jaccard to
# 4/19 = 0.21 and the family *splits* into a new cluster long before the 32-token
# cap is reachable. Recorded in the ledger as run 1, INVALID.
#
# Sizing the fixture against the gate instead of guessing: with a shared core of
# C tokens, a centroid of size M and a query contributing F new tokens, the
# cluster is reused only while C / (M + F) >= query_jaccard = 0.25. For the cap
# at M = 32 with F = 1 that needs C >= 33/4, so C = 9. Nine core tokens plus one
# fresh token per query grows one centroid to 33 and forces the drop.
#
# A low-reward noise family runs alongside it, because crystallization compares
# the cluster's Hoeffding LCB against an EXTERNAL baseline that needs ext_n >= 5
# non-cluster observations before it is used at all.
WORKER = r'''
import json, sys
from entroly.reward_crystallizer import RewardCrystallizer

W = {"w_recency": 0.1, "w_frequency": 0.1, "w_semantic": 0.7, "w_entropy": 0.1}
FRAGS = ["frag_a", "frag_b", "frag_c", "frag_d"]
NOISE_FRAGS = ["noise_a", "noise_b", "noise_c", "noise_d"]

CORE = ("selection budget knapsack fragment ranking "
        "receipt entropy dedup token")          # 9 identity tokens
NOISE_CORE = "quarterly invoice payroll ledger vendor"

# Phase 1: grow one centroid past the cap, with noise interleaved to supply the
# external baseline.
SEQ = []
for i in range(30):
    SEQ.append((f"{CORE} grow{i:03d}", 0.9, FRAGS))
    if i % 3 == 0:
        SEQ.append((f"{NOISE_CORE} misc{i:03d}", 0.05, NOISE_FRAGS))

# Phase 2: knife-edge probes. With the centroid pinned at 32, a probe of P
# tokens of which S are present passes only when 5S >= 32 + P. At P = 8 that
# requires all eight to have survived the drop, so one evicted token moves the
# probe into a brand-new cluster.
PROBES = []
for start in (0, 4, 8, 12, 16, 20, 24):
    PROBES.append((
        " ".join(f"grow{j:03d}" for j in range(start, start + 8)), 0.9, FRAGS,
    ))
PROBES.append((CORE, 0.9, FRAGS))                    # core-only control

c = RewardCrystallizer()
trace = []
for i, (q, r, frags) in enumerate(SEQ + PROBES):
    ev = c.observe(
        query=q, reward=r, weights=W,
        selected_fragment_ids=frags, baseline_reward=0.4,
    )
    trace.append({
        "i": i,
        "query": q,
        # cluster_id is a uuid4, so the stable surrogate for "which cluster did
        # this land in" is the number of clusters resident afterwards.
        "clusters_after": len(c._clusters),
        "event": None if ev is None else {
            "n_observations": getattr(ev, "n_observations", None),
            "mean_reward": round(getattr(ev, "mean_reward", 0.0), 6),
            "lower_bound": round(getattr(ev, "lower_bound", 0.0), 6),
            "common_terms": list(getattr(ev, "common_terms", []) or []),
            "top_fragments": list(getattr(ev, "top_fragments", []) or []),
            "weights": {k: round(v, 6) for k, v in
                        (getattr(ev, "weights", {}) or {}).items()},
        },
    })

# Partition: map each cluster to its sorted centroid and the queries it holds.
clusters = sorted(
    (
        sorted(cl.centroid_tokens),
        sorted(cl.centroid_fragments),
        list(cl.queries),
        cl.n_total,
    )
    for cl in c._clusters.values()
)
stats = dict(c.stats())
stats.pop("pending_events", None)  # drained state is not semantic here

print(json.dumps({
    "trace": trace,
    "clusters": clusters,
    "stats": stats,
}, sort_keys=True))
'''


def run(seed: str) -> dict:
    env = dict(os.environ)
    if seed == "random":
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = seed
    env["ENTROLY_NO_SELF_HEAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", WORKER],
        capture_output=True, text=True, cwd=str(ROOT), env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(f"seed {seed} failed:\n{proc.stderr[-2000:]}")
    # Engine logging goes to stderr; the payload is the last stdout line.
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    results = {}
    for seed in SEEDS:
        # "random" is run several times: without PYTHONHASHSEED each process
        # picks its own salt, so one sample cannot distinguish stable from
        # lucky.
        reps = 4 if seed == "random" else 1
        for rep in range(reps):
            key = seed if reps == 1 else f"{seed}#{rep}"
            results[key] = run(seed)
            print(f"  ran {key}")

    keys = list(results)
    ref = keys[0]

    def diff(field: str) -> list[str]:
        base = json.dumps(results[ref][field], sort_keys=True)
        return [k for k in keys[1:]
                if json.dumps(results[k][field], sort_keys=True) != base]

    # Event emission points are the decision-relevant output: where in the
    # sequence a candidate becomes authorized.
    def emission_points(r: dict) -> list[int]:
        return [t["i"] for t in r["trace"] if t["event"] is not None]

    emissions = {k: emission_points(results[k]) for k in keys}
    centroids = {
        k: [c[0] for c in results[k]["clusters"]] for k in keys
    }
    partitions = {
        k: [c[2] for c in results[k]["clusters"]] for k in keys
    }

    # Validity gate: a sweep where the mechanism never fired proves nothing. The
    # first run of this script passed every comparison with zero events and
    # every centroid below the cap, which is exactly the failure mode to refuse.
    max_centroid = max(
        max((len(c) for c in centroids[k]), default=0) for k in keys
    )
    validity = {
        "centroid_cap_exercised": max_centroid >= 32,
        "max_centroid_size": max_centroid,
        "events_emitted": {k: len(emissions[k]) for k in keys},
        "any_event_emitted": any(emissions[k] for k in keys),
    }
    validity["valid"] = (
        validity["centroid_cap_exercised"] and validity["any_event_emitted"]
    )

    report = {
        "baseline_commit": "55e75c88",
        "validity": validity,
        "seeds": keys,
        "n_clusters": {k: len(results[k]["clusters"]) for k in keys},
        "emission_points": emissions,
        "emission_points_identical": len(
            {json.dumps(v) for v in emissions.values()}
        ) == 1,
        "centroids_identical": len(
            {json.dumps(v, sort_keys=True) for v in centroids.values()}
        ) == 1,
        "partition_identical": len(
            {json.dumps(v, sort_keys=True) for v in partitions.values()}
        ) == 1,
        "trace_differs_from_reference": diff("trace"),
        "clusters_differ_from_reference": diff("clusters"),
        "stats_differ_from_reference": diff("stats"),
        "stats": {k: results[k]["stats"] for k in keys},
        "centroids": centroids,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("centroids", "stats")}, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
