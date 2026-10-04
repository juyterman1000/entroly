"""Crystallization state must be a function of its inputs, not of hash order.

``_find_or_create_cluster`` trimmed the centroid token bag with
``centroid_tokens.pop()``. ``set.pop`` returns whichever element the hash table
exposes first, and CPython salts ``str`` hashing per process, so an identical
48-observation sequence produced 4, 5 or 6 active clusters depending on
``PYTHONHASHSEED``, with different surviving centroids -- sometimes losing a
*core* identity term of the query family.

Cluster partition is the input to every future crystallization decision, so a
seed-dependent partition means "this query family earned a skill candidate" is
not reproducible. The pre-fix sweep did not show a *different candidate event*
being emitted; these tests pin the state property, which is the one that was
actually demonstrated to break.

The subprocess test is the real one: hash randomization is per-process, so it
cannot be exercised in-process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from entroly.reward_crystallizer import RewardCrystallizer, _Cluster

REPO = Path(__file__).resolve().parent.parent

# Sized against the membership gate rather than guessed. With a shared core of C
# tokens and a centroid of size M, a query adding F new tokens reuses the cluster
# only while C / (M + F) >= query_jaccard (0.25). Reaching the 32-token cap with
# F = 1 therefore needs C >= 33/4, i.e. 9 core tokens. An earlier fixture used 4
# and every centroid stalled at 16: the family split before the cap was ever
# touched, so the sweep passed vacuously.
CORE = ("selection budget knapsack fragment ranking "
        "receipt entropy dedup token")
NOISE_CORE = "quarterly invoice payroll ledger vendor"

_WORKER = r'''
import json
from entroly.reward_crystallizer import RewardCrystallizer

W = {"w_recency": 0.1, "w_frequency": 0.1, "w_semantic": 0.7, "w_entropy": 0.1}
FRAGS = ["frag_a", "frag_b", "frag_c", "frag_d"]
NOISE_FRAGS = ["noise_a", "noise_b", "noise_c", "noise_d"]
CORE = %r
NOISE_CORE = %r

seq = []
for i in range(30):
    seq.append((CORE + " grow%%03d" %% i, 0.9, FRAGS))
    if i %% 3 == 0:
        # Low-reward noise supplies the >=5 non-cluster observations the
        # external baseline needs before it is used at all.
        seq.append((NOISE_CORE + " misc%%03d" %% i, 0.05, NOISE_FRAGS))
# Knife-edge probes: with the centroid pinned at 32, a probe of P tokens of
# which S are resident passes only when 5S >= 32 + P, so at P = 8 a single
# evicted token moves the probe into a new cluster.
for start in (0, 4, 8, 12, 16, 20, 24):
    seq.append((" ".join("grow%%03d" %% j for j in range(start, start + 8)), 0.9, FRAGS))
seq.append((CORE, 0.9, FRAGS))

c = RewardCrystallizer()
emissions, payloads = [], []
for i, (q, r, frags) in enumerate(seq):
    ev = c.observe(query=q, reward=r, weights=W,
                   selected_fragment_ids=frags, baseline_reward=0.4)
    if ev is not None:
        emissions.append(i)
        payloads.append({
            "n_samples": getattr(ev, "n_samples", None),
            "common_terms": list(getattr(ev, "common_terms", []) or []),
            "fragment_recipe": list(getattr(ev, "fragment_recipe", []) or []),
            "sample_queries": list(getattr(ev, "sample_queries", []) or []),
            "weight_profile": {k: round(v, 6) for k, v in
                               (getattr(ev, "weight_profile", {}) or {}).items()},
        })

# cluster_id is a uuid4 and ts is wall-clock: neither is comparable. Everything
# below is semantic output.
clusters = sorted(
    (sorted(cl.centroid_tokens), sorted(cl.centroid_fragments),
     list(cl.queries), cl.n_total,
     sorted(cl.token_doc_counts.items()),
     RewardCrystallizer._common_terms(cl.queries[-16:], top_k=8),
     RewardCrystallizer._top_fragments(cl.window, top_k=8))
    for cl in c._clusters.values()
)
print(json.dumps({
    "clusters": clusters,
    "emissions": emissions,
    "payloads": payloads,
    "n_clusters": len(c._clusters),
    "max_centroid": max(len(cl.centroid_tokens) for cl in c._clusters.values()),
}, sort_keys=True))
''' % (CORE, NOISE_CORE)

SEEDS = ["0", "1", "42", "12345", "7919", "104729", None, None]


def _run(seed: str | None) -> dict:
    env = dict(os.environ)
    if seed is None:
        env.pop("PYTHONHASHSEED", None)   # unseeded: process picks its own salt
    else:
        env["PYTHONHASHSEED"] = seed
    env["ENTROLY_NO_SELF_HEAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER],
        capture_output=True, text=True, cwd=str(REPO), env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    # Engine logging goes to stderr; the payload is the last stdout line.
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_crystallization_is_identical_across_hash_seeds():
    runs = {f"{s}#{i}": _run(s) for i, s in enumerate(SEEDS)}
    ref_key, ref = next(iter(runs.items()))

    # Validity gate first. A sweep where the mechanism never fires proves
    # nothing, and that is exactly how the first version of this fixture passed.
    assert ref["max_centroid"] >= 32, (
        f"centroid cap never exercised (max {ref['max_centroid']}); the fixture "
        "is not reaching the trim path and the comparison below is vacuous"
    )
    assert ref["emissions"], "no candidate was emitted; fixture proves nothing"

    for field in ("clusters", "emissions", "payloads", "n_clusters"):
        base = json.dumps(ref[field], sort_keys=True)
        for key, run in runs.items():
            assert json.dumps(run[field], sort_keys=True) == base, (
                f"{field} differs between {ref_key} and {key} for identical input"
            )


def test_trim_drops_the_least_recurring_token_not_an_arbitrary_one():
    """The retention rule, asserted directly rather than through the sweep."""
    cl = _Cluster(
        cluster_id="c",
        centroid_tokens={f"core{i}" for i in range(32)} | {"driveby"},
        centroid_fragments=set(),
        token_doc_counts={f"core{i}": 9 for i in range(32)} | {"driveby": 1},
    )
    RewardCrystallizer._trim_centroid_locked(cl)

    assert "driveby" not in cl.centroid_tokens, "the one-off token must go first"
    assert len(cl.centroid_tokens) == 32
    # The count survives, so a term that keeps recurring can earn its way back.
    assert cl.token_doc_counts["driveby"] == 1


def test_trim_breaks_count_ties_lexically():
    cl = _Cluster(
        cluster_id="c",
        centroid_tokens={f"t{i:02d}" for i in range(34)},
        centroid_fragments=set(),
        token_doc_counts={f"t{i:02d}": 5 for i in range(34)},
    )
    RewardCrystallizer._trim_centroid_locked(cl)

    assert len(cl.centroid_tokens) == 32
    # All counts equal, so the two lexically smallest are the victims. The point
    # is that the choice is total and reproducible, not that it is clever.
    assert {"t00", "t01"}.isdisjoint(cl.centroid_tokens)


def test_trim_enforces_the_cap_when_several_tokens_arrive_at_once():
    """The old `if` + single pop left the documented bound unenforced."""
    cl = _Cluster(
        cluster_id="c",
        centroid_tokens={f"t{i:02d}" for i in range(40)},
        centroid_fragments=set(),
        token_doc_counts={f"t{i:02d}": i for i in range(40)},
    )
    RewardCrystallizer._trim_centroid_locked(cl)
    assert len(cl.centroid_tokens) == 32


def test_count_pruning_never_zeroes_a_resident_token():
    """A pruned count would reset to 0 and make a resident token the next victim."""
    cl = _Cluster(
        cluster_id="c",
        centroid_tokens={f"res{i}" for i in range(32)},
        centroid_fragments=set(),
        token_doc_counts=(
            {f"res{i}": 7 for i in range(32)}
            | {f"gone{i}": 1 for i in range(200)}
        ),
    )
    RewardCrystallizer._trim_centroid_locked(cl)

    for tok in cl.centroid_tokens:
        assert cl.token_doc_counts.get(tok, 0) > 0, f"{tok} lost its count"
    assert len(cl.token_doc_counts) <= 4 * 32


def test_set_pop_is_not_reintroduced():
    """Guard the executable code, not the prose.

    A substring scan matched this module's own docstring explaining the old
    rule, so the check parses instead: look for an actual zero-argument
    ``<something>.centroid_tokens.pop()`` call.
    """
    import ast
    import inspect

    import entroly.reward_crystallizer as module

    tree = ast.parse(inspect.getsource(module))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pop"
        and not node.args
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "centroid_tokens"
    ]
    assert not offenders, f"hash-ordered eviction is back at lines {offenders}"
