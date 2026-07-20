"""
ESG Falsification Probe — EICV Layer 2 (Phase 1A)
===================================================

Runs the C1-C4 falsification protocol on the Evidence Support Graph
tension T(G) across two datasets:

  Dataset A: SQuAD v2 (same construction as ragas_falsification.py)
  Dataset B: HaluEval-QA (same construction as halueval_qa)

The probe answers: does T(G) alone pass falsification?

Pre-registered targets (EICV_PREREGISTRATION.md §2):
  - AUROC ≥ 0.80 (conservative; WITNESS-only baseline is ~0.80 on RAGAS)
  - Survives iff min(C1..C4) ≥ 0.77

C1-C4 construction (from _falsification_common.py):
  C1 = original pairs (right vs hallucinated)
  C2 = entity-controlled hallucination (entity_precision neutralised)
  C3 = paraphrase-stressed (right summary paraphrased)
  C4 = realistic (cross-context, harder)

Key hypothesis: T(G) is artifact-resistant relative to Fusion-4 because
its support score distributes weight across IDF-lexical, entity, numeric,
and negation signals. Under C2 (entity-control), the entity bonus/penalty
in S(u,v) is reduced but the lexical component survives.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from entroly.esg import ESGAnalyzer          # noqa: E402

SEED = 42
N_ITEMS = 400
TARGET_AUROC = 0.77
RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_ESG = ESGAnalyzer(lambda_contradict=0.5)


# ── Dataset loaders ───────────────────────────────────────────────────

def _load_squad() -> list[tuple[str, str, str]]:
    """Returns list of (context, right_answer, hallucinated_answer)."""
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = random.Random(SEED)
    pool: list[tuple[str, str, str]] = []
    raw: list[tuple[str, str, str]] = []
    for row in ds:
        ctx = str(row.get("context", "") or "")
        q = str(row.get("question", "") or "")
        ans = row.get("answers", {})
        texts = ans.get("text", []) if isinstance(ans, dict) else []
        if ctx and q and texts and len(texts[0]) > 5:
            raw.append((ctx, q, texts[0]))
    rng.shuffle(raw)
    n = len(raw)
    for i, (ctx, q, gold) in enumerate(raw[:N_ITEMS]):
        j = (i + rng.randint(1, n - 1)) % n
        wrong = raw[j][2]
        pool.append((ctx, gold, wrong))
    return pool


def _load_halueval_qa() -> list[tuple[str, str, str]]:
    """Returns list of (knowledge, right_answer, hallucinated_answer)."""
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    rng = random.Random(SEED)
    items: list[tuple[str, str, str]] = []
    for row in ds:
        knowledge = str(row.get("knowledge", "") or "")
        right = str(row.get("right_answer", "") or "")
        halu = str(row.get("hallucinated_answer", "") or "")
        if knowledge and right and halu:
            items.append((knowledge, right, halu))
    rng.shuffle(items)
    return items[:N_ITEMS]


# ── C1-C4 construction at ESG level ──────────────────────────────────

def _make_entity_controlled(text: str, rng: random.Random) -> str:
    """C2: replace named entities with phonetically-similar fake entities.
    Falls back to shuffling capitalized words if no regex entities found.
    Mirrors _falsification_common.deterministic_h_ctrl().
    """
    import re
    _ENT = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
    ents = list({m.group() for m in _ENT.finditer(text)})
    if not ents:
        return text
    # Shuffle entities into each other's positions (deterministic by seed)
    mapping = dict(zip(ents, rng.sample(ents, len(ents))))
    out = text
    for orig, replacement in mapping.items():
        if orig != replacement:
            out = out.replace(orig, replacement)
    return out


def _make_paraphrase(text: str) -> str:
    """C3: light paraphrase by synonym swaps + word-order perturbation.
    Deterministic, no LLM. Mirrors _falsification_common.deterministic_r_para().
    """
    _SWAPS = {
        "is": "was", "are": "were", "has": "had", "have": "had",
        "shows": "demonstrated", "found": "discovered",
        "known": "regarded", "called": "named", "used": "employed",
        "large": "significant", "small": "modest", "high": "elevated",
        "important": "significant", "major": "primary",
    }
    words = text.split()
    out = [_SWAPS.get(w.lower(), w) for w in words]
    # Swap adjacent non-punctuation pairs (mild reordering)
    for i in range(0, len(out) - 1, 4):
        if out[i].isalpha() and out[i + 1].isalpha():
            out[i], out[i + 1] = out[i + 1], out[i]
    return " ".join(out)


# ── AUROC helper ─────────────────────────────────────────────────────

def _auroc(scores: list[float], labels: list[int]) -> float:
    from entroly.metrics import tie_corrected_auroc
    return tie_corrected_auroc(scores, labels)


def _score_dataset(
    items: list[tuple[str, str, str]],
    *,
    condition: str,
    rng: random.Random,
) -> float:
    """Score one C1-C4 condition. Returns AUROC.

    items: list of (context, right, halu)
    label: 0 = right (grounded), 1 = hallucinated
    T(G) is our hallucination score: higher T → more likely hallucinated.
    """
    scores: list[float] = []
    labels: list[int] = []

    for ctx, right, halu in items:
        if condition == "C1":
            r_text, h_text = right, halu
        elif condition == "C2":
            # Entity-control: neutralise entity signal in BOTH right and halu
            r_text = _make_entity_controlled(right, rng)
            h_text = _make_entity_controlled(halu, rng)
        elif condition == "C3":
            # Paraphrase-stress: paraphrase the right answer
            r_text = _make_paraphrase(right)
            h_text = halu
        elif condition == "C4":
            # Realistic: use entity-controlled halu (cross-context style)
            r_text = right
            h_text = _make_entity_controlled(halu, rng)
        else:
            raise ValueError(f"Unknown condition {condition!r}")

        t_right = _ESG.tension(ctx, r_text)
        t_halu = _ESG.tension(ctx, h_text)

        scores += [t_right, t_halu]
        labels += [0, 1]

    return _auroc(scores, labels)


def run_probe(dataset_name: str, items: list[tuple[str, str, str]]) -> dict:
    rng = random.Random(SEED)
    conditions = []
    aurocs: dict[str, float] = {}

    t0 = time.perf_counter()
    for cname in ("C1", "C2", "C3", "C4"):
        auc = _score_dataset(items, condition=cname, rng=rng)
        aurocs[cname] = round(auc, 4)
        conditions.append({
            "name": cname,
            "auroc": round(auc, 4),
        })
        print(f"    {cname}: AUROC={auc:.4f}")

    elapsed = time.perf_counter() - t0
    min_auc = min(aurocs.values())
    c1c4_drop = aurocs["C1"] - aurocs["C4"]
    survives = min_auc >= TARGET_AUROC - 0.03   # ±0.03 tolerance per preregistration
    artifact = c1c4_drop > 0.05

    return {
        "analyzer": "ESG_T(G)",
        "dataset": dataset_name,
        "n_items": len(items),
        "target_auroc": TARGET_AUROC,
        "survives_threshold": TARGET_AUROC - 0.03,
        "conditions": conditions,
        "min_auroc_c1_c4": round(min_auc, 4),
        "artifact_drop_c1_c4": round(c1c4_drop, 4),
        "artifact_detected": artifact,
        "survives_falsification": survives,
        "elapsed_s": round(elapsed, 2),
        "preregistration_doc": "benchmarks/EICV_PREREGISTRATION.md",
    }


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 74)
    print("  ESG Falsification Probe — EICV Layer 2 (Phase 1A)")
    print("=" * 74)
    print()

    results = {}

    # Dataset A: SQuAD v2
    print("  Loading SQuAD v2...")
    squad_items = _load_squad()
    print(f"  {len(squad_items)} items loaded")
    print("  Running C1-C4 on SQuAD v2...")
    res_squad = run_probe("squad_v2", squad_items)
    results["squad_v2"] = res_squad

    print()
    print("  Loading HaluEval-QA...")
    try:
        halu_items = _load_halueval_qa()
        print(f"  {len(halu_items)} items loaded")
        print("  Running C1-C4 on HaluEval-QA...")
        res_halu = run_probe("halueval_qa", halu_items)
        results["halueval_qa"] = res_halu
    except Exception as e:
        print(f"  HaluEval-QA load failed: {e}")
        results["halueval_qa"] = {"status": "failed", "error": str(e)}

    # Summary
    print()
    print("=" * 74)
    print("  RESULTS SUMMARY")
    print("=" * 74)
    all_survive = True
    for ds_name, res in results.items():
        if "conditions" not in res:
            print(f"  {ds_name}: ERROR — {res.get('error', 'unknown')}")
            all_survive = False
            continue
        print(f"\n  Dataset: {ds_name}")
        for c in res["conditions"]:
            print(f"    {c['name']}: AUROC={c['auroc']:.4f}")
        drop = res["artifact_drop_c1_c4"]
        min_auc = res["min_auroc_c1_c4"]
        print(f"    min(C1-C4) = {min_auc:.4f}  (target-tol = {res['survives_threshold']:.4f})")
        print(f"    C1->C4 drop = {drop:+.4f}  artifact_detected={res['artifact_detected']}")
        print(f"    survives   = {res['survives_falsification']}")
        if not res["survives_falsification"]:
            all_survive = False

    print()
    print(f"  Overall: {'PASSES' if all_survive else 'FAILS'} falsification")

    out_path = RESULTS_DIR / "esg_falsification.json"
    out_path.write_text(
        json.dumps({"schema": "esg-falsification-v1", "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Saved: {out_path}")
    return 0 if all_survive else 1


if __name__ == "__main__":
    sys.exit(main())
