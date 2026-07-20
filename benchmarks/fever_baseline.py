"""
FEVER Baseline (Phase 0 step 1.3)
==================================

FEVER (Thorne et al. NAACL 2018) — Fact Extraction and Verification.
Standard 3-class claim-verification benchmark with Wikipedia evidence.

Per EICV_PREREGISTRATION.md:
  - Target: label accuracy ≥ 0.75 (DeBERTa-large 0.74 / LLM 0.78)

This script ships the BASELINE — point AUROC + accuracy on (evidence, claim)
pairs from FEVER's NLI-style processed variants. The C1-C4 falsification
probe is a separate follow-up (Phase 0 step 1.6).

Dataset loading strategy
------------------------
FEVER's original dataset stores evidence as Wikipedia URLs, requiring a
Wikipedia pre-fetch. We use NLI-style processed variants that ship evidence
text inline. We try several HF dataset names in order, use the first one
that loads cleanly:
  - copenlu/fever_gold_evidence
  - pminervini/NLI-FEVER
  - lukasellinger/fever-nli
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

SEED = 42
N_SAMPLES = 1000
TARGET_ACC = 0.75
RESULTS_DIR = _THIS / "results"
OUT_PATH = RESULTS_DIR / "fever_baseline.json"

logging.basicConfig(level=logging.INFO,
                    format="[%(name)s] %(message)s")
log = logging.getLogger("fever_baseline")


# Order of dataset loaders to try. Each loader returns
# list[(evidence_text, claim_text, label_int)] where label is 0 = SUPPORTS,
# 1 = REFUTES, 2 = NOT_ENOUGH_INFO (or None to skip).
def _load_first_available_split(name: str, prefer: list[str]) -> Any:
    """Try a list of split names; return the first that loads."""
    from datasets import load_dataset
    last_err = None
    for sp in prefer:
        try:
            return load_dataset(name, split=sp)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if last_err:
        raise last_err
    return None


def _try_copenlu_gold() -> list[tuple[str, str, int]] | None:
    try:
        ds = _load_first_available_split(
            "copenlu/fever_gold_evidence",
            ["validation", "dev", "test", "train"],
        )
    except Exception as e:
        log.info("copenlu/fever_gold_evidence load failed: %s", e)
        return None
    out: list[tuple[str, str, int]] = []
    label_map = {"SUPPORTS": 0, "REFUTES": 1,
                 "NOT ENOUGH INFO": 2, "NOT_ENOUGH_INFO": 2}
    sample_keys = list(ds[0].keys()) if len(ds) else []
    log.info("copenlu sample keys: %s", sample_keys)
    for row in ds:
        claim = str(row.get("claim", "") or "")
        ev_field = (row.get("evidence") or row.get("evidence_text")
                    or row.get("context") or row.get("evidence_wiki_url") or "")
        if isinstance(ev_field, list):
            # Each element may be list-of-dicts or str
            parts = []
            for x in ev_field:
                if isinstance(x, dict):
                    parts.append(str(x.get("sentence", x.get("text", ""))))
                else:
                    parts.append(str(x))
            ev_text = "\n".join(p for p in parts if p)
        else:
            ev_text = str(ev_field)
        label_raw = str(row.get("label", "") or row.get("verdict", "") or "").upper().strip()
        label = label_map.get(label_raw)
        if claim and ev_text and label is not None:
            out.append((ev_text, claim, label))
    return out or None


def _try_nli_fever() -> list[tuple[str, str, int]] | None:
    try:
        ds = _load_first_available_split(
            "pminervini/NLI-FEVER",
            ["validation", "dev", "test", "train"],
        )
    except Exception as e:
        log.info("pminervini/NLI-FEVER load failed: %s", e)
        return None
    out: list[tuple[str, str, int]] = []
    # NLI label space: 0=entailment(supports), 1=neutral(NEI), 2=contradiction(refutes)
    # Remap to: 0 supports, 1 refutes, 2 NEI
    remap = {0: 0, 1: 2, 2: 1}
    sample_keys = list(ds[0].keys()) if len(ds) else []
    log.info("nli-fever sample keys: %s", sample_keys)
    for row in ds:
        premise = str(row.get("premise", row.get("evidence", row.get("context", ""))) or "")
        hypothesis = str(row.get("hypothesis", row.get("claim", "")) or "")
        lab = row.get("label", row.get("gold_label"))
        if isinstance(lab, str):
            lab = {"entailment": 0, "neutral": 1, "contradiction": 2,
                   "supports": 0, "refutes": 2, "nei": 1,
                   "not enough info": 1, "not_enough_info": 1}.get(lab.lower())
        if isinstance(lab, int):
            label = remap.get(lab)
        else:
            label = None
        if premise and hypothesis and label is not None:
            out.append((premise, hypothesis, label))
    return out or None


def _try_lukasellinger() -> list[tuple[str, str, int]] | None:
    try:
        ds = _load_first_available_split(
            "lukasellinger/fever-nli",
            ["validation", "dev", "test", "train"],
        )
    except Exception as e:
        log.info("lukasellinger/fever-nli load failed: %s", e)
        return None
    out: list[tuple[str, str, int]] = []
    sample_keys = list(ds[0].keys()) if len(ds) else []
    log.info("lukasellinger sample keys: %s", sample_keys)
    for row in ds:
        ev = str(row.get("evidence", row.get("premise", "")) or "")
        claim = str(row.get("claim", row.get("hypothesis", "")) or "")
        lab_raw = row.get("label", row.get("gold_label"))
        if isinstance(lab_raw, str):
            label = {"supports": 0, "refutes": 1, "nei": 2,
                     "not_enough_info": 2}.get(lab_raw.lower())
        else:
            label = lab_raw
        if ev and claim and isinstance(label, int):
            out.append((ev, claim, label))
    return out or None


def load_items() -> tuple[list[tuple[str, str, int]] | None, str]:
    for name, loader in [
        ("copenlu/fever_gold_evidence", _try_copenlu_gold),
        ("pminervini/NLI-FEVER",         _try_nli_fever),
        ("lukasellinger/fever-nli",     _try_lukasellinger),
    ]:
        items = loader()
        if items:
            print(f"  Loaded {len(items)} items from {name}")
            return items, name
    return None, ""


def _ci95(n: int, p: float) -> float:
    if n == 0:
        return 0.0
    z = 1.96
    denom = 1.0 + z * z / n
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return spread


def _auroc(scores: list[float], labels: list[int]) -> float:
    from entroly.metrics import tie_corrected_auroc
    return tie_corrected_auroc(scores, labels)


def main() -> int:
    print("=" * 74)
    print("  FEVER Baseline (Phase 0 step 1.3)")
    print("=" * 74)
    items, source = load_items()
    if items is None:
        print("  ERROR: no FEVER variant could be loaded")
        print("  Tried: copenlu/fever_gold_evidence, pminervini/NLI-FEVER, lukasellinger/fever-nli")
        return 1

    rng = random.Random(SEED)
    rng.shuffle(items)
    # For the binary detection AUROC use SUPPORTS (0) vs REFUTES (1) only.
    binary = [(ev, cl, lab) for (ev, cl, lab) in items if lab in (0, 1)]
    binary = binary[:N_SAMPLES]
    print(f"  Using {len(binary)} binary items (SUPPORTS=0, REFUTES=1) from {source}")
    if len(binary) < 20:
        print("  ERROR: not enough labelled items")
        return 1

    from entroly.witness import WitnessAnalyzer
    analyzer = WitnessAnalyzer(use_nli=False, force_python=True, profile="rag")

    # ── Score with WITNESS ──
    print("  Scoring with WITNESS Python path (force_python=True)...")
    t0 = time.perf_counter()
    scores: list[float] = []
    labels: list[int] = []
    n = len(binary)
    for i, (ev, cl, lab) in enumerate(binary):
        # We use the claim as the "output to verify" against the evidence
        # as context. Higher witness summary_score = more grounded.
        # For REFUTES (label=1, hallucinated) we want our detection score
        # to be HIGH, so we flip: hallucination_score = 1 - summary_score.
        res = analyzer.analyze(ev, cl)
        scores.append(1.0 - float(res.summary_score))
        labels.append(lab)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"  Scored in {elapsed:.1f}s ({1000*elapsed/n:.1f}ms/item)")

    # ── Metrics ──
    auroc = _auroc(scores, labels)
    # Best-threshold accuracy (sweep)
    paired = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    best_acc = 0.5
    best_t = 0.5
    tp = fp = 0
    for s, y in paired:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tn = n_neg - fp
        fn = n_pos - tp
        acc = (tp + tn) / max(n_pos + n_neg, 1)
        if acc > best_acc:
            best_acc = acc
            best_t = s
    # F1 at best threshold
    pred = [1 if s >= best_t else 0 for s in scores]
    tp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-9)) if (precision + recall) else 0.0
    ci = _ci95(len(labels), best_acc)

    print()
    print(f"  AUROC                = {auroc:.4f}")
    print(f"  Accuracy (best-thr)  = {best_acc:.4f} ± {ci:.4f}")
    print(f"  Precision            = {precision:.4f}")
    print(f"  Recall               = {recall:.4f}")
    print(f"  F1                   = {f1:.4f}")
    print(f"  ms/item              = {1000*elapsed/n:.2f}")
    print(f"  Target acc (preset)  = {TARGET_ACC}")
    print(f"  Target met           = {best_acc >= TARGET_ACC}")

    out = {
        "benchmark": "FEVER",
        "source_dataset": source,
        "n_items": len(binary),
        "seed": SEED,
        "preregistered_target_accuracy": TARGET_ACC,
        "preregistration_doc": "benchmarks/EICV_PREREGISTRATION.md",
        "ms_per_item": round(1000 * elapsed / n, 2),
        "metrics": {
            "auroc": round(auroc, 4),
            "accuracy_at_best_threshold": round(best_acc, 4),
            "accuracy_ci95": round(ci, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "best_threshold": round(best_t, 6),
        },
        "labels": {"0": "SUPPORTS", "1": "REFUTES"},
        "notes": (
            "Phase 0 baseline. No C1-C4 falsification probe yet (Phase 0 step 1.6). "
            "Scoring uses WITNESS only; full Fusion-4 (W+E+G+S) wiring is a follow-up."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
