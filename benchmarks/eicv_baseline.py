"""
EICV Baseline Aggregator — Phase 0
====================================

Phase 0 deliverable. Reads existing per-benchmark JSONs from `benchmarks/results/`
and emits one canonical `eicv_baseline.json` conforming to schema
`eicv-baseline-v1` (pre-registered in EICV_PREREGISTRATION.md).

What this does NOT do (yet):
  - Re-run benchmarks. It aggregates the most-recent existing JSONs.
  - Cover FActScore / FEVER / TRUE / HHEM. Adapters land as subsequent Phase 0 steps.
  - Compute the primary metric (Risk@Coverage worst-manifold). That requires
    EICV Phase 4 components.

The point of this script: produce ONE auditable file that captures the
*current honest baseline* before any EICV components land. Every later report
references this as the "before" state.

Usage:
    python benchmarks/eicv_baseline.py

Output:
    benchmarks/results/eicv_baseline.json
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "eicv-baseline-v1"
PREREG_FROZEN_COMMIT = "9587650c3f4288c7e163d7cfbdbe9a5b95ae1cf5"
RESULTS_DIR = Path(__file__).parent / "results"
OUT_PATH = RESULTS_DIR / "eicv_baseline.json"


# Pre-registered targets (mirror §2 of EICV_PREREGISTRATION.md).
# The document is authoritative; if these diverge, fix here.
PRIMARY_METRIC = {
    "name": "risk_at_coverage_80_worst_manifold",
    "target": 0.05,
    "computed_in_phase": 4,
}

TIER1_TARGETS = {
    "halueval_qa":            {"metric": "auroc",     "target": 0.85, "anchor": "DeBERTa-v3-large NLI ~0.85"},
    "halueval_dialogue":      {"metric": "f1",        "target": 0.68, "anchor": "NLI baselines ~0.78"},
    "halueval_summarization": {"metric": "f1",        "target": 0.55, "anchor": "NLI baselines ~0.65"},
    "truthfulqa":             {"metric": "auroc",     "target": 0.78, "anchor": "LLM-judge baselines"},
    "factscore":              {"metric": "atomic_precision", "target": 0.72, "anchor": "GPT-4 0.76 / smaller 0.60-0.65"},
    "fever":                  {"metric": "accuracy",  "target": 0.75, "anchor": "DeBERTa-large 0.74 / LLM 0.78"},
    "true":                   {"metric": "mean_auroc","target": 0.80, "anchor": "DeBERTa-v3-large 0.79"},
    "hhem":                   {"metric": "leaderboard_rank", "target": "top_decile", "anchor": "leaderboard-relative"},
    "ragas":                  {"metric": "faithfulness_auroc", "target": "report_only", "anchor": "RAGAS uses LLM-judge, no fair comparison"},
}

ROBUSTNESS_TARGETS = {
    "falsification_survived_auroc_drop": {"target_max": 0.03, "anchor": "Fusion-4 collapse 0.441 on HaluEval-QA"},
    "worst_manifold_ece":                {"target_max": 0.08, "anchor": "LLM-judge ~0.15, DeBERTa-NLI ~0.12"},
    "conformal_coverage_violation_a01": {"target_max": 0.02, "anchor": "Theoretical bound under exchangeability"},
    "cost_per_claim_ms_p99":            {"target_max": 20.0, "anchor": "LLM-judge ~500ms + cost"},
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
    except Exception:
        return "unknown"


def _entroly_version() -> str:
    try:
        from entroly import __version__
        return __version__
    except Exception:
        try:
            import tomllib
            data = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
            return data.get("project", {}).get("version", "unknown")
        except Exception:
            return "unknown"


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _read_json(name: str) -> tuple[dict | None, str | None]:
    """Return (parsed_json, file_hash). file_hash is short SHA256."""
    path = RESULTS_DIR / name
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), _file_hash(path)
    except (OSError, ValueError):
        return None, _file_hash(path)


# ── Per-benchmark adapters ──────────────────────────────────────────────


def _adapt_halueval_qa_10k(raw: dict | None, source_hash: str | None) -> dict:
    if raw is None:
        return {"status": "not_run", "reason": "halueval_10k_fusion.json missing"}
    w = raw.get("witness", {})
    f = raw.get("fusion", {})
    return {
        "status": "ok",
        "source_file": "halueval_10k_fusion.json",
        "source_hash": source_hash,
        "dataset": "pminervini/HaluEval",
        "config": "qa",
        "n_items": raw.get("n_items"),
        "n_decisions": raw.get("n_decisions"),
        "n_test": raw.get("n_test"),
        "metrics": {
            "witness": {
                "auroc":     w.get("auroc_test", w.get("auroc_full")),
                "ci95":      w.get("ci95"),
                "accuracy":  w.get("accuracy"),
                "threshold": w.get("threshold"),
            },
            "fusion": {
                "auroc":     f.get("auroc_test", f.get("auroc_full")),
                "ci95":      f.get("ci95"),
                "accuracy":  f.get("accuracy"),
                "threshold": f.get("threshold"),
                "weights":   f.get("weights"),
            },
        },
    }


def _adapt_truthfulqa(raw: dict | None, source_hash: str | None) -> dict:
    if raw is None:
        return {"status": "not_run", "reason": "truthfulqa_benchmark.json missing"}
    w = raw.get("witness", {})
    f = raw.get("fusion", {})
    return {
        "status": "ok",
        "source_file": "truthfulqa_benchmark.json",
        "source_hash": source_hash,
        "dataset": raw.get("dataset", "truthful_qa/generation"),
        "license": raw.get("license"),
        "n_questions": raw.get("n_questions"),
        "n_decisions": raw.get("n_decisions"),
        "seed": raw.get("seed"),
        "ms_per_decision": raw.get("ms_per_decision"),
        "metrics": {
            "witness": {
                "auroc":     w.get("auroc"),
                "accuracy":  w.get("accuracy"),
                "f1":        w.get("f1"),
                "precision": w.get("precision"),
                "recall":    w.get("recall"),
                "ci95_acc":  w.get("ci95"),
                "threshold": w.get("threshold"),
            },
            "fusion": {
                "auroc":     f.get("auroc"),
                "accuracy":  f.get("accuracy"),
                "f1":        f.get("f1"),
                "precision": f.get("precision"),
                "recall":    f.get("recall"),
                "weights":   f.get("weights"),
            },
        },
        "note": raw.get("note"),
    }


def _adapt_witness_slices(raw: list | None, source_hash: str | None) -> dict:
    """witness_benchmarks.json -> dialogue + summarization slices.

    Note: this file contains older n=200 runs (predates the v3 binding/residual
    gates that brought QA exposure 0.639 -> 0.340). Marked stale; the v3
    numbers are in witness_v3_report.md but not in a parseable JSON yet.
    """
    if raw is None or not isinstance(raw, list):
        return {"status": "not_run", "reason": "witness_benchmarks.json missing or unexpected shape"}
    by_name: dict[str, dict] = {}
    for entry in raw:
        if isinstance(entry, dict):
            name = entry.get("benchmark", "?")
            by_name[name] = entry
    def _slice(prefix: str) -> dict | None:
        for k, v in by_name.items():
            if k.startswith(prefix):
                return v
        return None
    def _adapt_one(entry: dict | None) -> dict:
        if entry is None:
            return {"status": "not_run"}
        return {
            "n_samples": entry.get("n_samples"),
            "tp": entry.get("tp"), "fp": entry.get("fp"),
            "fn": entry.get("fn"), "tn": entry.get("tn"),
            "precision": entry.get("precision"),
            "recall":    entry.get("recall"),
            "f1":        entry.get("f1"),
            "accuracy":  entry.get("accuracy"),
            "suppression_f1":           entry.get("suppression_f1"),
            "unsupported_exposure_rate":entry.get("unsupported_exposure_rate"),
            "supported_retention_rate": entry.get("supported_retention_rate"),
            "avg_ms":    entry.get("avg_ms"),
        }
    return {
        "status": "ok",
        "source_file": "witness_benchmarks.json",
        "source_hash": source_hash,
        "stale_warning": (
            "These n=200 numbers predate the v3 binding/residual gates that brought "
            "HaluEval-QA exposure 0.639 -> 0.340 (see witness_v3_report.md). "
            "Phase 0 step 1.1 will re-run on n=10k with the v3 gates."
        ),
        "halueval_qa_n200_stale":        _adapt_one(_slice("HaluEval-QA (string/local)")),
        "halueval_dialogue_n200":        _adapt_one(_slice("HaluEval-Dialogue (string/local)")),
        "halueval_summarization_n200":   _adapt_one(_slice("HaluEval-Summarization (string/local)")),
    }


def _adapt_ragas(raw: dict | None, source_hash: str | None) -> dict:
    if raw is None:
        return {"status": "not_run", "reason": "ragas_faithfulness_benchmark.json missing"}
    w = raw.get("witness", {})
    f = raw.get("fusion", {})
    return {
        "status": "ok",
        "source_file": "ragas_faithfulness_benchmark.json",
        "source_hash": source_hash,
        "dataset": raw.get("dataset"),
        "license": raw.get("license"),
        "n_questions": raw.get("n_questions"),
        "n_decisions": raw.get("n_decisions"),
        "seed": raw.get("seed"),
        "avg_claims_per_response": raw.get("avg_claims_per_response"),
        "ms_per_decision": raw.get("ms_per_decision"),
        "metrics": {
            "witness": {
                "auroc":     w.get("auroc"),
                "accuracy":  w.get("accuracy"),
                "f1":        w.get("f1"),
                "precision": w.get("precision"),
                "recall":    w.get("recall"),
                "ci95_acc":  w.get("ci95"),
            },
            "fusion": {
                "auroc":     f.get("auroc"),
                "accuracy":  f.get("accuracy"),
                "f1":        f.get("f1"),
                "precision": f.get("precision"),
                "recall":    f.get("recall"),
                "weights":   f.get("weights"),
            },
        },
        "ragas_comparison_note": raw.get("ragas_comparison", {}).get("note"),
    }


def _adapt_falsification_legacy(raw: dict | None, source_hash: str | None,
                                 *, source_file: str, applies_to: str) -> dict:
    """Adapter for fusion4_falsification.json (the original HaluEval-QA probe).
    Schema differs from the per-benchmark probes built under Phase 0 step 1.1."""
    if raw is None:
        return {"status": "not_run", "reason": f"{source_file} missing"}
    by_name = {c["name"]: c for c in raw.get("conditions", []) if isinstance(c, dict)}
    def _g(prefix: str, key: str) -> float | None:
        for n, c in by_name.items():
            if n.startswith(prefix):
                return c.get(key)
        return None
    fusion_aurocs = [c.get("fusion") for c in raw.get("conditions", []) if isinstance(c, dict)]
    fusion_aurocs = [x for x in fusion_aurocs if isinstance(x, (int, float))]
    c1 = _g("C1", "fusion"); c4 = _g("C4", "fusion")
    return {
        "status": "ok",
        "source_file": source_file,
        "source_hash": source_hash,
        "applies_to": applies_to,
        "weights":    raw.get("weights"),
        "n":          raw.get("n"),
        "backend":    raw.get("backend", "gpt-4o-mini"),  # legacy fusion4 was GPT
        "conditions": {
            "C1_original":        {"fusion_auroc": _g("C1", "fusion"), "witness_only": _g("C1", "witness_only"), "g_only": _g("C1", "g_only")},
            "C2_entity_ctrl":     {"fusion_auroc": _g("C2", "fusion"), "witness_only": _g("C2", "witness_only"), "g_only": _g("C2", "g_only")},
            "C3_paraphrase":      {"fusion_auroc": _g("C3", "fusion"), "witness_only": _g("C3", "witness_only"), "g_only": _g("C3", "g_only")},
            "C4_realistic":       {"fusion_auroc": _g("C4", "fusion"), "witness_only": _g("C4", "witness_only"), "g_only": _g("C4", "g_only")},
        },
        "min_fusion_auroc_c1_c4": min(fusion_aurocs) if fusion_aurocs else None,
        "max_fusion_auroc_c1_c4": max(fusion_aurocs) if fusion_aurocs else None,
        "artifact_drop_c1_c4":   (c1 - c4) if (c1 is not None and c4 is not None) else None,
        "artifact_detected":      raw.get("artifact"),
        "survives_falsification": raw.get("survives"),
        "survives_threshold":     None,
        "verdict":                raw.get("verdict"),
    }


def _adapt_falsification_v2(raw: dict | None, source_hash: str | None,
                            *, source_file: str, applies_to: str) -> dict:
    """Adapter for the per-benchmark falsification JSONs built under Phase 0
    step 1.1 by benchmarks/_falsification_common.py:run_probe()."""
    if raw is None:
        return {"status": "not_run", "reason": f"{source_file} missing"}
    conditions_list = raw.get("conditions", [])
    def _g(prefix: str, key: str) -> float | None:
        for c in conditions_list:
            if isinstance(c, dict) and c.get("name", "").startswith(prefix):
                return c.get(key)
        return None
    return {
        "status": "ok",
        "source_file": source_file,
        "source_hash": source_hash,
        "applies_to": applies_to,
        "weights":    raw.get("weights"),
        "n":          raw.get("n"),
        "backend":    raw.get("backend"),
        "conditions": {
            "C1_original":    {"fusion_auroc": _g("C1", "fusion"), "witness_only": _g("C1", "witness_only"), "g_only": _g("C1", "g_only")},
            "C2_entity_ctrl": {"fusion_auroc": _g("C2", "fusion"), "witness_only": _g("C2", "witness_only"), "g_only": _g("C2", "g_only")},
            "C3_paraphrase":  {"fusion_auroc": _g("C3", "fusion"), "witness_only": _g("C3", "witness_only"), "g_only": _g("C3", "g_only")},
            "C4_realistic":   {"fusion_auroc": _g("C4", "fusion"), "witness_only": _g("C4", "witness_only"), "g_only": _g("C4", "g_only")},
        },
        "min_fusion_auroc_c1_c4": raw.get("min_fusion_auroc_c1_c4"),
        "max_fusion_auroc_c1_c4": raw.get("max_fusion_auroc_c1_c4"),
        "artifact_drop_c1_c4":   raw.get("artifact_drop_c1_c4"),
        "artifact_detected":      raw.get("artifact_detected"),
        "survives_falsification": raw.get("survives_falsification"),
        "survives_threshold":     raw.get("survives_threshold"),
        "preregistered_target":   raw.get("preregistered_target"),
        "dataset_hash":           raw.get("dataset_hash"),
    }


def _adapt_simple_baseline(raw_and_hash: tuple[dict | None, str | None],
                            *, default_reason: str) -> dict:
    """Generic adapter for simple per-benchmark baseline JSONs.

    These are the minimal one-shot scripts (fever_baseline.py, etc.) that
    score WITNESS directly and write a flat JSON without falsification probes.
    They are 'headline_only' by definition until a per-benchmark falsification
    probe is also run.

    Args:
        raw_and_hash: return value of ``_read_json(filename)`` — a
            (parsed_dict, sha256_hex) tuple.  Either element may be None.
        default_reason: shown in the ``not_run.reason`` field when the JSON
            does not exist yet (e.g. "adapter_pending_wikipedia_fetcher").
    """
    raw, source_hash = raw_and_hash
    if raw is None:
        return {"status": "not_run", "reason": default_reason}
    # Infer a nice filename from the benchmark field if present
    bm_name = str(raw.get("benchmark", "unknown"))
    return {
        "status": "ok",
        "source_file": f"{bm_name.lower()}_baseline.json",
        "source_hash": source_hash,
        "benchmark": bm_name,
        "source_dataset": raw.get("source_dataset"),
        "n_items": raw.get("n_items"),
        "seed": raw.get("seed"),
        "preregistered_target": (
            raw.get("preregistered_target_accuracy")
            or raw.get("preregistered_target")
        ),
        "ms_per_item": raw.get("ms_per_item"),
        "metrics": raw.get("metrics"),
        "labels": raw.get("labels"),
        "notes": raw.get("notes"),
        "falsification_status": "not_yet_run",
    }


# ── Aggregation ────────────────────────────────────────────────────────


def _trust_classify(per_benchmark: dict[str, dict],
                    probes: dict[str, dict]) -> dict:
    """Apply EICV_PREREGISTRATION §4 reporting rules across all benchmarks.

    A benchmark result is `trust_defensible` only if a per-benchmark
    falsification probe has been run AND survives the C1-C4 protocol.
    Otherwise it is `headline_only` or `not_run`.
    """
    out = {"trust_defensible": [], "headline_only": [], "not_run": []}
    for name, payload in per_benchmark.items():
        status = payload.get("status")
        if status == "not_run":
            out["not_run"].append({
                "benchmark": name,
                "reason": payload.get("reason", "unknown"),
            })
            continue
        target_info = TIER1_TARGETS.get(name, {})
        probe = probes.get(name)
        if probe is None or probe.get("status") != "ok":
            out["headline_only"].append({
                "benchmark": name,
                "reason": "no_falsification_probe_for_this_benchmark",
            })
            continue
        # Prefer the AUROC target the probe was registered against (since the
        # falsification protocol measures AUROC across C1-C4). Fall back to
        # TIER1_TARGETS only when the probe didn't record one (legacy
        # fusion4 result).
        target = probe.get("preregistered_target") or target_info.get("target")
        min_auc = probe.get("min_fusion_auroc_c1_c4")
        if not isinstance(target, (int, float)) or not isinstance(min_auc, (int, float)):
            out["headline_only"].append({
                "benchmark": name,
                "reason": "target_or_probe_value_not_numeric",
                "probe_min_c1_c4": min_auc,
            })
            continue
        target_minus_tol = target - 0.03
        survives = min_auc >= target_minus_tol
        bucket = "trust_defensible" if survives else "headline_only"
        entry = {
            "benchmark": name,
            "min_auroc_c1_c4": round(min_auc, 4),
            "target_minus_tolerance": round(target_minus_tol, 4),
            "artifact_drop_c1_c4": probe.get("artifact_drop_c1_c4"),
            "backend": probe.get("backend"),
            "survives": survives,
        }
        if not survives:
            entry["reason"] = "falsification_did_not_survive"
        out[bucket].append(entry)
    return out


def build_baseline() -> dict:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    halueval_10k_raw, halueval_10k_hash = _read_json("halueval_10k_fusion.json")
    truthfulqa_raw,   truthfulqa_hash   = _read_json("truthfulqa_benchmark.json")
    witness_slices_raw, witness_slices_hash = _read_json("witness_benchmarks.json")
    ragas_raw,        ragas_hash        = _read_json("ragas_faithfulness_benchmark.json")
    # Per-benchmark falsification probes (Phase 0 step 1.1)
    fals_qa_raw,      fals_qa_hash      = _read_json("fusion4_falsification.json")
    fals_tqa_raw,     fals_tqa_hash     = _read_json("truthfulqa_falsification.json")
    fals_diag_raw,    fals_diag_hash    = _read_json("halueval_dialogue_falsification.json")
    fals_sum_raw,     fals_sum_hash     = _read_json("halueval_summarization_falsification.json")
    fals_ragas_raw,   fals_ragas_hash   = _read_json("ragas_falsification.json")

    slices_payload = _adapt_witness_slices(witness_slices_raw, witness_slices_hash)

    per_benchmark = {
        "halueval_qa":            _adapt_halueval_qa_10k(halueval_10k_raw, halueval_10k_hash),
        "halueval_dialogue":      {
            "status": "ok_stale_n200",
            "data": slices_payload.get("halueval_dialogue_n200"),
            "stale_warning": slices_payload.get("stale_warning"),
            "source_file": "witness_benchmarks.json",
            "source_hash": witness_slices_hash,
        } if slices_payload.get("status") == "ok" else {"status": "not_run"},
        "halueval_summarization": {
            "status": "ok_stale_n200",
            "data": slices_payload.get("halueval_summarization_n200"),
            "stale_warning": slices_payload.get("stale_warning"),
            "source_file": "witness_benchmarks.json",
            "source_hash": witness_slices_hash,
        } if slices_payload.get("status") == "ok" else {"status": "not_run"},
        "truthfulqa":             _adapt_truthfulqa(truthfulqa_raw, truthfulqa_hash),
        "ragas":                  _adapt_ragas(ragas_raw, ragas_hash),
        "factscore":              _adapt_simple_baseline(_read_json("factscore_baseline.json"),
                                                           default_reason="adapter_pending_wikipedia_fetcher"),
        "fever":                  _adapt_simple_baseline(_read_json("fever_baseline.json"),
                                                           default_reason="fever_baseline_not_yet_run"),
        "true":                   _adapt_simple_baseline(_read_json("true_baseline.json"),
                                                           default_reason="adapter_pending_11_subtasks"),
        "hhem":                   _adapt_simple_baseline(_read_json("hhem_submission.json"),
                                                           default_reason="adapter_pending_vectara_api"),
    }
    # Per-benchmark falsification probes
    probes = {
        "halueval_qa": _adapt_falsification_legacy(
            fals_qa_raw, fals_qa_hash,
            source_file="fusion4_falsification.json",
            applies_to="halueval_qa",
        ),
        "halueval_dialogue": _adapt_falsification_v2(
            fals_diag_raw, fals_diag_hash,
            source_file="halueval_dialogue_falsification.json",
            applies_to="halueval_dialogue",
        ),
        "halueval_summarization": _adapt_falsification_v2(
            fals_sum_raw, fals_sum_hash,
            source_file="halueval_summarization_falsification.json",
            applies_to="halueval_summarization",
        ),
        "truthfulqa": _adapt_falsification_v2(
            fals_tqa_raw, fals_tqa_hash,
            source_file="truthfulqa_falsification.json",
            applies_to="truthfulqa",
        ),
        "ragas": _adapt_falsification_v2(
            fals_ragas_raw, fals_ragas_hash,
            source_file="ragas_falsification.json",
            applies_to="ragas",
        ),
    }
    verdict = _trust_classify(per_benchmark, probes)

    return {
        "schema_version": SCHEMA_VERSION,
        "preregistration_doc": "benchmarks/EICV_PREREGISTRATION.md",
        "preregistration_frozen_commit": PREREG_FROZEN_COMMIT,
        "run": {
            "timestamp_utc":   ts,
            "git_commit":      _git_commit(),
            "entroly_version": _entroly_version(),
            "python_version":  sys.version.split()[0],
            "platform":        platform.platform(),
            "host":            platform.node(),
        },
        "preregistered_targets": {
            "primary":    PRIMARY_METRIC,
            "tier1":      TIER1_TARGETS,
            "robustness": ROBUSTNESS_TARGETS,
        },
        "benchmarks":           per_benchmark,
        "falsification_probes": probes,
        "verdict":              verdict,
        "summary": {
            "trust_defensible_count": len(verdict["trust_defensible"]),
            "headline_only_count":    len(verdict["headline_only"]),
            "not_run_count":          len(verdict["not_run"]),
            "primary_metric_value":   None,
            "primary_metric_target":  PRIMARY_METRIC["target"],
            "primary_metric_met":     None,
            "primary_metric_status":  "deferred_to_phase_4",
        },
    }


def main() -> int:
    out = build_baseline()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)

    print(f"Wrote {OUT_PATH}")
    print()
    print(f"Schema:               {out['schema_version']}")
    print(f"Run commit:           {out['run']['git_commit'][:12]}")
    print(f"Pre-registration:     {out['preregistration_frozen_commit'][:12]}")
    print(f"Entroly version:      {out['run']['entroly_version']}")
    print()
    print("=== Per-benchmark verdict (Phase 0 baseline) ===")
    v = out["verdict"]
    for item in v["trust_defensible"]:
        b = item["benchmark"]
        print(f"  [OK]  {b}: trust-defensible "
              f"(min C1-C4 = {item.get('min_auroc_c1_c4'):.4f}, "
              f"target-tol = {item.get('target_minus_tolerance'):.4f})")
    for item in v["headline_only"]:
        b = item["benchmark"]
        if item.get("min_auroc_c1_c4") is not None:
            drop = item.get("artifact_drop_c1_c4")
            drop_s = f"{drop:.4f}" if isinstance(drop, (int, float)) else "n/a"
            print(f"  [FAIL] {b}: falsification did NOT survive — "
                  f"min C1-C4 = {item['min_auroc_c1_c4']:.4f} < "
                  f"target-tol = {item['target_minus_tolerance']:.4f}, "
                  f"C1->C4 drop = {drop_s}")
        else:
            print(f"  [!]    {b}: headline only ({item.get('reason', '')})")
    for item in v["not_run"]:
        print(f"  [..]   {item['benchmark']}: not run ({item['reason']})")
    print()
    print(f"Primary metric (Risk@Coverage worst-manifold): "
          f"deferred to Phase 4 (EICV components not yet landed)")
    print()
    # Identify which steps remain
    remaining = []
    if not (RESULTS_DIR / "ragas_falsification.json").exists():
        remaining.append("RAGAS per-benchmark falsification probe (1.1)")
    if not (RESULTS_DIR / "factscore_baseline.json").exists():
        remaining.append("FActScore adapter (1.2)")
    if not (RESULTS_DIR / "fever_baseline.json").exists():
        remaining.append("FEVER adapter (1.3)")
    if not (RESULTS_DIR / "true_baseline.json").exists():
        remaining.append("TRUE benchmark adapter (1.4)")
    if not (RESULTS_DIR / "hhem_submission.json").exists():
        remaining.append("HHEM scaffolding (1.5)")
    if remaining:
        print("Next Phase 0 sub-steps:")
        for r in remaining:
            print(f"  - {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
