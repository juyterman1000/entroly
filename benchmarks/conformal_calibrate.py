"""Conformal Coverage Compression (C³) — Phase 1: offline calibration.

Computes per-stratum quantile thresholds δ̂(α) such that, for any
exchange-able test point, the probability that accuracy_drop ≤ δ̂(α)
is at least 1-α. This is split-conformal prediction (Vovk, Shafer &
Wasserman 2005) applied to context compression — to our knowledge, the
first such application that produces a per-response coverage certificate.

Per-stratum (Mondrian conformal, Vovk 2003): δ̂ is computed separately
for each content stratum (code, prose, structured QA, etc.) so the
guarantee tightens where compression is easier and loosens where it's
harder. The runtime header lookup then picks the right δ̂ by stratum.

Inputs
------
The runner reads per-sample audit records produced by
`benchmarks/audited_runner.py`. Each `*_audit.jsonl` is one line per
benchmark sample:
    {"index", "question", "expected", "baseline_pred",
     "baseline_correct", "entroly_pred", "entroly_correct",
     "baseline_total_tokens", "entroly_total_tokens", ...}

If no audit files exist, this runner WILL NOT silently fall back to
running benchmarks against an LLM — that costs API budget and time
which should be an explicit user decision. It prints a clear message
asking the user to run the audited runner first.

Outputs
-------
`entroly/data/conformal_calibration.json`:
{
  "spec_version": "c3-mondrian-conformal-v1",
  "calibrated_at": "2026-05-25T...",
  "git_sha": "...",
  "alphas": [0.05, 0.10, 0.20],
  "strata": {
     "<benchmark>": {
        "n": <int>,
        "delta_hat": {"0.05": <pp>, "0.10": <pp>, "0.20": <pp>},
        "non_conformity_scores": [...],   # raw per-sample for audit
        "calibration_source": "benchmarks/results/<benchmark>_audit.jsonl",
        "audit_sha256": "<first 12 chars of file content sha256>"
     }
  },
  "calibration_id": "<sha256 over all audit content, first 12 chars>"
}

Usage
-----
    python benchmarks/conformal_calibrate.py

To regenerate calibration after fresh runs:
    python benchmarks/audited_runner.py       # produces audit JSONLs
    python benchmarks/conformal_calibrate.py  # consumes them

Reference
---------
Vovk, V., Shafer, G., & Wasserman, L. (2005).
*Algorithmic Learning in a Random World*. Springer.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
DATA_DIR = ROOT / "entroly" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = DATA_DIR / "conformal_calibration.json"

ALPHAS = [0.05, 0.10, 0.20]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:12]


def _empirical_quantile(values: list[float], alpha: float) -> float:
    """Split-conformal δ̂: the ⌈(n+1)(1-α)⌉ / n empirical quantile.

    Vovk et al. 2005, Algorithm 2.1. Equivalent to:
        rank = ⌈(n+1)(1-α)⌉
        δ̂   = sorted(values)[rank - 1]
    Clipped to handle small-n edge cases where rank could exceed n.
    """
    n = len(values)
    if n == 0:
        return float("nan")
    rank = math.ceil((n + 1) * (1 - alpha))
    rank = min(rank, n)  # for small n, fall back to max observed
    return sorted(values)[rank - 1]


def _load_audit(path: Path) -> list[dict]:
    """Load per-sample audit records from a *_audit.jsonl file."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip malformed
    return out


def _accuracy_drop(rec: dict) -> float | None:
    """Per-sample non-conformity score = baseline_correct − entroly_correct.

    Both are 0/1, so the drop is in {-1, 0, +1}:
      - +1: baseline correct, entroly wrong   (accuracy hurt by compression)
      -  0: same outcome                       (compression preserved correctness)
      - -1: entroly correct, baseline wrong    (compression *helped*)

    We need per-sample numeric drops. Returns None if either field is
    missing (the sample is excluded from calibration).
    """
    b = rec.get("baseline_correct")
    e = rec.get("entroly_correct")
    if not isinstance(b, bool) or not isinstance(e, bool):
        return None
    return float(int(b) - int(e))


def calibrate_one(audit_path: Path, name: str) -> dict:
    records = _load_audit(audit_path)
    drops = [d for d in (_accuracy_drop(r) for r in records) if d is not None]
    if not drops:
        return {"n": 0, "error": "no valid samples", "source": str(audit_path.name)}

    delta_hat = {}
    for alpha in ALPHAS:
        q = _empirical_quantile(drops, alpha)
        # Express as percentage points (×100) for human-readable headers.
        delta_hat[f"{alpha:.2f}"] = round(q * 100.0, 2)

    return {
        "n": len(drops),
        "delta_hat_pp": delta_hat,
        "non_conformity_scores": drops,  # raw; reviewer can re-derive δ̂
        "n_baseline_correct": sum(1 for r in records if r.get("baseline_correct")),
        "n_entroly_correct": sum(1 for r in records if r.get("entroly_correct")),
        "calibration_source": str(audit_path.name),
        "audit_sha256": _file_sha256(audit_path),
    }


def main() -> int:
    audits = sorted(RESULTS_DIR.glob("*_audit.jsonl"))
    if not audits:
        print(
            "ERROR: no *_audit.jsonl files in benchmarks/results/.\n"
            "Run the audited benchmark runner first:\n"
            "    python benchmarks/audited_runner.py\n"
            "(costs ~$0.50 in OpenAI API, takes ~15 min).",
            file=sys.stderr,
        )
        return 2

    strata = {}
    combined_hash = hashlib.sha256()
    for audit in audits:
        name = audit.stem.replace("_audit", "")
        result = calibrate_one(audit, name)
        strata[name] = result
        combined_hash.update(audit.read_bytes())
        # ASCII-only output so the script runs on cp1252 Windows consoles
        # without UnicodeEncodeError. The JSON file uses correct math notation.
        print(
            f"  {name:12s} n={result['n']:>3}  "
            f"delta_hat(0.05)={result.get('delta_hat_pp', {}).get('0.05', 'n/a')}pp  "
            f"delta_hat(0.10)={result.get('delta_hat_pp', {}).get('0.10', 'n/a')}pp  "
            f"src={result['calibration_source']}",
            flush=True,
        )

    out = {
        "spec_version": "c3-mondrian-conformal-v1",
        "calibrated_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "alphas": ALPHAS,
        "method": (
            "split-conformal Vovk 2005, Mondrian-stratified per benchmark, "
            "non-conformity = baseline_correct - entroly_correct (0/1 per sample)"
        ),
        "n_strata": len(strata),
        "calibration_id": combined_hash.hexdigest()[:12],
        "strata": strata,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote: {OUT_PATH.relative_to(ROOT)}  "
          f"({len(strata)} strata, calibration_id={out['calibration_id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
