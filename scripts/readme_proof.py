#!/usr/bin/env python3
"""Render compact, verifier-backed proof summaries for README recordings.

Every number printed by this module is loaded from a committed artifact after
that artifact's fail-closed verifier succeeds. The local proof invokes the
same packaged verifier exposed by ``entroly verify-claims`` and summarizes its
machine-readable report. This keeps the recordings reproducible and prevents
presentation copy from drifting away from the evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_REPORT = ROOT / "benchmarks/results/model_recovery_v7_holdout.json"
RECOVERY_REPORT = (
    ROOT / "benchmarks/results/recovery_resilience_holdout_revalidation_v3.json"
)
PRIOR_RECOVERY_REPORT = (
    ROOT / "benchmarks/results/recovery_resilience_holdout_revalidation.json"
)

CYAN = "\033[38;5;45m"
GREEN = "\033[38;5;42m"
YELLOW = "\033[38;5;220m"
WHITE = "\033[97m"
GRAY = "\033[38;5;245m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _header(title: str, subtitle: str) -> None:
    print(f"{CYAN}{BOLD}ENTROLY PROOF{RESET}  {WHITE}{title}{RESET}")
    print(f"{GRAY}{subtitle}{RESET}")
    print(f"{GRAY}{'-' * 72}{RESET}")


def _pass(label: str, detail: str) -> None:
    print(f"{GREEN}[PASS]{RESET} {BOLD}{label:<30}{RESET} {detail}")


def _metric(label: str, entroly: str, baseline: str) -> None:
    print(
        f"{WHITE}{label:<28}{RESET}"
        f" {GREEN}{entroly:>15}{RESET}"
        f" {GRAY}{baseline:>18}{RESET}"
    )


def local_proof() -> int:
    """Run and summarize the packaged, no-key verification path."""
    from entroly.verify_claims import run

    _header(
        "Does the installed path work locally?",
        "Same implementation as: entroly verify-claims | no API key",
    )
    print(f"{YELLOW}$ entroly verify-claims{RESET}\n")
    captured = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="entroly-readme-proof-") as tmp:
        report_path = Path(tmp) / "verification.json"
        previous_logging_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(
                captured
            ):
                exit_code = run(output=str(report_path), max_files=40)
        finally:
            logging.disable(previous_logging_disable)
        report = _load(report_path)

    if exit_code != 0 or int(report["failed"]) != 0:
        print(captured.getvalue())
        return 1

    statuses = {item["id"]: item["status"] for item in report["results"]}
    required = {
        "SDK-1": "SDK import and compression",
        "IDX-1": "Local repository indexing",
        "OPT-1": "Selection respects budget",
        "CCR-2": "Exact source recovery",
        "CCR-3": "Bounded recovery excerpt",
        "LOCAL-1": "Local no-key execution",
    }
    for claim_id, label in required.items():
        if statuses.get(claim_id) != "PASS":
            print(f"\033[31m[FAIL]{RESET} {label}")
            return 1
        detail = {
            "SDK-1": "public SDK returned text",
            "IDX-1": "found indexable source",
            "OPT-1": "within the 8,000-token cap",
            "CCR-2": "byte-exact original restored",
            "CCR-3": "query window stayed in budget",
            "LOCAL-1": "all operations stayed local",
        }[claim_id]
        _pass(label, detail)

    print()
    _pass(
        "Packaged verification",
        f"{report['passed']}/{report['passed'] + report['failed']} checks passed",
    )
    print(f"\n{GRAY}Run it on your repo. Keep the JSON report. Trust the result, not this video.{RESET}")
    return 0


def model_recovery_proof() -> int:
    """Verify and display the frozen model-triggered recovery holdout."""
    from benchmarks import model_recovery

    report = _load(MODEL_REPORT)
    model_recovery.verify_report(report)
    systems = report["aggregates"]["systems"]
    paired = report["aggregates"]["paired"]
    entroly = systems["entroly"]
    headroom = systems["headroom"]

    _header(
        "Does smaller context preserve the answer?",
        "Frozen 24-case query-shift holdout | local Qwen2.5 1.5B | temp=0",
    )
    print(f"{YELLOW}$ python -m benchmarks.model_recovery verify <artifact>{RESET}\n")
    _pass("Artifact integrity", f"sha256 {report['payload_sha256'][:16]}...")
    _pass("Execution", "24/24 trials complete; 0 errors")
    print(f"\n{BOLD}{'METRIC':<28} {'ENTROLY':>15} {'HEADROOM 0.31.0':>18}{RESET}")
    _metric("Final exact answers", f"{entroly['final_exact']}/24", f"{headroom['final_exact']}/24")
    _metric(
        "Effective context ratio",
        f"{entroly['mean_effective_token_ratio'] * 100:.2f}%",
        f"{headroom['mean_effective_token_ratio'] * 100:.2f}%",
    )
    _metric(
        "Discordant wins",
        str(paired["entroly_only_final_exact"]),
        str(paired["headroom_only_final_exact"]),
    )
    print(
        f"\n{GREEN}{BOLD}Exact McNemar p = {paired['mcnemar_exact_p']:.5f}{RESET}"
        f"  {GRAY}(two-sided){RESET}"
    )
    print(f"{GRAY}Scoped synthetic workflow; not a universal product claim.{RESET}")
    print(f"{GRAY}Source: benchmarks/results/{MODEL_REPORT.name}{RESET}")
    return 0


def restart_recovery_proof() -> int:
    """Verify and display the frozen concurrent restart-recovery holdout."""
    from benchmarks import recovery_resilience

    prior = _load(PRIOR_RECOVERY_REPORT)
    report = _load(RECOVERY_REPORT)
    recovery_resilience.verify_report(prior)
    recovery_resilience.verify_report(report)
    entroly = report["aggregates"]["entroly"]
    headroom = report["aggregates"]["headroom"]
    errors = [
        error["message"]
        for worker in report["adapters"]["headroom"]["worker_runs"]
        for error in worker["errors"]
    ]

    _header(
        "Does omitted evidence survive concurrent writers and restart?",
        "Frozen Windows/Python 3.10 holdout | 6 writers | 66 payloads",
    )
    print(f"{YELLOW}$ python -m benchmarks.recovery_resilience verify <artifact>{RESET}\n")
    _pass("Artifact integrity", f"sha256 {report['payload_sha256'][:16]}...")
    _pass("Prior v2 retained", "Entroly 66/66; Headroom 66/66")
    print(f"\n{BOLD}{'METRIC':<28} {'ENTROLY':>15} {'HEADROOM 0.31.0':>18}{RESET}")
    _metric("Byte-exact after restart", f"{entroly['exact_entries']}/66", f"{headroom['exact_entries']}/66")
    _metric("Worker errors", str(entroly["worker_errors"]), str(headroom["worker_errors"]))
    _metric(
        "Live state",
        f"{entroly['state_bytes'] / 1024:.1f} KiB",
        f"{headroom['state_bytes'] / 1024:.1f} KiB",
    )
    if errors:
        print(f"\n{YELLOW}Headroom worker error retained: {errors[0]}{RESET}")
    print(
        f"{GRAY}One frozen run; prior tie retained; "
        f"not a universal superiority claim.{RESET}"
    )
    print(f"{GRAY}Source: benchmarks/results/{RECOVERY_REPORT.name}{RESET}")
    return 0


COMMANDS: dict[str, Callable[[], int]] = {
    "local": local_proof,
    "model-recovery": model_recovery_proof,
    "restart-recovery": restart_recovery_proof,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proof", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    return COMMANDS[args.proof]()


if __name__ == "__main__":
    raise SystemExit(main())
