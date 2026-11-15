"""Held-out real transitions for benchmarks/HANDOFF_EVIDENCE_PREREGISTRATION.md section 7.

Replays the 40 first-parent transitions of pallets/click ending at ``2247b35``.
At each commit ``i`` the predecessor records evidence (``entroly-work evidence
run``) and updates testmon data; the tree moves to ``i+1`` and each policy
decides. Ground truth is the actual rerun at ``i+1``.

Policies: M1 always reruns; M2 is ``entroly-work handoffs`` from ``main``
(``current`` flag); M3 is pytest-testmon 2.3.0 (tests executed); M4 is
``entroly-work evidence check`` from the branch under test.

Usage::

    python benchmarks/handoff_evidence_replay.py --click <clone> --venv <venv with click deps> \
        --entroly-bin <venv>/bin --main-bin <venv-main>/bin --out benchmarks/results/handoff_evidence_replay.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

END = "2247b35"
TRANSITIONS = 40
VERIFY = ["python", "-m", "pytest", "tests/test_options.py", "-q", "-p", "no:cacheprovider"]
AGENT_A = "claude-code:session-a"
AGENT_B = "codex:session-b"


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[subprocess.CompletedProcess, float]:
    started = time.perf_counter()
    done = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800)
    return done, time.perf_counter() - started


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _executed(output: str) -> int:
    return sum(int(n) for n, _ in re.findall(r"(\d+) (passed|failed|error|errors)\b", output))


def _failures(output: str) -> list[str]:
    return sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", output, re.M)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--click", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--entroly-bin", required=True)
    parser.add_argument("--main-bin", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    repo = Path(args.click).resolve()
    venv = Path(args.venv).resolve()
    entroly = Path(args.entroly_bin) / "entroly-work"
    main_tool = Path(args.main_bin) / "entroly-work"
    commits = _git(repo, "rev-list", "--first-parent", "--reverse", f"-{TRANSITIONS + 1}", END).split()
    assert len(commits) == TRANSITIONS + 1
    home = Path(args.state) / "home"
    home.mkdir(parents=True, exist_ok=True)

    def env(state: str) -> dict[str, str]:
        return {"PATH": f"{venv / 'bin'}:/usr/bin:/bin", "HOME": str(home), "LANG": "C.UTF-8",
                "ENTROLY_DIR": str(Path(args.state) / f"state-{state}")}

    def checkout(commit: str) -> None:
        _git(repo, "checkout", "-q", "-f", commit)
        _git(repo, "clean", "-q", "-fdx", "-e", ".testmondata")

    rows: list[dict[str, Any]] = []
    checkout(commits[0])
    _run([*VERIFY, "--testmon"], repo, env("branch"))  # seed testmon at the first commit
    for tool, state in ((entroly, "branch"), (main_tool, "main")):
        _run([str(tool), "--project", str(repo), "--json", "claim", "--agent", AGENT_A,
              "--task", "maintain options"], repo, env(state))
    workstream = json.loads(_run([str(main_tool), "--project", str(repo), "--json", "state"], repo,
                                 env("main"))[0].stdout)["unfinished"][0]["node_id"]
    for i in range(TRANSITIONS):
        before, after = commits[i], commits[i + 1]
        row: dict[str, Any] = {"index": i, "from": before[:10], "to": after[:10]}
        try:
            checkout(before)
            seed, _ = _run([*VERIFY, "--testmon"], repo, env("branch"))
            recorded, _ = _run([str(entroly), "--project", str(repo), "--json", "evidence", "run",
                                "--name", "options", "--", *VERIFY], repo, env("branch"))
            record = json.loads(recorded.stdout)
            row["record_trace_complete"] = record.get("trace", {}).get("complete")
            row["recorded_verdict"] = record.get("verdict")
            truth_before, _ = _run(VERIFY, repo, env("branch"))
            failures_before = _failures(truth_before.stdout)
            handoff = json.loads(_run([str(main_tool), "--project", str(repo), "--json", "handoff",
                                       "--workstream", workstream, "--from-agent", AGENT_A,
                                       "--to-agent", AGENT_B], repo, env("main"))[0].stdout)
            checkout(after)
            row["files_changed"] = len(_git(repo, "diff", "--name-only", before, after).split())
            check, seconds = _run([str(entroly), "--project", str(repo), "--json", "evidence", "check",
                                   "--record", record["record_id"]], repo, env("branch"))
            decision = json.loads(check.stdout)  # exit code 3 means "not reuse", not failure
            row["M4"] = {"decision": "reuse" if decision["decision"] == "reuse" else "revalidate",
                         "applicability": decision["decision"], "changed_count": decision.get("changed_count"),
                         "changed": [f"{c.get('kind')}:{c.get('key')}" for c in decision.get("changed", [])][:20],
                         "reasons": decision.get("reasons"),
                         "seconds": round(seconds, 3)}
            listed = json.loads(_run([str(main_tool), "--project", str(repo), "--json", "handoffs",
                                      "--to-agent", AGENT_B], repo, env("main"))[0].stdout)
            entry = next((h for h in listed.get("handoffs", []) if h["ref"] == handoff.get("handoff_ref")), None)
            row["M2"] = {"decision": "reuse" if entry and entry.get("current") else "revalidate"}
            row["M1"] = {"decision": "revalidate"}
            tm, seconds = _run([*VERIFY, "--testmon"], repo, env("branch"))
            executed = _executed(tm.stdout + tm.stderr)
            row["M3"] = {"decision": "reuse" if executed == 0 else "revalidate", "tests_executed": executed,
                         "seconds": round(seconds, 3)}
            truth, seconds = _run(VERIFY, repo, env("branch"))
            failures_after = _failures(truth.stdout)
            row["rerun"] = {"exit_code": truth.returncode, "tests": _executed(truth.stdout),
                            "failures": failures_after, "seconds": round(seconds, 3)}
            row["outcome_changed"] = failures_after != failures_before or (
                (truth.returncode == 0) != (truth_before.returncode == 0))
            del seed
        except Exception as exc:  # recorded, never dropped
            row["harness_error"] = f"{type(exc).__name__}: {exc}"[:2000]
        rows.append(row)
        print(f"{i + 1}/{TRANSITIONS} {row.get('M4', {}).get('decision')} M3={row.get('M3', {}).get('tests_executed')}",
              flush=True)

    summary: dict[str, Any] = {}
    valid = [r for r in rows if "harness_error" not in r]
    for method in ("M1", "M2", "M3", "M4"):
        reuses = [r for r in valid if r[method]["decision"] == "reuse"]
        summary[method] = {
            "transitions": len(valid),
            "command_reruns": len(valid) - len(reuses),
            "test_executions": sum(r[method]["tests_executed"] if method == "M3"
                                   else (r["rerun"]["tests"] if r[method]["decision"] == "revalidate" else 0)
                                   for r in valid),
            "consequential_stale": [r["index"] for r in reuses if r["outcome_changed"]],
        }
    payload = {
        "benchmark_module": "benchmarks/handoff_evidence_replay.py",
        "preregistration_sha256": hashlib.sha256(
            Path(__file__).with_name("HANDOFF_EVIDENCE_PREREGISTRATION.md").read_bytes()).hexdigest(),
        "repository": "pallets/click", "end_commit": END, "first_commit": commits[0][:10],
        "labels": {"repository": "real, held out (not used in development)", "provider": "not run"},
        "implementation_head": _git(Path(__file__).resolve().parents[1], "rev-parse", "HEAD").strip(),
        "harness_errors": [r["index"] for r in rows if "harness_error" in r],
        "transitions_with_outcome_change": [r["index"] for r in valid if r["outcome_changed"]],
        "summary": summary,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
