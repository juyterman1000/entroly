"""
Agentic Task Miner — Experiment 0 task source
=============================================

Preregistration: benchmarks/AGENTIC_TASKS_PREREGISTRATION.md §2.

Mines real repository histories for agent evaluation tasks. A task is a
commit where (a) source and test files changed together, (b) the touched
tests pass at the commit, and (c) the same tests fail when the source
change is reverted to the parent (tests kept at the fix state — the
SWE-bench fail-to-pass construction). The failing test is the oracle:
deterministic ground truth, no LLM judging.

Two phases, priced separately:

  discover  git-log scan for candidate commits (cheap, no checkouts)
  validate  per-candidate git worktree: run tests at fix, revert source
            only, rerun (bounded by --max-validate and per-run timeout)

Usage:
  python benchmarks/agentic_task_miner.py discover --repo . --out candidates.jsonl
  python benchmarks/agentic_task_miner.py validate --repo . \
      --candidates candidates.jsonl --out tasks.jsonl --max-validate 20
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

SOURCE_SUFFIXES = (".py", ".rs", ".ts", ".tsx", ".js", ".go", ".java")
DEFAULT_TEST_TIMEOUT = 300


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=check,
    )


def _is_test_file(path: str) -> bool:
    name = Path(path).name.lower()
    parts = {p.lower() for p in Path(path).parts}
    return (
        name.startswith("test_") or name.endswith("_test.py")
        or ".test." in name or ".spec." in name
        or "tests" in parts or "test" in parts
    )


def _is_source_file(path: str) -> bool:
    return path.endswith(SOURCE_SUFFIXES) and not _is_test_file(path)


@dataclass
class Candidate:
    sha: str
    date: str
    subject: str
    source_files: list[str]
    test_files: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class ValidatedTask:
    sha: str
    date: str
    subject: str
    source_files: list[str]
    test_files: list[str]
    test_command: list[str]
    pass_at_fix_s: float
    fail_at_revert_s: float
    status: str = "validated"
    notes: list[str] = field(default_factory=list)


def discover(repo: Path, *, max_commits: int = 500,
             max_files_per_commit: int = 20) -> list[Candidate]:
    """Scan history for commits touching source AND test files together."""
    log = _git(
        repo, "log", f"-{max_commits}", "--no-merges",
        "--pretty=format:%H%x1f%cI%x1f%s", "--name-only",
    ).stdout
    candidates: list[Candidate] = []
    for block in log.split("\n\n"):
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if not lines or "\x1f" not in lines[0]:
            continue
        sha, date, subject = lines[0].split("\x1f", 2)
        files = lines[1:]
        if len(files) > max_files_per_commit:
            continue  # bulk refactors make ambiguous tasks
        sources = [f for f in files if _is_source_file(f)]
        tests = [f for f in files if _is_test_file(f) and f.endswith(".py")]
        if sources and tests:
            candidates.append(Candidate(
                sha=sha, date=date, subject=subject,
                source_files=sources, test_files=tests,
            ))
    return candidates


def _run_tests(workdir: Path, test_files: list[str],
               timeout: int) -> tuple[bool, float]:
    """Run the touched tests; returns (passed, seconds)."""
    existing = [f for f in test_files if (workdir / f).exists()]
    if not existing:
        return False, 0.0
    cmd = [sys.executable, "-m", "pytest", *existing, "-x", "-q",
           "--timeout", str(min(timeout, 120))]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - t0
    return proc.returncode == 0, time.perf_counter() - t0


def validate(repo: Path, candidate: Candidate, *,
             timeout: int = DEFAULT_TEST_TIMEOUT) -> ValidatedTask | None:
    """Prove the candidate is a real task via the fail-to-pass construction.

    In an isolated worktree at the fix commit: the touched tests must pass;
    after reverting ONLY the source files to the parent (tests stay at the
    fix state), the same tests must fail.
    """
    worktree = Path(tempfile.mkdtemp(prefix="entroly-task-"))
    try:
        _git(repo, "worktree", "add", "--detach", str(worktree), candidate.sha)
        passed, t_pass = _run_tests(worktree, candidate.test_files, timeout)
        if not passed:
            return None  # tests don't pass even at the fix — not a usable oracle

        revert = _git(worktree, "checkout", f"{candidate.sha}^", "--",
                      *candidate.source_files, check=False)
        if revert.returncode != 0:
            return None  # source files new in this commit — nothing to revert to

        still_passes, t_fail = _run_tests(worktree, candidate.test_files, timeout)
        if still_passes:
            return None  # tests don't depend on the source change — no oracle

        return ValidatedTask(
            sha=candidate.sha, date=candidate.date, subject=candidate.subject,
            source_files=candidate.source_files, test_files=candidate.test_files,
            test_command=[sys.executable, "-m", "pytest", *candidate.test_files, "-x", "-q"],
            pass_at_fix_s=round(t_pass, 2), fail_at_revert_s=round(t_fail, 2),
        )
    finally:
        _git(repo, "worktree", "remove", "--force", str(worktree), check=False)
        shutil.rmtree(worktree, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover")
    d.add_argument("--repo", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--max-commits", type=int, default=500)

    v = sub.add_parser("validate")
    v.add_argument("--repo", required=True)
    v.add_argument("--candidates", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--max-validate", type=int, default=20)
    v.add_argument("--timeout", type=int, default=DEFAULT_TEST_TIMEOUT)

    args = parser.parse_args()
    repo = Path(args.repo).resolve()

    if args.cmd == "discover":
        found = discover(repo, max_commits=args.max_commits)
        Path(args.out).write_text(
            "".join(c.to_json() + "\n" for c in found), encoding="utf-8"
        )
        print(f"discovered {len(found)} candidates -> {args.out}")
        return 0

    candidates = [
        Candidate(**json.loads(line))
        for line in Path(args.candidates).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validated: list[ValidatedTask] = []
    for cand in candidates[: args.max_validate]:
        print(f"validating {cand.sha[:12]} {cand.subject[:60]!r} ...", flush=True)
        task = validate(repo, cand, timeout=args.timeout)
        if task is not None:
            validated.append(task)
            print(f"  VALIDATED (pass {task.pass_at_fix_s}s / fail {task.fail_at_revert_s}s)")
    Path(args.out).write_text(
        "".join(json.dumps(asdict(t), sort_keys=True) + "\n" for t in validated),
        encoding="utf-8",
    )
    print(f"validated {len(validated)}/{min(len(candidates), args.max_validate)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
