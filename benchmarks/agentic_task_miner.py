"""
Agentic Task Miner — Experiment 0 task source
=============================================

Preregistration: benchmarks/AGENTIC_TASKS_PREREGISTRATION.md §2.

Mines real repository histories for agent evaluation tasks. A task is a
commit where (a) source and test files changed together, (b) the touched
tests pass at the commit, and (c) the same tests fail with an assertion-style
test failure when the source change is reverted to the parent while tests stay
at the fix state. Collection failures, import failures, timeouts, missing tests,
and runner failures are rejected: they do not prove a behavioral fail-to-pass
oracle.

The passing and reverted states run in separate detached worktrees with pytest
cache disabled. This prevents the passing run from leaving generated files or
cache state that changes the reverted result.

Two phases, priced separately:

  discover  git-log scan for candidate commits (cheap, no checkouts)
  validate  two isolated worktrees per candidate (bounded by --max-validate
            and per-run timeout)

Usage:
  python benchmarks/agentic_task_miner.py discover --repo . --out candidates.jsonl
  python benchmarks/agentic_task_miner.py validate --repo . \
      --candidates candidates.jsonl --out tasks.jsonl --max-validate 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
        capture_output=True,
        text=True,
        check=check,
    )


def _is_test_file(path: str) -> bool:
    name = Path(path).name.lower()
    parts = {p.lower() for p in Path(path).parts}
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
        or "tests" in parts
        or "test" in parts
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


@dataclass(frozen=True)
class TestRun:
    passed: bool
    seconds: float
    returncode: int | None
    outcome: str
    output: str
    failure_signature: str | None = None


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
    fail_outcome: str
    fail_returncode: int
    failure_signature: str
    status: str = "validated"
    notes: list[str] = field(default_factory=list)


def discover(
    repo: Path,
    *,
    max_commits: int = 500,
    max_files_per_commit: int = 20,
) -> list[Candidate]:
    """Scan history for commits touching source AND Python test files."""
    log = _git(
        repo,
        "log",
        f"-{max_commits}",
        "--no-merges",
        "--pretty=format:%H%x1f%cI%x1f%s",
        "--name-only",
    ).stdout
    candidates: list[Candidate] = []
    for block in log.split("\n\n"):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        if not lines or "\x1f" not in lines[0]:
            continue
        sha, date, subject = lines[0].split("\x1f", 2)
        files = lines[1:]
        if len(files) > max_files_per_commit:
            continue
        sources = [path for path in files if _is_source_file(path)]
        tests = [
            path for path in files if _is_test_file(path) and path.endswith(".py")
        ]
        if sources and tests:
            candidates.append(
                Candidate(
                    sha=sha,
                    date=date,
                    subject=subject,
                    source_files=sources,
                    test_files=tests,
                )
            )
    return candidates


def _normalized_failure_signature(output: str) -> str:
    """Hash a stable-enough failure summary without paths or timings."""
    normalized = re.sub(r"/tmp/[^\s:]+", "<tmp>", output)
    normalized = re.sub(r"[A-Za-z]:\\[^\s:]+", "<tmp>", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?s\b", "<time>", normalized)
    normalized = "\n".join(
        line.strip() for line in normalized.splitlines() if line.strip()
    )[-4000:]
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def _classify_test_run(returncode: int, output: str) -> str:
    """Classify pytest outcomes so infrastructure errors are never oracles."""
    if returncode == 0:
        return "passed"
    if returncode == 5:
        return "no_tests_collected"

    lowered = output.lower()
    if "error during collection" in lowered or "syntaxerror" in lowered:
        return "collection_error"
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        return "import_error"
    if "internalerror" in lowered:
        return "pytest_internal_error"
    if "usageerror" in lowered or "unrecognized arguments" in lowered:
        return "runner_error"
    if "failed" in lowered and (
        "assertionerror" in lowered
        or "assert " in lowered
        or " short test summary info " in f" {lowered} "
    ):
        return "test_failure"
    return "unknown_failure"


def _run_tests(workdir: Path, test_files: list[str], timeout: int) -> TestRun:
    """Run touched tests with plugin autoload and pytest cache disabled."""
    existing = [path for path in test_files if (workdir / path).exists()]
    if not existing:
        return TestRun(False, 0.0, None, "missing_test_files", "")

    command = [
        sys.executable,
        "-m",
        "pytest",
        *existing,
        "-x",
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(
            part
            for part in (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout,
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr,
            )
            if part
        )
        return TestRun(
            False,
            time.perf_counter() - started,
            None,
            "timeout",
            output,
        )

    output = completed.stdout + "\n" + completed.stderr
    outcome = _classify_test_run(completed.returncode, output)
    signature = (
        _normalized_failure_signature(output)
        if outcome == "test_failure"
        else None
    )
    return TestRun(
        passed=completed.returncode == 0,
        seconds=time.perf_counter() - started,
        returncode=completed.returncode,
        outcome=outcome,
        output=output,
        failure_signature=signature,
    )


def _remove_worktree(repo: Path, worktree: Path) -> None:
    _git(repo, "worktree", "remove", "--force", str(worktree), check=False)
    shutil.rmtree(worktree, ignore_errors=True)


def validate(
    repo: Path,
    candidate: Candidate,
    *,
    timeout: int = DEFAULT_TEST_TIMEOUT,
) -> ValidatedTask | None:
    """Prove a behavioral fail-to-pass task in two isolated worktrees."""
    fix_worktree = Path(tempfile.mkdtemp(prefix="entroly-task-fix-"))
    revert_worktree = Path(tempfile.mkdtemp(prefix="entroly-task-revert-"))
    try:
        add_fix = _git(
            repo,
            "worktree",
            "add",
            "--detach",
            str(fix_worktree),
            candidate.sha,
            check=False,
        )
        if add_fix.returncode != 0:
            return None
        fixed = _run_tests(fix_worktree, candidate.test_files, timeout)
        if not fixed.passed:
            return None

        # Remove the passing worktree before constructing the reverted state.
        # This prevents pytest caches and generated files from crossing arms.
        _remove_worktree(repo, fix_worktree)

        add_revert = _git(
            repo,
            "worktree",
            "add",
            "--detach",
            str(revert_worktree),
            candidate.sha,
            check=False,
        )
        if add_revert.returncode != 0:
            return None
        revert = _git(
            revert_worktree,
            "checkout",
            f"{candidate.sha}^",
            "--",
            *candidate.source_files,
            check=False,
        )
        if revert.returncode != 0:
            return None

        failed = _run_tests(revert_worktree, candidate.test_files, timeout)
        if failed.outcome != "test_failure":
            return None
        if failed.returncode is None or failed.failure_signature is None:
            return None

        return ValidatedTask(
            sha=candidate.sha,
            date=candidate.date,
            subject=candidate.subject,
            source_files=candidate.source_files,
            test_files=candidate.test_files,
            test_command=[
                sys.executable,
                "-m",
                "pytest",
                *candidate.test_files,
                "-x",
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
            ],
            pass_at_fix_s=round(fixed.seconds, 2),
            fail_at_revert_s=round(failed.seconds, 2),
            fail_outcome=failed.outcome,
            fail_returncode=failed.returncode,
            failure_signature=failed.failure_signature,
            notes=[
                "pass and reverted states executed in separate worktrees",
                "pytest plugin autoload and cache disabled",
            ],
        )
    finally:
        _remove_worktree(repo, fix_worktree)
        _remove_worktree(repo, revert_worktree)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--repo", required=True)
    discover_parser.add_argument("--out", required=True)
    discover_parser.add_argument("--max-commits", type=int, default=500)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--repo", required=True)
    validate_parser.add_argument("--candidates", required=True)
    validate_parser.add_argument("--out", required=True)
    validate_parser.add_argument("--max-validate", type=int, default=20)
    validate_parser.add_argument("--timeout", type=int, default=DEFAULT_TEST_TIMEOUT)

    args = parser.parse_args()
    repo = Path(args.repo).resolve()

    if args.cmd == "discover":
        found = discover(repo, max_commits=args.max_commits)
        Path(args.out).write_text(
            "".join(candidate.to_json() + "\n" for candidate in found),
            encoding="utf-8",
        )
        print(f"discovered {len(found)} candidates -> {args.out}")
        return 0

    candidates = [
        Candidate(**json.loads(line))
        for line in Path(args.candidates).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validated: list[ValidatedTask] = []
    for candidate in candidates[: args.max_validate]:
        print(
            f"validating {candidate.sha[:12]} {candidate.subject[:60]!r} ...",
            flush=True,
        )
        task = validate(repo, candidate, timeout=args.timeout)
        if task is not None:
            validated.append(task)
            print(
                f"  VALIDATED (pass {task.pass_at_fix_s}s / "
                f"fail {task.fail_at_revert_s}s / {task.fail_outcome})"
            )
    Path(args.out).write_text(
        "".join(json.dumps(asdict(task), sort_keys=True) + "\n" for task in validated),
        encoding="utf-8",
    )
    print(
        f"validated {len(validated)}/{min(len(candidates), args.max_validate)} "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
