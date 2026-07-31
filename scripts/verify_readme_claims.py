#!/usr/bin/env python3
"""Fail when the README's first screen leans on evidence that cannot carry it.

The repository already computes ``headline_eligible`` for its benchmarks and
already pins two pilots to stay false. What it never checked is the inverse, and
more consequential, direction: whether a benchmark that failed its own
statistical gate is being used as a headline anyway.

Two gates, both mechanical:

1. **Ineligible evidence above the fold.** An artifact carrying
   ``headline_eligible: false`` may be cited in the README, but not inside the
   first screen unless an explicit experimental label sits next to it. A reader
   who stops at the banner must not leave with a number the benchmark itself
   refuses to certify.

2. **Unreachable evidence.** A benchmark whose measured module cannot be reached
   from any shipped entry point does not describe the installed product. Citing
   it in the first screen advertises code the user cannot run.

Neither gate reads prose or judges wording; both resolve to a JSON field or a
graph traversal, so they cannot drift with edits to marketing copy.

Usage::

    python scripts/verify_readme_claims.py
    python scripts/verify_readme_claims.py --readme README.md --first-screen 130
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Lines of README that constitute "the first screen" — the region a reader sees
# before scrolling, which the project's own guidance says must answer what the
# product does, why it differs, and what verified result it achieves.
DEFAULT_FIRST_SCREEN_LINES = 130

ARTIFACT_RE = re.compile(r"(benchmarks/results/[A-Za-z0-9_\-./]+\.json)")

# Wording that adequately marks a claim as not-yet-certified. Matched
# case-insensitively on the lines surrounding a citation.
EXPERIMENTAL_MARKERS = (
    "experimental",
    "not statistically conclusive",
    "not headline",
    "pilot",
    "preliminary",
    "research preview",
)
MARKER_WINDOW = 12  # lines either side of the citation to search for a label


def load_graph_module():
    spec = importlib.util.spec_from_file_location(
        "codebase_graph", REPO_ROOT / "scripts" / "codebase_graph.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unreachable_modules() -> set[str]:
    graph = load_graph_module()
    return set(graph.analyse(REPO_ROOT / "entroly")["unreachable"])


def benchmark_module_for(
    artifact: Path, payload: dict[str, object] | None = None
) -> Path | None:
    """Map results/<name>.json back to benchmarks/<name>.py."""
    declared = (payload or {}).get("benchmark_module")
    if isinstance(declared, str) and declared:
        candidate = REPO_ROOT / declared
        return candidate if candidate.is_file() else None
    candidate = REPO_ROOT / "benchmarks" / f"{artifact.stem}.py"
    return candidate if candidate.exists() else None


def modules_measured_by(benchmark: Path) -> set[str]:
    """Entroly modules a benchmark imports directly."""
    source = benchmark.read_text(encoding="utf-8", errors="replace")
    found = set()
    for match in re.finditer(r"^\s*from\s+(entroly[A-Za-z0-9_.]*)\s+import", source, re.M):
        found.add(match.group(1))
    for match in re.finditer(r"^\s*import\s+(entroly[A-Za-z0-9_.]*)", source, re.M):
        found.add(match.group(1))
    return found


def citations(readme_lines: list[str], limit: int) -> list[tuple[int, str]]:
    """Artifact paths cited within the first ``limit`` lines."""
    hits = []
    for number, line in enumerate(readme_lines[:limit], start=1):
        for match in ARTIFACT_RE.finditer(line):
            hits.append((number, match.group(1)))
    return hits


def has_nearby_label(readme_lines: list[str], line_number: int) -> bool:
    start = max(0, line_number - 1 - MARKER_WINDOW)
    end = min(len(readme_lines), line_number + MARKER_WINDOW)
    window = " ".join(readme_lines[start:end]).lower()
    return any(marker in window for marker in EXPERIMENTAL_MARKERS)


def checksum_matches(artifact: Path) -> bool:
    checksum_path = artifact.with_suffix(artifact.suffix + ".sha256")
    if not checksum_path.is_file():
        return False
    fields = checksum_path.read_text(encoding="ascii", errors="replace").split()
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        return False
    canonical = artifact.read_bytes().replace(b"\r\n", b"\n")
    return fields[0] == hashlib.sha256(canonical).hexdigest()


def evidence_metadata_failures(
    relative: str, artifact: Path, payload: dict[str, object]
) -> list[str]:
    """Validate the minimum provenance needed for a prominent public result."""
    failures: list[str] = []
    if not isinstance(payload.get("headline_eligible"), bool):
        failures.append("missing boolean headline_eligible")
    if not isinstance(payload.get("claim_scope"), str) or not payload["claim_scope"].strip():
        failures.append("missing claim_scope")

    sample_size = payload.get("sample_size")
    if not isinstance(sample_size, dict) or not any(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in sample_size.values()
    ):
        failures.append("missing positive sample_size denominator")

    command = payload.get("reproduction_command")
    if not isinstance(command, str) or relative not in command:
        failures.append("reproduction_command does not name the cited artifact")

    implementation = payload.get("implementation")
    commit = implementation.get("commit") if isinstance(implementation, dict) else None
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        failures.append("implementation.commit is not a pinned 40-character SHA")

    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or not limitations or not all(
        isinstance(item, str) and item.strip() for item in limitations
    ):
        failures.append("missing non-empty limitations")

    benchmark_module = payload.get("benchmark_module")
    if not isinstance(benchmark_module, str) or not (
        REPO_ROOT / benchmark_module
    ).is_file():
        failures.append("benchmark_module is missing or does not exist")
    else:
        expected_harness = payload.get("harness_sha256")
        actual_harness = hashlib.sha256(
            (REPO_ROOT / benchmark_module).read_bytes()
        ).hexdigest()
        if expected_harness != actual_harness:
            failures.append("harness_sha256 does not match benchmark_module")

    if not checksum_matches(artifact):
        failures.append("missing or mismatched .sha256 sidecar")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument("--first-screen", type=int, default=DEFAULT_FIRST_SCREEN_LINES)
    args = parser.parse_args()

    readme_lines = args.readme.read_text(encoding="utf-8").splitlines()
    cited = citations(readme_lines, args.first_screen)
    if not cited:
        print(f"no benchmark artifacts cited in the first {args.first_screen} README lines")
        return 0

    failures: list[str] = []
    unreachable = unreachable_modules()

    for line_number, relative in cited:
        artifact = REPO_ROOT / relative
        if not artifact.exists():
            failures.append(f"{args.readme.name}:{line_number} cites missing artifact {relative}")
            continue

        try:
            payload = json.loads(
                artifact.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(f"{relative} is not valid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"{relative} must contain a JSON object")
            continue

        for problem in evidence_metadata_failures(relative, artifact, payload):
            failures.append(
                f"{args.readme.name}:{line_number} cites {relative}: {problem}."
            )

        # Gate 1 — the artifact's own statistical verdict.
        if payload.get("headline_eligible") is False and not has_nearby_label(
            readme_lines, line_number
        ):
            reason = payload.get("calibration", {}).get("reason", "gate not met")
            failures.append(
                f"{args.readme.name}:{line_number} presents {relative} in the first screen, "
                f"but that artifact sets headline_eligible=false ({reason}). "
                f"Move it below the fold or label it experimental."
            )

        # Gate 2 — does the cited benchmark describe shipped code?
        benchmark = benchmark_module_for(artifact, payload)
        if benchmark is None:
            continue
        stranded = sorted(modules_measured_by(benchmark) & unreachable)
        if stranded:
            failures.append(
                f"{args.readme.name}:{line_number} cites {relative}, whose benchmark "
                f"measures {', '.join(stranded)} — unreachable from every shipped entry "
                f"point, so the result does not describe the installed product."
            )

    if failures:
        print(f"README CLAIM GATE FAILED ({len(failures)} problems)\n")
        for failure in failures:
            print(f"  - {failure}\n")
        return 1

    print(f"README claim gate OK — {len(cited)} first-screen citation(s) verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
