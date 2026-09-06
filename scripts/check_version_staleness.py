"""Fail if anything still *declares* or *pins* a version older than the master.

Every release, some version surface gets left behind. `bump_version.py` rewrites
an explicit list of files, so a surface added later is silently skipped and keeps
the previous release's number while everything around it moves. Nothing notices,
because a stale version is valid JSON, valid TOML and a valid string literal.

Real misses this exists to prevent, all found by hand:

* `.claude-plugin/plugin.json` sat one release behind the `manifest.json` beside
  it, which *was* in the list.
* Six Codex/Gemini agent bundles and extension manifests were never in the list.
* Three functional-test suites printed a banner reading ``Entroly v0.2.0`` --
  eighty releases stale, because a literal inside ``print()`` has nothing
  checking it.
* ``BENCHMARKS.md`` declared ``entroly-core v0.9.0``, written in the v1.0
  founding commit and never touched again.
* The same "on the v0.19.x roadmap" sentence sat in three packaging READMEs;
  fixing two by hand missed the third.
* Four workflow-dispatch descriptions offered ``for example 1.0.25`` to whoever
  was triggering a manual publish.

## The rule

A version string must equal the master if it **declares or pins the product as
it is now**. It may be older if it **records something that happened** -- a
release note, a measurement, a test fixture, a past-tense sentence.

That distinction is the whole design. Sweeping for "any old version number"
produces 559 hits, of which ~550 are correct history: release notes for 1.0.47
should say 1.0.47 forever, a benchmark that ran on 1.0.59 must keep saying so,
and a lock file records `serde 1.0.4` because that is the version of serde.
Rewriting those would destroy provenance and break the build. So archives are
identified by path, and everything outside them is checked syntactically.

Usage:
    python scripts/check_version_staleness.py          # non-zero if stale
    python scripts/check_version_staleness.py --list   # show every classified hit
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MASTER = re.search(
    r'__version__\s*=\s*"([^"]+)"',
    (ROOT / "entroly" / "__init__.py").read_text(encoding="utf-8"),
).group(1)
MASTER_TUPLE = tuple(int(p) for p in MASTER.split("."))

# ── Archives: paths whose whole purpose is to record the past ────────────────
#
# A file here may name any version. `docs/releases/v1.0.47.md` is *about* 1.0.47;
# `benchmarks/results/` records which engine produced a measurement; a test
# fixture is the input proving the bump script rewrites 1.0.39 to 1.0.40.
ARCHIVE_PREFIXES = (
    "docs/releases/",            # release notes are permanently about their release
    "docs/research/",
    "docs/investigations/",      # dated investigation records
    "docs/reviews/",
    "benchmarks/results/",       # measurement outputs
    "tests/",                    # fixtures and narrative docstrings
    "entroly-core/tests/",
    ".entroly/",                 # engine's own runtime state
)
ARCHIVE_SUFFIXES = (
    "Cargo.lock",                # nested versions belong to third-party crates
    "package-lock.json",
    "CHANGELOG.md",
)
ARCHIVE_SUBSTRINGS = (
    "node_modules",
    "/protocol",                 # benchmarks/*_protocol*.json pin a measured run
    "_protocol",
    # Documents whose filename marks them as a dated record of one PR or
    # experiment. `PR352_CODEBASE_UNDERSTANDING_EVIDENCE.md` quotes the
    # dependency pins as they stood at review time, to explain a release-
    # ordering hazard that existed then -- rewriting those to today's version
    # would destroy the finding the document exists to record.
    "_EVIDENCE",
    "_PREREGISTRATION",
)

# ── Pinned artifacts: version tracks a *published* file, not the product ─────
#
# These carry a URL and a verified checksum for an artifact that exists. Moving
# them before that release ships points at a 404 with a hash matching nothing.
PINNED_ARTIFACTS = {
    "packaging/scoop/entroly.json",
    "packaging/homebrew/entroly.rb",
}

BINARY_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".ico", ".woff", ".woff2",
    ".zip", ".mcpb", ".svg", ".pdf", ".whl", ".gz",
)

# ── What counts as declaring or pinning ──────────────────────────────────────
DECLARATION_PATTERNS = (
    # A manifest declaring its own version.
    (re.compile(r'^\s*"version"\s*:\s*"(\d+\.\d+\.\d+)"', re.M), "manifest version field"),
    (re.compile(r'^\s*version\s*[:=]\s*"?(\d+\.\d+\.\d+)"?\s*$', re.M), "manifest version field"),
    # An install pin for one of our own distributions.
    (re.compile(r'entroly(?:-core|-mcp|-wasm|-openclaw)?\s*[><~=]=\s*["\']?(\d+\.\d+\.\d+)'),
     "install pin"),
    (re.compile(r'entroly(?:-mcp|-wasm)?@(\d+\.\d+\.\d+)'), "npm pin"),
    # A container tag.
    (re.compile(r'ghcr\.io/[^\s:]+:(\d+\.\d+\.\d+)'), "container tag"),
    # A placeholder shown to someone filling in a form.
    (re.compile(r'(?:for example|Example:)\s+v?(\d+\.\d+\.\d+)', re.I), "user-facing example"),
    # A hardcoded product banner.
    (re.compile(r'Entroly\s+v(\d+\.\d+\.\d+)'), "product banner"),
    # A bare "name version" statement with no operator and no JSON field --
    # the shape of `Engine version: \`entroly-core v0.9.0\``, which was written
    # in the v1.0 founding commit and never moved. Restricted to a labelled
    # declaration so ordinary prose ("shipped in Entroly 1.0.57") is not caught;
    # that phrasing is past tense and belongs to the historical markers below.
    (re.compile(
        r'(?:version|Version)\s*[:=]\s*`?entroly(?:-core|-engine|-mcp|-wasm|-openclaw)?\s+v?(\d+\.\d+\.\d+)'
    ), "labelled engine version"),
)

# Past-tense context: the line records history even outside an archive path.
HISTORICAL_MARKERS = re.compile(
    r"\b(previously|used to|was written|shipped in|as of|until this release|"
    r"historically|no longer|had outlived|left behind|never moved|founding commit|"
    r"predates?|earlier version|before this|regression|caught|measured on)\b",
    re.I,
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=str(ROOT),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    if result.returncode != 0:
        raise SystemExit(f"git ls-files failed: {result.stderr.strip()[:200]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _is_archive(rel: str) -> bool:
    return (
        rel.startswith(ARCHIVE_PREFIXES)
        or rel.endswith(ARCHIVE_SUFFIXES)
        or any(s in rel for s in ARCHIVE_SUBSTRINGS)
    )


def scan() -> tuple[list[str], list[str]]:
    """Return (stale, allowed) — stale entries are declarations below master."""
    stale: list[str] = []
    allowed: list[str] = []

    for rel in _tracked_files():
        if rel.endswith(BINARY_SUFFIXES):
            continue
        path = ROOT / rel
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()

        for pattern, kind in DECLARATION_PATTERNS:
            for match in pattern.finditer(text):
                version = match.group(1)
                try:
                    parsed = tuple(int(p) for p in version.split("."))
                except ValueError:
                    continue
                if parsed >= MASTER_TUPLE:
                    continue
                index = text[: match.start()].count("\n")
                line = lines[index].strip() if index < len(lines) else ""
                entry = f"{rel}:{index + 1}  [{kind}] {version}  {line[:88]}"

                if _is_archive(rel):
                    allowed.append(f"archive        {entry}")
                elif rel in PINNED_ARTIFACTS:
                    allowed.append(f"pinned artifact{entry}")
                elif HISTORICAL_MARKERS.search(line):
                    allowed.append(f"past tense     {entry}")
                else:
                    stale.append(entry)
    return stale, allowed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="also print everything classified as legitimately historical")
    args = parser.parse_args()

    stale, allowed = scan()

    if args.list:
        print(f"Allowed (history, fixtures, pinned artifacts): {len(allowed)}")
        for entry in allowed:
            print(f"  {entry}")
        print()

    if stale:
        print(f"{len(stale)} version declaration(s) older than master {MASTER}:\n")
        for entry in stale:
            print(f"  {entry}")
        print(
            "\nEach of these declares or pins the product as it is now, so it must "
            f"read {MASTER}.\n"
            "If one genuinely records the past, phrase it in past tense or move it "
            "under an archive path; do not add it to an ignore list."
        )
        return 1

    print(f"No version declaration is older than master {MASTER}.")
    print(f"({len(allowed)} historical references checked and allowed.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
