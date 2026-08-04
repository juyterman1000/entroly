#!/usr/bin/env python3
"""Every benchmark artifact cited as public evidence must be traceable.

`scripts/verify_readme_claims.py` enforces full provenance, but only for
artifacts cited within the first ~130 lines of README.md. Most of the project's
evidence is cited below that line or in docs/, and was therefore unchecked: of
22 artifacts cited as public evidence, 2 carried a harness hash and a checksum
sidecar and 20 carried neither.

Unsealed does not mean wrong. It means unverifiable -- nothing ties the numbers
to the code that produced them, and an edited JSON would not be detected. That
is the gap a fabricated harness would exploit: emit a plausible artifact, cite
it below the first screen, and no check objects.

This script closes the gap as a **ratchet** rather than a cliff. Sealing an
existing artifact honestly requires re-running its benchmark, which is not
always possible (some need provider credentials or a model). Writing a sidecar
for an artifact whose provenance nobody verified would assert exactly the kind
of unearned confidence this repository exists to avoid. So:

  * artifacts already cited without provenance are recorded in
    docs/evidence-provenance-debt.json and reported, not failed;
  * any **newly** cited artifact must be sealed, or the check fails;
  * an entry that is no longer cited, or that has since been sealed, must be
    removed from the debt file, so the list can only shrink.

Usage:
    python scripts/verify_evidence_provenance.py
    python scripts/verify_evidence_provenance.py --update-debt   # after sealing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEBT_FILE = REPO_ROOT / "docs" / "evidence-provenance-debt.json"

ARTIFACT_RE = re.compile(r"(benchmarks/results/[A-Za-z0-9_\-./]+\.json)")

# Documents whose citations count as public evidence claims.
EVIDENCE_DOCS = (
    "README.md",
    "PYPI_README.md",
    "docs/BENCHMARKS.md",
    "docs/public-evidence.md",
    "docs/limitations.md",
    "docs/for-teams.md",
)


def portable_text_sha256(path: Path) -> str:
    """Hash with normalised line endings, matching verify_readme_claims.py."""
    canonical = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def find_citations() -> dict[str, list[str]]:
    """Map each cited artifact to the places claiming it."""
    cited: dict[str, list[str]] = {}
    for doc in EVIDENCE_DOCS:
        path = REPO_ROOT / doc
        if not path.is_file():
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in ARTIFACT_RE.finditer(line):
                cited.setdefault(match.group(1), []).append(f"{doc}:{number}")
    return cited


def seal_status(relative: str) -> tuple[bool, list[str]]:
    """Report whether an artifact is fully traceable, and what is missing."""
    artifact = REPO_ROOT / relative
    missing: list[str] = []

    if not artifact.is_file():
        return False, ["artifact does not exist"]

    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, [f"not valid JSON: {exc}"]

    if not isinstance(payload, dict):
        return False, ["not a JSON object"]

    module = payload.get("benchmark_module")
    if not module:
        missing.append("benchmark_module")
    elif not (REPO_ROOT / module).is_file():
        missing.append(f"benchmark_module points at missing {module}")

    claimed = payload.get("harness_sha256")
    if not claimed:
        missing.append("harness_sha256")
    elif module and (REPO_ROOT / module).is_file():
        actual = portable_text_sha256(REPO_ROOT / module)
        if actual != claimed:
            missing.append("harness_sha256 does not match benchmark_module")

    sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
    if not sidecar.is_file():
        missing.append(".sha256 sidecar")
    else:
        recorded = sidecar.read_text(encoding="ascii", errors="replace").split()
        canonical = artifact.read_bytes().replace(b"\r\n", b"\n")
        if not recorded or recorded[0] != hashlib.sha256(canonical).hexdigest():
            missing.append(".sha256 sidecar does not match the artifact")

    limitations = payload.get("limitations")
    if not (isinstance(limitations, list) and limitations):
        missing.append("limitations")

    return (not missing), missing


def load_debt() -> set[str]:
    if not DEBT_FILE.is_file():
        return set()
    payload = json.loads(DEBT_FILE.read_text(encoding="utf-8"))
    return set(payload.get("unsealed", []))


def write_debt(unsealed: set[str]) -> None:
    DEBT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEBT_FILE.write_text(
        json.dumps(
            {
                "purpose": (
                    "Benchmark artifacts cited as public evidence that carry "
                    "no verifiable link to the code that produced them. "
                    "Unsealed means unverifiable, not known-wrong. This list "
                    "may only shrink: newly cited artifacts must be sealed. "
                    "Seal an entry by re-running its benchmark so the artifact "
                    "records benchmark_module, harness_sha256 and limitations, "
                    "and by writing its .sha256 sidecar -- never by hand."
                ),
                "enforced_by": "scripts/verify_evidence_provenance.py",
                "unsealed": sorted(unsealed),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-debt",
        action="store_true",
        help="Rewrite the debt file from the current state (only ever to shrink it).",
    )
    args = parser.parse_args()

    cited = find_citations()
    sealed: list[str] = []
    unsealed: dict[str, list[str]] = {}

    for relative in sorted(cited):
        ok, missing = seal_status(relative)
        if ok:
            sealed.append(relative)
        else:
            unsealed[relative] = missing

    known_debt = load_debt()
    current_unsealed = set(unsealed)

    if args.update_debt:
        bootstrapping = not DEBT_FILE.is_file()
        # Once the ledger exists it may only shrink; otherwise "update" would
        # become a way to launder a new unsealed artifact into acceptance.
        if not bootstrapping and (current_unsealed - known_debt):
            print("refusing to grow the debt file; seal these instead:")
            for item in sorted(current_unsealed - known_debt):
                print(f"  {item}")
            return 1
        write_debt(current_unsealed)
        verb = "created" if bootstrapping else "updated"
        print(f"debt file {verb}: {len(current_unsealed)} unsealed artifact(s)")
        return 0

    failures: list[str] = []

    # A newly cited artifact must arrive sealed.
    for relative in sorted(current_unsealed - known_debt):
        where = ", ".join(cited[relative][:3])
        failures.append(
            f"{relative} is cited ({where}) without provenance and is not "
            f"recorded as existing debt. Missing: {', '.join(unsealed[relative])}"
        )

    # Debt that no longer applies must be removed, so the list stays honest.
    for relative in sorted(known_debt - current_unsealed):
        if relative in cited:
            failures.append(
                f"{relative} is now sealed; remove it from {DEBT_FILE.name} "
                "(run --update-debt)"
            )
        else:
            failures.append(
                f"{relative} is no longer cited anywhere; remove it from "
                f"{DEBT_FILE.name} (run --update-debt)"
            )

    print(f"cited as public evidence : {len(cited)}")
    print(f"fully traceable          : {len(sealed)}")
    print(f"unsealed (recorded debt) : {len(current_unsealed)}")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL {failure}")
        return 1

    print("\nno newly cited artifact lacks provenance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
