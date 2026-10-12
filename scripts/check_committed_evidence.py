#!/usr/bin/env python3
"""A file that commits to its own bytes may not be rewritten in passing.

Some artifacts carry a digest of their own contents. Editing one is therefore
not an ordinary edit: the bytes and the commitment have to move together, and
every published citation of that commitment has to move with them.

A repository-wide pass that neutralised named external products rewrote a key
inside ``benchmarks/results/model_recovery_v7_holdout.json`` and re-sorted its
object keys. The embedded ``payload_sha256`` was recomputed, so the artifact
stayed internally consistent and nothing failed. The digest published in
``docs/benchmarks/model-triggered-recovery.md`` was not updated, so for several
releases the hash a reader was told to check matched nothing. Nothing measured
had changed -- the damage was entirely to the integrity chain, which is the one
part a reader cannot verify any other way.

That is why self-consistency is not the check. A rewrite that recomputes the
embedded digest **is** self-consistent; a valid self-hash establishes internal
consistency, not authenticity. The citation is what goes stale, so the citation
is what this verifies.

Three properties, over a discovered set rather than a maintained list:

1. **Self-committing artifacts are discovered by testing the property.** Any
   top-level ``*_sha256`` field whose value equals the canonical digest of the
   rest of the file is a self-commitment. Nothing is hardcoded, so a new
   artifact or a new field name is covered the day it lands.
2. **Sidecars match.** A ``X.sha256`` beside ``X`` must be ``X``'s digest.
3. **Published citations match.** Where a document cites an artifact and quotes
   a digest under that artifact's own field name, the live claim must be the
   artifact's current digest. Later mentions may record history -- superseded
   digests are provenance, and deleting them would destroy the audit trail --
   so only the first claim is read as current.

Legitimate changes to committed evidence already have a route:
``scripts/verify_evidence_provenance.py`` seals an artifact by re-running its
benchmark, never by hand. This script does not replace that; it stops an
unrelated pass from silently invalidating a published commitment.

Usage:
    python scripts/check_committed_evidence.py
    python scripts/check_committed_evidence.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent

# A digest claim written against a named field, e.g. ``payload_sha256 = ab12...``.
# Backticks, bold markers and colons are all in use across the docs.
_CLAIM = re.compile(
    r"(?P<field>[a-z0-9_]*sha256)\s*[:=]\s*[`*\"']{0,2}(?P<digest>[0-9a-f]{64})",
    re.IGNORECASE,
)
_DOC_SUFFIXES = {".md", ".markdown"}


def canonical_digest(payload: dict, exclude: str) -> str:
    """Digest of ``payload`` with ``exclude`` removed, canonically serialised."""
    unsigned = {key: value for key, value in payload.items() if key != exclude}
    canonical = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a sidecar's artifact using the sealing convention that wrote it.

    The convention normalises line endings, so on a CRLF checkout a raw-byte
    hash disagrees with every sidecar in the tree -- an earlier draft of this
    script reported all seven as corrupt. It is imported rather than restated
    so there is one definition; a second copy could drift and manufacture
    exactly the false corruption reports this script exists to distinguish from
    real ones. A failed import is fatal for the same reason: hashing by some
    other rule would be worse than not checking.
    """
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    from verify_evidence_provenance import portable_text_sha256

    return portable_text_sha256(path)


@dataclass(frozen=True)
class Committed:
    """An artifact that commits to its own bytes."""

    path: Path
    field: str
    digest: str

    @property
    def relative(self) -> str:
        return self.path.relative_to(ROOT).as_posix()


def _tracked(pattern: str) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", pattern],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return sorted(ROOT.glob(pattern))
    return [ROOT / line for line in out.splitlines() if line.strip()]


def discover_committed() -> list[Committed]:
    """Find artifacts that commit to their own bytes, by testing the property.

    A field is a self-commitment only if removing it and hashing the remainder
    reproduces its value. Fields like ``harness_sha256`` or ``source_sha256``
    digest something else and are correctly ignored.
    """
    found: list[Committed] = []
    for path in _tracked("*.json"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for field, value in payload.items():
            if not (
                isinstance(field, str)
                and field.lower().endswith("sha256")
                and isinstance(value, str)
                and len(value) == 64
            ):
                continue
            if canonical_digest(payload, field) == value.lower():
                found.append(Committed(path, field, value.lower()))
    return found


def check_sidecars() -> list[str]:
    problems: list[str] = []
    for sidecar in _tracked("*.sha256"):
        artifact = sidecar.with_suffix("")
        if not artifact.is_file():
            problems.append(f"{sidecar.relative_to(ROOT).as_posix()}: no artifact beside it")
            continue
        recorded = sidecar.read_text(encoding="ascii", errors="replace").split()
        actual = file_digest(artifact)
        if not recorded or recorded[0].lower() != actual:
            problems.append(
                f"{artifact.relative_to(ROOT).as_posix()}: sidecar records "
                f"{(recorded[0] if recorded else '<empty>')[:16]}..., file hashes {actual[:16]}..."
            )
    return problems


def check_citations(committed: list[Committed]) -> list[str]:
    """A published digest must name the artifact that is on disk."""
    docs = [
        path
        for path in _tracked("*")
        if path.suffix.lower() in _DOC_SUFFIXES and path.is_file()
    ]
    problems: list[str] = []
    for artifact in committed:
        name = artifact.path.name
        for doc in docs:
            try:
                text = doc.read_text(encoding="utf-8")
            except OSError:
                continue
            if name not in text:
                continue
            claims = [
                match.group("digest").lower()
                for match in _CLAIM.finditer(text)
                if match.group("field").lower() == artifact.field.lower()
            ]
            if not claims or claims[0] == artifact.digest:
                continue
            problems.append(
                f"{doc.relative_to(ROOT).as_posix()} publishes "
                f"{artifact.field} = {claims[0][:16]}... for {artifact.relative}, "
                f"which hashes to {artifact.digest[:16]}..."
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="print the discovered artifacts and exit"
    )
    args = parser.parse_args()

    committed = discover_committed()
    if args.list:
        for artifact in sorted(committed, key=lambda item: item.relative):
            print(f"{artifact.relative}  [{artifact.field}]  {artifact.digest[:16]}...")
        for sidecar in sorted(_tracked("*.sha256")):
            print(f"{sidecar.relative_to(ROOT).as_posix()}  [sidecar]")
        return 0

    if not committed:
        print(
            "committed-evidence check: found no self-committing artifact. "
            "Discovery is broken, or every artifact lost its commitment.",
            file=sys.stderr,
        )
        return 1

    problems = check_sidecars() + check_citations(committed)
    if problems:
        print("committed-evidence check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nAn artifact that carries its own digest cannot be edited in passing. "
            "Move the bytes, the commitment and every published citation together, "
            "and record why in the page that cites it. To re-seal an artifact "
            "honestly, re-run its benchmark "
            "(scripts/verify_evidence_provenance.py) -- never write a digest by hand.",
            file=sys.stderr,
        )
        return 1

    print(
        f"committed-evidence check passed "
        f"({len(committed)} self-committing artifact(s), "
        f"{len(_tracked('*.sha256'))} sidecar(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
