#!/usr/bin/env python3
"""Measure whether Context Receipt fragments are byte-faithful to their source.

Primary metric
--------------
``byte_span_exact`` — the fraction of ingested fragments for which

    source_bytes[byte_start:byte_end] == fragment_text.encode("utf-8")

This is the property the product advertises as "exact recovery": a receipt
promises an omitted fragment can be returned as the original bytes. If a
fragment's recorded byte range does not slice its own text back out of the
untouched source, that promise cannot be kept for that fragment.

Secondary metric
----------------
``verbatim`` — the fraction of fragments whose text appears anywhere in the
source document. A fragment failing this has been *altered*, not merely
mis-addressed: the agent is shown text that does not exist in the repository.
A fragment trimmed only at its edges would still be a substring, so a failed
verbatim check specifically indicates injected or altered interior bytes.

Both are reported per language, because failure is not uniformly distributed —
comment syntax and indentation drive it.

Corpus
------
The corpus is read from a **pinned git ref** (``BASELINE_REF``), not from the
working tree. This matters for three reasons:

* **Immutability.** The measurement describes entroly at one commit. Adding
  documentation, tests, or the artifacts this benchmark itself writes cannot
  move a published number, because none of them exist at the baseline.
* **Platform independence.** Git blobs are LF-normalized, so a Windows checkout
  with ``core.autocrlf=true`` and a Linux checkout yield identical bytes and
  therefore identical results. Reading the working tree would not.
* **Auditability.** Anyone can reproduce the exact input with
  ``git cat-file blob <ref>:<path>``.

Inclusion (deterministic, no sampling): tracked at ``BASELINE_REF``, sorted;
suffix in ``LANGUAGES``; decodes as strict UTF-8; non-empty after strip; at most
``MAX_BYTES`` bytes. Every excluded file is recorded with its reason, so the
denominator is auditable.

Usage
-----
    python -m benchmarks.receipt_fragment_fidelity run \
        --out benchmarks/results/receipt_fragment_fidelity_prefix.json
    python -m benchmarks.receipt_fragment_fidelity verify \
        benchmarks/results/receipt_fragment_fidelity_prefix.json
    python -m benchmarks.receipt_fragment_fidelity sdk-probe \
        --out benchmarks/results/receipt_fragment_fidelity_sdk_prefix.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path

from entroly.context_receipts import ingest_documents as ingest_via_default
from entroly.context_receipts.ingest import ingest_documents as ingest_via_python

SCHEMA_VERSION = "receipt-fragment-fidelity.v3"

# `entroly.context_receipts.ingest_documents` dispatches to the Rust core when it
# is available (prefer_rust defaults to True), so importing the pure-Python
# module directly measures the *fallback*, not what a user with a native wheel
# actually runs. Both are measured; a claim may only cite the default path.
BACKENDS = ("default", "python")
MAX_BYTES = 400_000

# entroly 1.0.69 — the shipped state this defect was found in, and the last
# commit before any of this investigation's own files existed. Pinned so the
# published numbers can never drift as the repository grows.
BASELINE_REF = "1ecf1e093348068539f9e1463826209c966ed535"

LANGUAGES: dict[str, str] = {
    ".c": "C", ".cpp": "C++", ".cs": "C#", ".css": "CSS", ".go": "Go",
    ".h": "C", ".html": "HTML", ".java": "Java", ".js": "JavaScript",
    ".json": "JSON", ".jsx": "JavaScript", ".kt": "Kotlin", ".md": "Markdown",
    ".mjs": "JavaScript", ".ps1": "PowerShell", ".py": "Python",
    ".rb": "Ruby", ".rs": "Rust", ".rst": "reStructuredText", ".sh": "Shell",
    ".sql": "SQL", ".swift": "Swift", ".toml": "TOML", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".xml": "XML", ".yaml": "YAML", ".yml": "YAML",
}

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── reading the pinned baseline ──────────────────────────────────────────────


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def baseline_paths(ref: str) -> list[str]:
    return sorted(_git("ls-tree", "-r", "--name-only", ref).splitlines())


def read_baseline_blobs(ref: str, paths: list[str]) -> dict[str, bytes]:
    """Read many blobs through a single `git cat-file --batch` process.

    Spawning one subprocess per file costs minutes on Windows for a corpus this
    size; batching keeps a full run to seconds.

    The request is written from a separate thread. A full corpus request is
    ~81 KB, which exceeds the typical 64 KB pipe buffer: writing it inline would
    block once the buffer filled, while git simultaneously blocked writing blob
    contents to a stdout nobody was draining — a deadlock that hangs the run
    rather than failing it.
    """
    if not paths:
        return {}

    proc = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdin and proc.stdout

    def _feed() -> None:
        try:
            proc.stdin.write("".join(f"{ref}:{path}\n" for path in paths).encode("utf-8"))
            proc.stdin.flush()
        finally:
            proc.stdin.close()

    writer = threading.Thread(target=_feed, daemon=True)
    writer.start()

    blobs: dict[str, bytes] = {}
    for path in paths:
        header = proc.stdout.readline().decode("utf-8", errors="replace").strip()
        if not header or header.endswith(("missing", "ambiguous")):
            continue
        try:
            size = int(header.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        # read() on a pipe can return short; loop until the object is complete.
        buffer = bytearray()
        while len(buffer) < size:
            block = proc.stdout.read(size - len(buffer))
            if not block:
                break
            buffer.extend(block)
        proc.stdout.read(1)  # trailing newline git appends after each object
        blobs[path] = bytes(buffer)

    writer.join(timeout=30)
    proc.stdout.close()
    proc.wait()
    return blobs


# ── corpus ───────────────────────────────────────────────────────────────────


def build_corpus(ref: str = BASELINE_REF) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Return ``(included, excluded)`` records under the declared rules."""
    candidates = [p for p in baseline_paths(ref) if Path(p).suffix.lower() in LANGUAGES]
    blobs = read_baseline_blobs(ref, candidates)

    included: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []

    for rel in candidates:
        raw = blobs.get(rel)
        if raw is None:
            excluded.append({"path": rel, "reason": "unreadable_at_baseline_ref"})
            continue
        if len(raw) > MAX_BYTES:
            excluded.append({"path": rel, "reason": f"larger_than_{MAX_BYTES}_bytes"})
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            excluded.append({"path": rel, "reason": "not_strict_utf8"})
            continue
        if not text.strip():
            excluded.append({"path": rel, "reason": "empty_after_strip"})
            continue
        included.append({
            "path": rel,
            "language": LANGUAGES[Path(rel).suffix.lower()],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        })

    return included, excluded


# ── measurement ──────────────────────────────────────────────────────────────


def _chunks_for(rel: str, text: str, backend: str) -> list[dict[str, object]]:
    """Return chunk dicts from one backend, normalising the two return shapes."""
    if backend == "python":
        index = ingest_via_python([(rel, text)])
        return [
            {"text": c.text, "byte_start": c.byte_start, "byte_end": c.byte_end}
            for c in index.chunks
        ]

    index = ingest_via_default([(rel, text)])
    chunks = index["chunks"] if isinstance(index, dict) else index.chunks
    normalised = []
    for c in chunks:
        if isinstance(c, dict):
            normalised.append(
                {"text": c["text"], "byte_start": c["byte_start"], "byte_end": c["byte_end"]}
            )
        else:
            normalised.append(
                {"text": c.text, "byte_start": c.byte_start, "byte_end": c.byte_end}
            )
    return normalised


def measure_file(rel: str, raw: bytes, text: str, backend: str = "default") -> dict[str, int]:
    """Count fragments, verbatim hits and byte-exact hits for one file."""
    fragments = verbatim = byte_exact = 0
    try:
        chunks = _chunks_for(rel, text, backend)
    except BaseException:  # noqa: BLE001 - a native panic must not abort the corpus
        # A backend that crashes on real input is a fidelity failure, not an
        # excluded file. Recorded as a file that produced no usable fragments.
        return {"fragments": 0, "verbatim": 0, "byte_span_exact": 0, "backend_error": 1}

    for chunk in chunks:
        fragments += 1
        if chunk["text"] in text:
            verbatim += 1
        if raw[chunk["byte_start"]:chunk["byte_end"]] == chunk["text"].encode("utf-8"):
            byte_exact += 1

    return {
        "fragments": fragments,
        "verbatim": verbatim,
        "byte_span_exact": byte_exact,
        "backend_error": 0,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def run_measurement(ref: str = BASELINE_REF, backend: str = "default") -> dict[str, object]:
    included, excluded = build_corpus(ref)
    blobs = read_baseline_blobs(ref, [str(r["path"]) for r in included])

    per_language: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "fragments": 0, "verbatim": 0, "byte_span_exact": 0, "backend_error": 0}
    )
    per_file: list[dict[str, object]] = []

    for record in included:
        rel = str(record["path"])
        raw = blobs[rel]
        counts = measure_file(rel, raw, raw.decode("utf-8"), backend)

        bucket = per_language[str(record["language"])]
        bucket["files"] += 1
        for key in ("fragments", "verbatim", "byte_span_exact", "backend_error"):
            bucket[key] += counts[key]
        per_file.append({**record, **counts})

    totals = {"files": len(included), "fragments": 0, "verbatim": 0, "byte_span_exact": 0, "backend_error": 0}
    for bucket in per_language.values():
        for key in ("fragments", "verbatim", "byte_span_exact", "backend_error"):
            totals[key] += bucket[key]

    languages = {
        name: {
            **counts,
            "verbatim_rate": _rate(counts["verbatim"], counts["fragments"]),
            "byte_span_exact_rate": _rate(counts["byte_span_exact"], counts["fragments"]),
        }
        for name, counts in sorted(per_language.items())
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_ref": ref,
        "backend": backend,
        "environment": {
            # The corpus comes from the pinned ref, so the checkout state cannot
            # affect the result; recorded only to describe where it was run.
            "entroly_version": __import__("entroly").__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "corpus_rules": {
            "source": f"git ls-tree -r {ref[:12]}, sorted",
            "languages": sorted(set(LANGUAGES.values())),
            "max_bytes": MAX_BYTES,
            "encoding": "strict utf-8",
            "newlines": "as stored in git (LF), identical on every platform",
            "sampling": "none — every file matching the rules is measured",
        },
        "totals": {
            **totals,
            "verbatim_rate": _rate(totals["verbatim"], totals["fragments"]),
            "byte_span_exact_rate": _rate(totals["byte_span_exact"], totals["fragments"]),
        },
        "languages": languages,
        "excluded": excluded,
        "files": per_file,
    }


# ── verification ─────────────────────────────────────────────────────────────


def verify(artifact_path: Path) -> int:
    """Reproduce a committed artifact from the baseline ref it recorded.

    Pinned to the artifact's own file manifest and its own ref, so the result is
    independent of the current checkout, the working tree, and the platform.
    """
    stored = json.loads(artifact_path.read_text(encoding="utf-8"))
    if stored.get("schema_version") != SCHEMA_VERSION:
        print(f"FAIL: schema {stored.get('schema_version')} != {SCHEMA_VERSION}")
        return 1

    ref = stored.get("baseline_ref", "")
    backend = stored.get("backend", "default")
    failures: list[str] = []
    recomputed = {"files": 0, "fragments": 0, "verbatim": 0, "byte_span_exact": 0}

    paths = [str(f["path"]) for f in stored["files"]]
    blobs = read_baseline_blobs(ref, paths)

    for record in stored["files"]:
        rel = str(record["path"])
        raw = blobs.get(rel)
        if raw is None:
            failures.append(f"missing at {ref[:12]}: {rel}")
            continue
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            failures.append(f"content changed at {ref[:12]}: {rel}")
            continue

        counts = measure_file(rel, raw, raw.decode("utf-8"), backend)
        recomputed["files"] += 1
        for key in ("fragments", "verbatim", "byte_span_exact"):
            recomputed[key] += counts[key]
            if counts[key] != record[key]:
                failures.append(f"{rel}: {key} recomputed {counts[key]} != stored {record[key]}")

    for key in ("files", "fragments", "verbatim", "byte_span_exact"):
        if recomputed[key] != stored["totals"][key]:
            failures.append(
                f"totals.{key}: recomputed {recomputed[key]} != stored {stored['totals'][key]}"
            )

    if failures:
        print(f"VERIFY FAILED ({len(failures)} problems)")
        for line in failures[:20]:
            print(f"  - {line}")
        return 1

    totals = stored["totals"]
    print("VERIFY OK")
    print(f"  baseline  : {ref}")
    print(f"  backend   : {backend}")
    print(f"  files     : {totals['files']}")
    print(f"  fragments : {totals['fragments']}")
    print(f"  verbatim  : {totals['verbatim']}/{totals['fragments']} ({totals['verbatim_rate']:.1%})")
    print(
        f"  byte-exact: {totals['byte_span_exact']}/{totals['fragments']} "
        f"({totals['byte_span_exact_rate']:.1%})"
    )
    return 0


# ── public-path probe ────────────────────────────────────────────────────────

# Fixed inputs so the probe is a stable, citable measurement rather than an
# ad-hoc terminal run. Both files are indentation-heavy Python with comments,
# read from the pinned baseline for the same reproducibility reasons.
PROBE_FILES = ["entroly/esg.py", "entroly/evidence_locked_compression.py"]
PROBE_QUERY = "how is evidence coverage enforced fail-closed"
PROBE_BUDGET = 900


def sdk_probe(ref: str = BASELINE_REF) -> dict[str, object]:
    """Exercise recovery through the public SDK exactly as a user would.

    Checks three independent properties per recovered fragment:
      * the returned text exists verbatim in the original file
      * the receipt fingerprint equals sha256 of the returned text
      * the receipt fingerprint equals sha256 of the original byte span

    A receipt is only auditable if a third party can recompute at least one of
    the last two from public data.
    """
    from entroly import sdk

    blobs = read_baseline_blobs(ref, PROBE_FILES)
    sources = {rel: blobs[rel].decode("utf-8") for rel in PROBE_FILES}

    receipt = sdk.create_context_receipt(
        list(sources.items()), query=PROBE_QUERY, budget=PROBE_BUDGET, recoverable=True
    )

    results = []
    for entry in receipt["omitted_context"]:
        chunk_id = entry["chunk_id"]
        fingerprint = entry["fingerprint"]
        for recovered in sdk.recover_receipt_omission(receipt, chunk_id):
            if recovered["chunk_id"] != chunk_id:
                continue
            text = recovered["text"]
            source = sources.get(recovered["source_path"], "")
            digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

            offset = source.find(text[:60])
            span = source[offset: offset + len(text)] if offset >= 0 else ""
            span_digest = (
                "sha256:" + hashlib.sha256(span.encode("utf-8")).hexdigest() if span else ""
            )

            results.append({
                "chunk_id": chunk_id,
                "verbatim_in_source": text in source,
                "fingerprint_matches_recovered": digest == fingerprint,
                "fingerprint_matches_original_span": span_digest == fingerprint,
            })

    total = len(results)
    return {
        "schema_version": "receipt-sdk-probe.v2",
        "baseline_ref": ref,
        "environment": {
            "entroly_version": __import__("entroly").__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "inputs": {
            "files": PROBE_FILES,
            "query": PROBE_QUERY,
            "budget": PROBE_BUDGET,
            "file_sha256": {
                rel: hashlib.sha256(blobs[rel]).hexdigest() for rel in PROBE_FILES
            },
        },
        "totals": {
            "recovered_fragments": total,
            "verbatim_in_source": sum(r["verbatim_in_source"] for r in results),
            "fingerprint_matches_recovered": sum(
                r["fingerprint_matches_recovered"] for r in results
            ),
            "fingerprint_matches_original_span": sum(
                r["fingerprint_matches_original_span"] for r in results
            ),
        },
        "fragments": results,
    }


# ── reporting ────────────────────────────────────────────────────────────────


def render(report: dict[str, object]) -> str:
    lines = []
    totals = report["totals"]  # type: ignore[index]
    env = report["environment"]  # type: ignore[index]
    lines.append(
        f"baseline {str(report['baseline_ref'])[:12]}  backend={report.get('backend', '?')}  "
        f"entroly {env['entroly_version']}  python {env['python']}"
    )
    lines.append("")
    header = (
        f"{'language':16s} {'files':>6s} {'frags':>7s} {'verbatim':>9s} "
        f"{'rate':>8s} {'byteexact':>10s} {'rate':>8s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, counts in sorted(
        report["languages"].items(), key=lambda kv: kv[1]["byte_span_exact_rate"]  # type: ignore[index]
    ):
        lines.append(
            f"{name:16s} {counts['files']:6d} {counts['fragments']:7d} "
            f"{counts['verbatim']:9d} {counts['verbatim_rate']:8.1%} "
            f"{counts['byte_span_exact']:10d} {counts['byte_span_exact_rate']:8.1%}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':16s} {totals['files']:6d} {totals['fragments']:7d} "
        f"{totals['verbatim']:9d} {totals['verbatim_rate']:8.1%} "
        f"{totals['byte_span_exact']:10d} {totals['byte_span_exact_rate']:8.1%}"
    )
    lines.append("")
    lines.append(f"excluded files: {len(report['excluded'])}")  # type: ignore[arg-type]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run", help="measure the baseline corpus and write the artifact")
    run_cmd.add_argument("--out", type=Path, required=True)
    run_cmd.add_argument("--ref", default=BASELINE_REF)
    run_cmd.add_argument("--backend", choices=BACKENDS, default="default")
    verify_cmd = sub.add_parser("verify", help="re-check a committed corpus artifact")
    verify_cmd.add_argument("artifact", type=Path)
    probe_cmd = sub.add_parser(
        "sdk-probe", help="measure recovery through the public SDK end to end"
    )
    probe_cmd.add_argument("--out", type=Path, required=True)
    probe_cmd.add_argument("--ref", default=BASELINE_REF)
    args = parser.parse_args()

    if args.command == "verify":
        return verify(args.artifact)

    if args.command == "sdk-probe":
        probe = sdk_probe(args.ref)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(probe, indent=1, sort_keys=True), encoding="utf-8")
        totals = probe["totals"]  # type: ignore[index]
        n = totals["recovered_fragments"]
        print(f"public SDK recovery probe @ baseline {str(probe['baseline_ref'])[:12]}")
        print(f"  recovered fragments                : {n}")
        print(f"  verbatim in source                 : {totals['verbatim_in_source']}/{n}")
        print(f"  fingerprint == sha256(recovered)   : {totals['fingerprint_matches_recovered']}/{n}")
        print(f"  fingerprint == sha256(original)    : {totals['fingerprint_matches_original_span']}/{n}")
        print(f"\nwrote {args.out}")
        return 0

    report = run_measurement(args.ref, args.backend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(render(report))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
