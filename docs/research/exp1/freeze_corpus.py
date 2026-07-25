"""Build a versioned, runtime-bound corpus for the Exp 1 selector probe.

The corpus is built from a clean, explicitly named Git checkout using a fresh,
non-persistent Entroly engine. It never reuses a caller's warm index.

Usage:
    python docs/research/exp1/freeze_corpus.py \
        docs/research/exp1/frozen_corpus.json --source .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path

CORPUS_SCHEMA = "entroly.research.frozen-corpus.v2"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git(source: str, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", source, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}"
        )
    return result.stdout.strip()


def _git_optional(source: str, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", source, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _source_identity(source: str) -> dict[str, str]:
    commit = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    if not _COMMIT_RE.fullmatch(commit) or not _COMMIT_RE.fullmatch(tree):
        raise RuntimeError("source checkout did not resolve to full Git object identities")
    dirty = _git(source, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError(
            "source checkout has tracked modifications; commit or stash them before freezing"
        )
    remote = _git_optional(source, "config", "--get", "remote.origin.url")
    return {"source_commit": commit, "source_tree": tree, "source_remote": remote}


def main(out: str, source: str) -> int:
    source = os.path.realpath(source)
    if not os.path.isdir(source):
        raise ValueError(f"source checkout does not exist: {source}")

    from entroly import __version__ as entroly_version
    from entroly.auto_index import auto_index
    from entroly.config import EntrolyConfig
    from entroly.native_status import (
        QCCR_SYMBOLS,
        native_status,
        native_status_message,
    )
    from entroly.server import EntrolyEngine

    status = native_status(QCCR_SYMBOLS)
    if not status.ok or not status.path or not status.version:
        raise RuntimeError(
            native_status_message(status, feature="reproducibility corpus freeze")
        )
    identity = _source_identity(source)
    with tempfile.TemporaryDirectory(prefix="entroly_exp1_") as checkpoint:
        engine = EntrolyEngine(
            config=EntrolyConfig(
                use_persistent_index=False,
                checkpoint_dir=Path(checkpoint),
            )
        )
        result = auto_index(engine, project_dir=source, force=True)
        if result.get("status") != "indexed":
            raise RuntimeError(f"fresh corpus indexing failed: {result}")
        if result.get("skipped_too_large") or result.get("skipped_unreadable"):
            raise RuntimeError(
                "fresh corpus indexing was incomplete: "
                f"{result.get('skipped_too_large', 0)} oversized, "
                f"{result.get('skipped_unreadable', 0)} unreadable"
            )
        if not engine._use_rust or engine._rust is None:
            raise RuntimeError("reproducibility corpus freeze requires the native engine")
        fragments = [
            {
                "source": str(fragment.get("source") or ""),
                "content": str(fragment.get("content") or ""),
                "fragment_id": str(fragment.get("fragment_id") or ""),
                "feedback_multiplier": 1.0,
            }
            for fragment in engine._rust.export_fragments()
        ]

    if not fragments:
        raise RuntimeError("fresh corpus indexing produced no fragments")
    canonical = json.dumps(
        fragments,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact = {
        "schema_version": CORPUS_SCHEMA,
        "metadata": {
            **identity,
            "entroly_version": entroly_version,
            "native_version": status.version,
            "native_module_sha256": _sha256_file(status.path),
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "fragment_count": len(fragments),
            "fragments_sha256": hashlib.sha256(canonical).hexdigest(),
        },
        "fragments": fragments,
    }
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(artifact, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
    print(
        f"froze {len(fragments)} fragments from {identity['source_commit']} -> {output}"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="frozen_corpus.json")
    parser.add_argument("--source", default=os.getcwd())
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.output, arguments.source))
