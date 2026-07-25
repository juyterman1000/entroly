"""Experiment 1: strict ordered-selection reproducibility matrix.

Every condition runs in a fresh subprocess. The result digest includes source,
content bytes, origin-fragment identities, and rank. Failed or malformed
conditions make the experiment invalid and return non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE = os.path.join(HERE, "capture_selection.py")
CORPUS = os.path.join(HERE, "frozen_corpus.json")
BUDGET = "2000"
QUERY = "where does the proxy inject compressed context into requests"
CORPUS_SCHEMA = "entroly.research.frozen-corpus.v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _result_digest(order: list[dict]) -> str:
    blob = json.dumps(
        [
            (
                item["rank"],
                item["source"],
                item["content_sha"],
                item["content_len"],
                item["source_fragment_ids"],
            )
            for item in order
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _validate_capture(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("capture returned a non-object result")
    order = payload.get("order")
    if not isinstance(order, list) or payload.get("n") != len(order):
        raise RuntimeError("capture returned an invalid ordered-selection contract")
    for rank, item in enumerate(order):
        if (
            not isinstance(item, dict)
            or item.get("rank") != rank
            or not isinstance(item.get("source"), str)
            or not _SHA256_RE.fullmatch(str(item.get("content_sha") or ""))
            or not isinstance(item.get("content_len"), int)
            or isinstance(item.get("content_len"), bool)
            or item["content_len"] < 0
            or not isinstance(item.get("source_fragment_ids"), list)
            or any(
                not isinstance(origin_id, str) or not origin_id
                for origin_id in item["source_fragment_ids"]
            )
        ):
            raise RuntimeError(f"capture returned an invalid selection item at rank {rank}")
    expected = _result_digest(order)
    if payload.get("digest") != expected:
        raise RuntimeError("capture digest does not bind its ordered selection")
    return payload


def run_capture(corpus_path: str, env_extra: dict[str, str]) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    env.setdefault("ENTROLY_SOURCE", os.getcwd())
    try:
        result = subprocess.run(
            [sys.executable, "-u", CAPTURE, corpus_path, BUDGET, QUERY],
            capture_output=True,
            text=True,
            env=env,
            cwd=os.getcwd(),
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("capture timed out after 300 seconds") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"capture exited {result.returncode}: "
            f"{(result.stderr or result.stdout)[-500:]}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"capture emitted no JSON result: {result.stderr[-500:]}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"capture emitted invalid JSON: {exc}") from exc
    return _validate_capture(payload)


def selection_keys(result: dict) -> list[tuple]:
    """Stable multiset identities; duplicate fragments remain distinguishable."""
    occurrences: Counter[tuple] = Counter()
    keys: list[tuple] = []
    for item in result["order"]:
        identity = (
            item["source"],
            item["content_sha"],
            tuple(item["source_fragment_ids"]),
        )
        occurrence = occurrences[identity]
        occurrences[identity] += 1
        keys.append((*identity, occurrence))
    return keys


def jaccard(a: list[tuple], b: list[tuple]) -> float:
    left, right = set(a), set(b)
    return len(left & right) / len(left | right) if (left | right) else 1.0


def kendall_tau(a: list[tuple], b: list[tuple]) -> float | None:
    """Kendall tau over unique selected-fragment identities common to both."""
    right = set(b)
    common = [item for item in a if item in right]
    if len(common) < 2:
        return None
    rank_a = {item: index for index, item in enumerate(a)}
    rank_b = {item: index for index, item in enumerate(b)}
    concordant = discordant = 0
    for left_index in range(len(common)):
        for right_index in range(left_index + 1, len(common)):
            left, right_item = common[left_index], common[right_index]
            sign = (rank_a[left] - rank_a[right_item]) * (
                rank_b[left] - rank_b[right_item]
            )
            concordant += sign > 0
            discordant += sign < 0
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def _load_corpus_artifact() -> dict:
    with open(CORPUS, encoding="utf-8") as corpus_file:
        artifact = json.load(corpus_file)
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != CORPUS_SCHEMA
        or not isinstance(artifact.get("metadata"), dict)
        or not isinstance(artifact.get("fragments"), list)
    ):
        raise ValueError(f"frozen corpus does not use {CORPUS_SCHEMA}")
    return artifact


def _write_variant(artifact: dict, path: str) -> str:
    fragments = artifact["fragments"]
    canonical = json.dumps(
        fragments,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    metadata = artifact["metadata"]
    metadata["fragment_count"] = len(fragments)
    metadata["fragments_sha256"] = hashlib.sha256(canonical).hexdigest()
    with open(path, "w", encoding="utf-8", newline="\n") as variant_file:
        json.dump(artifact, variant_file, indent=2, ensure_ascii=False)
        variant_file.write("\n")
    return path


def permuted_corpus(seed: int, directory: str) -> str:
    artifact = _load_corpus_artifact()
    random.Random(seed).shuffle(artifact["fragments"])
    return _write_variant(
        artifact,
        os.path.join(directory, f"frozen_corpus_perm{seed}.json"),
    )


def reversed_corpus(directory: str) -> str:
    artifact = _load_corpus_artifact()
    artifact["fragments"].reverse()
    return _write_variant(
        artifact,
        os.path.join(directory, "frozen_corpus_rev.json"),
    )


def _write_result(path: str | None, artifact: dict) -> None:
    if path is None:
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as destination:
        json.dump(artifact, destination, indent=2, ensure_ascii=False)
        destination.write("\n")


def main(output_path: str | None = None) -> int:
    source_artifact = _load_corpus_artifact()
    with tempfile.TemporaryDirectory(prefix="entroly_exp1_variants_") as variants:
        conditions: list[tuple[str, str, dict[str, str]]] = [
            ("baseline", CORPUS, {}),
            ("baseline-2", CORPUS, {}),
            ("hashseed=0", CORPUS, {"PYTHONHASHSEED": "0"}),
            ("hashseed=1", CORPUS, {"PYTHONHASHSEED": "1"}),
            ("hashseed=42", CORPUS, {"PYTHONHASHSEED": "42"}),
            ("hashseed=random", CORPUS, {"PYTHONHASHSEED": "random"}),
            (
                "threads=1",
                CORPUS,
                {"RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            ),
            (
                "threads=2",
                CORPUS,
                {"RAYON_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"},
            ),
            (
                "threads=8",
                CORPUS,
                {"RAYON_NUM_THREADS": "8", "OMP_NUM_THREADS": "8"},
            ),
            ("insert-perm-1", permuted_corpus(1, variants), {}),
            ("insert-perm-2", permuted_corpus(2, variants), {}),
            ("insert-perm-3", permuted_corpus(7, variants), {}),
            ("insert-reversed", reversed_corpus(variants), {}),
        ]

        results: dict[str, dict] = {}
        failures: dict[str, str] = {}
        for label, corpus, env in conditions:
            try:
                results[label] = run_capture(corpus, env)
                result = results[label]
                print(
                    f"  ran {label:18s} n={result['n']:2d} "
                    f"digest={result['digest'][:12]}"
                )
            except Exception as exc:
                failures[label] = f"{type(exc).__name__}: {exc}"
                print(f"  FAIL {label:18s}: {exc}")

    result_artifact: dict = {
        "schema_version": "entroly.research.repro-result.v2",
        "protocol": {
            "query": QUERY,
            "budget": int(BUDGET),
            "declared_conditions": [label for label, _corpus, _env in conditions],
        },
        "corpus_metadata": source_artifact["metadata"],
        "conditions": results,
        "failures": failures,
        "comparisons": [],
        "valid": False,
    }
    if "baseline" not in results:
        print("\n=== verdict ===")
        print("  => INVALID: baseline condition failed")
        _write_result(output_path, result_artifact)
        return 1
    baseline = results["baseline"]
    baseline_keys = selection_keys(baseline)
    baseline_digest = baseline["digest"]
    print("\n=== reproducibility matrix (vs baseline) ===")
    print(f"{'condition':18s} {'Jaccard':>8s} {'tau':>7s} {'byte':>6s}  n")
    complete = len(results) == len(conditions)
    all_byte = complete
    all_set = complete
    for label, result in results.items():
        if label == "baseline":
            continue
        current_keys = selection_keys(result)
        overlap = jaccard(baseline_keys, current_keys)
        tau = kendall_tau(baseline_keys, current_keys)
        byte_identical = result["digest"] == baseline_digest
        all_byte &= byte_identical
        all_set &= overlap == 1.0
        result_artifact["comparisons"].append(
            {
                "condition": label,
                "fragment_identity_jaccard": overlap,
                "kendall_tau": tau,
                "byte_identical": byte_identical,
                "selection_count": result["n"],
            }
        )
        tau_text = "n/a" if tau is None else f"{tau:+.3f}"
        print(
            f"{label:18s} {overlap:8.3f} {tau_text:>7s} "
            f"{str(byte_identical):>6s}  {result['n']}"
        )

    print("\n=== verdict ===")
    print(f"  all declared conditions completed:   {complete}")
    print(f"  set-identical across all conditions:  {all_set}")
    print(f"  byte-identical across all conditions: {all_byte}")
    if failures:
        print(f"  failed conditions: {', '.join(sorted(failures))}")
    if complete and all_byte and all_set:
        print("  => STRICT DETERMINISTIC on the tested runtime and architecture")
    elif complete and all_set:
        print("  => SET-stable but order or bytes vary on at least one axis")
    else:
        print("  => NOT ESTABLISHED: a condition failed or selection diverged")
    valid = complete and all_byte and all_set
    result_artifact["valid"] = valid
    result_artifact["verdict"] = {
        "complete": complete,
        "set_identical": all_set,
        "byte_identical": all_byte,
    }
    _write_result(output_path, result_artifact)
    return 0 if valid else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.output))
