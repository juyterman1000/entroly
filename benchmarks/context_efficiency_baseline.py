"""Build frozen, auditable baseline manifests for context-efficiency trials.

The built-in head-tail baseline is a transparent calibration control, not a
state-of-the-art competitor. The same manifest contract accepts outputs from
version-pinned external systems without importing them into Entroly's runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.context_efficiency_openai import _load_longbench

BASELINE_SCHEMA_VERSION = "entroly.context-efficiency-baseline.v1"
HEAD_TAIL_VERSION = "head-tail-o200k-v1"
HEAD_TAIL_MARKER = "\n\n[... frozen head-tail baseline omission ...]\n\n"


def head_tail_context(text: str, *, token_budget: int, encoding: Any) -> str:
    """Retain equal token slices from both boundaries under an exact token cap."""
    if token_budget < 8:
        raise ValueError("token_budget must be at least 8")
    tokens = encoding.encode(text)
    if len(tokens) <= token_budget:
        return text
    marker_tokens = encoding.encode(HEAD_TAIL_MARKER)
    retained_budget = token_budget - len(marker_tokens)
    if retained_budget < 2:
        raise ValueError("token_budget is too small for the omission marker")
    head_size = retained_budget // 2
    tail_size = retained_budget - head_size
    selected = (
        encoding.decode(tokens[:head_size])
        + HEAD_TAIL_MARKER
        + encoding.decode(tokens[-tail_size:])
    )
    if len(encoding.encode(selected)) > token_budget:
        raise RuntimeError("head-tail baseline exceeded its declared token budget")
    return selected


def build_head_tail_manifest(
    *, items: list[Any], token_budget: int, encoding: Any, implementation_sha256: str
) -> dict[str, Any]:
    tasks = []
    for item in items:
        selected = head_tail_context(
            item.context, token_budget=token_budget, encoding=encoding
        )
        tasks.append(
            {
                "task_id": item.task_id,
                "source_context_sha256": hashlib.sha256(
                    item.context.encode("utf-8")
                ).hexdigest(),
                "selected_context": selected,
                "selected_context_sha256": hashlib.sha256(
                    selected.encode("utf-8")
                ).hexdigest(),
            }
        )
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "condition": "algorithmic_baseline",
        "baseline": {
            "name": "frozen head-tail truncation",
            "version": HEAD_TAIL_VERSION,
            "source": "benchmarks/context_efficiency_baseline.py",
            "config": {
                "head_fraction": 0.5,
                "implementation_sha256": implementation_sha256,
                "token_budget": token_budget,
                "tokenizer": "tiktoken:o200k_base",
            },
        },
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument(
        "--selection", choices=("random", "shortest-context"), default="random"
    )
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.output.exists():
        parser.error(f"{args.output} already exists")

    import tiktoken

    items = _load_longbench(args.samples, args.selection)
    implementation_path = Path(__file__)
    manifest = build_head_tail_manifest(
        items=items,
        token_budget=args.budget,
        encoding=tiktoken.get_encoding("o200k_base"),
        implementation_sha256=hashlib.sha256(implementation_path.read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} with {len(items)} frozen baseline tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
