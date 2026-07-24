"""Capture Entroly's ordered context selection for a frozen corpus + query.

Deterministic-mode reproducibility probe. Loads a FROZEN fragment corpus (so
index state is not a variable), runs the validated QCCR selector, and emits a
canonical serialization of the ordered selection S_r = [(source, rank,
content_sha, source_fragment_ids)] plus its digest h(S).

Usage:
    python capture_selection.py <frozen_corpus.json> <budget> <query...>
Emits one JSON line to stdout: {"digest": ..., "n": ..., "order": [...]}
Honors env for the determinism axes: PYTHONHASHSEED, RAYON_NUM_THREADS,
OMP_NUM_THREADS, ENTROLY_* — set by the orchestrator, not here.
"""
from __future__ import annotations

import hashlib
import json
import sys


def canonical(selected: list[dict]) -> dict:
    order = []
    for rank, frag in enumerate(selected):
        content = frag.get("content") or ""
        order.append(
            {
                "rank": rank,
                "source": str(frag.get("source") or ""),
                "content_sha": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "content_len": len(content),
                "source_fragment_ids": list(frag.get("source_fragment_ids") or []),
            }
        )
    # h(S): digest over the ordered selection (source + content identity + order)
    blob = json.dumps(
        [(o["rank"], o["source"], o["content_sha"]) for o in order],
        separators=(",", ":"),
    )
    return {"digest": hashlib.sha256(blob.encode("utf-8")).hexdigest(), "n": len(order), "order": order}


def main() -> int:
    frozen_path = sys.argv[1]
    budget = int(sys.argv[2])
    query = " ".join(sys.argv[3:])
    with open(frozen_path, encoding="utf-8") as fh:
        fragments = json.load(fh)
    from entroly import qccr

    selected = qccr.select(fragments, budget, query)
    print(json.dumps(canonical(selected)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
