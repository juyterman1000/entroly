"""Freeze the live Entroly index into a corpus JSON for reproducibility tests.

Isolates the SELECTOR from index state: every reproducibility run reads this
same frozen corpus, so any divergence is attributable to the selector, not to
a changing index.

    python docs/research/exp1/freeze_corpus.py docs/research/exp1/frozen_corpus.json
"""
from __future__ import annotations

import json
import os
import sys


def main(out: str) -> int:
    os.environ.setdefault("ENTROLY_SOURCE", os.getcwd())
    from entroly.server import EntrolyConfig, EntrolyEngine

    engine = EntrolyEngine(config=EntrolyConfig())
    engine._ensure_index_loaded()
    fragments = [
        {
            "source": f.get("source", ""),
            "content": f.get("content", ""),
            "fragment_id": f.get("fragment_id", ""),
            "feedback_multiplier": 1.0,
        }
        for f in engine._rust.export_fragments()
    ]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(fragments, fh)
    print(f"froze {len(fragments)} fragments -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "frozen_corpus.json"))
