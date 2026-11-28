"""Replay LoCoMo C1–C4 evidence retrieval with MemoryOS and BM25.

Both selectors get the same text turns and estimated token budget. Each query
starts from the same post-ingestion memory state. Results measure annotated
dialogue evidence recall and omit model calls, answer generation, and costs.

The public dataset used for the baseline is snap-research/locomo at commit
3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376, data/locomo10.json. The report
records the input hash and all exclusions. Run with:

    python -m benchmarks.locomo_retrieval locomo10.json --output report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from entroly.memory import MemoryOS, _estimate_tokens


SESSION = re.compile(r"session_(\d+)")
WORD = re.compile(r"\b\w+\b", re.UNICODE)
BM25_K1 = 1.5
BM25_B = 0.75


def _terms(text: str) -> Counter[str]:
    return Counter(WORD.findall(text.casefold()))


def _bm25_index(turns: dict[str, str]):
    terms = {source: _terms(text) for source, text in turns.items()}
    frequencies = Counter(term for counts in terms.values() for term in counts)
    lengths = {source: counts.total() for source, counts in terms.items()}
    mean_length = (statistics.mean(lengths.values()) if lengths else 0.0) or 1.0
    return terms, frequencies, lengths, mean_length


def _bm25_select(query: str, costs: dict[str, int], index, budget: int) -> tuple[list[str], int]:
    terms, frequencies, lengths, mean_length = index
    weights = {
        term: math.log1p((len(terms) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
        for term in _terms(query)
        if term in frequencies
    }
    scores = {}
    for source, counts in terms.items():
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * lengths[source] / mean_length)
        scores[source] = sum(
            weight * (1.0 + BM25_K1) * counts[term] / (counts[term] + norm)
            for term, weight in weights.items()
            if term in counts
        )
    selected, used = [], 0
    for source, score in sorted(scores.items(), key=lambda item: -item[1]):
        if score <= 0:
            break
        if used + costs[source] <= budget:
            selected.append(source)
            used += costs[source]
    return selected, used


def _metrics(selected: list[str], evidence: set[str], tokens: int, seconds: float) -> dict:
    hits = evidence.intersection(selected)
    return {
        "selected_count": len(selected),
        "recovered_evidence": sorted(hits),
        "any_evidence": bool(hits),
        "all_evidence": hits == evidence,
        "evidence_recall": len(hits) / len(evidence),
        "estimated_tokens": tokens,
        "retrieval_ms": round(seconds * 1000, 3),
    }


def run(dataset: list[dict], budget: int) -> dict:
    rows, exclusions = [], []
    blocked_turns = 0
    for sample in dataset:
        conversation = sample["conversation"]
        order = sorted(
            (int(match[1]), key) for key in conversation if (match := SESSION.fullmatch(key))
        )
        turns = {
            turn["dia_id"]: f"{turn['speaker']}: {turn['text']}".strip()
            for _, key in order for turn in conversation[key] if turn.get("text")
        }
        if not turns:
            raise ValueError(f"no dialogue turns in sample {sample['sample_id']}")
        memory = MemoryOS(default_budget=budget)
        accepted_turns = {}
        for source, text in turns.items():
            try:
                memory.remember(text, source=source, tier="episodic")
            except ValueError:
                if memory.scan_safety(text).allowed:
                    raise
                blocked_turns += 1
            else:
                accepted_turns[source] = text
        index = _bm25_index(accepted_turns)
        costs = {source: _estimate_tokens(text) for source, text in accepted_turns.items()}
        snapshot = memory.snapshot()
        for question_index, qa in enumerate(sample["qa"]):
            row = {"sample_id": sample["sample_id"], "question_index": question_index, "category": qa["category"]}
            evidence = set(qa.get("evidence", ()))
            if row["category"] not in {1, 2, 3, 4} or not evidence or evidence.difference(turns):
                reason = "out_of_scope_category" if row["category"] not in {1, 2, 3, 4} else "unresolvable_evidence"
                exclusions.append({**row, "reason": reason})
                continue

            fresh = MemoryOS.from_snapshot(snapshot)
            start = time.perf_counter()
            context = fresh.recall(qa["question"], budget=budget)
            memory_result = _metrics([item.source for item in context.selected], evidence, context.used_tokens, time.perf_counter() - start)
            start = time.perf_counter()
            selected, used = _bm25_select(qa["question"], costs, index, budget)
            bm25_result = _metrics(selected, evidence, used, time.perf_counter() - start)
            rows.append({**row, "evidence": sorted(evidence), "memory_os": memory_result, "bm25": bm25_result})

    summary = {
        method: {
            "macro_evidence_recall": statistics.mean(row[method]["evidence_recall"] for row in rows),
            "any_evidence_queries": sum(row[method]["any_evidence"] for row in rows),
            "all_evidence_queries": sum(row[method]["all_evidence"] for row in rows),
            "mean_estimated_tokens": statistics.mean(row[method]["estimated_tokens"] for row in rows),
            "median_retrieval_ms": statistics.median(row[method]["retrieval_ms"] for row in rows),
        }
        for method in ("memory_os", "bm25")
    } if rows else {}
    return {"budget": budget, "scored_queries": len(rows), "blocked_memory_turns": blocked_turns,
            "summary": summary, "exclusions": exclusions, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")
    raw = args.dataset.read_bytes()
    report = {"dataset_sha256": hashlib.sha256(raw).hexdigest(), **run(json.loads(raw), args.budget)}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scored_queries": report["scored_queries"], "summary": report["summary"]}, indent=2))
    return 0 if report["scored_queries"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
