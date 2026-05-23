"""
Entroly NeedleInAHaystack Heatmap
==================================

Generates the iconic NIAH heatmap visualization showing Entroly's
retrieval accuracy across different context lengths and needle positions.

Produces a publishable PNG/SVG that shows:
  - X axis: Context length (4K → 128K tokens)
  - Y axis: Needle depth (0% top → 100% bottom)
  - Color: Retrieval accuracy (green=perfect, red=failed)

Two heatmaps side-by-side:
  Left:  Baseline (raw context → LLM)
  Right: Entroly (compressed context → LLM)

Usage:
    python -m bench.needle_heatmap --model gemini-2.0-flash --output needle_results.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def generate_heatmap(
    results: dict,
    output_path: str = "needle_heatmap.png",
    title: str = "NeedleInAHaystack: Entroly vs Baseline",
):
    """Generate side-by-side heatmap from benchmark results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  Install matplotlib: pip install matplotlib")
        return

    sizes = sorted(set(r["size"] for r in results["baseline"]))
    depths = sorted(set(r["depth"] for r in results["baseline"]))

    def build_matrix(data):
        matrix = []
        for depth in depths:
            row = []
            for size in sizes:
                matches = [r for r in data if r["size"] == size and r["depth"] == depth]
                if matches:
                    row.append(matches[0]["score"])
                else:
                    row.append(0)
            matrix.append(row)
        return matrix

    baseline_matrix = build_matrix(results["baseline"])
    entroly_matrix = build_matrix(results["entroly"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    size_labels = [f"{s//1000}K" for s in sizes]
    depth_labels = [f"{int(d*100)}%" for d in depths]

    cmap = plt.cm.RdYlGn

    im1 = ax1.imshow(baseline_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax1.set_title("Baseline (No Compression)", fontsize=14)
    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("Needle Depth")
    ax1.set_xticks(range(len(sizes)))
    ax1.set_xticklabels(size_labels)
    ax1.set_yticks(range(len(depths)))
    ax1.set_yticklabels(depth_labels)

    # Annotate cells
    for i in range(len(depths)):
        for j in range(len(sizes)):
            v = baseline_matrix[i][j]
            color = "white" if v < 0.5 else "black"
            ax1.text(j, i, f"{v:.0%}", ha="center", va="center", color=color, fontsize=10)

    im2 = ax2.imshow(entroly_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax2.set_title("Entroly Compressed", fontsize=14)
    ax2.set_xlabel("Context Length")
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels(size_labels)
    ax2.set_yticks(range(len(depths)))
    ax2.set_yticklabels(depth_labels)

    for i in range(len(depths)):
        for j in range(len(sizes)):
            v = entroly_matrix[i][j]
            color = "white" if v < 0.5 else "black"
            ax2.text(j, i, f"{v:.0%}", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="Retrieval Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved heatmap to {output_path}")


def run_needle_sweep(
    model: str = "gpt-4o-mini",
    sizes: list[int] | None = None,
    depths: list[float] | None = None,
    budget: int = 50_000,
) -> dict:
    """Run NeedleInAHaystack sweep across sizes and depths."""
    from bench.accuracy import _call_llm, _compress_messages, _generate_haystack

    if sizes is None:
        sizes = [4_000, 8_000, 16_000, 32_000, 64_000, 128_000]
    if depths is None:
        depths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    needle = "The secret passphrase for Project Aurora is 'crystalline-nebula-7742'."
    question = "What is the secret passphrase for Project Aurora? Answer with just the passphrase."
    expected = "crystalline-nebula-7742"

    baseline_results = []
    entroly_results = []

    total = len(sizes) * len(depths)
    count = 0

    for size in sizes:
        for depth in depths:
            count += 1
            haystack = _generate_haystack(size, needle, depth)
            messages = [
                {"role": "system", "content": f"Context:\n{haystack}"},
                {"role": "user", "content": question},
            ]

            # Baseline
            try:
                resp, _, _ = _call_llm(messages, model, max_tokens=100)
                score = 1.0 if expected in resp.lower() else 0.0
            except Exception:
                score = 0.0
            baseline_results.append({"size": size, "depth": depth, "score": score})

            # Entroly
            try:
                compressed = _compress_messages(messages, budget)
                resp, _, _ = _call_llm(compressed, model, max_tokens=100)
                score = 1.0 if expected in resp.lower() else 0.0
            except Exception:
                score = 0.0
            entroly_results.append({"size": size, "depth": depth, "score": score})

            pct = count / total * 100
            print(f"  [{count}/{total}] {pct:.0f}% — size={size//1000}K depth={depth:.0%}")

    return {"baseline": baseline_results, "entroly": entroly_results}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NeedleInAHaystack Heatmap Generator")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output", type=str, default="docs/assets/needle_heatmap.png")
    parser.add_argument("--budget", type=int, default=50_000)
    parser.add_argument("--results", type=str, default=None, help="Load existing results JSON")
    args = parser.parse_args()

    if args.results:
        with open(args.results) as f:
            results = json.load(f)
    else:
        print(f"\n  NeedleInAHaystack Sweep — {args.model}")
        print("  " + "=" * 50)
        results = run_needle_sweep(model=args.model, budget=args.budget)

        # Save raw results
        results_path = Path(args.output).with_suffix(".json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved raw results to {results_path}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate_heatmap(results, args.output)


if __name__ == "__main__":
    main()
