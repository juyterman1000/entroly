"""Generate a public benchmark table for Entroly compression.

The table is generated from the deterministic scoreboard scenarios, so published
numbers stay reproducible and do not require API calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.compression_proxy_scoreboard import run_scoreboard


def render_markdown() -> str:
    report = run_scoreboard()
    lines = [
        "# Entroly Compression Proxy Benchmark Table",
        "",
        "Generated from deterministic local scenarios. No LLM/API calls are used.",
        "",
        f"Overall status: **{'PASS' if report.passed else 'FAIL'}**",
        f"Mean savings ratio: **{report.mean_savings_ratio:.1%}**",
        "",
        "| Scenario | Original tokens | Compressed tokens | Savings | Evidence preserved | Receipt emitted |",
        "|---|---:|---:|---:|---|---|",
    ]
    for scenario in report.scenarios:
        lines.append(
            "| {name} | {orig:,} | {comp:,} | {saved:.1%} | {evidence} | {receipt} |".format(
                name=scenario.name,
                orig=scenario.original_tokens,
                comp=scenario.compressed_tokens,
                saved=scenario.savings_ratio,
                evidence="yes" if scenario.evidence_preserved else "no",
                receipt="yes" if scenario.receipt_emitted else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The benchmark gates two things at once: token savings and preservation of answer-critical evidence.",
            "A scenario fails if Entroly saves tokens but drops the evidence needed to answer the user.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m benchmarks.compression_public_table --write docs/benchmarks/compression-proxy-table.md",
            "python -m benchmarks.compression_proxy_scoreboard --json",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Entroly compression benchmark table")
    parser.add_argument("--write", type=str, default="", help="Optional markdown output path")
    args = parser.parse_args()
    markdown = render_markdown()
    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
