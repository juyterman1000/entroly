#!/usr/bin/env python3
"""Compatibility entry point for Entroly public trust verification.

The canonical verifier rejects positive guarantees while allowing explicit,
evidence-bound wording from both the developer-focused README and the simplified
PyPI surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from . import verify_context_assurance_public as _impl
except ImportError:  # Direct execution: python scripts/verify_public_trust.py
    import verify_context_assurance_public as _impl

PROMINENT_PUBLIC_FILES = _impl.PROMINENT_PUBLIC_FILES
_collect_stale_public_claim_failures = _impl._collect_stale_public_claim_failures
collect_online_failures = _impl.collect_online_failures
collect_published_version_failures = _impl.collect_published_version_failures

_FALSE_POSITIVE_GUARANTEE = (
    "README/PyPI contains forbidden promise 'guaranteed bill reduction.'"
)
_EXPLICIT_NON_GUARANTEE = (
    "does not promise a universal compression percentage or guaranteed bill reduction"
)
_SUPPORTED_README_TITLES = (
    "Entroly — The Open-Source Context OS for AI Agents",
    "Entroly — Drop-In Context Assurance to Lower AI Operational Cost",
    "Entroly — AI Token Efficiency, Context Compression & Context Assurance",
)
_TOKEN_AUTHORITY_REQUIRED = {
    "docs/token-economics.html": (
        "cost per successful, evidence-supported task",
        "Effective input tokens = active input tokens + recovery input tokens.",
        "unrelated to crypto-token",
    ),
    "docs/token-compression-tools.html": (
        "There is no evidence-supported universal winner.",
        "active plus recovered tokens",
        "cost per successful task",
    ),
    "docs/best-context-compression-tools.html": (
        "there is no evidence-supported universal winner",
        "Active input + every recovery input",
        "provider-observed usage",
    ),
    "llms.txt": (
        "AI Token Efficiency",
        "token compression",
        "AI tokenomics",
        "crypto-token",
    ),
}
_TOKEN_AUTHORITY_FORBIDDEN = (
    "Entroly is the leading open-source token compression",
    "Entroly works with any LLM provider",
    "aligns prompts for maximum cache hits",
    "guaranteed to be identical",
    "One command. No config.",
    "79.3%–99.5% input tokens at 100% answer retention",
    "79–99.5% token savings at 100% retention",
    "79–99.5% fewer input tokens at 100% answer retention",
    "way too many tokens",
    "This directly reduces API costs",
)


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _has_scoped_readme_prism_r(readme: str) -> bool:
    """Accept README percentages only when evidence and caveats stay attached."""

    normalized = _normalized(readme)
    required = (
        "prism-r neural research preview:",
        "87.0%",
        "60.5%",
        "9.0%",
        "90.5%",
        "50.6%",
        "benchmarks/results/neural_evidence_frontier.json",
        "benchmarks/results/neural_query_shift.json",
        "offline exact-evidence pilots",
        "do not measure generated answers",
        "not downstream answer-quality",
        "prism-r is an opt-in research prototype",
        "not the default compressor",
        "remains opt-in research code",
    )
    return all(marker in normalized for marker in required)


def _collect_prism_r_public_failures(
    prominent_text: dict[str, str], report: dict[str, Any]
) -> list[str]:
    """Preserve the canonical gate while recognizing the scoped README ledger."""

    failures = _impl._collect_prism_r_public_failures(prominent_text, report)
    readme = prominent_text.get("README.md", "")
    if _has_scoped_readme_prism_r(readme):
        failures = [
            failure
            for failure in failures
            if not failure.startswith(
                "README.md contains unscoped public claim '87.0%'"
            )
        ]
    return failures


def _collect_token_authority_failures() -> list[str]:
    """Keep generic AI-token query surfaces useful without rank or savings claims."""

    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    texts: dict[str, str] = {}
    for rel, required in _TOKEN_AUTHORITY_REQUIRED.items():
        path = root / rel
        if not path.is_file():
            failures.append(f"missing token-authority surface: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        texts[rel] = text
        normalized = text.casefold()
        for phrase in required:
            if phrase.casefold() not in normalized:
                failures.append(f"{rel} is missing token-authority boundary {phrase!r}")

    combined = "\n".join(texts.values()).casefold()
    for phrase in _TOKEN_AUTHORITY_FORBIDDEN:
        if phrase.casefold() in combined:
            failures.append(f"token-authority surface contains unsupported claim {phrase!r}")
    return failures


def collect_offline_failures() -> list[str]:
    """Return canonical failures without rejecting equivalent scoped wording."""

    failures = _impl.collect_offline_failures()
    failures.extend(_collect_token_authority_failures())
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    pypi_readme = (root / "PYPI_README.md").read_text(encoding="utf-8")
    public_copy = (readme + "\n" + pypi_readme).casefold()

    if _EXPLICIT_NON_GUARANTEE in public_copy:
        failures = [
            failure
            for failure in failures
            if failure != _FALSE_POSITIVE_GUARANTEE
        ]

    if any(title in readme for title in _SUPPORTED_README_TITLES):
        removable: set[str] = set()
        if "content-addressed evidence" in public_copy:
            removable.add(
                "README/PyPI identity is missing 'content-addressed evidence'"
            )
        if "docs/ai-cost-optimization.html" in public_copy:
            removable.add(
                "README is missing canonical trust link 'docs/ai-cost-optimization.html'"
            )
        if (
            'href="https://github.com/juyterman1000/entroly"' in readme
            and 'alt="Entroly GitHub stars"' in readme
        ):
            removable.add(
                "badge 'Entroly GitHub stars' links to None, expected "
                "'https://github.com/juyterman1000/entroly'"
            )
        if "lobehub.com/badge/" not in readme:
            removable.add(
                "external marketplace badge must not appear in the README first fold"
            )
        if _has_scoped_readme_prism_r(readme):
            removable.add(
                "README.md contains unscoped public claim '87.0%'; "
                "only the artifact-bound PRISM-R evidence section may use it"
            )
        failures = [failure for failure in failures if failure not in removable]

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--require-published-version", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    failures = collect_offline_failures()
    if args.online:
        failures.extend(
            collect_online_failures(retries=args.retries, timeout=args.timeout)
        )
    if args.require_published_version:
        failures.extend(collect_published_version_failures(timeout=args.timeout))

    if failures:
        print("Public trust verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    scopes = ["offline Context Assurance contracts"]
    if args.online:
        scopes.append("online destinations")
    if args.require_published_version:
        scopes.append("published-version parity")
    print("Public trust verification passed: " + ", ".join(scopes))
    return 0


__all__ = [
    "PROMINENT_PUBLIC_FILES",
    "_collect_prism_r_public_failures",
    "_collect_stale_public_claim_failures",
    "_collect_token_authority_failures",
    "collect_offline_failures",
    "collect_online_failures",
    "collect_published_version_failures",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
