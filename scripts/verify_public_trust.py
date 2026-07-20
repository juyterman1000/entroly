#!/usr/bin/env python3
"""Verify Entroly's prominent public trust contracts.

The default checks are deterministic and offline. ``--online`` additionally
checks canonical public destinations with bounded retries; it does not treat a
marketplace HTTP 200 as marketplace validation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.neural_query_shift import verify_report as verify_query_shift_report  # noqa: E402

CANONICAL_LINKS = {
    "Entroly on PyPI": "https://pypi.org/project/entroly/",
    "Entroly on npm": "https://www.npmjs.com/package/entroly",
    "Apache-2.0 license": "LICENSE",
    "Context Commit conformance evidence": "benchmarks/results/context_commit_conformance.json",
    "WITNESS HaluEval-QA evidence": "benchmarks/results/halueval_qa_faithful.json",
    "Python, optional Rust, and WASM runtimes": "docs/product-surface.md",
    "Measure token savings on your workload": "#proof",
    "Entroly repository and GitHub stars": "https://github.com/juyterman1000/entroly",
    "Entroly Discord community": "https://juyterman1000.github.io/entroly/docs/discord.html",
    "Current external Entroly status on LobeHub": "https://lobehub.com/mcp/juyterman1000-entroly",
}

ONLINE_DESTINATIONS = {
    "PyPI": "https://pypi.org/project/entroly/",
    "npm registry metadata": "https://registry.npmjs.org/entroly/latest",
    "GitHub": "https://github.com/juyterman1000/entroly",
    "Discord landing": "https://juyterman1000.github.io/entroly/docs/discord.html",
    "LobeHub listing": "https://lobehub.com/mcp/juyterman1000-entroly?activeTab=score",
}

PROMINENT_PUBLIC_FILES = (
    "README.md",
    "PYPI_README.md",
    "docs/index.html",
    "docs/discord.html",
    "docs/mcp-server-guide.html",
    "docs/first-run-trust.md",
    "docs/public-evidence.md",
    "docs/marketing/registry_submissions.md",
    "docs/marketing/tutorial_devto.md",
    "docs/marketing/tutorial_reddit.md",
)

RETIRED_MARKETING_PAGES = (
    "docs/best-context-compression-tools.html",
    "docs/cursor-token-usage-fix.html",
    "docs/how-to-reduce-claude-api-costs.html",
    "docs/reduce-llm-api-costs.html",
    "docs/prompt-compression.html",
    "docs/hallucination-guard.html",
    "docs/prevent-ai-hallucinations.html",
    "docs/dashboard.html",
    "docs/token-optimization.html",
    "docs/what-is-context-rot.html",
)

RETIRED_SETUP_PAGES = (
    "docs/cursor-context-guide.html",
    "docs/claude-code-setup.html",
)

TRANSLATED_READMES = (
    "docs/i18n/README.de.md",
    "docs/i18n/README.es.md",
    "docs/i18n/README.fr.md",
    "docs/i18n/README.hi.md",
    "docs/i18n/README.ja.md",
    "docs/i18n/README.ko.md",
    "docs/i18n/README.pt-BR.md",
    "docs/i18n/README.ru.md",
    "docs/i18n/README.zh-CN.md",
)

CLAIM_SENSITIVE_PUBLIC_FILES = (
    *RETIRED_MARKETING_PAGES,
    *RETIRED_SETUP_PAGES,
    *TRANSLATED_READMES,
    "docs/context-engineering.html",
    "docs/DETAILS.md",
    "docs/for-teams.md",
    "docs/marketing/launch_playbook.md",
    "docs/marketing/entroly_vs_headroom_seo.md",
    "docs/generate_demo.py",
    "docs/assets/demo.svg",
    "docs/assets/demo_animated.svg",
    "docs/assets/demo.html",
    "docs/assets/value.svg",
    "BIPT.md",
)

STALE_PUBLIC_CLAIMS = {
    "70–95%": "universal token or billing range",
    "70-95%": "universal token or billing range",
    "78% fewer": "unscoped token-reduction headline",
    "statistically equivalent": "unsupported equivalence conclusion",
    "equivalent to gpt-4o-mini": "unsupported cross-protocol conclusion",
    "0.844 auroc": "retired exploratory metric",
    "same accuracy": "answer-quality guarantee",
    "verifies every output": "universal verifier coverage",
    "checks every llm response": "universal verifier coverage",
    "all your files": "universal repository-visibility claim",
    "all 847 files": "simulated repository-visibility claim",
    "zero config changes": "unverified integration-friction claim",
    "the first hallucination detector": "unverified novelty claim",
    "why nobody else has this": "unverified competitor-wide claim",
    "the only system": "unverified exclusivity claim",
    "no prior art": "unverified prior-art claim",
    "mathematical breakthrough": "unverified research headline",
}

PRISM_R_PUBLIC_FILES = {
    "README.md": "benchmarks/results/neural_query_shift.json",
    "docs/public-evidence.md": "../benchmarks/results/neural_query_shift.json",
}


class _BadgeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._href: str | None = None
        self.badges: dict[str, str | None] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a":
            self._href = values.get("href")
        elif tag == "img":
            src = values.get("src") or ""
            if "img.shields.io" in src or "lobehub.com/badge/" in src:
                self.badges[values.get("alt") or src] = self._href

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._href = None


def _read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _collect_prism_r_public_failures(
    prominent_text: dict[str, str], report: dict[str, Any]
) -> list[str]:
    """Bind the bounded PRISM-R pilot copy to its recomputable artifact.

    ``87.0%`` was historically banned as an unsupported aggregate savings
    headline.  The PRISM-R artifact also happens to produce 87.0%, but for a
    narrower exact-evidence metric.  Scope and provenance must therefore be
    checked together instead of treating the numeric string as the claim.
    """

    failures: list[str] = []
    try:
        verify_query_shift_report(report)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"PRISM-R query-shift artifact failed internal verification: {exc}"]

    metrics = report["metrics"]
    protocol = report["protocol"]
    expected_protocol = {
        "active_ratio": 0.25,
        "dataset": "rajpurkar/squad_v2",
        "offline_only": True,
        "query_shift": "q2 is a different question with an answer in a different sentence",
        "trials": 200,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            failures.append(
                f"PRISM-R query-shift protocol {key}={protocol.get(key)!r}, "
                f"expected {expected!r}"
            )
    if report.get("pilot_gate_passed") is not True:
        failures.append("PRISM-R public pilot requires pilot_gate_passed=true")
    if report.get("headline_eligible") is not False:
        failures.append("PRISM-R pilot must remain headline_eligible=false")

    rendered_metrics = {
        f"{metrics['prism_r_q1_retention']:.1%}": "current-query exact evidence",
        f"{metrics['lexical_q1_retention']:.1%}": "lexical current-query evidence",
        f"{metrics['prism_r_future_q2_retention']:.1%}": "unseen future evidence",
        f"{metrics['prism_r_q2_after_exact_rehydration']:.1%}": (
            "future evidence after exact rehydration"
        ),
        f"{metrics['active_plus_rehydration_ratio_approx']:.1%}": (
            "approximate active plus recovered context"
        ),
    }
    required_scope = {
        "README.md": (
            "offline exact-evidence pilots",
            "not\ndownstream answer-quality",
            "opt-in research code",
        ),
        "docs/public-evidence.md": (
            "exact answer-string retention",
            "do not measure generated answers",
            "not the default compressor",
        ),
    }
    for path, artifact_link in PRISM_R_PUBLIC_FILES.items():
        text = prominent_text.get(path, "")
        if "PRISM-R" not in text:
            failures.append(f"{path} is missing the scoped PRISM-R pilot section")
            continue
        if artifact_link not in text:
            failures.append(f"{path} does not link the PRISM-R query-shift artifact")
        for rendered, label in rendered_metrics.items():
            if rendered not in text:
                failures.append(
                    f"{path} is missing artifact-backed PRISM-R value {rendered} ({label})"
                )
        for phrase in required_scope[path]:
            if phrase not in text:
                failures.append(f"{path} is missing PRISM-R scope language {phrase!r}")

    for path, text in prominent_text.items():
        if path not in PRISM_R_PUBLIC_FILES and "87.0%" in text:
            failures.append(
                f"{path} contains unscoped public claim '87.0%'; "
                "only the artifact-bound PRISM-R pilot may use it"
            )
    return failures


def _collect_stale_public_claim_failures(
    public_text: dict[str, str],
) -> list[str]:
    """Reject retired universal claims from trust-sensitive public surfaces."""

    failures: list[str] = []
    for path, text in public_text.items():
        folded = text.casefold()
        for phrase, reason in STALE_PUBLIC_CLAIMS.items():
            if phrase.casefold() in folded:
                failures.append(
                    f"{path} contains stale public claim {phrase!r}: {reason}"
                )
    return failures


def collect_offline_failures() -> list[str]:
    failures: list[str] = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pypi_readme = (ROOT / "PYPI_README.md").read_text(encoding="utf-8")

    parser = _BadgeParser()
    parser.feed(readme)
    for alt, expected_href in CANONICAL_LINKS.items():
        actual = parser.badges.get(alt)
        if actual != expected_href:
            failures.append(f"badge {alt!r} links to {actual!r}, expected {expected_href!r}")
    unlinked = sorted(alt for alt, href in parser.badges.items() if not href)
    if unlinked:
        failures.append(f"unlinked public badges: {unlinked}")

    marketplace_heading = readme.find("### Marketplace status")
    lobehub_badge = readme.find("https://lobehub.com/badge/mcp/juyterman1000-entroly")
    if marketplace_heading < 0 or lobehub_badge < marketplace_heading:
        failures.append("LobeHub badge must stay in the contextualized marketplace section")

    banned_readme_claims = {
        "every answer gets a receipt": "receipt coverage is integration-specific",
        "Nothing is lost": "recovery depends on retained state",
        "statistically ties": "benchmark protocols must not be conflated",
        "70–95%": "universal savings range has no single supporting artifact",
        "99.1%": "unsupported aggregate token headline",
        "96.7%": "unsupported aggregate token headline",
        "0.844": "STAVE exploratory protocol is not the faithful headline",
    }
    for phrase, reason in banned_readme_claims.items():
        if phrase in readme:
            failures.append(f"README contains banned claim {phrase!r}: {reason}")

    prominent_text = {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in PROMINENT_PUBLIC_FILES
    }
    banned_prominent_claims = {
        "70–95%": "universal savings range",
        "70-95%": "universal savings range",
        "99.1%": "unsupported aggregate token headline",
        "96.7%": "unsupported aggregate token headline",
        "statistically ties": "unsupported cross-protocol comparison",
        "AUROC 0.84": "exploratory STAVE result used as a headline",
        "~3 ms": "latency headline without the matching protocol",
        "Zero Hallucinations": "verification cannot guarantee zero hallucinations",
        "zero hallucinations": "verification cannot guarantee zero hallucinations",
        "30-Second Install": "unverified setup-time promise",
    }
    for path, text in prominent_text.items():
        for phrase, reason in banned_prominent_claims.items():
            if phrase in text:
                failures.append(f"{path} contains banned public claim {phrase!r}: {reason}")

    neural_query_shift = _read_json("benchmarks/results/neural_query_shift.json")
    failures.extend(
        _collect_prism_r_public_failures(prominent_text, neural_query_shift)
    )

    mcp_guide = prominent_text["docs/mcp-server-guide.html"]
    unsupported_serve_flags = (
        "entroly serve --transport",
        "entroly serve --quality",
        "entroly serve --port",
    )
    for command in unsupported_serve_flags:
        if command in mcp_guide:
            failures.append(f"MCP guide advertises unsupported CLI syntax: {command}")

    sitemap = (ROOT / "docs/sitemap.xml").read_text(encoding="utf-8")
    redirect = "https://github.com/juyterman1000/entroly/blob/main/docs/public-evidence.md"
    for path in RETIRED_MARKETING_PAGES:
        retired = (ROOT / path).read_text(encoding="utf-8")
        if 'name="robots" content="noindex,nofollow"' not in retired:
            failures.append(f"retired marketing page is indexable: {path}")
        if f'http-equiv="refresh" content="0; url={redirect}"' not in retired:
            failures.append(f"retired marketing page does not redirect to evidence: {path}")
        if Path(path).name in sitemap:
            failures.append(f"retired marketing page remains in sitemap: {path}")
        if f'href="/entroly/{path}"' in prominent_text["docs/index.html"]:
            failures.append(f"homepage still promotes retired marketing page: {path}")

    setup_redirect = "https://juyterman1000.github.io/entroly/docs/mcp-server-guide.html"
    for path in RETIRED_SETUP_PAGES:
        retired = (ROOT / path).read_text(encoding="utf-8")
        if 'name="robots" content="noindex,nofollow"' not in retired:
            failures.append(f"retired setup page is indexable: {path}")
        if f'http-equiv="refresh" content="0; url={setup_redirect}"' not in retired:
            failures.append(f"retired setup page does not redirect to MCP guide: {path}")
        if Path(path).name in sitemap:
            failures.append(f"retired setup page remains in sitemap: {path}")
        if f'href="/entroly/{path}"' in prominent_text["docs/index.html"]:
            failures.append(f"homepage still promotes retired setup page: {path}")

    for path in TRANSLATED_READMES:
        translated = (ROOT / path).read_text(encoding="utf-8")
        for required in ("entroly verify-claims", "../public-evidence.md", "../limitations.md"):
            if required not in translated:
                failures.append(f"{path} is missing translated trust link {required!r}")

    claim_sensitive_text = {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in CLAIM_SENSITIVE_PUBLIC_FILES
    }
    failures.extend(_collect_stale_public_claim_failures(claim_sensitive_text))

    forbidden_promotions = (
        "https://huggingface.co/spaces/entroly/entroly-context-compression",
        "https://juyterman1000.github.io/entroly/docs/dashboard.html",
    )
    for url in forbidden_promotions:
        if url in readme:
            failures.append(f"primary README promotes quarantined surface: {url}")

    forbidden_pypi_routes = (
        "uvx --from entroly entroly serve",
        "npx -y entroly-mcp serve",
        '"args": ["--from", "entroly", "entroly", "serve"]',
    )
    for route in forbidden_pypi_routes:
        if route in pypi_readme:
            failures.append(f"PYPI_README.md still advertises Docker-first MCP route: {route}")
    if "`entroly serve` is a different deployment path" not in pypi_readme:
        failures.append("PYPI_README.md must explain the Docker-first serve command")

    manifest = _read_json("server.json")
    for package in manifest.get("packages", []):
        if package.get("packageArguments"):
            failures.append(
                f"server.json {package.get('identifier')} must use argument-free stdio registration"
            )

    conformance = _read_json("benchmarks/results/context_commit_conformance.json")
    aggregate = conformance["aggregate"]
    conformance_expected = {
        "cases": 128,
        "deterministic_replay_rate": 1.0,
        "omitted_chunks_verified": 576,
        "tamper_trials": 768,
        "tamper_detection_rate": 1.0,
    }
    for key, expected in conformance_expected.items():
        if aggregate.get(key) != expected:
            failures.append(f"Context Commit artifact {key}={aggregate.get(key)!r}, expected {expected!r}")

    faithful = _read_json("benchmarks/results/halueval_qa_faithful.json")
    witness = faithful["witness"]
    shared = faithful["witness_on_gpt_sample"]
    gpt4o = next(row for row in faithful["gpt"] if row["model"] == "gpt-4o-mini")
    expected_values = {
        "0.7976": witness.get("auroc_full"),
        "84.92%": 100 * witness["test_accuracy_calibrated"]["accuracy"],
        "20,000": 2 * witness["n_items"],
        "16,000": witness["test_accuracy_calibrated"]["n"],
        "86.58%": 100 * shared["accuracy"],
        "86.25%": 100 * gpt4o["accuracy"],
        "1,200": shared["n"],
    }
    for rendered, value in expected_values.items():
        if rendered not in readme:
            failures.append(f"README is missing faithful benchmark value {rendered} ({value!r})")

    package_names = {
        _read_json("entroly/npm-alias/package.json")["name"],
        _read_json("entroly/npm/package.json")["name"],
        _read_json("entroly-wasm/package.json")["name"],
    }
    if package_names != {"entroly", "entroly-mcp", "entroly-wasm"}:
        failures.append(f"unexpected npm package identity set: {sorted(package_names)}")

    for artifact in ("mcp-publisher", "mcp-publisher.exe", "mcp-publisher.tar.gz"):
        if (ROOT / artifact).exists():
            failures.append(f"unsigned registry publisher artifact committed at repo root: {artifact}")

    publisher_workflow = (ROOT / ".github/workflows/publish-mcp-registry.yml").read_text(
        encoding="utf-8"
    )
    publisher_digest = "ab128162b0616090b47cf245afe0a23f3ef08936fdce19074f5ba0a4469281ac"
    if publisher_digest not in publisher_workflow:
        failures.append("MCP publisher workflow is missing the reviewed v1.7.9 Linux digest")
    if "sha256sum --check --strict" not in publisher_workflow:
        failures.append("MCP publisher workflow does not fail closed on checksum mismatch")
    if "| tar" in publisher_workflow:
        failures.append("MCP publisher workflow streams an unverified download into tar")

    return failures


def collect_online_failures(*, retries: int = 3, timeout: float = 15.0) -> list[str]:
    failures: list[str] = []
    for name, url in ONLINE_DESTINATIONS.items():
        error = "unknown failure"
        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "entroly-public-trust-check/1"},
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if 200 <= response.status < 400:
                        error = ""
                        break
                    error = f"HTTP {response.status}"
            except (OSError, urllib.error.URLError) as exc:
                error = str(exc)
            if attempt + 1 < retries:
                time.sleep(1 + attempt)
        if error:
            failures.append(f"{name} destination failed after {retries} attempts: {error}")
    return failures


def collect_published_version_failures(*, timeout: float = 15.0) -> list[str]:
    expected = _read_json("server.json")["version"]
    sources = {
        "PyPI": ("https://pypi.org/pypi/entroly/json", lambda data: data["info"]["version"]),
        "npm": ("https://registry.npmjs.org/entroly/latest", lambda data: data["version"]),
    }
    failures: list[str] = []
    for name, (url, extract) in sources.items():
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "entroly-public-trust-check/1"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                published = str(extract(json.load(response)))
        except (KeyError, OSError, TypeError, urllib.error.URLError, ValueError) as exc:
            failures.append(f"could not read {name} published version: {exc}")
            continue
        if published != expected:
            failures.append(f"{name} latest is {published}, expected release {expected}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument(
        "--require-published-version",
        action="store_true",
        help="Require PyPI and npm latest versions to match server.json.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    failures = collect_offline_failures()
    if args.online:
        failures.extend(collect_online_failures(retries=args.retries, timeout=args.timeout))
    if args.require_published_version:
        failures.extend(collect_published_version_failures(timeout=args.timeout))

    if failures:
        print("Public trust verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    scopes = ["offline contracts"]
    if args.online:
        scopes.append("online destinations")
    if args.require_published_version:
        scopes.append("published version parity")
    scope = ", ".join(scopes)
    print(f"Public trust verification passed ({scope}).")
    print("This check validates declared public contracts; it does not certify every product claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
