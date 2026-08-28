"""A public claim must be measured on the version the public installs.

The README carries two badges sourced from benchmark artifacts:
`5,117/5,117 source spans verified` and `13/13 SDK recovery exact`. Both were
measured on entroly 1.0.69 while 1.0.79 shipped -- ten patch versions of
selection, receipt and engine changes between the evidence and the product.

The numbers happened to still hold when re-measured. That is luck, not a
control. A claim of provability is worth exactly as much as the freshness of
its proof, and nothing in the repository noticed the drift.

Scoped to artifacts the README actually cites, not everything flagged
`headline_eligible`. That marker means "could carry a headline"; being linked
from the README means "is a public claim right now", and only the second earns
a freshness requirement.

The distinction is load-bearing. `language_symbol_coverage.json` is
headline-eligible and deliberately stamps `tree_sitter_language_pack` rather
than an Entroly version, because its claim is about the grammar pack. Holding
it to the package version would fail it forever for being correct, and a gate
that cries wolf gets deleted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def _shipped_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match, "pyproject.toml has no version"
    return match.group(1)


def _artifacts() -> list[tuple[Path, dict]]:
    out = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(payload, dict):
            out.append((path, payload))
    return out


def _published_artifacts() -> list[tuple[Path, dict]]:
    """Artifacts the README links to -- the claims a visitor actually sees."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return [
        (path, payload) for path, payload in _artifacts()
        if payload.get("headline_eligible") is True and path.name in readme
    ]


_headline_artifacts = _published_artifacts


def test_there_are_headline_artifacts_to_check():
    # Guards the premise. If the marker is renamed, the version test below
    # would pass vacuously while every public claim went unchecked.
    assert _headline_artifacts(), (
        "the README cites no headline artifact; either a badge was removed or "
        "this gate is now checking nothing"
    )


@pytest.mark.parametrize(
    "name",
    [p.name for p, _ in _headline_artifacts()] or ["<none>"],
)
def test_headline_claims_are_measured_on_the_shipped_version(name):
    shipped = _shipped_version()
    match = next((d for p, d in _headline_artifacts() if p.name == name), None)
    assert match is not None, f"{name} is no longer headline_eligible"

    measured = (match.get("environment") or {}).get("entroly_version")
    assert measured, f"{name} claims a headline without recording which version produced it"
    assert measured == shipped, (
        f"{name} advertises a public number measured on {measured} while "
        f"{shipped} ships. Re-run the benchmark, or drop headline_eligible "
        "until it is re-measured -- a provability claim cannot rest on a "
        "proof for code nobody is running."
    )


def test_headline_artifacts_state_their_limitations():
    # The badges strip all context down to a bare number, so the artifact
    # behind them has to carry the caveats a reader would otherwise never see.
    for path, payload in _headline_artifacts():
        assert payload.get("limitations"), (
            f"{path.name} is used as a public claim but records no limitations"
        )
        assert payload.get("claim_scope"), (
            f"{path.name} is used as a public claim but does not say what it claims"
        )
