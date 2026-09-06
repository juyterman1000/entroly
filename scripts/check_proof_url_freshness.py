"""Verify that every recorded external listing still mentions Entroly.

`docs/distribution/targets.json` records a `proof_url` for each target that
reached `submitted`, `published` or `rejected`, and `check_distribution_surface`
requires that URL to exist. Nothing ever rechecked it. An upstream maintainer
can drop the entry, revert the PR, or restructure the list, and the registry
goes on asserting a listing that is no longer there.

A plain reachability check does not help, and would be worse than nothing here.
Every recorded proof URL points at a README in someone else's repository:

    https://github.com/tensorchord/Awesome-LLMOps/blob/main/README.md

That page returns 200 whether or not Entroly appears anywhere in it. A link
checker would stay green through exactly the failure it is supposed to catch.

So this fetches the *content* and looks for the project name. The claim being
tested is "our entry is still in that document", which is the claim the registry
actually makes.

Exit codes are deliberately three-way, because "the listing is gone" and "GitHub
timed out" call for different reactions and collapsing them trains people to
ignore the check:

    0  every listing was found
    1  a document was fetched and does not mention Entroly -- act on this
    2  a document could not be fetched -- inconclusive, not a verdict

Usage:
    python scripts/check_proof_url_freshness.py
    python scripts/check_proof_url_freshness.py --offline   # parse only, no network
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "docs" / "distribution" / "targets.json"

# Matched case-insensitively against the fetched document.
NEEDLE = "entroly"

# Statuses whose proof_url asserts a live external listing. A rejected target
# points at a decision thread, which stays readable but need not name the
# project in the rendered text, so it is recorded and not asserted on.
ASSERTED_STATUSES = {"submitted", "published"}

_TIMEOUT_SECONDS = 30
_USER_AGENT = "entroly-distribution-freshness-check"


def _raw_url(url: str) -> str:
    """Prefer raw text over rendered HTML for GitHub blob links.

    The rendered page splits long lines across markup, so a substring search can
    miss a name that is plainly present. The raw file is what the listing
    actually says.
    """
    match = re.match(
        r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$", url
    )
    if match:
        owner, repo, ref, path = match.groups()
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    return url


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="replace")


def _targets() -> list[dict]:
    document = json.loads(TARGETS.read_text(encoding="utf-8"))
    return document["targets"] if isinstance(document, dict) else document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true",
        help="parse the registry and report what would be checked, without network",
    )
    args = parser.parse_args()

    checkable = [
        t for t in _targets()
        if t.get("proof_url") and t.get("status") in ASSERTED_STATUSES
    ]

    if not checkable:
        print("No target claims a live external listing; nothing to verify.")
        return 0

    if args.offline:
        print(f"Would verify {len(checkable)} listing(s):")
        for target in checkable:
            print(f"  {target['id']}: {target['proof_url']}")
        return 0

    missing: list[str] = []
    unreachable: list[str] = []

    for target in checkable:
        identifier = target["id"]
        url = target["proof_url"]
        try:
            body = _fetch(_raw_url(url))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            unreachable.append(f"{identifier}: {url}\n      {type(exc).__name__}: {exc}")
            print(f"  ?  {identifier}: could not fetch ({type(exc).__name__})")
            continue

        if NEEDLE in body.lower():
            print(f"  OK {identifier}: listing present")
        else:
            missing.append(
                f"{identifier}: {url}\n"
                f"      fetched {len(body)} chars, no occurrence of {NEEDLE!r}"
            )
            print(f"  !! {identifier}: LISTING GONE")

    if missing:
        print(
            f"\n{len(missing)} recorded listing(s) no longer mention Entroly.\n"
            "The registry is asserting external proof that is not there:\n"
        )
        for entry in missing:
            print(f"    {entry}")
        print(
            "\nEither the entry was removed upstream (set the target back and "
            "record why) or the proof_url is wrong."
        )
        return 1

    if unreachable:
        print(
            f"\n{len(unreachable)} listing(s) could not be fetched. This is "
            "inconclusive, not a failure:\n"
        )
        for entry in unreachable:
            print(f"    {entry}")
        return 2

    print(f"\nAll {len(checkable)} recorded listings still mention Entroly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
