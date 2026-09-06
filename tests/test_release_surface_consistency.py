"""Every release surface must agree with the master version.

Version drift is silent and only surfaces downstream, usually as a published
artifact that does not exist. Observed on this repository: the codebase read
1.0.67 across thirteen surfaces while the Homebrew formula still pointed at
1.0.64, the OpenClaw install docs told users `pip install "entroly>=1.0.65"`,
and no `entroly-v1.0.67` tag existed at all -- so PyPI, npm, ClawHub and GitHub
Releases were all still serving 1.0.66. The first sign of trouble was a ClawHub
error, three surfaces and one release cycle later.

`entroly/__init__.py` is the master. Anything that pins a version for a user --
package manifests, install instructions -- must match it.

Two categories are deliberately excluded, because changing them would be wrong
rather than merely unnecessary:

  * historical records (`docs/releases/*`, benchmark evidence matrices) state
    which version something was measured or released at; rewriting them would
    falsify provenance;
  * the Homebrew formula must reference an sdist that actually exists on PyPI,
    so it legitimately trails the repository until a release publishes.
"""

from __future__ import annotations

import io
import json
import re
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _master_version() -> str:
    text = (ROOT / "entroly" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([0-9]+\.[0-9]+\.[0-9]+)["\']', text)
    assert match, "entroly/__init__.py must define __version__"
    return match.group(1)


MASTER = _master_version()

JSON_MANIFESTS = (
    ".claude-plugin/plugin.json",
    "entroly/npm/package.json",
    "entroly/npm-alias/package.json",
    "entroly-wasm/package.json",
    "integrations/codex/entroly/.codex-plugin/plugin.json",
    "integrations/codex/entroly/entroly-bundle.json",
    "integrations/codex/entroly/skills/entroly-evidence-operations/entroly-bundle.json",
    "integrations/gemini/entroly/gemini-extension.json",
    "integrations/gemini/entroly/entroly-bundle.json",
    "integrations/openclaw/package.json",
    "server.json",
    "skills/entroly-evidence-operations/entroly-bundle.json",
)

TOML_MANIFESTS = (
    "pyproject.toml",
    "entroly/pyproject.toml",
    "entroly-core/pyproject.toml",
    "entroly-core/Cargo.toml",
    "entroly-qccr/Cargo.toml",
    "entroly-wasm/Cargo.toml",
)


@pytest.mark.parametrize("relative", JSON_MANIFESTS)
def test_json_manifest_matches_master_version(relative: str) -> None:
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} not present in this checkout")
    version = json.loads(path.read_text(encoding="utf-8")).get("version")
    assert version == MASTER, (
        f"{relative} is {version!r}, entroly/__init__.py is {MASTER!r}"
    )


@pytest.mark.parametrize("relative", TOML_MANIFESTS)
def test_toml_manifest_matches_master_version(relative: str) -> None:
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} not present in this checkout")
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*["\']([0-9]+\.[0-9]+\.[0-9]+)["\']', text, re.M)
    assert match, f"{relative} has no top-level version"
    assert match.group(1) == MASTER, (
        f"{relative} is {match.group(1)!r}, entroly/__init__.py is {MASTER!r}"
    )


def _code_without_comments(path: Path) -> str:
    """Return `path`'s source with comment tokens removed.

    Scanning raw text conflates two different things: a version *constant*,
    which must track the master version, and a version *mentioned in prose*,
    which records what was true of some earlier release.

    This module's own docstring already draws that line for files -- historical
    records "state which version something was measured or released at;
    rewriting them would falsify provenance". The same applies inside a file.
    `native_status.py` explains that published entroly-core 1.0.78 reports
    `version_ok` while predating `WorkGraph`; rewriting that to 1.0.79 would
    make the comment false and lose the reason the gate exists.

    Tokenizing rather than stripping from the first `#` keeps a `#` inside a
    string literal intact, so a version in real code is still caught.
    """
    source = path.read_text(encoding="utf-8")
    kept: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.COMMENT:
                kept.append(token.string)
    except (tokenize.TokenError, IndentationError):  # pragma: no cover - defensive
        return source
    return "\n".join(kept)


def test_native_status_matches_master_version() -> None:
    """The native gate's version constants must track the master version.

    Comments are excluded deliberately -- see `_code_without_comments`. String
    literals are not: a user-facing install hint such as
    `Install entroly-core>=X` is a surface someone acts on, so it stays in scope.
    """
    path = ROOT / "entroly" / "native_status.py"
    found = set(re.findall(r"1\.0\.[0-9]+", _code_without_comments(path)))
    stale = {v for v in found if v != MASTER}
    assert not stale, (
        f"entroly/native_status.py references {sorted(stale)} in code, master is {MASTER}"
    )


def test_install_instructions_do_not_pin_an_older_version() -> None:
    """Docs must not tell users to install a version older than we ship.

    Caught `pip install "entroly>=1.0.65"` in the OpenClaw README two releases
    after 1.0.65, which silently steers users onto a stale floor.
    """
    pattern = re.compile(r'entroly[>=~]+=?["\']?(\d+\.\d+\.\d+)')
    offenders: list[str] = []
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT).as_posix()
        # `.entroly/` is the engine's own runtime state -- vault beliefs and
        # ledger objects it wrote while indexing this repository. It quotes
        # source text, so it trips this check without being a surface anyone
        # installs from.
        if any(part in rel for part in ("node_modules", "docs/releases/",
                                        "docs/research", "benchmarks/results",
                                        ".entroly/", ".git/")):
            continue
        for pinned in pattern.findall(path.read_text(encoding="utf-8", errors="ignore")):
            if tuple(map(int, pinned.split("."))) < tuple(map(int, MASTER.split("."))):
                offenders.append(f"{rel}: pins {pinned}, master is {MASTER}")
    assert not offenders, "stale install pins:\n  " + "\n  ".join(offenders)
