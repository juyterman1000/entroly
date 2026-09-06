"""No file may declare or pin a version older than the runtime master.

Runs `scripts/check_version_staleness.py` as a test so the sweep executes on
every pull request rather than only when someone remembers to run it.

The distinction the checker draws is the whole point, and it is worth restating
where a reader will find it: a version string must equal the master if it
**declares or pins the product as it is now**, and may be older if it **records
something that happened**. Release notes for 1.0.47 say 1.0.47 forever, a
benchmark that ran on 1.0.59 must keep saying so, and a lock file records
`serde 1.0.4` because that is serde's version. A sweep that ignores this finds
631 "stale" strings and is useless; a sweep that respects it found the 12 real
ones.

Do not silence a failure by adding a path to the archive list. Archives are
paths whose purpose is recording the past. If a live declaration is genuinely
about history, phrase it in past tense -- the checker recognises that too.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_version_staleness.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300,
    )


def test_no_declaration_is_older_than_the_master_version():
    result = _run()
    assert result.returncode == 0, (
        "version declarations older than the master version:\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_the_checker_is_actually_inspecting_the_tree():
    """Guard the guard.

    If the sweep stopped finding files -- a broken `git ls-files`, a changed
    layout -- it would report success while checking nothing. This asserts it
    still classified a substantial number of historical references, which it
    can only do by having read them.
    """
    result = _run("--list")
    assert result.returncode == 0, result.stdout
    assert "historical references checked" in result.stdout
    count = int(result.stdout.split("(")[-1].split(" historical")[0])
    assert count > 300, (
        f"only {count} references classified; the sweep is not reading the "
        f"repository and this gate would pass without checking anything"
    )


def test_a_stale_declaration_is_detected(tmp_path, monkeypatch):
    """The checker must fail on a real regression, not just pass on a clean tree.

    Exercised against a manifest that was genuinely left behind by a past bump:
    `.claude-plugin/plugin.json` sat one release behind the `manifest.json`
    directly beside it, which the bump script did rewrite.
    """
    target = REPO_ROOT / ".claude-plugin" / "plugin.json"
    original = target.read_text(encoding="utf-8")
    assert '"version"' in original

    import json
    document = json.loads(original)
    current = document["version"]
    major, minor, patch = (int(p) for p in current.split("."))
    stale = f"{major}.{minor}.{max(patch - 1, 0)}"

    try:
        target.write_text(original.replace(f'"{current}"', f'"{stale}"', 1), encoding="utf-8")
        result = _run()
        assert result.returncode != 0, (
            "the checker passed while .claude-plugin/plugin.json declared "
            f"{stale} against master {current}"
        )
        assert "plugin.json" in result.stdout, (
            f"the failure did not name the offending file:\n{result.stdout}"
        )
    finally:
        target.write_text(original, encoding="utf-8")
        assert target.read_text(encoding="utf-8") == original
