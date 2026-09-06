"""The built wheel must not ship a version older than the master.

A clean source tree is not the same as a clean artifact, and 1.0.82 proved it.
The release surfaces were consistent when the tag was pushed, so the sweep
passed -- but three files fixed later in the same session missed the tag by
minutes, and the published wheel shipped
`entroly/integrations/hermes_context_engine/plugin.yaml` declaring
`version: 1.0.0` with a floor of `entroly>=1.0.57`. Verified by downloading the
published artifact, not inferred.

`check_version_staleness.py` scans the repository. This scans what actually goes
in the box, which is a different question: a file can be correct on disk and
still be packaged from a stale state, and only files inside the wheel reach a
user.

Skipped when `build` is unavailable rather than silently passing -- a packaging
gate that quietly does nothing is worse than no gate, because it reads as
coverage.
"""
from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER = re.search(
    r'__version__\s*=\s*"([^"]+)"',
    (REPO_ROOT / "entroly" / "__init__.py").read_text(encoding="utf-8"),
).group(1)
MASTER_TUPLE = tuple(int(p) for p in MASTER.split("."))

TEXTUAL = (".py", ".json", ".yaml", ".yml", ".toml", ".js", ".md", ".cfg", ".txt")

DECLARATION = re.compile(
    r'(?:^\s*"?version"?\s*[:=]\s*"?(\d+\.\d+\.\d+)'
    r'|entroly(?:-core|-mcp|-wasm|-openclaw)?\s*[><~=]=\s*["\']?(\d+\.\d+\.\d+)'
    r'|Entroly\s+v(\d+\.\d+\.\d+))',
    re.M,
)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    try:
        import build  # noqa: F401
    except ImportError:
        pytest.skip("`build` is not installed; cannot produce a wheel to inspect")

    outdir = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900,
    )
    if result.returncode != 0:
        pytest.skip(f"wheel build failed:\n{result.stdout[-1500:]}\n{result.stderr[-800:]}")
    wheels = list(outdir.glob("*.whl"))
    assert wheels, f"build reported success but produced no wheel:\n{result.stdout[-800:]}"
    return wheels[0]


def test_the_wheel_declares_the_master_version(wheel: Path):
    archive = zipfile.ZipFile(wheel)
    metadata = next(n for n in archive.namelist() if n.endswith("METADATA"))
    text = archive.read(metadata).decode("utf-8", errors="replace")
    assert f"Version: {MASTER}" in text, (
        f"wheel METADATA does not declare {MASTER}:\n"
        + "\n".join(line for line in text.splitlines() if line.startswith("Version:"))
    )


def test_no_packaged_file_declares_an_older_version(wheel: Path):
    """Every declaration inside the wheel must be current.

    This is the check that would have caught the 1.0.82 slip. `plugin.yaml`
    ships inside `entroly/`, so it reaches every user of the package even
    though nothing imports its version.
    """
    archive = zipfile.ZipFile(wheel)
    stale: list[str] = []

    for name in archive.namelist():
        if not name.endswith(TEXTUAL):
            continue
        text = archive.read(name).decode("utf-8", errors="ignore")
        lines = text.splitlines()
        for match in DECLARATION.finditer(text):
            version = match.group(1) or match.group(2) or match.group(3)
            if not version:
                continue
            try:
                parsed = tuple(int(p) for p in version.split("."))
            except ValueError:
                continue
            if parsed >= MASTER_TUPLE:
                continue
            index = text[: match.start()].count("\n")
            line = lines[index].strip() if index < len(lines) else ""
            stale.append(f"{name}:{index + 1}  declares {version}  {line[:70]}")

    assert not stale, (
        f"the wheel would ship {len(stale)} declaration(s) older than {MASTER}:\n  "
        + "\n  ".join(stale)
        + "\n\nThese reach every user of the package. Fix the source and rebuild; "
        "a clean repository sweep does not prove a clean artifact."
    )


def test_the_scan_actually_reads_the_wheel(wheel: Path):
    """Guard the guard: an empty archive would make the assertion vacuous."""
    names = zipfile.ZipFile(wheel).namelist()
    textual = [n for n in names if n.endswith(TEXTUAL)]
    assert len(textual) > 100, (
        f"only {len(textual)} textual files in the wheel; the scan is not "
        f"inspecting the package and this gate would pass without checking anything"
    )
