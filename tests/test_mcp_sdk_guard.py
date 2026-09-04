"""The MCP entry points must report why the SDK is unusable, not guess.

Every MCP server factory used to guard its import with ``except ImportError``
and report "MCP SDK not installed. Install with: pip install mcp". Both halves
fail in the case operators actually hit:

* ``mcp`` 2.x removed ``mcp.server.fastmcp`` (FastMCP was renamed MCPServer), so
  the import raises ``ModuleNotFoundError`` while the SDK *is* installed -- the
  message asserts something false.
* ``pip install mcp`` resolves 2.x, so following the advice reinstalls the
  version that does not work. The remedy makes the failure permanent.

``pyproject.toml`` pins ``mcp>=1.28.1,<2``, so a normal install is unaffected.
These tests cover the environments that are not normal installs.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entroly import mcp_sdk  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_an_in_range_sdk_is_not_blamed_on_the_2x_rename(monkeypatch):
    """Three states, three causes -- do not assert a cause that cannot apply.

    When ``mcp`` is installed *within* the supported range and the submodule is
    still missing, the install is broken. Blaming the 2.x rename there is the
    same defect this module exists to fix, pointed the other way: it names a
    cause that did not happen and sends the operator to the wrong remedy.
    """
    monkeypatch.setattr(mcp_sdk, "installed_version", lambda: "1.29.1")
    message = mcp_sdk.unavailable_reason()

    assert "1.29.1" in message
    assert "not installed" not in message, message
    assert "2.x" not in message, (
        f"blamed the 2.x rename for an in-range install: {message}"
    )
    assert mcp_sdk.REQUIRED_SPEC in message


def test_absent_sdk_and_incompatible_sdk_get_different_messages(monkeypatch):
    monkeypatch.setattr(mcp_sdk, "installed_version", lambda: None)
    absent = mcp_sdk.unavailable_reason()

    monkeypatch.setattr(mcp_sdk, "installed_version", lambda: "2.1.1")
    incompatible = mcp_sdk.unavailable_reason()

    assert absent != incompatible
    assert "2.x" in incompatible, incompatible

    # The absent case may say "not installed"; the incompatible case may not,
    # because it is untrue and sends the operator down the wrong path.
    assert "not installed" in absent
    assert "not installed" not in incompatible
    assert "2.1.1" in incompatible, "the incompatible message must name what is installed"

    # Both must give advice that actually resolves the failure, which means the
    # pin -- never a bare `pip install mcp`, which resolves 2.x.
    for message in (absent, incompatible):
        assert mcp_sdk.REQUIRED_SPEC in message, message
        assert not re.search(r"pip install ['\"]?mcp['\"]?(?![>=<~])", message), (
            f"advice resolves to an incompatible major version: {message}"
        )


def _project_dependencies() -> list[str]:
    """Extract ``[project].dependencies`` without tomllib.

    ``requires-python`` is ``>=3.10`` and ``tomllib`` arrived in 3.11, so
    skipping when it is missing would make this gate vacuous on the minimum
    supported interpreter -- the one most likely to be running in CI's oldest
    matrix leg. The array is scanned directly instead. Scoping to the array
    matters: ``[project].keywords`` also contains a bare ``"mcp"``.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    start = text.index("dependencies = [")
    end = text.index("]", start)
    return re.findall(r'"([^"]+)"', text[start:end])


def test_required_spec_matches_the_pin_in_pyproject():
    """A hardcoded spec that drifts from the pin gives stale install advice."""
    deps = _project_dependencies()
    assert deps, "failed to parse [project].dependencies"
    pins = [d for d in deps if re.match(r"^mcp\b", d.strip())]

    assert len(pins) == 1, f"expected exactly one mcp pin, found {pins}"
    declared = pins[0].replace(" ", "")
    assert declared == mcp_sdk.REQUIRED_SPEC, (
        f"pyproject pins {declared!r} but mcp_sdk.REQUIRED_SPEC is "
        f"{mcp_sdk.REQUIRED_SPEC!r}; the install advice would be stale"
    )


def test_no_entry_point_reintroduces_the_misleading_advice():
    """Catch the next entry point that copies the old guard.

    Four factories carried the same wrong message. Fixing them once does not
    stop a fifth from being written, so assert the property over the tree
    instead of over the four files that happened to have it.
    """
    offenders: list[str] = []
    bare_advice = re.compile(r"pip install ['\"]?mcp['\"]?(?![>=<~\w-])")

    for path in (REPO_ROOT / "entroly").rglob("*.py"):
        if path.name == "mcp_sdk.py":
            continue  # documents the anti-pattern in prose, by design
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if bare_advice.search(line):
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "these lines advise `pip install mcp`, which resolves mcp 2.x and does "
        "not provide mcp.server.fastmcp; use entroly.mcp_sdk.load_fastmcp():\n  "
        + "\n  ".join(offenders)
    )


def test_load_fastmcp_raises_the_explained_reason(monkeypatch):
    """The loader must fail closed, carrying the accurate reason."""
    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name == "mcp.server.fastmcp":
            raise ModuleNotFoundError("No module named 'mcp.server.fastmcp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    monkeypatch.setattr(mcp_sdk, "installed_version", lambda: "2.1.1")

    with pytest.raises(RuntimeError) as excinfo:
        mcp_sdk.load_fastmcp()

    assert "2.1.1" in str(excinfo.value)
    assert mcp_sdk.REQUIRED_SPEC in str(excinfo.value)
