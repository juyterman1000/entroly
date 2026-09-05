"""Load the MCP server class, and explain accurately when it cannot be loaded.

Every MCP entry point guarded ``from mcp.server.fastmcp import FastMCP`` with
``except ImportError`` and reported "MCP SDK not installed. Install with: pip
install mcp". That message is wrong in the case operators are most likely to
hit, and its advice makes the failure permanent:

``mcp`` 2.x renamed ``FastMCP`` to ``MCPServer`` and dropped the
``mcp.server.fastmcp`` module, so importing it raises ``ModuleNotFoundError``
(an ``ImportError`` subclass) even though the SDK *is* installed. The operator
is then told to run ``pip install mcp``, which installs 2.x -- reproducing the
exact failure they were trying to fix.

``pyproject.toml`` pins ``mcp>=1.28.1,<2``, so a normal ``pip install entroly``
resolves a working SDK. This module exists for everyone else: an environment
where ``mcp`` was installed separately or upgraded past the pin. Failing closed
is correct; failing closed with a false reason is not.

This module deliberately imports nothing from ``entroly``. The package has an
import cycle spanning ``entroly/__init__`` and the proxy stack, and a leaf with
no internal imports cannot join it.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

#: The pin declared in ``pyproject.toml``. ``test_mcp_sdk_guard`` asserts the
#: two agree, so this string cannot drift into giving stale install advice.
REQUIRED_SPEC = "mcp>=1.28.1,<2"


def installed_version() -> str | None:
    """Return the installed ``mcp`` version, or ``None`` when absent."""
    try:
        return version("mcp")
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - metadata backends vary
        return None


def _major(version: str) -> int | None:
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, AttributeError):
        return None


def unavailable_reason() -> str:
    """Explain why the MCP SDK cannot be used, distinguishing the causes.

    Three distinct states, three messages. Naming the 2.x rename
    unconditionally would be the same defect this module exists to fix: when
    ``mcp`` is installed at a supported version and the submodule is still
    missing, the cause is a broken or partial install, and blaming a rename
    that did not happen sends the operator to the wrong remedy.
    """
    found = installed_version()
    if found is None:
        return (
            f"MCP SDK not installed. Install with: pip install '{REQUIRED_SPEC}'"
        )

    major = _major(found)
    if major is not None and major >= 2:
        return (
            f"MCP SDK {found} is installed but does not provide "
            f"'mcp.server.fastmcp'. Entroly requires {REQUIRED_SPEC}; mcp 2.x "
            f"renamed FastMCP to MCPServer. "
            f"Fix with: pip install '{REQUIRED_SPEC}'"
        )

    return (
        f"MCP SDK {found} is installed and within {REQUIRED_SPEC}, but "
        f"'mcp.server.fastmcp' could not be imported, so the install is "
        f"incomplete or broken. "
        f"Reinstall with: pip install --force-reinstall '{REQUIRED_SPEC}'"
    )


def load_fastmcp():
    """Return the ``FastMCP`` class, or raise ``RuntimeError`` with the reason."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise RuntimeError(unavailable_reason()) from None
    return FastMCP


def load_context():
    """Return the MCP ``Context`` class, or ``None`` when the SDK is unusable.

    Used for type annotations at import time, where failing closed would break
    importing the module at all. The server's ``create_mcp_server`` reports the
    real error when a server is actually requested.
    """
    try:
        from mcp.server.fastmcp import Context

        return Context
    except ImportError:
        return None
