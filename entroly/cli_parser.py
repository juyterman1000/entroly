"""Argument definitions for the public Entroly CLI.

Runtime dispatch and command handlers remain in :mod:`entroly.cli`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from .cli_parser_core import _add_core_commands
from .cli_parser_measurement import _add_lifecycle_and_measurement_commands
from .cli_parser_agents import _add_agent_and_governance_commands
from .cli_parser_runtime import _add_tools_and_runtime_commands


#: Printed at the end of ``entroly --help``.
#:
#: The one-time welcome in :func:`entroly.cli._check_first_run` writes this path
#: to **stderr** and never writes it again, while ``--help`` writes 68 commands
#: to stdout. A user who pipes stdout, redirects stderr, or just scrolls past
#: loses the getting-started path for good -- and that banner signs off with
#: "Run entroly --help anytime", which until now contained none of it. Measured
#: on a fresh install: second invocation emitted 0 bytes of guidance, and
#: ``--help`` matched none of "verify-claims", "simulate", "Step 1" or
#: "claude mcp add".
#:
#: Deliberately plain text: importing ``entroly.cli`` here for its colour class
#: would add a module-level edge into the import cycle that already contains
#: every shipped entry point.
GETTING_STARTED = """
getting started (local, no API key needed):
  entroly verify-claims   Step 1: verify SDK, indexing, optimization, recovery
  entroly simulate        Step 2: estimate savings on this repo, without an LLM

use it with Claude Code:
  claude mcp add entroly -- entroly

if something looks wrong:
  entroly doctor          Diagnose common issues
  entroly status          Check whether the server or proxy is running

Pass --query "<your task>" to simulate or demo to measure your own workload.
The built-in sample queries cover auth, error handling and module structure,
and may not match this repository.

Documentation: https://github.com/juyterman1000/entroly
"""


def build_cli_parser(
    *,
    version: str,
    wrap_agent_names: Callable[[], str],
    reason_options: tuple[tuple[str, str], ...],
    benefit_options: tuple[tuple[str, str], ...],
    surface_options: tuple[tuple[str, str], ...],
    duration_options: tuple[tuple[str, str], ...],
) -> argparse.ArgumentParser:
    """Create the same command tree used by the CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="entroly",
        description="\u26a1 Entroly \u2014 Information-theoretic context optimization for AI coding agents",
        epilog=GETTING_STARTED,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", "-V", action="version",
        version=f"entroly {version}",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_core_commands(subparsers)
    _add_lifecycle_and_measurement_commands(
        subparsers,
        reason_options=reason_options,
        benefit_options=benefit_options,
        surface_options=surface_options,
        duration_options=duration_options,
    )
    _add_agent_and_governance_commands(subparsers, wrap_agent_names)
    _add_tools_and_runtime_commands(subparsers)
    return parser
