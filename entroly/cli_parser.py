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
