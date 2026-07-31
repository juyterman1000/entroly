"""
Programmatic embedding example: run entroly's MCP server in-process.

Most users run entroly's MCP server via the CLI:

    entroly serve

This example shows how to embed and customize the same MCP server
programmatically — useful when you want to wrap entroly inside a larger
Python process (a test harness, a multi-tenant gateway, a custom
integration with a non-standard transport).

Usage:

    python -m entroly.integrate_entroly_mcp

    # Or import and call programmatically:
    from entroly.integrate_entroly_mcp import run_mcp_server
    run_mcp_server()

For the typical case (stdio MCP server in a separate process), just use
the CLI — it's strictly simpler. This file exists as a reference for the
embedding case.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("entroly.integrate_entroly_mcp")


def run_mcp_server() -> None:
    """Boot entroly's MCP server in this Python process.

    Uses the FastMCP transport over stdio by default — the standard
    MCP server protocol that Claude Code, Cursor, Windsurf, and other
    MCP-aware IDEs expect.
    """
    from entroly.server import create_mcp_server

    mcp, _engine = create_mcp_server()
    logger.info("Starting entroly MCP server (stdio transport)")
    mcp.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_mcp_server()
