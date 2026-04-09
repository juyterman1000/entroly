"""
Entroly — Information-Theoretic Context Optimization for Agentic AI
========================================================================

An MCP server that mathematically optimizes what goes into an LLM's
context window. Uses knapsack dynamic programming, Shannon entropy scoring,
SimHash deduplication, and predictive pre-fetching to cut token costs by
50–70% while improving agent accuracy.

Quick Setup (Cursor)::

    Add to .cursor/mcp.json:
    {
      "mcpServers": {
        "entroly": {
          "command": "entroly"
        }
      }
    }

Quick Setup (Claude Code)::

    claude mcp add entroly -- entroly

"""

__version__ = "0.6.4"

# SDK: 3-line integration for any AI application
try:
    from .sdk import compress, compress_messages  # noqa: F401
except ImportError:
    pass  # Graceful degradation if dependencies missing
