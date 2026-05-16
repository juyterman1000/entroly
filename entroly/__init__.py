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

__version__ = "0.19.4"

try:
    from .sdk import compress, compress_messages, verify  # noqa: F401
except ImportError:
    pass  # Graceful degradation if dependencies missing

# Verification SDK: hallucination detection + suppression
try:
    from .verifiers import trace_provenance, forge_loop  # noqa: F401
except ImportError:
    pass  # Verifiers are available but don't block core functionality

try:
    from .witness import WitnessAnalyzer  # noqa: F401
except ImportError:
    pass
