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

__version__ = "0.19.6"

try:
    from .sdk import compress, compress_messages, verify  # noqa: F401
    from .sdk import detect_hallucination, optimize  # noqa: F401
except ImportError:
    pass  # Graceful degradation if dependencies missing

# engine_s6 — public file-localization API (used internally by SDK / MCP /
# proxy / CLI; also callable directly for advanced agent integrations).
try:
    from .file_localizer import localize_files, localize_fragments  # noqa: F401
except ImportError:
    pass

# Verification SDK: hallucination detection + suppression
try:
    from .verifiers import trace_provenance, forge_loop  # noqa: F401
except ImportError:
    pass  # Verifiers are available but don't block core functionality

try:
    from .witness import WitnessAnalyzer  # noqa: F401
except ImportError:
    pass
