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

__version__ = "1.0.6"

try:
    from .sdk import compress, compress_messages, verify  # noqa: F401
    from .sdk import detect_hallucination, optimize  # noqa: F401
    from .sdk import eicv_verify, eicv_suppress  # noqa: F401
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

# EICV — Evidence-Invariant Causal Verification.
# Deterministic hallucination detector. See benchmarks/results/ for the
# specific datasets, splits, and accuracy numbers achieved in our test
# runs. No claims of superiority over any specific external system.
try:
    from .eicv import EICVAnalyzer, EICVCertificate  # noqa: F401
    from .eicv_suppressor import EICVSuppressor  # noqa: F401
    from .esg import ESGAnalyzer, compute_tension  # noqa: F401
except ImportError:
    pass

# STAVE — Semantic Triplet Alignment Via Extraction.
# Binary-relational hallucination verifier. Novel: checks whether the
# *relationship* between entities in the answer matches knowledge, not
# just whether individual tokens appear. Wrong-slot gate: 100% precision.
# Already wired into WitnessAnalyzer (use_stave=True by default).
try:
    from .verifiers.stave import stave_verify, stave_risk  # noqa: F401
except ImportError:
    pass

# Local NLI — zero-API entailment using DeBERTa-v3-small (~80MB).
# Enable: WitnessAnalyzer(use_local_nli=True) or ENTROLY_LOCAL_NLI=1.
try:
    from .verifiers.local_nli import nli_score, is_available as nli_available  # noqa: F401
except ImportError:
    pass
