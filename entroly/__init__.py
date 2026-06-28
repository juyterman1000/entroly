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

__version__ = "1.0.30"

try:
    from .sdk import compress, compress_messages, verify  # noqa: F401
    from .sdk import detect_hallucination, optimize  # noqa: F401
    from .sdk import eicv_verify, eicv_suppress  # noqa: F401
    from .sdk import (  # noqa: F401
        context_receipt_from_path,
        create_context_receipt,
        explain_receipt_omission,
        render_context_receipt,
    )
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

# ── Breakthrough Inventions ──────────────────────────────────────────

# ESC — Entropic Shell Codec.
# Universal shell compressor: one algorithm replaces 95+ per-tool regex
# patterns.  Uses Shannon entropy scoring + structural classification +
# SimHash dedup + knapsack DP selection.  Works on ANY CLI tool output.
try:
    from .shell_codec import esc_compress, ESCResult  # noqa: F401
except ImportError:
    pass

# ELC — Evidence-Locked Compression.
# Compress around evidence, never through evidence: anchors, query matches,
# outliers, omitted-span receipts, and JSON schema+examples.
try:
    from .evidence_locked_compression import (  # noqa: F401
        CompressionReceipt,
        CompressionResult,
        OmittedSpan,
        compress_evidence_locked,
        compress_payload_messages,
        detect_heavy_content_type,
    )
except ImportError:
    pass

# Compression Proxy — provider-light request payload compression surface.
try:
    from .compression_proxy import (  # noqa: F401
        ProxyCompressionReceipt,
        ProxyCompressionResult,
        compress_proxy_payload,
        compress_proxy_payload_from_env,
    )
except ImportError:
    pass

# Compression Retrieval Store — local recoverability for omitted spans.
try:
    from .compression_retrieval_store import (  # noqa: F401
        CompressionRetrievalStore,
        StoredCompression,
        StoredSpan,
    )
except ImportError:
    pass

# Live HTTP proxy installer. This is inert unless
# ENTROLY_COMPRESSION_PROXY_MODE=elc is set before importing Entroly.
try:
    from .compression_proxy_live import install_live_compression_proxy  # noqa: F401

    install_live_compression_proxy()
except ImportError:
    pass

# SRP — Semantic Resolution Protocol.
# Information-optimal file reads: automatically selects per-block resolution
# (FULL/MEDIUM/LOW/SKIP) based on query relevance and token budget.
# Replaces fixed read modes with budget-driven optimization.
try:
    from .semantic_resolution import resolve as srp_resolve, SRPResult  # noqa: F401
except ImportError:
    pass

# CMWP — Causal Memory is handled by hippocampus LTM + Ebbinghaus decay.
# See entroly/long_term_memory.py and entroly/checkpoint.py.

# WVH — Witness-Verified Handoff.
# Multi-agent handoff protocol with built-in hallucination filtering.
# Strips contradicted claims before they propagate to downstream agents.
try:
    from .verified_handoff import handoff as wvh_handoff, HandoffBundle  # noqa: F401
except ImportError:
    pass

# ACF — Adversarial Context Firewall.
# End-to-end content security: prompt injection detection (20+ patterns),
# Unicode steganography detection, base64 payload detection, repetition
# flooding detection, cryptographic pipeline integrity verification.
try:
    from .context_firewall import scan as acf_scan, sanitize as acf_sanitize  # noqa: F401
    from .context_firewall import IntegrityChain  # noqa: F401
except ImportError:
    pass

# Cache Aligner — provider KV-cache prefix stabilizer (lever #3).
# Hashes injected context and stabilizes it across requests so Anthropic's
# 90% / OpenAI's 50% cached-prefix discounts actually hit. Documented in the
# README as `from entroly import CacheAligner`.
try:
    from .cache_aligner import CacheAligner  # noqa: F401
except ImportError:
    pass

# Control plane -- read-only request planning and transform compliance audit.
try:
    from .control_plane import (  # noqa: F401
        ControlAudit,
        ControlPlaneDecision,
        EntrolyCompressionDecision,
        audit_request_transform,
        canonical_json_dumps,
        plan_request,
        stable_request_fingerprint,
    )
except ImportError:
    pass

try:
    from .image_optimizer import (  # noqa: F401
        ImageOptimizationDecision,
        ImageTokenEstimate,
        estimate_image_tokens,
        estimate_image_tokens_from_dimensions,
        optimize_image_bytes,
        plan_image_optimization,
    )
except ImportError:
    pass

# Cost Cortex — price-aware budget clamp + recoverability ledger (the
# control-plane cost organ). Available to SDK/pip users directly:
#   from entroly import clamp_injected_budget, ProviderPrice, ContextLedger
try:
    from .cost_cortex import (  # noqa: F401
        ContextDecision,
        ContextLedger,
        CostBudget,
        Decision,
        ProviderPrice,
        clamp_injected_budget,
    )
except ImportError:
    pass

# Entroly Memory OS — public, dependency-free memory-control facade.
# Gives users a clean product surface while the native Rust memory stack is
# exposed more deeply over time.
try:
    from .memory import (  # noqa: F401
        MemoryContext,
        MemoryEntry,
        MemoryOS,
        MemorySafetyFinding,
        MemorySafetyResult,
        OmittedMemory,
        SelectedMemory,
    )
except ImportError:
    pass
