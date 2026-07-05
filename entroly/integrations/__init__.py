"""Entroly integrations package - drop-in middleware for agent frameworks."""

from __future__ import annotations

try:
    from .ebbiforge import (  # noqa: F401
        EbbiforgeAuditResult,
        EbbiforgeEntrolyBridge,
        EbbiforgeProvenanceTurn,
        run_swarm_with_entroly,
        summarize_ebbiforge_anomalies,
    )
except ImportError:
    pass
