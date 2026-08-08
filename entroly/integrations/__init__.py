"""Entroly integrations package - drop-in middleware for agent frameworks."""

from __future__ import annotations

from .gateway import (  # noqa: F401
    CompressionGatewayClient,
    GatewayCompression,
    GatewayError,
    GatewayReceipt,
    wrap_anthropic,
    wrap_openai,
)

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
