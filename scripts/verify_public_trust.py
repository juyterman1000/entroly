#!/usr/bin/env python3
"""Compatibility entry point for Entroly public trust verification."""

from __future__ import annotations

try:
    from .verify_context_assurance_public import (
        PROMINENT_PUBLIC_FILES,
        _collect_prism_r_public_failures,
        _collect_stale_public_claim_failures,
        collect_offline_failures,
        collect_online_failures,
        collect_published_version_failures,
        main,
    )
except ImportError:  # Direct execution: python scripts/verify_public_trust.py
    from verify_context_assurance_public import (
        PROMINENT_PUBLIC_FILES,
        _collect_prism_r_public_failures,
        _collect_stale_public_claim_failures,
        collect_offline_failures,
        collect_online_failures,
        collect_published_version_failures,
        main,
    )

__all__ = [
    "PROMINENT_PUBLIC_FILES",
    "_collect_prism_r_public_failures",
    "_collect_stale_public_claim_failures",
    "collect_offline_failures",
    "collect_online_failures",
    "collect_published_version_failures",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
