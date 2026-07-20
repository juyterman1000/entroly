"""Entroly ContextEngine plugin for Hermes Agent.

Implements the Hermes ``ContextEngine`` ABC so that Hermes users can replace
the built-in ``ContextCompressor`` with Entroly's query-conditioned selection,
receipts, and verification.

Installation (Hermes side)::

    pip install -U entroly
    # Copy this directory to plugins/context_engine/entroly/  in your Hermes install
    # Then set config.yaml:
    #   context:
    #     engine: "entroly"

No hermes-agent dependency is required at install time.  When Hermes is
present, the real ABC is inherited; otherwise a compatible protocol stub
is used so the module remains importable for testing and SDK use.
"""

from .engine import EntrolyContextEngine

__all__ = ["EntrolyContextEngine"]
