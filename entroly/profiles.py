"""
Entroly Configuration Profiles
==============================

To ensure developers get the right behavior without needing to tune 20
different boolean flags, we expose three high-level profiles.
"""

from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "safe": {
        "compression_ratio": 0.50,     # Conservative — keep 50%
        "recovery_enabled": True,      # Always recover on verification failure
        "verification": "full",        # WITNESS + STAVE + BIPT
        "prism_learning": False,       # No weight adaptation
        "vault_coupling": False,       # No belief injection
        "model_routing": "fixed",      # Use caller's model
    },
    "balanced": {
        "compression_ratio": 0.30,     # Standard — keep 30%
        "recovery_enabled": True,
        "verification": "witness",     # WITNESS only (fast)
        "prism_learning": True,        # Learn from outcomes
        "vault_coupling": True,        # Beliefs enter context
        "model_routing": "adaptive",   # RAVS routes to cheapest capable
    },
    "max": {
        "compression_ratio": 0.15,     # Aggressive — keep 15%
        "recovery_enabled": True,
        "verification": "full",
        "prism_learning": True,
        "vault_coupling": True,
        "model_routing": "adaptive",
    },
}

def get_profile(name: str) -> dict[str, Any]:
    """Retrieve the settings for a given profile name."""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choose from: {list(PROFILES.keys())}")
    return PROFILES[name]
