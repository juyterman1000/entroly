"""
Entroly Proxy Configuration
============================

Configuration for the prompt compiler proxy.
All settings have sensible defaults and can be overridden via environment variables.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Model name → context window size (tokens)
MODEL_CONTEXT_WINDOWS = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-pro": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # Anthropic
    "claude-opus-4-6": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-haiku-20240307": 200_000,
}

_DEFAULT_CONTEXT_WINDOW = 128_000


def context_window_for_model(model: str) -> int:
    """Look up context window size for a model name, with fuzzy prefix matching."""
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]
    # Fuzzy: match by prefix (e.g. "gpt-4o-2024-08-06" matches "gpt-4o")
    for prefix, size in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(prefix):
            return size
    return _DEFAULT_CONTEXT_WINDOW


@dataclass
class ProxyConfig:
    """Configuration for the entroly prompt compiler proxy."""

    port: int = 9377
    host: str = "127.0.0.1"

    openai_base_url: str = "https://api.openai.com"
    anthropic_base_url: str = "https://api.anthropic.com"

    # Fraction of model context window to use for injected context (0.0-1.0)
    context_fraction: float = 0.15

    enable_query_refinement: bool = True
    enable_ltm: bool = True
    enable_security_scan: bool = True
    enable_temperature_calibration: bool = True
    enable_trajectory_convergence: bool = True

    # EGTC v2 coefficients (overridable by autotune daemon via tuning_config.json)
    fisher_scale: float = 0.55
    trajectory_c_min: float = 0.6
    trajectory_lambda: float = 0.07

    @classmethod
    def from_env(cls) -> ProxyConfig:
        """Create config from environment variables, with tuning_config.json overlay."""
        config = cls(
            port=int(os.environ.get("ENTROLY_PROXY_PORT", "9377")),
            host=os.environ.get("ENTROLY_PROXY_HOST", "127.0.0.1"),
            openai_base_url=os.environ.get(
                "ENTROLY_OPENAI_BASE", "https://api.openai.com"
            ),
            anthropic_base_url=os.environ.get(
                "ENTROLY_ANTHROPIC_BASE", "https://api.anthropic.com"
            ),
            context_fraction=float(
                os.environ.get("ENTROLY_CONTEXT_FRACTION", "0.15")
            ),
            enable_temperature_calibration=(
                os.environ.get("ENTROLY_TEMPERATURE_CALIBRATION", "1") != "0"
            ),
            enable_trajectory_convergence=(
                os.environ.get("ENTROLY_TRAJECTORY_CONVERGENCE", "1") != "0"
            ),
            fisher_scale=float(
                os.environ.get("ENTROLY_FISHER_SCALE", "0.55")
            ),
            trajectory_c_min=float(
                os.environ.get("ENTROLY_TRAJECTORY_CMIN", "0.6")
            ),
            trajectory_lambda=float(
                os.environ.get("ENTROLY_TRAJECTORY_LAMBDA", "0.07")
            ),
        )
        # Overlay EGTC coefficients from tuning_config.json (written by autotune)
        config._load_egtc_from_tuning_config()
        return config

    def _load_egtc_from_tuning_config(self) -> None:
        """Load EGTC coefficients from tuning_config.json if present."""
        tc_path = Path(__file__).parent / "tuning_config.json"
        if not tc_path.exists():
            return
        try:
            with open(tc_path) as f:
                tc = json.load(f)
            egtc = tc.get("egtc", {})
            if not egtc:
                return
            if "fisher_scale" in egtc:
                self.fisher_scale = float(egtc["fisher_scale"])
            if "trajectory_c_min" in egtc:
                self.trajectory_c_min = float(egtc["trajectory_c_min"])
            if "trajectory_lambda" in egtc:
                self.trajectory_lambda = float(egtc["trajectory_lambda"])
            logging.getLogger("entroly.proxy").debug(
                f"EGTC coefficients loaded from tuning_config.json: {egtc}"
            )
        except Exception:
            pass  # non-critical: fall back to defaults
