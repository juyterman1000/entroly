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


# ══════════════════════════════════════════════════════════════════════
# Single-Dial Quality Profiles
# ══════════════════════════════════════════════════════════════════════
#
# Single-knob control: one `quality` parameter [0, 1] that auto-derives
# all tuning parameters via Pareto-front interpolation.
#
# quality=0.0 → "speed":   minimal budget, aggressive pruning, fast
# quality=0.5 → "balanced": current defaults
# quality=1.0 → "quality":  generous budget, full diversity, thorough
#
# For each numeric param p:
#   p(q) = p_speed × (1 - q) + p_quality × q
#
# Boolean features are enabled above a threshold:
#   feature(q) = q >= threshold
#
# Profiles are defined as dicts for easy extension and autotune overlay.
# ══════════════════════════════════════════════════════════════════════

_PROFILE_SPEED = {
    "context_fraction": 0.08,
    "ecdb_min_budget": 200,
    "ecdb_max_fraction": 0.15,
    "ecdb_sigmoid_steepness": 4.0,    # aggressive budget scaling
    "ecdb_sigmoid_base": 0.3,
    "ecdb_sigmoid_range": 1.0,
    "ecdb_codebase_divisor": 400.0,   # less codebase scaling
    "ecdb_codebase_cap": 1.5,
    "ios_skeleton_info_factor": 0.50,  # skeletons carry less info (aggressive)
    "ios_reference_info_factor": 0.10,
    "ios_diversity_floor": 0.05,
    "fisher_scale": 0.45,
    "trajectory_c_min": 0.5,          # faster convergence
    "trajectory_lambda": 0.10,
    "egtc_alpha": 1.2,
    "egtc_gamma": 1.5,
    "egtc_epsilon": 0.6,
    # Thresholds for features (enabled when quality >= threshold)
    "_thresh_multi_resolution": 0.3,
    "_thresh_diversity": 0.2,
    "_thresh_hierarchical": 0.4,
    "_thresh_ltm": 0.3,
    "_thresh_trajectory": 0.3,
}

_PROFILE_QUALITY = {
    "context_fraction": 0.25,
    "ecdb_min_budget": 800,
    "ecdb_max_fraction": 0.45,
    "ecdb_sigmoid_steepness": 2.0,    # gentler budget scaling
    "ecdb_sigmoid_base": 0.6,
    "ecdb_sigmoid_range": 2.0,
    "ecdb_codebase_divisor": 100.0,   # more codebase scaling
    "ecdb_codebase_cap": 3.0,
    "ios_skeleton_info_factor": 0.80,  # skeletons carry more info
    "ios_reference_info_factor": 0.25,
    "ios_diversity_floor": 0.15,
    "fisher_scale": 0.65,
    "trajectory_c_min": 0.7,          # gentler convergence
    "trajectory_lambda": 0.05,
    "egtc_alpha": 2.0,
    "egtc_gamma": 1.0,
    "egtc_epsilon": 0.4,
    "_thresh_multi_resolution": 0.3,
    "_thresh_diversity": 0.2,
    "_thresh_hierarchical": 0.4,
    "_thresh_ltm": 0.3,
    "_thresh_trajectory": 0.3,
}


# Named quality presets for human-friendly --quality flag
QUALITY_PRESETS = {
    "speed": 0.0,
    "fast": 0.25,
    "balanced": 0.5,
    "quality": 0.8,
    "max": 1.0,
}


def resolve_quality(value: str) -> float:
    """Accept either a named preset or a float 0.0-1.0."""
    if value in QUALITY_PRESETS:
        return QUALITY_PRESETS[value]
    try:
        q = float(value)
        if 0.0 <= q <= 1.0:
            return q
        raise ValueError(f"quality must be 0.0-1.0, got {q}")
    except ValueError as e:
        if "could not convert" in str(e) or "quality must be" in str(e):
            valid = ", ".join(QUALITY_PRESETS.keys())
            raise ValueError(
                f"Unknown quality preset '{value}'. Valid: {valid} or 0.0-1.0"
            ) from None
        raise


def _interpolate_profiles(quality: float) -> dict:
    """Linearly interpolate between speed and quality profiles."""
    q = max(0.0, min(1.0, quality))
    result = {}
    for key in _PROFILE_SPEED:
        if key.startswith("_"):
            continue
        s = _PROFILE_SPEED[key]
        qv = _PROFILE_QUALITY[key]
        if isinstance(s, int) and isinstance(qv, int):
            result[key] = int(s * (1 - q) + qv * q)
        else:
            result[key] = round(float(s) * (1 - q) + float(qv) * q, 6)
    return result


@dataclass
class ProxyConfig:
    """Configuration for the entroly prompt compiler proxy.

    Supports two configuration modes:
      1. Explicit: set each parameter individually via env vars
      2. Single-dial: set ENTROLY_QUALITY=[0,1] to auto-derive all params
    """

    port: int = 9377
    host: str = "127.0.0.1"

    openai_base_url: str = "https://api.openai.com"
    anthropic_base_url: str = "https://api.anthropic.com"

    # Single-dial quality knob: None = use explicit params, 0-1 = auto-derive
    quality: float | None = None

    # Fraction of model context window to use for injected context (0.0-1.0)
    context_fraction: float = 0.15

    enable_query_refinement: bool = True
    enable_ltm: bool = True
    enable_security_scan: bool = True
    enable_temperature_calibration: bool = True
    enable_trajectory_convergence: bool = True
    enable_prompt_directives: bool = True
    enable_hierarchical_compression: bool = True
    enable_conversation_compression: bool = True

    # Context window size (auto-detected per model, this is the fallback)
    context_window: int = 128_000

    # IOS: Information-Optimal Selection
    enable_ios: bool = True
    enable_ios_diversity: bool = True
    enable_ios_multi_resolution: bool = True

    # ECDB: Entropy-Calibrated Dynamic Budget
    enable_dynamic_budget: bool = True
    ecdb_min_budget: int = 500
    ecdb_max_fraction: float = 0.30
    ecdb_sigmoid_steepness: float = 3.0
    ecdb_sigmoid_base: float = 0.5
    ecdb_sigmoid_range: float = 1.5
    ecdb_codebase_divisor: float = 200.0
    ecdb_codebase_cap: float = 2.0

    # IOS: tunable info factors and diversity floor
    ios_skeleton_info_factor: float = 0.70
    ios_reference_info_factor: float = 0.15
    ios_diversity_floor: float = 0.10

    # EGTC v2 coefficients (overridable by autotune daemon via tuning_config.json)
    fisher_scale: float = 0.55
    egtc_alpha: float = 1.6       # vagueness coefficient
    egtc_gamma: float = 1.2       # sufficiency coefficient
    egtc_epsilon: float = 0.5     # dispersion coefficient
    trajectory_c_min: float = 0.6
    trajectory_lambda: float = 0.07

    @classmethod
    def from_env(cls) -> ProxyConfig:
        """Create config from environment variables, with tuning_config.json overlay.

        Supports single-dial mode: set ENTROLY_QUALITY=0.0–1.0 to auto-derive
        all numeric params from Pareto-interpolated profiles.
        """
        quality_env = os.environ.get("ENTROLY_QUALITY")
        quality = float(quality_env) if quality_env else None

        config = cls(
            port=int(os.environ.get("ENTROLY_PROXY_PORT", "9377")),
            host=os.environ.get("ENTROLY_PROXY_HOST", "127.0.0.1"),
            quality=quality,
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
            enable_conversation_compression=(
                os.environ.get("ENTROLY_CONVERSATION_COMPRESSION", "1") != "0"
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

        # Single-dial mode: auto-derive params from quality knob
        if quality is not None:
            config._apply_quality_dial(quality)

        # Overlay tunable coefficients from tuning_config.json (written by autotune)
        config._load_tuned_coefficients()
        return config

    def _apply_quality_dial(self, quality: float) -> None:
        """Apply single-dial quality interpolation.

        Derives all numeric tuning parameters from the quality knob
        via linear interpolation between speed (q=0) and quality (q=1)
        profiles on the Pareto front of the speed-accuracy tradeoff.

        Boolean features are enabled when quality >= their threshold.
        """
        q = max(0.0, min(1.0, quality))
        params = _interpolate_profiles(q)

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Boolean features: enabled above threshold
        self.enable_ios_multi_resolution = q >= _PROFILE_SPEED.get("_thresh_multi_resolution", 0.3)
        self.enable_ios_diversity = q >= _PROFILE_SPEED.get("_thresh_diversity", 0.2)
        self.enable_hierarchical_compression = q >= _PROFILE_SPEED.get("_thresh_hierarchical", 0.4)
        self.enable_ltm = q >= _PROFILE_SPEED.get("_thresh_ltm", 0.3)
        self.enable_trajectory_convergence = q >= _PROFILE_SPEED.get("_thresh_trajectory", 0.3)

        logger = logging.getLogger("entroly.proxy")
        logger.info(f"Single-dial quality={q:.2f}: context_fraction={self.context_fraction:.3f}, "
                     f"ecdb_min_budget={self.ecdb_min_budget}, diversity={self.enable_ios_diversity}")

    def _load_tuned_coefficients(self) -> None:
        """Load tunable coefficients from tuning_config.json if present.

        Overlays EGTC, IOS, and ECDB params from the autotune-managed config.
        Each param falls back to the dataclass default if absent.
        """
        tc_path = Path(__file__).parent / "tuning_config.json"
        if not tc_path.exists():
            return
        try:
            with open(tc_path) as f:
                tc = json.load(f)
        except Exception:
            return  # non-critical

        logger = logging.getLogger("entroly.proxy")

        # EGTC coefficients
        egtc = tc.get("egtc", {})
        if egtc:
            for key, attr in (
                ("fisher_scale", "fisher_scale"),
                ("alpha", "egtc_alpha"),
                ("gamma", "egtc_gamma"),
                ("epsilon", "egtc_epsilon"),
                ("trajectory_c_min", "trajectory_c_min"),
                ("trajectory_lambda", "trajectory_lambda"),
            ):
                if key in egtc:
                    setattr(self, attr, float(egtc[key]))
            logger.debug(f"EGTC coefficients from tuning_config.json: {egtc}")

        # IOS coefficients
        ios = tc.get("ios", {})
        if ios:
            if "skeleton_info_factor" in ios:
                self.ios_skeleton_info_factor = float(ios["skeleton_info_factor"])
            if "reference_info_factor" in ios:
                self.ios_reference_info_factor = float(ios["reference_info_factor"])
            if "diversity_floor" in ios:
                self.ios_diversity_floor = float(ios["diversity_floor"])
            logger.debug(f"IOS coefficients from tuning_config.json: {ios}")

        # ECDB coefficients
        ecdb = tc.get("ecdb", {})
        if ecdb:
            for key in (
                "min_budget", "max_fraction", "sigmoid_steepness",
                "sigmoid_base", "sigmoid_range",
                "codebase_divisor", "codebase_cap",
            ):
                attr = f"ecdb_{key}"
                if key in ecdb and hasattr(self, attr):
                    val = ecdb[key]
                    setattr(self, attr, int(val) if key == "min_budget" else float(val))
            logger.debug(f"ECDB coefficients from tuning_config.json: {ecdb}")
