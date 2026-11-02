"""An assumption-based prefill energy proxy for input-token reduction.

Reducing input tokens removes prefill work. Prefill is the forward pass over
the prompt, and its cost is approximately ``2 * P * T`` floating-point
operations for ``P`` parameters and ``T`` tokens -- one multiply and one add
per parameter per token. This dense-model approximation omits attention and
serving effects; it does not establish net electricity savings.

Scope is deliberately narrow. This models prefill only, because prefill is what
input-token reduction may avoid. Decode, prompt-cache hits, retries, local
selection and verification, and shared-server utilization are not modeled.
Shortening a prompt can affect decode work and the generated response as well.

The result is modeled, not measured: Entroly runs locally and cannot observe a
provider's accelerators. Every input is therefore stated and overridable, and
the arithmetic is simple enough to check by hand. A reader who disagrees with an
assumption can substitute their own and recompute, which is the only honest way
to publish a number nobody can independently instrument.

Reported in kilowatt-hours. Carbon is intentionally not derived: converting kWh
to emissions requires a grid-intensity factor that varies by region, hour and
methodology, and inventing one would attach a contestable number to an
otherwise checkable one.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

# Illustrative assumptions, not a conservative bound on deployed hardware.
_DEFAULT_PARAMS_B = 70.0          # billions of parameters
_DEFAULT_PEAK_TFLOPS = 989.0      # BF16 dense, no sparsity
_DEFAULT_MFU = 0.40               # achieved fraction of peak during prefill
_DEFAULT_TDP_WATTS = 700.0

_FLOPS_PER_PARAM_PER_TOKEN = 2    # one multiply, one add
_JOULES_PER_KWH = 3_600_000.0


def _sig(value: float, digits: int = 12) -> float:
    """Round to significant figures, so small magnitudes survive.

    These quantities span many orders of magnitude -- a single request avoids
    microwatt-hours, a fleet-year avoids megawatt-hours -- and a fixed decimal
    count cannot serve both ends of that range.
    """
    if value == 0 or not math.isfinite(value):
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return round(value, -(exponent - digits + 1))


def _env_float(name: str, fallback: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    try:
        value = float(raw)
    except ValueError:
        return fallback
    return value if math.isfinite(value) and value > 0 else fallback


@dataclass(frozen=True)
class EnergyAssumptions:
    """The inputs a reader needs in order to disagree with the result."""

    model_params_billions: float = _DEFAULT_PARAMS_B
    accelerator_peak_tflops: float = _DEFAULT_PEAK_TFLOPS
    model_flops_utilization: float = _DEFAULT_MFU
    accelerator_tdp_watts: float = _DEFAULT_TDP_WATTS

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            try:
                valid = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0
                )
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite and positive")
        if self.model_flops_utilization > 1:
            raise ValueError("model_flops_utilization must be <= 1")

    @classmethod
    def from_env(cls) -> "EnergyAssumptions":
        """Assumptions with environment overrides applied.

        Overridable because the defaults cannot be right for everyone: a team
        running a 7B model on their own hardware and a team calling a frontier
        API differ by more than an order of magnitude, and a single baked-in
        number would be wrong for both.
        """
        return cls(
            model_params_billions=_env_float(
                "ENTROLY_ENERGY_MODEL_PARAMS_B", _DEFAULT_PARAMS_B),
            accelerator_peak_tflops=_env_float(
                "ENTROLY_ENERGY_PEAK_TFLOPS", _DEFAULT_PEAK_TFLOPS),
            model_flops_utilization=min(
                1.0, _env_float("ENTROLY_ENERGY_MFU", _DEFAULT_MFU)),
            accelerator_tdp_watts=_env_float(
                "ENTROLY_ENERGY_TDP_WATTS", _DEFAULT_TDP_WATTS),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_params_billions": self.model_params_billions,
            "accelerator_peak_tflops": self.accelerator_peak_tflops,
            "model_flops_utilization": self.model_flops_utilization,
            "accelerator_tdp_watts": self.accelerator_tdp_watts,
        }


def energy_for_tokens(
    tokens_saved: int, assumptions: EnergyAssumptions | None = None
) -> dict[str, Any]:
    """Model a prefill energy proxy for ``tokens_saved`` input tokens.

    Returns the intermediate quantities as well as the answer. A single kWh
    figure invites belief; showing the FLOPs and accelerator-seconds it came
    from invites checking, and this number exists to be checked.
    """
    cfg = assumptions or EnergyAssumptions.from_env()
    tokens = max(0, int(tokens_saved))

    params = cfg.model_params_billions * 1e9
    flops = _FLOPS_PER_PARAM_PER_TOKEN * params * tokens
    effective_flops_per_second = (
        cfg.accelerator_peak_tflops * 1e12 * cfg.model_flops_utilization
    )
    if not math.isfinite(effective_flops_per_second) or effective_flops_per_second <= 0:
        raise ValueError("effective prefill throughput must be finite and positive")
    seconds = flops / effective_flops_per_second
    kwh = seconds * cfg.accelerator_tdp_watts / _JOULES_PER_KWH
    if not all(math.isfinite(value) for value in (flops, seconds, kwh)):
        raise ValueError("energy estimate exceeds finite arithmetic range")
    if tokens and any(value <= 0 for value in (flops, seconds, kwh)):
        raise ValueError("energy estimate underflows finite arithmetic range")

    # Rounded to significant figures rather than decimal places. A fixed
    # decimal count silently zeroes small results -- a thousand tokens is
    # ~7e-5 kWh, which six decimals distorts and eight would still degrade --
    # and callers summing many periods would accumulate that error. Display
    # code rounds for the reader; the payload keeps the value.
    return {
        "tokens_saved": tokens,
        "petaflops_avoided": _sig(flops / 1e15),
        "accelerator_seconds_avoided": _sig(seconds),
        "kwh_avoided": _sig(kwh),
        "basis": "prefill_only",
        "measured": False,
        "assumptions": cfg.as_dict(),
        "method": (
            "prefill FLOPs = 2 * parameters * tokens; accelerator-seconds = "
            "FLOPs / (peak_tflops * MFU); kWh = seconds * TDP / 3.6e6. Prefill "
            "proxy only; attention, decode, prompt caching, retries, local "
            "processing and serving utilization are not modeled. TDP is an "
            "assumed power input, not measured draw. This is not a measurement "
            "of provider electricity or demonstrated net energy savings."
        ),
    }


__all__ = [
    "EnergyAssumptions",
    "energy_for_tokens",
]
