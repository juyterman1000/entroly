"""Public sufficiency surface with named, fail-closed calibration."""

from __future__ import annotations

from . import sufficiency_legacy as _legacy
from .sufficiency_hotfix import (
    CalibrationPolicy,
    SufficiencyCertificate,
    certify,
    certify_selection,
)

for _name, _value in vars(_legacy).items():
    if not _name.startswith("__") and _name not in {
        "SufficiencyCertificate",
        "certify",
        "certify_selection",
    }:
        globals()[_name] = _value
