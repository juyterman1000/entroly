"""Compatibility wrapper around the Rust-backed QCCR binding."""

from __future__ import annotations

from . import qccr_legacy as _legacy
from .qccr_hotfix import attach_sufficiency as _safe_attach_sufficiency

for _name, _value in vars(_legacy).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

# The legacy select() function resolves this name in its defining module.
_legacy._attach_sufficiency = _safe_attach_sufficiency
_attach_sufficiency = _safe_attach_sufficiency
