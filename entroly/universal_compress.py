"""Compatibility wrapper for the universal compressor trust hotfix."""

from __future__ import annotations

from . import universal_compress_legacy as _legacy
from .universal_compress_hotfix import (
    _compress_log_universal as _safe_compress_log_universal,
    _is_load_bearing_key,
    _json_to_schema as _safe_json_to_schema,
    _log_template as _safe_log_template,
    _normalise_json_key,
)

for _name, _value in vars(_legacy).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_legacy._json_to_schema = _safe_json_to_schema
_legacy._log_template = _safe_log_template
_legacy._compress_log_universal = _safe_compress_log_universal
_legacy._is_load_bearing_key = _is_load_bearing_key
_legacy._normalise_json_key = _normalise_json_key

_json_to_schema = _safe_json_to_schema
_log_template = _safe_log_template
_compress_log_universal = _safe_compress_log_universal
