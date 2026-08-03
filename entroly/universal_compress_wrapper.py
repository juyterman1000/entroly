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

# Preserve the complete established API, including private helpers used by
# codecs and compatibility tests. Dunder module identity is intentionally not
# copied.
for _name, _value in vars(_legacy).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

# Install corrections into the legacy module's globals. Existing function
# objects such as universal_compress() resolve their helper names there.
_legacy._json_to_schema = _safe_json_to_schema
_legacy._log_template = _safe_log_template
_legacy._compress_log_universal = _safe_compress_log_universal
_legacy._is_load_bearing_key = _is_load_bearing_key
_legacy._normalise_json_key = _normalise_json_key

# Export corrected helpers from the public module as well.
_json_to_schema = _safe_json_to_schema
_log_template = _safe_log_template
_compress_log_universal = _safe_compress_log_universal
