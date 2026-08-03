"""Compatibility wrapper for the universal compressor trust hotfix.

Mutable public helpers are synchronized before each entry-point call so tests
and integrations that patch module-level compression hooks keep working.
"""

from __future__ import annotations

from typing import Any

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

_json_to_schema = _safe_json_to_schema
_log_template = _safe_log_template
_compress_log_universal = _safe_compress_log_universal


def _sync_public_hooks() -> None:
    _legacy._json_to_schema = globals()["_json_to_schema"]
    _legacy._log_template = globals()["_log_template"]
    _legacy._compress_log_universal = globals()["_compress_log_universal"]
    _legacy._is_load_bearing_key = globals()["_is_load_bearing_key"]
    _legacy._normalise_json_key = globals()["_normalise_json_key"]


def universal_compress(
    content: str,
    target_ratio: float = 0.3,
    content_type: str | None = None,
    **kwargs: Any,
):
    _sync_public_hooks()
    return _legacy.universal_compress(
        content,
        target_ratio,
        content_type,
        **kwargs,
    )


_sync_public_hooks()
