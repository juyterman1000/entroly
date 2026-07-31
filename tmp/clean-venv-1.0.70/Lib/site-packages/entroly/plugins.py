"""Third-party verifier plugins via Python entry points.

Any installed distribution may register a verifier:

    # pyproject.toml of entroly-myverifier
    [project.entry-points."entroly.verifier"]
    myverifier = "entroly_myverifier:verify"

The callable receives ``(evidence: str, claim: str)`` and returns a dict —
recommended keys: ``verdict`` ("supported" | "contradicted" | "uncertain"),
``confidence`` (float 0–1), plus anything else the plugin wants surfaced.

Trust contract (deliberate, do not weaken):

- Plugins are **additive observations**. Their results are attached to the
  verification report under ``plugins`` with full attribution; they never
  override the core fail-closed verdict, so a rogue or buggy plugin cannot
  weaken verification.
- A plugin that raises is recorded as ``{"status": "error"}`` and skipped —
  discovery and execution never break core verification.
- Installing the plugin package IS the opt-in (same model as pytest plugins);
  ``ENTROLY_PLUGINS=0`` is the kill-switch that disables discovery entirely.
- Plugins are local installed code; nothing is fetched remotely.
"""

from __future__ import annotations

import logging
import os
import time
from importlib.metadata import entry_points
from typing import Any, Callable

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "entroly.verifier"
_MAX_RESULT_KEYS = 32  # bound a plugin's surface in the report

_cache: dict[str, Callable[[str, str], Any]] | None = None


def plugins_enabled() -> bool:
    return os.environ.get("ENTROLY_PLUGINS", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def discover_verifiers(*, refresh: bool = False) -> dict[str, Callable[[str, str], Any]]:
    """Load all ``entroly.verifier`` entry points. Broken ones are skipped."""
    global _cache
    if _cache is not None and not refresh:
        return _cache
    found: dict[str, Callable[[str, str], Any]] = {}
    if plugins_enabled():
        try:
            eps = entry_points(group=ENTRY_POINT_GROUP)
        except Exception as exc:  # metadata backend failure — never fatal
            logger.warning("verifier plugin discovery failed: %s", exc)
            eps = ()
        for ep in eps:
            try:
                fn = ep.load()
                if callable(fn):
                    found[ep.name] = fn
                else:
                    logger.warning("verifier plugin %r is not callable; skipped", ep.name)
            except Exception as exc:
                logger.warning("verifier plugin %r failed to load: %s", ep.name, exc)
    _cache = found
    return found


def _sanitize(result: Any) -> dict[str, Any]:
    """Coerce a plugin result into a bounded, JSON-safe dict."""
    if not isinstance(result, dict):
        return {"status": "ok", "result": str(result)[:500]}
    out: dict[str, Any] = {}
    for i, (k, v) in enumerate(result.items()):
        if i >= _MAX_RESULT_KEYS:
            out["_truncated"] = True
            break
        key = str(k)[:64]
        if isinstance(v, (int, float, bool)) or v is None:
            out[key] = v
        else:
            out[key] = str(v)[:500]
    out.setdefault("status", "ok")
    return out


def run_verifier_plugins(evidence: str, claim: str) -> dict[str, dict[str, Any]]:
    """Run every discovered verifier; return attributed, sanitized results.

    Returns ``{}`` when no plugins are installed (the common case), so callers
    can attach the result only when non-empty.
    """
    results: dict[str, dict[str, Any]] = {}
    for name, fn in discover_verifiers().items():
        t0 = time.perf_counter()
        try:
            raw = fn(evidence, claim)
            entry = _sanitize(raw)
        except Exception as exc:
            entry = {"status": "error", "error": str(exc)[:200]}
        entry["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        results[name] = entry
    return results
