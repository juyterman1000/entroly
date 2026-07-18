"""
Entroly Value Tracker — Evidence-classified value across sessions
==================================================================

Tracks cumulative value without mixing incompatible evidence classes:
  - Provider-bound input tokens reduced and modeled API cost avoidance
  - Local-only SDK/MCP/npm token reduction with no dollar claim
  - Legacy or unknown-source history preserved as unclassified
  - Requests optimized
  - Daily/weekly/monthly aggregates for trend charts
  - Context confidence score (real-time)

Data persists to ~/.entroly/value_tracker.json and survives proxy restarts.
The dashboard reads this for trend charts; the /confidence endpoint reads
it for IDE status bar widgets.

Thread-safe: all writes go through a lock + atomic file write.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("entroly.value_tracker")

# ── Per-model pricing (USD per 1K tokens, input + output) ────────────────
#
# Output-aware: input and output tokens are priced separately (output is
# typically 3–5× input). Context-compression savings are input-priced;
# response-distillation savings are output-priced — pass
# estimate_cost(..., kind="output") for the latter.
#
# LIVE / overridable: these bundled rates are a dated snapshot. Refresh them —
# or set your negotiated rates — WITHOUT a code release by providing a local
# JSON via ENTROLY_PRICING_FILE or ~/.entroly/pricing.json:
#     {"as_of": "2026-06",
#      "default": {"input": 0.003, "output": 0.009},
#      "models": {"gpt-4o": {"input": 0.0025, "output": 0.01}}}
# Loading is local-only (no network) and fail-open to these defaults.
_PRICING_AS_OF = "2026-05"

_MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o1-pro": {"input": 0.015, "output": 0.06},
    "o3": {"input": 0.01, "output": 0.04},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    "o4-mini": {"input": 0.0011, "output": 0.0044},
    # Anthropic
    "claude-opus-4": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4": {"input": 0.003, "output": 0.015},
    "claude-haiku-4": {"input": 0.0008, "output": 0.004},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
    # Google
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-2.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
}

_DEFAULT_PRICING = {"input": 0.003, "output": 0.009}  # conservative fallback

# Backward-compat: input-only view for any legacy reader / test.
_MODEL_COSTS_PER_1K = {k: v["input"] for k, v in _MODEL_PRICING.items()}
_DEFAULT_COST_PER_1K = _DEFAULT_PRICING["input"]

# Aliases so old/new Anthropic names map to the same rate as the JS runtime.
_MODEL_ALIASES = {
    "claude-3-opus": "claude-opus-4",
    "claude-3-sonnet": "claude-sonnet-4",
    "claude-3-haiku": "claude-haiku-4",
    "claude-3.5-sonnet": "claude-3-5-sonnet",
    "claude-3.5-haiku": "claude-3-5-haiku",
}

# ── Evolution Budget Guardrail ──────────────────────────────────────────
# The evolution daemon may spend only τ% of provider-classified modeled cost
# avoidance on LLM synthesis. Local-only and legacy estimates cannot fund it.
# Budget(t) = τ · S_provider(t) − C_spent(t).
EVOLUTION_TAX_RATE = 0.05


_PRICING_CACHE: dict[str, Any] | None = None


def _pricing() -> dict[str, Any]:
    """Merged pricing: bundled defaults overlaid with an optional local override
    file (``ENTROLY_PRICING_FILE`` or ``~/.entroly/pricing.json``). Lets rates be
    refreshed or set to negotiated prices without a code release. Local-only (no
    network); fail-open to bundled defaults. Cached after first load."""
    global _PRICING_CACHE
    if _PRICING_CACHE is not None:
        return _PRICING_CACHE
    merged: dict[str, Any] = {
        "as_of": _PRICING_AS_OF,
        "source": "bundled",
        "default": dict(_DEFAULT_PRICING),
        "models": {k: dict(v) for k, v in _MODEL_PRICING.items()},
    }
    path = os.environ.get("ENTROLY_PRICING_FILE") or str(
        Path.home() / ".entroly" / "pricing.json"
    )
    try:
        p = Path(path)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            for m, rates in (data.get("models") or {}).items():
                if isinstance(rates, dict):
                    merged["models"].setdefault(str(m), {}).update(
                        {k: float(rates[k]) for k in ("input", "output") if k in rates}
                    )
            if isinstance(data.get("default"), dict):
                merged["default"].update(
                    {k: float(data["default"][k]) for k in ("input", "output") if k in data["default"]}
                )
            if data.get("as_of"):
                merged["as_of"] = str(data["as_of"])
            merged["source"] = str(p)
    except Exception as e:  # noqa: BLE001 — pricing override is best-effort
        logger.debug("pricing override load failed (%s); using bundled defaults", e)
    _PRICING_CACHE = merged
    return merged


def reset_pricing_cache() -> None:
    """Drop the cached merged pricing (e.g. after writing a new override file)."""
    global _PRICING_CACHE
    _PRICING_CACHE = None


def pricing_provenance() -> dict[str, str]:
    """Where the active rates come from — for audit/report surfaces."""
    p = _pricing()
    return {"as_of": str(p.get("as_of", _PRICING_AS_OF)), "source": str(p.get("source", "bundled"))}


def _has_priced_model(model: str) -> bool:
    """Return whether the active catalog explicitly prices ``model``."""
    if not model:
        return False
    pricing = _pricing()
    normalized = model.lower()
    for alias, canonical in _MODEL_ALIASES.items():
        if normalized.startswith(alias):
            normalized = canonical + normalized[len(alias):]
            break
    return any(
        normalized.startswith(prefix)
        for prefix in pricing.get("models", {})
    )


def estimate_cost(tokens_saved: int, model: str = "", kind: str = "input") -> float:
    """Estimate USD saved for ``tokens_saved`` tokens of a model.

    ``kind`` selects the rate: ``"input"`` (default — context-compression
    savings) or ``"output"`` (response-distillation savings). Longest-prefix
    match avoids 'gpt-4o' eating 'gpt-4o-mini'; aliases map old/new names to one
    rate. Rates come from :func:`_pricing` (bundled defaults + optional local
    override). Unknown models log a warning so the budget invariant stays honest.
    """
    pricing = _pricing()
    rates = dict(pricing["default"])
    matched = False
    if model:
        m = model.lower()
        for alias, canonical in _MODEL_ALIASES.items():
            if m.startswith(alias):
                m = canonical + m[len(alias):]
                break
        # Longest prefix wins.
        for prefix in sorted(pricing["models"].keys(), key=len, reverse=True):
            if m.startswith(prefix):
                rates = pricing["models"][prefix]
                matched = True
                break
        if not matched:
            logger.warning("unknown model %r; falling back to default pricing", model)
    rate = rates.get(kind)
    if rate is None:
        rate = rates.get("input", _DEFAULT_PRICING["input"])
    return (tokens_saved / 1000.0) * float(rate)


def _day_key(ts: float | None = None) -> str:
    """Return YYYY-MM-DD for a timestamp (or now)."""
    t = time.gmtime(ts or time.time())
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"


def _week_key(ts: float | None = None) -> str:
    """Return YYYY-WNN for a timestamp (or now)."""
    t = time.gmtime(ts or time.time())
    # ISO week number
    import datetime
    dt = datetime.date(t.tm_year, t.tm_mon, t.tm_mday)
    iso = dt.isocalendar()
    return f"{iso[0]:04d}-W{iso[1]:02d}"


def _month_key(ts: float | None = None) -> str:
    """Return YYYY-MM for a timestamp (or now)."""
    t = time.gmtime(ts or time.time())
    return f"{t.tm_year:04d}-{t.tm_mon:02d}"


class ValueTracker:
    """Persistent, thread-safe tracker for lifetime Entroly value.

    Stores cumulative stats + daily/weekly/monthly breakdowns.
    Survives proxy restarts via atomic JSON file writes.
    """

    _FILE_NAME = "value_tracker.json"
    _ACTIVITY_NAME = "activity.jsonl"
    _MAX_DAILY_ENTRIES = 90    # ~3 months of daily data
    _MAX_WEEKLY_ENTRIES = 52   # ~1 year
    _MAX_MONTHLY_ENTRIES = 24  # ~2 years
    _MAX_ACTIVITY = 200        # bounded cross-process live feed
    _SCHEMA_VERSION = 4
    _PROVIDER_SOURCES = frozenset({"provider", "proxy", "gateway"})
    _LOCAL_SOURCES = frozenset({"sdk", "npm", "mcp", "local"})

    @staticmethod
    def _default_dir() -> Path:
        """The shared telemetry directory. Honors ENTROLY_DIR so the
        Python and Node (npm) runtimes write to the SAME place — without
        this they diverge (~/.entroly vs cwd/.entroly) and no cross-mode
        dashboard ever sees data."""
        env = os.environ.get("ENTROLY_DIR")
        return Path(env) if env else (Path.home() / ".entroly")

    def __init__(self, data_dir: Path | None = None):
        self._dir = data_dir or self._default_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / self._FILE_NAME
        self._activity_path = self._dir / self._ACTIVITY_NAME
        self._lock = threading.Lock()
        self._data = self._load()
        self._activity: list[dict[str, Any]] = self._load_activity()
        # mtime fingerprints so reader processes (the dashboard) can go
        # live on writes made by OTHER processes (proxy/MCP/npm).
        self._data_mtime: float = self._mtime(self._path)
        self._activity_mtime: float = self._mtime(self._activity_path)

        # In-memory snapshot for fast reads (updated on every record)
        self._last_confidence: float = 0.0
        self._last_coverage_pct: float = 0.0
        self._session_requests: int = 0
        self._session_tokens_saved: int = 0
        self._session_cost_saved: float = 0.0
        self._session_provider_requests: int = 0
        self._session_provider_tokens_saved: int = 0
        self._session_provider_cost_avoided: float = 0.0
        self._session_local_operations: int = 0
        self._session_local_tokens_reduced: int = 0

    @staticmethod
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    def _load(self) -> dict[str, Any]:
        """Load tracker data from disk, or return fresh defaults.

        Migrates older (v2) files forward by back-filling any missing
        keys so a long-lived install never loses history on upgrade."""
        if self._path.exists():
            try:
                raw = self._path.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict) and "version" in data:
                    return self._migrate(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Value tracker load failed, starting fresh: %s", e)
        return self._defaults()

    @classmethod
    def _migrate(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate without presenting old mixed counters as provider savings."""
        base = cls._defaults()
        lt = data.setdefault("lifetime", {})
        previous_version = int(data.get("version", 0) or 0)
        if previous_version < 4:
            # v3 mixed proxy, SDK, MCP, and npm reductions in the same totals.
            # Preserve that history, but quarantine it as unclassified because
            # there is no honest way to reconstruct which operations reached a
            # paid provider after the fact.
            lt.setdefault(
                "unclassified_tokens_reduced", int(lt.get("tokens_saved", 0) or 0)
            )
            lt.setdefault(
                "unclassified_cost_estimate_usd",
                float(lt.get("cost_saved_usd", 0.0) or 0.0),
            )
            lt.setdefault(
                "unclassified_operations",
                int(lt.get("requests_optimized", 0) or 0),
            )
            # The legacy dollar field is retained as a compatibility alias,
            # but v4 narrows it to provider-classified value. The old mixed
            # estimate remains available above as unclassified history.
            lt["cost_saved_usd"] = float(
                lt.get("provider_cost_avoided_usd", 0.0) or 0.0
            )
        for k, v in base["lifetime"].items():
            lt.setdefault(k, v)
        for bucket in ("daily", "weekly", "monthly"):
            period = data.setdefault(bucket, {})
            if previous_version < 4 and isinstance(period, dict):
                for row in period.values():
                    if not isinstance(row, dict):
                        continue
                    row.setdefault(
                        "unclassified_tokens_reduced",
                        int(row.get("tokens_saved", 0) or 0),
                    )
                    row.setdefault(
                        "unclassified_cost_estimate_usd",
                        float(row.get("cost_saved", 0.0) or 0.0),
                    )
                    row.setdefault(
                        "unclassified_operations",
                        int(row.get("requests", 0) or 0),
                    )
                    row["cost_saved"] = float(
                        row.get("provider_cost_avoided_usd", 0.0) or 0.0
                    )
                    for field, default in cls._empty_period().items():
                        row.setdefault(field, default)
        data["version"] = cls._SCHEMA_VERSION
        return data

    @classmethod
    def _defaults(cls) -> dict[str, Any]:
        return {
            "version": cls._SCHEMA_VERSION,
            "lifetime": {
                "tokens_saved": 0,
                "cost_saved_usd": 0.0,
                "requests_optimized": 0,
                "requests_total": 0,
                "duplicates_caught": 0,
                # v4: evidence classes. Only provider-bound requests can
                # produce a public dollar-cost-avoidance claim.
                "provider_tokens_saved": 0,
                "provider_cost_avoided_usd": 0.0,
                "provider_requests": 0,
                "provider_requests_optimized": 0,
                "provider_unpriced_tokens": 0,
                "provider_unpriced_requests": 0,
                "local_tokens_reduced": 0,
                "local_operations": 0,
                "unclassified_tokens_reduced": 0,
                "unclassified_cost_estimate_usd": 0.0,
                "unclassified_operations": 0,
                "first_seen": time.time(),
                "last_seen": time.time(),
                # Evolution budget accounting (Pillar 1)
                "evolution_spent_usd": 0.0,
                "evolution_attempts": 0,
                "evolution_successes": 0,
                # v3: hallucination + model-routing value (WITNESS / RAVS)
                "hallucinations_blocked": 0,
                "routing_saved_usd": 0.0,
                "routing_decisions": 0,
                # Belief-conditioned compression (H(X | beliefs))
                "beliefs_conditioned_fragments": 0,
                "belief_conditioning_passes": 0,
            },
            "daily": {},    # "YYYY-MM-DD" -> {tokens_saved, cost_saved, requests}
            "weekly": {},   # "YYYY-WNN" -> {tokens_saved, cost_saved, requests}
            "monthly": {},  # "YYYY-MM" -> {tokens_saved, cost_saved, requests}
        }

    def _load_activity(self) -> list[dict[str, Any]]:
        """Load the bounded cross-process activity feed (JSONL)."""
        if not self._activity_path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            for line in self._activity_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError as e:
            logger.debug("activity load failed: %s", e)
        # Enforce time-based retention on disk telemetry (privacy-by-default).
        try:
            from entroly.privacy import retention_cutoff_ts

            cutoff = retention_cutoff_ts()
            if cutoff is not None:
                out = [
                    row for row in out
                    if float(row.get("ts", 0) or 0) >= cutoff
                ]
        except Exception:
            pass

        return out[-self._MAX_ACTIVITY:]

    def _save(self) -> None:
        """Atomic write: write to temp file then rename (no partial writes)."""
        try:
            content = json.dumps(self._data, indent=2)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._dir), suffix=".tmp", prefix="vt_"
            )
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                # Atomic rename (POSIX) or replace (Windows)
                if os.name == "nt":
                    # Windows: os.replace is atomic
                    os.replace(tmp_path, str(self._path))
                else:
                    os.rename(tmp_path, str(self._path))
            except Exception:
                os.close(fd) if not os.get_inheritable(fd) else None
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except OSError as e:
            logger.debug("Value tracker save failed: %s", e)
        self._data_mtime = self._mtime(self._path)

    def _save_activity(self) -> None:
        """Atomically rewrite the bounded activity JSONL (≤200 lines, so
        whole-file rewrite is cheap and crash-safe via temp+rename)."""
        try:
            # Enforce time-based retention before size bounding.
            try:
                from entroly.privacy import retention_cutoff_ts

                cutoff = retention_cutoff_ts()
                if cutoff is not None:
                    self._activity = [
                        row for row in self._activity
                        if float(row.get("ts", 0) or 0) >= cutoff
                    ]
            except Exception:
                pass

            self._activity = self._activity[-self._MAX_ACTIVITY:]
            content = "\n".join(json.dumps(e, separators=(",", ":"))
                                for e in self._activity)
            if content:
                content += "\n"
            fd, tmp = tempfile.mkstemp(dir=str(self._dir),
                                       suffix=".tmp", prefix="act_")
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                os.replace(tmp, str(self._activity_path))
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        except OSError as e:
            logger.debug("activity save failed: %s", e)
        self._activity_mtime = self._mtime(self._activity_path)

    def reload_if_changed(self) -> bool:
        """Re-read disk state IF another process advanced the files.

        This is the fix for the cross-process dashboard: the writer
        (proxy/MCP/npm) and the reader (`entroly dashboard`) are usually
        different processes, and the singleton otherwise froze its
        in-memory copy at startup. Readers call this each poll; it is a
        no-op (one cheap stat) when nothing changed. Per-process session
        counters are intentionally preserved (they are not on disk)."""
        changed = False
        with self._lock:
            dm = self._mtime(self._path)
            if dm > self._data_mtime:
                self._data = self._load()
                self._data_mtime = dm
                changed = True
            am = self._mtime(self._activity_path)
            if am > self._activity_mtime:
                self._activity = self._load_activity()
                self._activity_mtime = am
                changed = True
        return changed

    def record_event(
        self,
        kind: str,
        summary: str,
        *,
        source: str = "",
        tokens_saved: int = 0,
        cost_saved_usd: float = 0.0,
        model: str = "",
        **extra: Any,
    ) -> None:
        """Append one row to the persistent, cross-process activity feed.

        `kind` is a short tag the dashboard groups on: "optimize",
        "hallucination", "routing", "compress". Fail-open: telemetry must
        never break the caller."""
        try:
            row: dict[str, Any] = {
                "ts": round(time.time(), 3),
                "kind": str(kind),
                "summary": str(summary)[:240],
            }
            if source:
                row["source"] = source
            if tokens_saved:
                row["tokens_saved"] = int(tokens_saved)
            if cost_saved_usd:
                row["cost_saved_usd"] = round(float(cost_saved_usd), 6)
            if model:
                row["model"] = model
            for k, v in extra.items():
                if isinstance(v, (str, int, float, bool)):
                    row[k] = v
            with self._lock:
                self._activity.append(row)
                self._save_activity()
        except Exception as e:  # noqa: BLE001
            logger.debug("record_event failed (non-fatal): %s", e)

    def record_hallucination_blocked(
        self, n: int = 1, *, source: str = "", detail: str = ""
    ) -> None:
        """WITNESS suppressed `n` unsupported claims before they reached
        the user. Fail-open."""
        try:
            with self._lock:
                self._data["lifetime"]["hallucinations_blocked"] = (
                    self._data["lifetime"].get("hallucinations_blocked", 0)
                    + int(n)
                )
                self._save()
            self.record_event(
                "hallucination",
                detail or f"Blocked {n} unsupported claim(s)",
                source=source, blocked=int(n),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("record_hallucination_blocked failed: %s", e)

    def record_routing_saving(
        self, cost_saved_usd: float, *, source: str = "",
        chosen_model: str = "", detail: str = "",
    ) -> None:
        """RAVS routed to a cheaper capable model; record the $ avoided.
        Fail-open."""
        try:
            with self._lock:
                lt = self._data["lifetime"]
                lt["routing_saved_usd"] = round(
                    lt.get("routing_saved_usd", 0.0)
                    + float(cost_saved_usd), 6)
                lt["routing_decisions"] = lt.get("routing_decisions", 0) + 1
                self._save()
            self.record_event(
                "routing",
                detail or f"Routed to {chosen_model or 'cheaper model'}",
                source=source, cost_saved_usd=float(cost_saved_usd),
                model=chosen_model,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("record_routing_saving failed: %s", e)

    def record_belief_conditioning(
        self, n_fragments: int, *, source: str = "", detail: str = ""
    ) -> None:
        """Belief-conditioned compression discounted `n_fragments` candidate
        fragments that merely restated already-known vault beliefs
        (H(X | beliefs)), freeing budget for novel content. Fail-open."""
        try:
            if n_fragments <= 0:
                return
            with self._lock:
                lt = self._data["lifetime"]
                lt["beliefs_conditioned_fragments"] = (
                    lt.get("beliefs_conditioned_fragments", 0) + int(n_fragments)
                )
                lt["belief_conditioning_passes"] = (
                    lt.get("belief_conditioning_passes", 0) + 1
                )
                self._save()
            self.record_event(
                "belief_conditioning",
                detail or f"Discounted {n_fragments} fragment(s) restating known beliefs",
                source=source, fragments_discounted=int(n_fragments),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("record_belief_conditioning failed: %s", e)

    def get_activity(self, last_n: int = 50) -> list[dict[str, Any]]:
        """Most-recent-first slice of the cross-process activity feed."""
        with self._lock:
            return list(reversed(self._activity[-last_n:]))

    def _trim_history(self) -> None:
        """Keep history within size limits."""
        for key, limit in [
            ("daily", self._MAX_DAILY_ENTRIES),
            ("weekly", self._MAX_WEEKLY_ENTRIES),
            ("monthly", self._MAX_MONTHLY_ENTRIES),
        ]:
            bucket = self._data.get(key, {})
            if len(bucket) > limit:
                sorted_keys = sorted(bucket.keys())
                for old_key in sorted_keys[: len(sorted_keys) - limit]:
                    del bucket[old_key]

    @classmethod
    def _channel(cls, source: str) -> str:
        normalized = (source or "unclassified").strip().lower()
        if normalized in cls._PROVIDER_SOURCES:
            return "provider"
        if normalized in cls._LOCAL_SOURCES:
            return "local"
        return "unclassified"

    @staticmethod
    def _empty_period() -> dict[str, int | float]:
        return {
            "tokens_saved": 0,
            "cost_saved": 0.0,
            "requests": 0,
            "provider_tokens_saved": 0,
            "provider_cost_avoided_usd": 0.0,
            "provider_requests": 0,
            "provider_requests_optimized": 0,
            "provider_unpriced_tokens": 0,
            "provider_unpriced_requests": 0,
            "local_tokens_reduced": 0,
            "local_operations": 0,
            "unclassified_tokens_reduced": 0,
            "unclassified_cost_estimate_usd": 0.0,
            "unclassified_operations": 0,
        }

    def _record_period(
        self,
        bucket_name: str,
        key: str,
        *,
        tokens: int,
        cost: float,
        channel: str,
        optimized: bool,
        provider_priced: bool,
    ) -> None:
        bucket = self._data.setdefault(bucket_name, {})
        row = bucket.setdefault(key, self._empty_period())
        for field, default in self._empty_period().items():
            row.setdefault(field, default)

        row["tokens_saved"] += tokens
        row["requests"] += 1
        if channel == "provider":
            row["cost_saved"] = round(float(row["cost_saved"]) + cost, 6)
            row["provider_tokens_saved"] += tokens
            row["provider_cost_avoided_usd"] = round(
                float(row["provider_cost_avoided_usd"]) + cost, 6
            )
            row["provider_requests"] += 1
            if optimized:
                row["provider_requests_optimized"] += 1
            if not provider_priced:
                row["provider_unpriced_tokens"] += tokens
                row["provider_unpriced_requests"] += 1
        elif channel == "local":
            row["local_tokens_reduced"] += tokens
            row["local_operations"] += 1
        else:
            row["unclassified_tokens_reduced"] += tokens
            row["unclassified_cost_estimate_usd"] = round(
                float(row["unclassified_cost_estimate_usd"]) + cost, 6
            )
            row["unclassified_operations"] += 1

    def record(
        self,
        tokens_saved: int,
        model: str = "",
        duplicates: int = 0,
        optimized: bool = True,
        coverage_pct: float = 0.0,
        confidence: float = 0.0,
        source: str = "unclassified",
    ) -> None:
        """Record an optimization without overstating its economic evidence.

        ``source="proxy"`` records a provider-bound request whose pre/post
        token counts may support modeled API input-cost avoidance. SDK, npm,
        MCP, and local operations record token reduction only because the
        tracker cannot prove their output was sent to a paid provider.
        """
        tokens_saved = max(0, int(tokens_saved))
        channel = self._channel(source)
        estimated_cost = estimate_cost(tokens_saved, model)
        provider_priced = channel != "provider" or _has_priced_model(model)
        cost = estimated_cost if provider_priced else 0.0
        now = time.time()
        day = _day_key(now)
        week = _week_key(now)
        month = _month_key(now)

        with self._lock:
            lt = self._data["lifetime"]
            lt["tokens_saved"] += tokens_saved
            lt["requests_total"] += 1
            if optimized:
                lt["requests_optimized"] += 1
            lt["duplicates_caught"] += duplicates
            lt["last_seen"] = now

            if channel == "provider":
                lt["cost_saved_usd"] = round(
                    lt["cost_saved_usd"] + cost, 6
                )
                lt["provider_tokens_saved"] += tokens_saved
                lt["provider_cost_avoided_usd"] = round(
                    lt["provider_cost_avoided_usd"] + cost, 6
                )
                lt["provider_requests"] += 1
                if optimized:
                    lt["provider_requests_optimized"] += 1
                if not provider_priced:
                    lt["provider_unpriced_tokens"] += tokens_saved
                    lt["provider_unpriced_requests"] += 1
            elif channel == "local":
                lt["local_tokens_reduced"] += tokens_saved
                lt["local_operations"] += 1
            else:
                lt["unclassified_tokens_reduced"] += tokens_saved
                lt["unclassified_cost_estimate_usd"] = round(
                    lt["unclassified_cost_estimate_usd"] + estimated_cost, 6
                )
                lt["unclassified_operations"] += 1

            self._record_period(
                "daily", day, tokens=tokens_saved, cost=cost,
                channel=channel, optimized=optimized,
                provider_priced=provider_priced,
            )
            self._record_period(
                "weekly", week, tokens=tokens_saved, cost=cost,
                channel=channel, optimized=optimized,
                provider_priced=provider_priced,
            )
            self._record_period(
                "monthly", month, tokens=tokens_saved, cost=cost,
                channel=channel, optimized=optimized,
                provider_priced=provider_priced,
            )

            # Session counters (in-memory only)
            self._session_requests += 1
            self._session_tokens_saved += tokens_saved
            if channel == "provider":
                self._session_cost_saved += cost
                self._session_provider_requests += 1
                self._session_provider_tokens_saved += tokens_saved
                self._session_provider_cost_avoided += cost
            elif channel == "local":
                self._session_local_operations += 1
                self._session_local_tokens_reduced += tokens_saved
            self._last_confidence = confidence
            self._last_coverage_pct = coverage_pct

            self._trim_history()
            self._save()

            # Light up the cross-process live feed for free. Append
            # in-lock to keep ordering; persist outside the heavy path.
            self._activity.append({
                "ts": round(now, 3),
                "kind": "optimize",
                "summary": (f"Optimized request: saved {tokens_saved:,} "
                            f"tokens" + (f" ({model})" if model else "")),
                "tokens_saved": int(tokens_saved),
                "cost_saved_usd": round(cost if channel == "provider" else 0.0, 6),
                "modeled_cost_avoided_usd": round(
                    cost if channel == "provider" else 0.0, 6
                ),
                "model": model or "",
                "duplicates": int(duplicates),
                "source": source or "unclassified",
                "measurement_channel": channel,
            })
            self._save_activity()

    def get_lifetime(self) -> dict[str, Any]:
        """Return lifetime cumulative stats."""
        with self._lock:
            return dict(self._data.get("lifetime", {}))

    def get_daily(self, last_n: int = 30) -> list[dict[str, Any]]:
        """Return last N days of daily stats, sorted ascending."""
        with self._lock:
            d = self._data.get("daily", {})
            keys = sorted(d.keys())[-last_n:]
            return [{"date": k, **d[k]} for k in keys]

    def get_weekly(self, last_n: int = 12) -> list[dict[str, Any]]:
        """Return last N weeks of stats, sorted ascending."""
        with self._lock:
            w = self._data.get("weekly", {})
            keys = sorted(w.keys())[-last_n:]
            return [{"week": k, **w[k]} for k in keys]

    def get_monthly(self, last_n: int = 12) -> list[dict[str, Any]]:
        """Return last N months of stats, sorted ascending."""
        with self._lock:
            m = self._data.get("monthly", {})
            keys = sorted(m.keys())[-last_n:]
            return [{"month": k, **m[k]} for k in keys]

    def get_session(self) -> dict[str, Any]:
        """Return current session stats (since proxy start)."""
        with self._lock:
            return {
                "requests": self._session_requests,
                "tokens_saved": self._session_tokens_saved,
                "cost_saved_usd": round(self._session_cost_saved, 4),
                "provider_requests": self._session_provider_requests,
                "provider_tokens_saved": self._session_provider_tokens_saved,
                "provider_cost_avoided_usd": round(
                    self._session_provider_cost_avoided, 4
                ),
                "local_operations": self._session_local_operations,
                "local_tokens_reduced": self._session_local_tokens_reduced,
            }

    def get_confidence(self) -> dict[str, Any]:
        """Return real-time confidence snapshot for IDE widgets.

        This is the single endpoint an IDE status bar polls.
        """
        with self._lock:
            lt = self._data.get("lifetime", {})
            today = _day_key()
            today_data = self._data.get("daily", {}).get(today, {})
            return {
                "confidence": round(self._last_confidence, 4),
                "coverage_pct": round(self._last_coverage_pct, 2),
                "session": {
                    "requests": self._session_provider_requests,
                    "tokens_saved": self._session_provider_tokens_saved,
                    "cost_saved_usd": round(
                        self._session_provider_cost_avoided, 4
                    ),
                    "local_operations": self._session_local_operations,
                    "local_tokens_reduced": self._session_local_tokens_reduced,
                },
                "today": {
                    "tokens_saved": today_data.get("provider_tokens_saved", 0),
                    "cost_saved_usd": today_data.get(
                        "provider_cost_avoided_usd", 0.0
                    ),
                    "requests": today_data.get(
                        "provider_requests_optimized", 0
                    ),
                    "local_tokens_reduced": today_data.get(
                        "local_tokens_reduced", 0
                    ),
                    "local_operations": today_data.get("local_operations", 0),
                },
                "lifetime": {
                    "tokens_saved": lt.get("provider_tokens_saved", 0),
                    "cost_saved_usd": lt.get(
                        "provider_cost_avoided_usd", 0.0
                    ),
                    "requests_optimized": lt.get(
                        "provider_requests_optimized", 0
                    ),
                    "provider_requests": lt.get("provider_requests", 0),
                    "provider_unpriced_tokens": lt.get(
                        "provider_unpriced_tokens", 0
                    ),
                    "provider_unpriced_requests": lt.get(
                        "provider_unpriced_requests", 0
                    ),
                    "local_tokens_reduced": lt.get("local_tokens_reduced", 0),
                    "local_operations": lt.get("local_operations", 0),
                    "hallucinations_blocked": lt.get(
                        "hallucinations_blocked", 0),
                    "routing_saved_usd": lt.get("routing_saved_usd", 0.0),
                },
                "status": "active" if self._session_requests > 0 else "idle",
            }

    def get_trends(self) -> dict[str, Any]:
        """Return all trend data for dashboard charts."""
        return {
            "daily": self.get_daily(30),
            "weekly": self.get_weekly(12),
            "monthly": self.get_monthly(12),
            "lifetime": self.get_lifetime(),
            "session": self.get_session(),
            "activity": self.get_activity(50),
        }

    def get_value_receipt(self) -> dict[str, Any]:
        """Return a machine-readable, evidence-classified value summary."""
        self.reload_if_changed()
        lifetime = self.get_lifetime()
        daily = self.get_daily(90)
        provider_days = sum(
            1 for row in daily if int(row.get("provider_requests", 0) or 0) > 0
        )
        local_days = sum(
            1 for row in daily if int(row.get("local_operations", 0) or 0) > 0
        )
        pricing = pricing_provenance()
        return {
            "schema_version": "entroly.value-receipt.v1",
            "provider_path": {
                "requests_observed": int(
                    lifetime.get("provider_requests", 0) or 0
                ),
                "requests_optimized": int(
                    lifetime.get("provider_requests_optimized", 0) or 0
                ),
                "input_tokens_reduced": int(
                    lifetime.get("provider_tokens_saved", 0) or 0
                ),
                "modeled_input_cost_avoided_usd": round(
                    float(lifetime.get("provider_cost_avoided_usd", 0.0) or 0.0),
                    6,
                ),
                "active_days": provider_days,
                "unpriced_requests": int(
                    lifetime.get("provider_unpriced_requests", 0) or 0
                ),
                "unpriced_input_tokens": int(
                    lifetime.get("provider_unpriced_tokens", 0) or 0
                ),
                "evidence": (
                    "Pre/post local token counts on requests Entroly handled "
                    "on a provider-bound proxy path. Dollar value applies the "
                    "recorded model's configured input rate; it is not an invoice. "
                    "Requests without an explicit catalog match remain unpriced."
                ),
            },
            "local_operations": {
                "operations": int(lifetime.get("local_operations", 0) or 0),
                "tokens_reduced": int(
                    lifetime.get("local_tokens_reduced", 0) or 0
                ),
                "active_days": local_days,
                "dollar_claimed_usd": 0.0,
                "evidence": (
                    "SDK, MCP, npm, and other local reductions. Entroly does not "
                    "claim dollar savings because it cannot prove the output was "
                    "sent to a paid provider."
                ),
            },
            "legacy_unclassified": {
                "operations": int(
                    lifetime.get("unclassified_operations", 0) or 0
                ),
                "tokens_reduced": int(
                    lifetime.get("unclassified_tokens_reduced", 0) or 0
                ),
                "historical_cost_estimate_usd": round(
                    float(
                        lifetime.get("unclassified_cost_estimate_usd", 0.0) or 0.0
                    ),
                    6,
                ),
                "dollar_claimed_usd": 0.0,
                "evidence": (
                    "History recorded before source classification or by an "
                    "unknown caller. Preserved, but excluded from public savings."
                ),
            },
            "trust_signals": {
                "unsupported_claims_blocked": int(
                    lifetime.get("hallucinations_blocked", 0) or 0
                ),
                "routing_decisions": int(
                    lifetime.get("routing_decisions", 0) or 0
                ),
                "modeled_routing_cost_avoided_usd": round(
                    float(lifetime.get("routing_saved_usd", 0.0) or 0.0), 6
                ),
            },
            "pricing": pricing,
            "generated_at_unix": round(time.time(), 3),
        }

    # ── Evolution Budget Guardrail (Pillar 1) ─────────────────────────────

    def get_evolution_budget(self) -> dict[str, Any]:
        """Return the available evolution budget.

        Budget = τ · provider_classified_cost_avoidance − total_spent
        The system can spend at most τ% (5%) of provider-classified modeled
        cost avoidance on LLM-based skill synthesis. Local-only and legacy
        estimates cannot fund it.

        Returns:
            {
                "available_usd": float,   # remaining evolution budget
                "total_earned_usd": float, # τ · provider cost avoidance
                "total_spent_usd": float,  # already debited
                "can_evolve": bool,        # available > 0
                "tax_rate": float,         # τ
            }
        """
        with self._lock:
            lt = self._data.get("lifetime", {})
            lifetime_saved = lt.get("provider_cost_avoided_usd", 0.0)
            total_spent = lt.get("evolution_spent_usd", 0.0)
            total_earned = lifetime_saved * EVOLUTION_TAX_RATE
            available = max(0.0, total_earned - total_spent)

            return {
                "available_usd": round(available, 6),
                "total_earned_usd": round(total_earned, 6),
                "total_spent_usd": round(total_spent, 6),
                "can_evolve": available > 0.001,  # > 0.1 cent floor
                "tax_rate": EVOLUTION_TAX_RATE,
            }

    def record_evolution_spend(
        self,
        cost_usd: float,
        success: bool = False,
    ) -> dict[str, Any]:
        """Debit an evolution attempt from the budget.

        Called by the evolution daemon after an LLM-based synthesis attempt.
        Only succeeds if the budget allows it (fail-safe: checks again here).

        Args:
            cost_usd: Cost of this evolution attempt.
            success: Whether the synthesized skill passed benchmarks.

        Returns:
            {"status": "recorded" | "rejected", "remaining_usd": float}
        """
        with self._lock:
            lt = self._data.get("lifetime", {})
            lifetime_saved = lt.get("provider_cost_avoided_usd", 0.0)
            current_spent = lt.get("evolution_spent_usd", 0.0)
            total_earned = lifetime_saved * EVOLUTION_TAX_RATE
            available = total_earned - current_spent

            if cost_usd > available + 0.001:  # 0.1 cent tolerance
                logger.warning(
                    "Evolution spend rejected: $%.4f requested, $%.4f available",
                    cost_usd, available,
                )
                return {
                    "status": "rejected",
                    "remaining_usd": round(max(0.0, available), 6),
                }

            lt["evolution_spent_usd"] = round(current_spent + cost_usd, 6)
            lt["evolution_attempts"] = lt.get("evolution_attempts", 0) + 1
            if success:
                lt["evolution_successes"] = lt.get("evolution_successes", 0) + 1

            self._save()

            remaining = max(0.0, total_earned - lt["evolution_spent_usd"])
            logger.info(
                "Evolution spend recorded: $%.4f (remaining: $%.4f, success=%s)",
                cost_usd, remaining, success,
            )
            return {
                "status": "recorded",
                "remaining_usd": round(remaining, 6),
            }


# ── Module-level singleton (lazy-init) ───────────────────────────────────

_tracker: ValueTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> ValueTracker:
    """Get or create the global ValueTracker singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = ValueTracker()
    return _tracker
